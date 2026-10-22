"""HTTP endpoints for camera PPG (include with ``app.include_router(camera_api.router)``).

* ``POST /camera/video`` - tier "video": multipart upload of a stock-camera recording.
* ``POST /camera/trace`` - tier "capture": JSON per-frame RGB means from the browser capture page.

Everything received is stored under ``data/camera/<UTC stamp>-<mode>[-trace]/`` (personal health
data; ``data/`` is gitignored) next to the ``result.json`` that is also returned.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.datastructures import UploadFile as StarletteUploadFile

from . import camera

router = APIRouter(prefix="/camera", tags=["camera"])

CAMERA_DATA = Path(__file__).resolve().parent.parent / "data" / "camera"
MAX_VIDEO_BYTES = 500 * 1024 * 1024
MAX_TRACE_BYTES = 32 * 1024 * 1024
MAX_TRACE_SAMPLES = 200_000  # ~55 min at 60 fps
# capture.js puts two alternative timestamp arrays in meta: a 120 s capture at 30 fps is ~72 KB,
# at 60 fps ~145 KB, so 64 KB rejected the page's own 120 s option. 1 MB still bounds it.
MAX_META_BYTES = 1024 * 1024
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".3gp", ".mkv", ".webm", ".avi"}
CHUNK = 1024 * 1024


def _new_dir(suffix: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = CAMERA_DATA / f"{stamp}-{suffix}"
    for i in range(1000):
        d = base if i == 0 else base.with_name(f"{base.name}-{i}")
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise HTTPException(500, "could not allocate a storage directory")


def _ext(filename: str | None, content_type: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext in VIDEO_EXTS:
        return ext
    by_type = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
               "video/3gpp": ".3gp", "video/x-matroska": ".mkv"}
    return by_type.get((content_type or "").split(";")[0].strip().lower(), ".mp4")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, allow_nan=False))


_VIDEO_OPENAPI = {"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object", "required": ["file", "mode"],
    "properties": {"file": {"type": "string", "format": "binary", "description": "video, max 500 MB"},
                   "mode": {"type": "string", "enum": list(camera.MODES)},
                   "deep": {"type": "boolean", "default": True,
                            "description": "face mode: also run open-rppg models"}}}}}}}


def _form_bool(v: Any, default: bool) -> bool:
    if v is None or v == "":
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise HTTPException(422, f"deep must be a boolean, got {v!r}")


@router.post("/video", openapi_extra=_VIDEO_OPENAPI)
async def camera_video(request: Request) -> dict:
    """Analyse an uploaded video (tier "video"). Multipart fields: ``file``, ``mode``, optional ``deep``.

    The body is parsed by hand (not via ``UploadFile`` parameters) so oversized uploads are
    refused from the Content-Length header before anything is read.
    """
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > MAX_VIDEO_BYTES + 1024 * 1024):
        raise HTTPException(413, f"upload larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
    if not request.headers.get("content-type", "").startswith("multipart/form-data"):
        raise HTTPException(415, "expected multipart/form-data with fields file and mode")
    async with request.form(max_files=1, max_fields=8) as form:
        mode = form.get("mode")
        file = form.get("file")
        deep = _form_bool(form.get("deep"), True)
        if mode not in camera.MODES:
            raise HTTPException(422, f"mode must be one of {list(camera.MODES)}")
        if not isinstance(file, StarletteUploadFile):
            raise HTTPException(422, "missing file field")
        ctype = file.content_type or ""
        if ctype and not (ctype.startswith("video/") or ctype == "application/octet-stream"):
            raise HTTPException(415, f"expected a video, got {ctype}")
        out_dir = _new_dir(mode)
        video = out_dir / f"video{_ext(file.filename, ctype)}"
        size = 0
        try:
            with video.open("wb") as fh:
                while chunk := await file.read(CHUNK):
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise HTTPException(413, f"upload larger than {MAX_VIDEO_BYTES // (1024 * 1024)} MB")
                    fh.write(chunk)
            if size == 0:
                raise HTTPException(422, "empty file")
        except HTTPException:
            video.unlink(missing_ok=True)
            _cleanup(out_dir)
            raise
        upload = {"filename": file.filename, "content_type": ctype, "bytes": size,
                  "stored_as": str(video.relative_to(CAMERA_DATA))}
    try:
        result = await anyio.to_thread.run_sync(
            lambda: camera.analyze_video(video, mode, tier="video", deep=deep, trace_out=out_dir / "trace.json"))
    except Exception as exc:  # undecodable file, too few frames, ...
        err = {"tier": "video", "mode": mode, "error": f"{type(exc).__name__}: {exc}", "upload": upload,
               "received_utc": datetime.now(UTC).isoformat(), "id": out_dir.name}
        _write_json(out_dir / "result.json", err)
        raise HTTPException(422, err["error"]) from exc
    result["upload"] = upload
    result["received_utc"] = datetime.now(UTC).isoformat()
    result["id"] = out_dir.name
    _write_json(out_dir / "result.json", result)
    return result


def _cleanup(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass


class TraceIn(BaseModel):
    mode: Literal["finger", "face"]
    t_ms: list[float] = Field(min_length=10, max_length=MAX_TRACE_SAMPLES)
    r: list[float] = Field(min_length=10, max_length=MAX_TRACE_SAMPLES)
    g: list[float] = Field(min_length=10, max_length=MAX_TRACE_SAMPLES)
    b: list[float] = Field(min_length=10, max_length=MAX_TRACE_SAMPLES)
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("t_ms", "r", "g", "b")
    @classmethod
    def _finite(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("values must be finite numbers")
        return v

    @field_validator("r", "g", "b")
    @classmethod
    def _range(cls, v: list[float]) -> list[float]:
        if min(v) < 0 or max(v) > 255:
            raise ValueError("channel means must be within 0..255")
        return v

    @field_validator("meta")
    @classmethod
    def _small_meta(cls, v: dict) -> dict:
        if len(json.dumps(v, default=str)) > MAX_META_BYTES:
            raise ValueError(f"meta larger than {MAX_META_BYTES} bytes")
        return v


_CLIP_KEY = re.compile(r"^clipped_fraction(_[rgb]|_any)?$")


def _clip_from_meta(meta: dict) -> float | dict | None:
    """Saturation info the capture page may send, as share of ROI pixels with any channel >= 250:
    ``meta.saturation.mean_fraction`` (what capture.js sends) or ``clipped_fraction[_any|_r|_g|_b]``."""
    def frac(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1

    vals = {k: v for k, v in meta.items() if _CLIP_KEY.match(k) and frac(v)}
    sat = meta.get("saturation")
    if not vals and isinstance(sat, dict) and frac(sat.get("mean_fraction")):
        return {"any": float(sat["mean_fraction"])}
    if not vals:
        return None
    out = {"any": vals.get("clipped_fraction_any", vals.get("clipped_fraction"))}
    for c in "rgb":
        if f"clipped_fraction_{c}" in vals:
            out[c] = vals[f"clipped_fraction_{c}"]
    return out


@router.post("/trace", openapi_extra={"requestBody": {"required": True, "content": {
    "application/json": {"schema": TraceIn.model_json_schema()}}}})
async def camera_trace(request: Request) -> dict:
    """Analyse an RGB trace from the capture page (tier "capture").

    Body: ``{"mode": "finger"|"face", "t_ms": [...], "r": [...], "g": [...], "b": [...], "meta": {...}}``.
    """
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_TRACE_BYTES:
        raise HTTPException(413, f"body larger than {MAX_TRACE_BYTES // (1024 * 1024)} MB")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_TRACE_BYTES:
            raise HTTPException(413, f"body larger than {MAX_TRACE_BYTES // (1024 * 1024)} MB")
    try:
        tr = TraceIn.model_validate_json(bytes(body))
    except ValidationError as exc:
        raise HTTPException(422, json.loads(exc.json(include_url=False, include_input=False))) from exc
    if not (len(tr.t_ms) == len(tr.r) == len(tr.g) == len(tr.b)):
        raise HTTPException(422, "t_ms, r, g, b must have equal length")
    span_s = (max(tr.t_ms) - min(tr.t_ms)) / 1000
    if span_s > 3600:
        raise HTTPException(422, "trace longer than 1 h")
    out_dir = _new_dir(f"{tr.mode}-trace")
    _write_json(out_dir / "trace.json", camera._jsonable(tr.model_dump()))
    t_s = [x / 1000.0 for x in tr.t_ms]
    try:
        result = await anyio.to_thread.run_sync(
            lambda: camera.analyze_trace(t_s, tr.r, tr.g, tr.b, tr.mode, "capture",
                                         clipped_fraction=_clip_from_meta(tr.meta)))
    except ValueError as exc:
        err = {"tier": "capture", "mode": tr.mode, "error": str(exc), "meta": camera._jsonable(tr.meta),
               "received_utc": datetime.now(UTC).isoformat()}
        _write_json(out_dir / "result.json", err)
        raise HTTPException(422, str(exc)) from exc
    result["meta"] = camera._jsonable(tr.meta)
    result["received_utc"] = datetime.now(UTC).isoformat()
    result["id"] = out_dir.name
    _write_json(out_dir / "result.json", result)
    return result
