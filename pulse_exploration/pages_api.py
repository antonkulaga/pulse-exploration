"""Phone-facing web pages for the two phone tiers.

- ``GET /``        index: explains the tiers and links both pages.
- ``GET /upload``  tier "video": record with the stock camera app, upload the file (plain HTTP is fine).
- ``GET /capture`` tier "capture": the page drives the camera (torch, locked exposure) and sends only
  per-frame colour means; getUserMedia needs HTTPS, see scripts/make_cert.py.
- ``GET /static/{name}`` the page assets (flat directory, no subfolders).

The pages call the camera API on the same origin: ``POST /camera/video`` (multipart ``file`` + ``mode``)
and ``POST /camera/trace`` (JSON ``{mode, t_ms, r, g, b, meta}``). This module only serves files; the
orchestrator includes ``router`` in the main app.
"""

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

STATIC = Path(__file__).resolve().parent / "static"

# Pages are tiny and change often while prototyping: let the phone revalidate on every load.
_HEADERS = {"Cache-Control": "no-cache"}
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".json": "application/json",
}
# One path segment, no leading dot: rules out "..", hidden files and subdirectories.
_SAFE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*$")

router = APIRouter(tags=["pages"])


def _serve(name: str) -> FileResponse:
    path = STATIC / name
    media_type = _TYPES.get(path.suffix)
    if media_type is None or not path.is_file() or path.resolve().parent != STATIC:
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type=media_type, headers=_HEADERS)


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return _serve("index.html")


@router.get("/upload", include_in_schema=False)
def upload_page() -> FileResponse:
    return _serve("upload.html")


@router.get("/capture", include_in_schema=False)
def capture_page() -> FileResponse:
    return _serve("capture.html")


@router.get("/static/{name}", include_in_schema=False)
def static_file(name: str) -> FileResponse:
    if not _SAFE.match(name):
        raise HTTPException(404, "not found")
    return _serve(name)
