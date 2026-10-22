"""Camera photoplethysmography (PPG) from phone video or browser-captured RGB traces.

Two measurement modes:

* ``finger`` - contact PPG: fingertip pressed on the rear camera with the torch on. The frame is
  red-dominant and darkens slightly with every heartbeat. Pulse = inverted red channel (green when
  red is saturated, or when green has the better SNR).
* ``face`` - remote PPG (rPPG): average skin colour of forehead + cheeks. Pulse = POS
  (Wang et al. 2017) and CHROM (de Haan & Jeanne 2013), both reported. Optional pretrained models
  via the ``open-rppg`` package (see ``run_deep_models``).

Two tiers say how the data was captured (every result carries it):

* ``video``   - stock camera app recording uploaded as a file (compressed, auto exposure).
* ``capture`` - browser page with getUserMedia, torch and locked exposure, sending RGB means.

Public functions: ``extract_trace``, ``analyze_trace``, ``analyze_video``, ``run_deep_models``.
Results are plain JSON-serialisable dicts (NaN/inf become ``None``).

Face detector: OpenCV YuNet (``cv2.FaceDetectorYN``). MediaPipe 1.x does have Python 3.13 wheels,
but it pins ``opencv-contrib-python``, which collides with ``opencv-python`` (required by
open-rppg; both install the same ``cv2`` package), and its Tasks API needs a separate model
download anyway. YuNet's ONNX model (232 KB, MIT licence, Shiqi Yu / OpenCV Zoo) is fetched once
from the opencv_zoo repository at tag 4.10.0 into ``data/models/`` and checked by SHA-256.

open-rppg 0.1.1 imports ``pkg_resources``, which setuptools removed in 82, hence the
``setuptools<82`` pin in pyproject.toml. It runs on CPU (PyPI jaxlib has no CUDA).

This is a research prototype, not a medical device.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import math
import threading
import time
import urllib.request
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as sps
from scipy.interpolate import PchipInterpolator
from scipy.stats import skew

log = logging.getLogger(__name__)

ANALYSIS_VERSION = "camera-ppg-0.1"
MODES = ("finger", "face")
TIERS = ("video", "capture")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "data" / "models"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/4.10.0/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_LICENSE = "MIT (Copyright (c) 2020 Shiqi Yu), opencv/opencv_zoo models/face_detection_yunet"

MIN_HR_S = 20.0  # shorter recordings get no HR
MIN_HRV_S = 60.0  # shorter recordings get no HRV
MIN_RESP_S = 30.0  # shorter recordings get no respiratory rate
GRID_FS = 60.0  # uniform resampling rate (Hz) used for all analysis
HR_BAND = (0.65, 3.5)  # Hz, 39-210 bpm
RESP_BAND = (0.1, 0.6)  # Hz, 6-36 breaths/min
MAX_SIDE = 640  # frames are downscaled (area averaging) so the longer side is at most this
SATURATED = 250  # 8-bit level counted as clipped
LOW_FPS = 24.0
FACE_MAX_REL_AMPLITUDE = 0.012  # peak-to-peak share of the green skin level; facial rPPG is ~0.1-1 %
# FacePhys is excluded on purpose: its weights carry a Privacy Protection Addendum (FacePhys-Release
# LICENSE, 2026-08-03) that forbids inference on a server, which "cannot be waived by consent" and
# applies to the copy bundled in open-rppg. It may only run on the phone. ME-rPPG's repo is MIT.
DEEP_MODELS = ("ME-chunk.rlap",)
SERVER_FORBIDDEN_MODELS = frozenset({"FacePhys.rlap", "FacePhys.pure", "FacePhys"})
DEEP_CROP = 128  # face crops handed to open-rppg (largest model input is 128x128)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / Path / Fraction values; NaN and inf become None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating, Fraction)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _r(x: Any, nd: int = 2) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _warn(warns: list, code: str, message: str, severity: str = "warning") -> None:
    if not any(w["code"] == code for w in warns):
        warns.append({"code": code, "severity": severity, "message": message})


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    sos = sps.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1:
        return x.copy()
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, np.ones(n) / n, mode="valid")


def _parabolic(y: np.ndarray, i: int) -> float:
    """Sub-sample offset of a local extremum at index i."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    d = a - 2 * b + c
    return 0.0 if d == 0 else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def _spectrum(x: np.ndarray, fs: float, seg_s: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    nper = int(min(len(x), seg_s * fs))
    nfft = int(2 ** math.ceil(math.log2(max(nper, fs / 0.005))))
    return sps.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, nfft=nfft, detrend="constant")


def _spectral_peak(x: np.ndarray, fs: float, band: tuple[float, float],
                   seg_s: float = 20.0) -> tuple[float | None, float | None]:
    """(peak frequency in Hz, share of in-band power within +-0.05 Hz of the peak)."""
    if len(x) < 8 or not np.any(np.isfinite(x)) or np.std(x) == 0:
        return None, None
    f, p = _spectrum(x, fs, seg_s)
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or p[m].sum() <= 0:
        return None, None
    fm, pm = f[m], p[m]
    i = int(np.argmax(pm))
    df = f[1] - f[0]
    peak = fm[i] + _parabolic(pm, i) * df
    near = np.abs(fm - fm[i]) <= 0.05
    return float(peak), float(pm[near].sum() / pm.sum())


def _snr_db(x: np.ndarray, fs: float, f0: float | None) -> float | None:
    """de Haan & Jeanne SNR: power at HR fundamental (+-0.1 Hz) and 1st harmonic (+-0.2 Hz)
    versus the remaining power in the 0.65-4 Hz band."""
    if f0 is None or len(x) < 8 or np.std(x) == 0:
        return None
    f, p = _spectrum(x, fs)
    band = (f >= HR_BAND[0]) & (f <= 4.0)
    sig = band & ((np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2))
    noise = band & ~sig
    ps, pn = p[sig].sum(), p[noise].sum()
    if ps <= 0 or pn <= 0:
        return None
    return float(10 * np.log10(ps / pn))


# --------------------------------------------------------------------------------------------
# face detection (YuNet)
# --------------------------------------------------------------------------------------------

_YUNET_LOCK = threading.Lock()


def ensure_yunet_model(models_dir: Path = MODELS_DIR) -> Path:
    """Return the local YuNet ONNX path, fetching it from opencv_zoo on first use."""
    path = Path(models_dir) / YUNET_FILE
    with _YUNET_LOCK:
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == YUNET_SHA256:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("fetching YuNet face detector from %s", YUNET_URL)
        with urllib.request.urlopen(YUNET_URL, timeout=60) as resp:
            data = resp.read(5_000_000)
        digest = hashlib.sha256(data).hexdigest()
        if digest != YUNET_SHA256:
            raise RuntimeError(f"YuNet model checksum mismatch ({digest})")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path


class _FaceTracker:
    """YuNet detections every few frames, smoothed, turned into forehead + cheek rectangles."""

    def __init__(self, width: int, height: int, every: int = 5, smooth: float = 0.3):
        import cv2

        try:  # OpenCV 5 logs a harmless "Targets are not supported by the new graph engine"
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:  # pragma: no cover
            pass
        self.det = cv2.FaceDetectorYN.create(str(ensure_yunet_model()), "", (width, height),
                                             score_threshold=0.6, nms_threshold=0.3, top_k=50)
        self.w, self.h, self.every, self.alpha = width, height, every, smooth
        self.state: np.ndarray | None = None  # smoothed [x, y, w, h, 10 landmark coords]
        self.n_attempts = 0
        self.n_found = 0
        self.score_sum = 0.0

    def update(self, idx: int, rgb: np.ndarray) -> None:
        if idx % self.every:
            return
        import cv2

        self.n_attempts += 1
        _, faces = self.det.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) == 0:
            return
        best = faces[np.argmax(faces[:, 2] * faces[:, 3])]  # largest face
        self.n_found += 1
        self.score_sum += float(best[14])
        v = best[:14].astype(np.float64)
        self.state = v if self.state is None else (1 - self.alpha) * self.state + self.alpha * v

    def rois(self) -> list[tuple[int, int, int, int]]:
        """Forehead and both cheeks as (x, y, w, h) in the frame the tracker sees."""
        x, y, w, h = self.state[:4]
        e1, e2 = self.state[4:6], self.state[6:8]  # YuNet: right eye, left eye (of the subject)
        nose, m1, m2 = self.state[8:10], self.state[10:12], self.state[12:14]
        left_eye, right_eye = (e1, e2) if e1[0] <= e2[0] else (e2, e1)  # image left / right
        d = max(right_eye[0] - left_eye[0], 0.25 * w)
        eye_y = (left_eye[1] + right_eye[1]) / 2
        cx = (left_eye[0] + right_eye[0]) / 2
        mouth_y = (m1[1] + m2[1]) / 2
        boxes = []
        top = max(y, eye_y - 0.95 * d)
        bottom = eye_y - 0.40 * d
        if bottom - top >= 0.15 * d:
            boxes.append((cx - 0.55 * d, top, 1.1 * d, bottom - top))  # forehead
        ctop = eye_y + 0.30 * d
        cbottom = max(ctop + 0.25 * d, min(mouth_y - 0.05 * d, nose[1] + 0.25 * d))
        cw = 0.45 * d
        boxes.append((min(left_eye[0] - 0.10 * d, nose[0] - 0.25 * d - cw), ctop, cw, cbottom - ctop))
        boxes.append((max(right_eye[0] + 0.10 * d - cw, nose[0] + 0.25 * d), ctop, cw, cbottom - ctop))
        out = []
        for bx, by, bw, bh in boxes:
            x0, y0 = int(np.clip(bx, 0, self.w - 1)), int(np.clip(by, 0, self.h - 1))
            x1, y1 = int(np.clip(bx + bw, x0 + 1, self.w)), int(np.clip(by + bh, y0 + 1, self.h))
            out.append((x0, y0, x1 - x0, y1 - y0))
        return out


# --------------------------------------------------------------------------------------------
# trace extraction
# --------------------------------------------------------------------------------------------

def _rotate(img: np.ndarray, rotation: int) -> np.ndarray:
    """Apply display-matrix rotation (PyAV ``frame.rotation``, counter-clockwise degrees)."""
    rot = (-int(round(rotation))) % 360
    if rot == 90:
        return np.rot90(img, k=-1)
    if rot == 180:
        return img[::-1, ::-1]
    if rot == 270:
        return np.rot90(img, k=1)
    return img


def _centre_box(w: int, h: int, frac: float) -> tuple[int, int, int, int]:
    bw, bh = int(w * frac), int(h * frac)
    return ((w - bw) // 2, (h - bh) // 2, bw, bh)


def _region_stats(img: np.ndarray, boxes) -> tuple[np.ndarray, int, np.ndarray, int]:
    """Sum of RGB, pixel count, per-channel saturated count, any-channel saturated count."""
    s = np.zeros(3)
    sat = np.zeros(3)
    sat_any = 0
    n = 0
    for x, y, w, h in boxes:
        px = img[y:y + h, x:x + w]
        if px.size == 0:
            continue
        s += px.reshape(-1, 3).sum(axis=0, dtype=np.float64)
        hi = px >= SATURATED
        sat += hi.reshape(-1, 3).sum(axis=0)
        sat_any += int(hi.any(axis=2).sum())
        n += px.shape[0] * px.shape[1]
    return s, n, sat, sat_any


def extract_trace(video_path: str | Path, mode: str, roi: tuple[int, int, int, int] | None = None,
                  *, max_side: int = MAX_SIDE, detect_every: int = 5) -> dict:
    """Per-frame mean R, G, B of the measurement region of a video.

    Timestamps come from the container's presentation timestamps (PyAV ``frame.time``), so
    variable frame rate and dropped frames show up as they are. ``roi=(x, y, w, h)`` in pixels of
    the upright (rotation-corrected) full-resolution frame overrides the automatic region:
    the central 50 % box in finger mode, forehead + cheeks from YuNet in face mode.
    """
    _check_mode(mode)
    import av

    warns: list[dict] = []
    t_list, rgb_list, clip_any, clip_ch = [], [], [], []
    face_rgb, face_clip_any, face_clip_ch = [], [], []
    tracker: _FaceTracker | None = None
    tracker_error = None
    roi_info: dict[str, Any] = {}
    face_boxes_first = None
    key_t: list[float] = []
    n_no_pts = 0
    t_decode = time.perf_counter()
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("file has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate
        fps_nominal = float(rate) if rate else None
        cc = stream.codec_context
        src_w, src_h = int(cc.width), int(cc.height)
        scale = min(1.0, max_side / max(src_w, src_h))
        dw, dh = max(2, int(round(src_w * scale / 2)) * 2), max(2, int(round(src_h * scale / 2)) * 2)
        rotation = 0
        boxes = None
        for idx, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                n_no_pts += 1
                continue
            if boxes is None:
                rotation = int(frame.rotation or 0)
            img = frame.reformat(width=dw, height=dh, format="rgb24", interpolation="AREA").to_ndarray()
            img = _rotate(img, rotation)
            H, W = img.shape[:2]
            if boxes is None:  # first frame: decide the static region (finger / override / fallback)
                full_w, full_h = (src_h, src_w) if W != dw else (src_w, src_h)
                fx, fy = (W / full_w), (H / full_h)
                if roi is not None:
                    x, y, w, h = (int(v) for v in roi)
                    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > full_w or y + h > full_h:
                        raise ValueError(f"roi {tuple(roi)} is not inside the {full_w}x{full_h} frame")
                    bx = (int(x * fx), int(y * fy), max(1, int(w * fx)), max(1, int(h * fy)))
                    boxes = [bx]
                    roi_info = {"source": "override", "boxes": [list(map(int, roi))]}
                else:
                    boxes = [_centre_box(W, H, 0.5)]
                    roi_info = {"source": "centre" if mode == "finger" else "centre_fallback",
                                "boxes": [[int(v / f) for v, f in zip(boxes[0], (fx, fy, fx, fy))]]}
                    if mode == "face":
                        try:
                            tracker = _FaceTracker(W, H, every=detect_every)
                        except Exception as exc:  # model missing / offline
                            tracker_error = f"{type(exc).__name__}: {exc}"
                roi_info["frame_size"] = [full_w, full_h]
                roi_info["analysis_scale"] = round(fx, 4)
            s, n, sat, sat_any = _region_stats(img, boxes)
            if frame.key_frame:
                key_t.append(float(frame.time))
            t_list.append(float(frame.time))
            rgb_list.append(s / max(n, 1))
            clip_ch.append(sat / max(n, 1))
            clip_any.append(sat_any / max(n, 1))
            if tracker is not None:
                tracker.update(idx, img)
                if tracker.state is not None:
                    fb = tracker.rois()
                    s2, n2, sat2, sat_any2 = _region_stats(img, fb)
                    face_rgb.append(s2 / max(n2, 1))
                    face_clip_ch.append(sat2 / max(n2, 1))
                    face_clip_any.append(sat_any2 / max(n2, 1))
                    if face_boxes_first is None:
                        face_boxes_first = [[int(b[0] / fx), int(b[1] / fy), int(b[2] / fx), int(b[3] / fy)]
                                            for b in fb]
                else:
                    face_rgb.append(None)
                    face_clip_ch.append(None)
                    face_clip_any.append(None)
        codec = cc.name
        bit_rate = stream.bit_rate or container.bit_rate
    decode_s = time.perf_counter() - t_decode
    if len(t_list) < 2:
        raise ValueError("video has fewer than 2 decodable frames")

    t = np.asarray(t_list)
    rgb = np.asarray(rgb_list)
    cany = np.asarray(clip_any)
    cch = np.asarray(clip_ch)
    if mode == "face" and roi is None:
        found = [i for i, v in enumerate(face_rgb) if v is not None]
        if tracker is None:
            _warn(warns, "face_detector_unavailable",
                  f"Face detector unavailable ({tracker_error}); used a centred box instead.")
        elif not found:
            _warn(warns, "no_face_detected",
                  "No face was found; used a centred box. Results are unlikely to be meaningful.")
        else:
            first = found[0]
            if t[first] - t[0] > 2.0:
                _warn(warns, "face_found_late",
                      f"Face first detected after {t[first] - t[0]:.1f} s; earlier frames dropped.")
            t, rgb = t[first:], np.asarray(face_rgb[first:], float)
            cany = np.asarray(face_clip_any[first:], float)
            cch = np.asarray(face_clip_ch[first:], float)
            roi_info["source"] = "yunet_forehead_cheeks"
            roi_info["boxes"] = face_boxes_first
            roi_info["detector"] = {"name": "YuNet (OpenCV FaceDetectorYN)", "model": YUNET_FILE,
                                    "license": YUNET_LICENSE,
                                    "detected_fraction": round(tracker.n_found / max(tracker.n_attempts, 1), 3),
                                    "mean_score": _r(tracker.score_sum / max(tracker.n_found, 1), 3)}
            if tracker.n_found / max(tracker.n_attempts, 1) < 0.8:
                _warn(warns, "face_lost", "The face was not detected in more than 20 % of checks; "
                                          "the last known position was used.")
    order = np.argsort(t, kind="stable")
    t, rgb, cany, cch = t[order], rgb[order], cany[order], cch[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    n_dup = int((~keep).sum())
    t, rgb, cany, cch = t[keep], rgb[keep], cany[keep], cch[keep]
    t = t - t[0]
    dt = np.diff(t)
    med = float(np.median(dt)) if len(dt) else float("nan")
    out = {
        "mode": mode,
        "t_s": t,
        "r": rgb[:, 0], "g": rgb[:, 1], "b": rgb[:, 2],
        "n_frames": int(len(t)),
        "duration_s": float(t[-1]),
        "fps_nominal": fps_nominal,
        "fps_measured": float((len(t) - 1) / t[-1]) if t[-1] > 0 else None,
        "frame_interval_ms_median": med * 1000,
        "frame_time_jitter_ms": float(np.std(dt) * 1000) if len(dt) else None,
        "dropped_frames": int((dt > 1.5 * med).sum()) if len(dt) else 0,
        "duplicate_timestamps": n_dup,
        "frames_without_pts": n_no_pts,
        "clipped_fraction": float(np.mean(cany)),
        "clipped_fraction_channels": {"r": float(np.mean(cch[:, 0])), "g": float(np.mean(cch[:, 1])),
                                      "b": float(np.mean(cch[:, 2]))},
        "roi": roi_info,
        "video": {"codec": codec, "width": src_w, "height": src_h, "rotation": rotation,
                  "bit_rate": int(bit_rate) if bit_rate else None, "decode_s": round(decode_s, 2),
                  "keyframes": len(key_t),
                  "keyframe_interval_s": (float(np.median(np.diff(key_t))) if len(key_t) >= 3 else None)},
        "warnings": warns,
    }
    return out


# --------------------------------------------------------------------------------------------
# pulse extraction
# --------------------------------------------------------------------------------------------

def _pos(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (Wang et al. 2017), overlap-added over 1.6 s windows."""
    n = rgb.shape[0]
    l = int(round(win_s * fs))
    if n < l + 1:
        l = max(4, n // 2)
    out = np.zeros(n)
    m = n - l + 1  # number of windows (one per sample)
    for c0 in range(0, m, 20_000):  # chunks bound memory for long traces
        c1 = min(m, c0 + 20_000)
        win = np.lib.stride_tricks.sliding_window_view(rgb[c0:c1 + l - 1], l, axis=0)  # (c, 3, l)
        cn = win / (win.mean(axis=2, keepdims=True) + 1e-9)
        s0 = cn[:, 1] - cn[:, 2]
        s1 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]
        alpha = s0.std(axis=1, keepdims=True) / (s1.std(axis=1, keepdims=True) + 1e-12)
        h = s0 + alpha * s1
        h -= h.mean(axis=1, keepdims=True)
        for k in range(l):
            out[c0 + k:c0 + k + h.shape[0]] += h[:, k]
    return out


def _chrom(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013), Hann-windowed overlap-add with per-window alpha."""
    mm = np.stack([_moving_mean(rgb[:, c], int(win_s * fs)) for c in range(3)], axis=1)
    cn = rgb / (mm + 1e-9)
    xs = 3 * cn[:, 0] - 2 * cn[:, 1]
    ys = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
    xf = _bandpass(xs - xs.mean(), fs, *HR_BAND)
    yf = _bandpass(ys - ys.mean(), fs, *HR_BAND)
    l = int(round(win_s * fs)) // 2 * 2
    step = l // 2
    win = np.hanning(l)
    out = np.zeros(len(xf))
    starts = list(range(0, max(1, len(xf) - l + 1), step))
    if starts[-1] + l < len(xf):
        starts.append(len(xf) - l)  # cover the tail
    for start in starts:
        xw, yw = xf[start:start + l], yf[start:start + l]
        a = xw.std() / (yw.std() + 1e-12)
        out[start:start + l] += (xw - a * yw) * win[:len(xw)]
    return out


def _detect_peaks(x: np.ndarray, fs: float, hr_hint_bpm: float | None) -> np.ndarray:
    """Systolic peak times (s, sub-sample refined) of a band-passed, upright pulse signal."""
    period = 60.0 / hr_hint_bpm if hr_hint_bpm else 0.8
    dist = max(1, int(0.5 * period * fs))
    prom = 0.3 * np.std(x)
    idx, _ = sps.find_peaks(x, distance=dist, prominence=prom)
    return np.array([(i + _parabolic(x, i)) / fs for i in idx])


def _clean_ibis(peaks_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """IBIs (ms) and their end times, physiological range + local-median outlier rejection."""
    if len(peaks_t) < 3:
        return np.array([]), np.array([]), 1.0
    ibi = np.diff(peaks_t) * 1000
    tt = peaks_t[1:]
    ok = (ibi >= 300) & (ibi <= 2000)
    med = np.array([np.median(ibi[max(0, i - 3):i + 4]) for i in range(len(ibi))])
    ok &= np.abs(ibi - med) <= 0.25 * med
    return ibi[ok], tt[ok], float(1 - ok.mean())


def _template_corr(x: np.ndarray, fs: float, peaks_t: np.ndarray) -> float | None:
    """Mean correlation of each beat with the average beat (Orphanidou-style template SQI)."""
    if len(peaks_t) < 4:
        return None
    period = float(np.median(np.diff(peaks_t)))
    pre, post = int(0.35 * period * fs), int(0.65 * period * fs)
    beats = []
    for pt in peaks_t:
        i = int(round(pt * fs))
        if i - pre >= 0 and i + post < len(x):
            beats.append(x[i - pre:i + post])
    if len(beats) < 3:
        return None
    beats = np.asarray(beats)
    tmpl = beats.mean(axis=0)
    cs = [np.corrcoef(b, tmpl)[0, 1] for b in beats if np.std(b) > 0]
    return float(np.mean(cs)) if cs else None


def _windowed_skewness(x: np.ndarray, fs: float, win_s: float = 5.0) -> float | None:
    """Elgendi (2016) skewness SQI, median over 5 s windows (positive for clean upright PPG)."""
    l = int(win_s * fs)
    vals = [skew(x[i:i + l]) for i in range(0, len(x) - l + 1, l)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals and len(x) > 8:
        vals = [skew(x)]
    return float(np.median(vals)) if vals else None


def _respiration(raw: np.ndarray, filt: np.ndarray, fs: float, peaks_t: np.ndarray,
                 ibi_ms: np.ndarray, ibi_t: np.ndarray) -> dict:
    """Breathing rate from baseline (RIIV), amplitude (RIAV) and frequency (RIFV) modulation."""
    est: dict[str, dict] = {}
    # baseline wander of the un-band-passed pulse signal
    base = _bandpass(raw - raw.mean(), fs, *RESP_BAND, order=2)
    f, share = _spectral_peak(base, fs, RESP_BAND, seg_s=30)
    est["baseline"] = {"rate_brpm": _r(f * 60 if f else None, 1), "peak_share": _r(share, 3)}
    fs4 = 4.0
    if len(peaks_t) >= 8:
        tg = np.arange(peaks_t[1], peaks_t[-1], 1 / fs4)
        # pulse amplitude: peak minus preceding trough
        amps, at = [], []
        for a, b in zip(peaks_t[:-1], peaks_t[1:]):
            ia, ib = int(a * fs), int(b * fs)
            if ib - ia > 2:
                seg = filt[ia:ib + 1]
                amps.append(filt[min(ib, len(filt) - 1)] - seg.min())
                at.append(b)
        if len(amps) >= 8 and len(tg) > 16:
            ser = PchipInterpolator(at, amps)(tg)
            f, share = _spectral_peak(ser - ser.mean(), fs4, RESP_BAND, seg_s=30)
            est["amplitude"] = {"rate_brpm": _r(f * 60 if f else None, 1), "peak_share": _r(share, 3)}
    if len(ibi_ms) >= 8:
        tg = np.arange(ibi_t[0], ibi_t[-1], 1 / fs4)
        if len(tg) > 16:
            ser = PchipInterpolator(ibi_t, ibi_ms)(tg)
            f, share = _spectral_peak(ser - ser.mean(), fs4, RESP_BAND, seg_s=30)
            est["frequency"] = {"rate_brpm": _r(f * 60 if f else None, 1), "peak_share": _r(share, 3)}
    good = [e["rate_brpm"] for e in est.values() if e["rate_brpm"] is not None
            and (e["peak_share"] or 0) >= 0.15]
    rate = float(np.median(good)) if good else None
    agree = len(good) >= 2 and (max(good) - min(good)) <= 3.0
    return {"rate_brpm": _r(rate, 1), "reliable": bool(agree), "estimates": est,
            "method": "median of baseline/amplitude/frequency modulation estimates with a clear "
                      "spectral peak (share >= 0.15 of 6-36 /min band power)"}


def _hrv(ibi_ms: np.ndarray, ibi_t: np.ndarray, duration_s: float) -> dict | None:
    import neurokit2 as nk
    import pandas as pd

    if len(ibi_ms) < 30:
        return None
    rri = {"RRI": np.asarray(ibi_ms, float), "RRI_Time": np.asarray(ibi_t, float)}
    out: dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        td = nk.hrv_time(rri)
        keys = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_SDSD", "HRV_pNN50", "HRV_pNN20",
                "HRV_MedianNN", "HRV_IQRNN"]
        out.update({k.removeprefix("HRV_"): _r(td[k].iloc[0], 2) for k in keys if k in td})
        if duration_s >= 120:
            try:
                fd = nk.hrv_frequency(rri)
                for k in ["HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn", "HRV_HFn"]:
                    if k in fd:
                        v = fd[k].iloc[0]
                        out[k.removeprefix("HRV_")] = _r(v, 5) if not pd.isna(v) else None
            except Exception as exc:  # pragma: no cover - neurokit edge cases
                out["frequency_error"] = str(exc)
    out["n_intervals"] = int(len(ibi_ms))
    out["library"] = f"neurokit2 {importlib.metadata.version('neurokit2')}"
    return out


# --------------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------------

def _method_summary(x: np.ndarray, fs: float) -> dict:
    f0, share = _spectral_peak(x, fs, HR_BAND)
    hr_fft = f0 * 60 if f0 else None
    peaks = _detect_peaks(x, fs, hr_fft)
    ibi, _, _ = _clean_ibis(peaks)
    return {"hr_fft_bpm": hr_fft, "hr_peaks_bpm": 60000 / ibi.mean() if len(ibi) else None,
            "snr_db": _snr_db(x, fs, f0), "n_peaks": len(peaks), "_peaks": peaks, "_f0": f0}


def analyze_trace(t_s, r, g, b, mode: str, tier: str, *, clipped_fraction: float | dict | None = None,
                  extra_warnings: list[dict] | None = None) -> dict:
    """Heart rate, IBIs, HRV, breathing rate and quality from per-frame RGB means.

    ``t_s`` are frame times in seconds (any origin, need not be uniform). ``clipped_fraction`` is
    the share of saturated ROI pixels, either a float or ``{"any", "r", "g", "b"}``; when absent
    a red channel sitting at >= 250 is taken as clipped. HR needs >= 20 s, HRV >= 60 s.
    """
    _check_mode(mode)
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    t = np.asarray(t_s, float).ravel()
    ch = [np.asarray(c, float).ravel() for c in (r, g, b)]
    if not (len(t) == len(ch[0]) == len(ch[1]) == len(ch[2])):
        raise ValueError("t_s, r, g, b must have equal length")
    ok = np.isfinite(t) & np.all([np.isfinite(c) for c in ch], axis=0)
    t, ch = t[ok], [c[ok] for c in ch]
    order = np.argsort(t, kind="stable")
    t, ch = t[order], [c[order] for c in ch]
    keep = np.concatenate([[True], np.diff(t) > 0]) if len(t) else np.array([], bool)
    t, ch = t[keep], [c[keep] for c in ch]
    if len(t) < 10 or t[-1] - t[0] < 2.0:
        raise ValueError("need at least 10 samples spanning 2 s")
    t = t - t[0]
    r, g, b = ch
    warns: list[dict] = [dict(w) for w in (extra_warnings or [])]
    dt = np.diff(t)
    med_dt = float(np.median(dt))
    duration = float(t[-1] + med_dt)  # n frames cover n frame intervals
    fps_measured = (len(t) - 1) / float(t[-1])
    jitter_ms = float(np.std(dt) * 1000)
    dropped = int((dt > 1.5 * med_dt).sum())

    # --- capture-level warnings -----------------------------------------------------------
    if duration < MIN_HR_S:
        _warn(warns, "too_short", f"Recording is {duration:.1f} s; at least {MIN_HR_S:.0f} s are needed "
                                  "for heart rate (60 s for HRV).")
    elif duration < MIN_HRV_S:
        _warn(warns, "hrv_needs_60s", f"Recording is {duration:.0f} s; HRV needs at least "
                                      f"{MIN_HRV_S:.0f} s and is not reported.", "info")
    if fps_measured < LOW_FPS:
        _warn(warns, "low_fps", f"Measured frame rate is {fps_measured:.1f} fps (< {LOW_FPS:.0f}); "
                                "beat timing and HRV are degraded.")
    if jitter_ms > 0.2 * med_dt * 1000 or dropped > 0.02 * len(t):
        _warn(warns, "irregular_frame_times", f"Frame times are irregular (jitter {jitter_ms:.1f} ms, "
                                              f"{dropped} gaps > 1.5 frame intervals).")
    if isinstance(clipped_fraction, dict):
        clip_any = clipped_fraction.get("any")
        clip_r = clipped_fraction.get("r")
    else:
        clip_any, clip_r = clipped_fraction, None
    red_clipped = bool((clip_r is not None and clip_r >= 0.25) or np.median(r) >= SATURATED)
    if (not red_clipped and mode == "finger" and clip_r is None and clip_any is not None and clip_any >= 0.25
            and np.median(r) > 1.3 * np.median(g)):
        # The capture page only reports "any channel >= 250"; under the torch that channel is red,
        # so apply the same 25 % rule as the per-channel video path.
        red_clipped = True
    if clip_any is None and red_clipped:
        clip_any = float(np.mean(r >= SATURATED))
    if (clip_any is not None and clip_any > 0.05) or red_clipped:
        share = f"{100 * clip_any:.0f} % of ROI pixels" if clip_any is not None else "the red channel"
        _warn(warns, "clipped", f"{share} saturated" + (" (red channel clipped: reduce exposure/ISO or "
                                "press less hard)" if red_clipped else "") + ".")

    # --- uniform grid ------------------------------------------------------------------------
    fs = GRID_FS
    tg = np.arange(0.0, float(t[-1]), 1.0 / fs)
    rgb = np.stack([PchipInterpolator(t, c)(tg) for c in (r, g, b)], axis=1)
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    if np.median(lum) < 15:
        _warn(warns, "too_dark", "The image is very dark; turn the torch on / add light.")
    # exposure: slow drift and abrupt steps of overall brightness
    sm = _moving_mean(lum, int(2 * fs))
    core = sm[int(fs):-int(fs)] if len(sm) > 4 * fs else sm
    drift_pct = float((core.max() - core.min()) / max(core.mean(), 1e-9) * 100)
    d_raw = np.diff(0.299 * r + 0.587 * g + 0.114 * b)
    mad = float(np.median(np.abs(d_raw - np.median(d_raw)))) * 1.4826 if len(d_raw) else 0.0
    steps = int((np.abs(d_raw) > max(0.02 * float(np.mean(lum)), 8 * mad)).sum())
    if drift_pct > 10:
        _warn(warns, "exposure_drift", f"Brightness drifts by {drift_pct:.0f} % (auto exposure / white "
                                       "balance or changing contact pressure).")
    if steps > 0:
        _warn(warns, "exposure_steps", f"{steps} abrupt brightness jump(s), typical of auto exposure; "
                                       "lock exposure (capture tier) or keep still.")

    # --- pulse signals -----------------------------------------------------------------------
    methods: dict[str, np.ndarray] = {}
    raws: dict[str, np.ndarray] = {}
    finger_missing = False
    if mode == "finger":
        finger_missing = bool(np.median(r) < 1.3 * np.median(g))
        if finger_missing:
            _warn(warns, "finger_not_detected", "Frame is not red-dominant: the fingertip may not cover "
                                                "the lens or the torch is off.")
        for name, c in (("red", rgb[:, 0]), ("green", rgb[:, 1])):
            raw = -(c / max(float(c.mean()), 1e-9))  # inverted: more blood -> less light
            raws[name] = raw
            methods[name] = _bandpass(raw - raw.mean(), fs, *HR_BAND)
    else:
        for name, fn in (("pos", _pos), ("chrom", _chrom)):
            s = fn(rgb, fs)
            raws[name] = s
            methods[name] = _bandpass(s - s.mean(), fs, *HR_BAND)
        # orient upright (systolic peak up) using inverted green as reference
        ref = _bandpass(-(rgb[:, 1] - rgb[:, 1].mean()), fs, *HR_BAND)
        for name in list(methods):
            if np.corrcoef(methods[name], ref)[0, 1] < 0:
                methods[name], raws[name] = -methods[name], -raws[name]

    summaries = {k: _method_summary(v, fs) for k, v in methods.items()}
    if mode == "finger":
        snr_r = summaries["red"]["snr_db"] if summaries["red"]["snr_db"] is not None else -99.0
        snr_g = summaries["green"]["snr_db"] if summaries["green"]["snr_db"] is not None else -99.0
        if red_clipped:
            primary, reason = "green", "red channel clipped"
        elif snr_g > snr_r + 3.0:
            primary, reason = "green", "green SNR > red SNR + 3 dB"
        else:
            primary, reason = "red", "default (inverted red)"
    else:
        primary = max(("pos", "chrom"), key=lambda k: summaries[k]["snr_db"] if summaries[k]["snr_db"]
                      is not None else -99)
        reason = "higher SNR of POS / CHROM"
    x = methods[primary]
    sm_p = summaries[primary]
    peaks_t = sm_p["_peaks"]
    ibi, ibi_t, artifact = _clean_ibis(peaks_t)
    hr_fft, hr_peaks = sm_p["hr_fft_bpm"], sm_p["hr_peaks_bpm"]
    snr = sm_p["snr_db"]

    # --- quality -----------------------------------------------------------------------------
    tc = _template_corr(x, fs, peaks_t)
    sk = _windowed_skewness(x, fs)
    quality: dict[str, Any] = {"snr_db": _r(snr), "template_corr": _r(tc, 3), "skewness": _r(sk, 3),
                               "ibi_artifact_fraction": _r(artifact, 3),
                               "exposure_drift_pct": _r(drift_pct, 2), "exposure_steps": steps}
    if mode == "finger":
        c = rgb[:, 0 if primary == "red" else 1]
        amps = []
        for a, bb in zip(peaks_t[:-1], peaks_t[1:]):
            seg = c[int(a * fs):int(bb * fs) + 1]
            if len(seg) > 2:
                amps.append((seg.max() - seg.min()) / seg.mean())
        quality["perfusion_index_pct"] = _r(np.median(amps) * 100 if amps else None, 3)
    good = snr is not None and snr >= 3 and (tc or 0) >= 0.85 and artifact <= 0.1
    fair = snr is not None and snr >= 0 and (tc or 0) >= 0.6 and artifact <= 0.25
    quality["label"] = "good" if good else "fair" if fair else "poor"
    if finger_missing:
        # A periodic scene (flicker, an animated emulator camera) can look like a perfect pulse; without a
        # fingertip on the lens the number is not a heart rate, however clean it looks.
        quality["label"] = "poor"
    if mode == "face":
        # Facial rPPG changes skin brightness by well under 1 %; a cardiac-band swing several times that is
        # motion, flicker or a changing scene, not blood volume.
        gb = _bandpass(rgb[:, 1] - rgb[:, 1].mean(), fs, *HR_BAND)
        rel = float((np.percentile(gb, 95) - np.percentile(gb, 5)) / max(float(rgb[:, 1].mean()), 1e-9))
        quality["face_amplitude_pct"] = _r(rel * 100, 3)
        if rel > FACE_MAX_REL_AMPLITUDE:
            quality["label"] = "poor"
            _warn(warns, "implausible_face_amplitude",
                  f"The pulsatile brightness change is {rel * 100:.1f} % of the skin level; a real face pulse is "
                  "well under 1 %, so this is motion, flicker or a changing scene.")
    if quality["label"] == "poor":
        _warn(warns, "low_quality", "Signal quality is poor (low SNR / irregular beats); treat values "
                                    "as unreliable.")
    if hr_fft and hr_peaks and abs(hr_fft - hr_peaks) > 5:
        _warn(warns, "hr_disagreement", f"Spectral HR ({hr_fft:.0f}) and beat-to-beat HR ({hr_peaks:.0f}) "
                                        "disagree by more than 5 bpm.")

    # --- HR / HRV / respiration gates -------------------------------------------------------
    hr_ok = duration >= MIN_HR_S
    hrv = None
    hrv_reliable = False
    if mode == "face":
        hrv_note = "HRV from face video is not reliable (compression, motion, low SNR)."
    elif duration < MIN_HRV_S:
        hrv_note = f"HRV needs >= {MIN_HRV_S:.0f} s of recording."
    else:
        hrv_note = None
    if duration >= MIN_HRV_S:
        try:
            hrv = _hrv(ibi, ibi_t, duration)
        except Exception as exc:  # pragma: no cover
            hrv_note = f"HRV computation failed: {exc}"
        if hrv is None and hrv_note is None:
            hrv_note = "Too few clean beats for HRV."
        hrv_reliable = bool(hrv is not None and mode == "finger" and quality["label"] == "good"
                            and fps_measured >= LOW_FPS)
        if hrv is not None and not hrv_reliable and hrv_note is None:
            hrv_note = "HRV shown for exploration only: signal quality or frame rate is insufficient."
    resp = (_respiration(raws[primary], x, fs, peaks_t, ibi, ibi_t) if duration >= MIN_RESP_S else
            {"rate_brpm": None, "reliable": False, "estimates": {},
             "note": f"needs >= {MIN_RESP_S:.0f} s"})

    # --- waveform for plotting -----------------------------------------------------------------
    plot_fs = min(25.0, 4000.0 / max(duration, 1e-9))
    step = max(1, int(round(fs / plot_fs)))
    z = (x - x.mean()) / (x.std() + 1e-12)
    waveform = {"fs": round(fs / step, 3), "t_s": np.round(tg[::step], 3), "values": np.round(z[::step], 3),
                "peaks_t_s": np.round(peaks_t, 3), "signal": primary}

    result = {
        "analysis_version": ANALYSIS_VERSION,
        "tier": tier,
        "mode": mode,
        "duration_s": _r(duration),
        "n_samples": int(len(t)),
        "fps_measured": _r(fps_measured),
        "frame_interval_ms_median": _r(med_dt * 1000),
        "frame_time_jitter_ms": _r(jitter_ms),
        "dropped_frames": dropped,
        "resampled_fs": fs,
        "hr_bpm": _r(hr_fft, 1) if hr_ok else None,
        "hr_fft_bpm": _r(hr_fft, 1) if hr_ok else None,
        "hr_peaks_bpm": _r(hr_peaks, 1) if hr_ok else None,
        "primary_method": primary,
        "primary_reason": reason,
        "methods": {k: {"hr_fft_bpm": _r(v["hr_fft_bpm"], 1) if hr_ok else None,
                        "hr_peaks_bpm": _r(v["hr_peaks_bpm"], 1) if hr_ok else None,
                        "snr_db": _r(v["snr_db"]), "n_peaks": v["n_peaks"]} for k, v in summaries.items()},
        "n_beats": int(len(peaks_t)),
        "ibi_ms": np.round(ibi, 1) if hr_ok else None,
        "ibi_t_s": np.round(ibi_t, 3) if hr_ok else None,
        "hrv": hrv,
        "hrv_reliable": hrv_reliable,
        "hrv_note": hrv_note,
        "respiration": resp,
        "quality": quality,
        "channel_means": {"r": _r(np.mean(r)), "g": _r(np.mean(g)), "b": _r(np.mean(b))},
        "clipped_fraction": _r(clip_any, 4),
        "red_clipped": red_clipped,
        "warnings": warns,
        "waveform": waveform,
    }
    return _jsonable(result)


# --------------------------------------------------------------------------------------------
# open-rppg (pretrained models)
# --------------------------------------------------------------------------------------------

_DEEP_LOCK = threading.Lock()


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(f"{record.levelname}: {record.getMessage()}")


def _decode_crops(video_path: Path, roi: tuple[int, int, int, int],
                  size: int = DEEP_CROP) -> tuple[list[np.ndarray], list[float]]:
    import av
    import cv2

    crops, ts = [], []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            img = _rotate(frame.to_ndarray(format="rgb24"), int(frame.rotation or 0))
            x, y, w, h = roi
            crop = np.ascontiguousarray(img[y:y + h, x:x + w])
            if crop.size == 0:
                raise ValueError("roi lies outside the frame")
            crops.append(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA))
            ts.append(float(frame.time))
    return crops, ts


def deep_available() -> dict:
    """Whether open-rppg can be imported (cheap check without importing jax)."""
    try:
        return {"available": True, "version": importlib.metadata.version("open-rppg")}
    except importlib.metadata.PackageNotFoundError:
        return {"available": False, "reason": "open-rppg is not installed"}


def run_deep_models(video_path: str | Path, roi: tuple[int, int, int, int] | None = None,
                    models: tuple[str, ...] | list[str] = DEEP_MODELS) -> dict:
    """Zero-shot pretrained rPPG models from open-rppg (JAX/Keras, weights bundled in the wheel).

    Without ``roi`` the package's own BlazeFace detector finds the face (``Model.process_video``);
    with ``roi`` the given box is cropped and fed frame by frame with its real timestamps.
    Each model is timed (load incl. JIT compile, inference). HRV from these is never reliable.
    """
    out: dict[str, Any] = {"package": "open-rppg", "models": {}}
    skipped = [m for m in models if m in SERVER_FORBIDDEN_MODELS]
    models = [m for m in models if m not in SERVER_FORBIDDEN_MODELS]
    if skipped:
        out["skipped"] = {m: "licence forbids server-side inference (on-device only)" for m in skipped}
    t0 = time.perf_counter()
    try:
        import rppg  # noqa: PLC0415 - heavy import (jax + keras), ~10-15 s the first time
    except Exception as exc:
        out.update(available=False, reason=f"open-rppg could not be imported: {type(exc).__name__}: {exc}")
        return _jsonable(out)
    out.update(available=True, version=importlib.metadata.version("open-rppg"),
               import_s=round(time.perf_counter() - t0, 2),
               face_source="roi_override" if roi is not None else "open-rppg BlazeFace detector",
               license_note="open-rppg code is MIT; pretrained weights fall under each paper's terms. "
                            "FacePhys is never run here: its licence allows on-device inference only.")
    try:
        import jax

        out["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:  # pragma: no cover
        pass
    crops = ts = None
    if roi is not None:
        t1 = time.perf_counter()
        crops, ts = _decode_crops(Path(video_path), roi)
        out["crop_decode_s"] = round(time.perf_counter() - t1, 2)
    rlog = logging.getLogger("open-rppg")
    for name in models:
        rec: dict[str, Any] = {}
        if name not in rppg.supported_models:
            rec["error"] = f"not in this open-rppg version (supported: {rppg.supported_models})"
            out["models"][name] = rec
            continue
        handler = _ListHandler()
        rlog.addHandler(handler)
        try:
            with _DEEP_LOCK:
                t1 = time.perf_counter()
                model = rppg.Model(name)
                rec["load_s"] = round(time.perf_counter() - t1, 2)
                t2 = time.perf_counter()
                if crops is None:
                    res = model.process_video(str(video_path))
                else:
                    with model:
                        for img, tt in zip(crops, ts):
                            model.update_face(img, tt)
                    res = model.hr(return_hrv=False)
                rec["infer_s"] = round(time.perf_counter() - t2, 2)
                if res is not None and res.get("hr") is not None and crops is not None:
                    try:
                        res["hrv"] = (model.hr(return_hrv=True) or {}).get("hrv", {})
                    except Exception:
                        res["hrv"] = {}
                bvp, bts = model.bvp()
                stats = dict(getattr(model, "statistic", {}))
            if res is None or res.get("hr") is None:
                rec["error"] = "model produced no signal (no face / too short)"
            else:
                rec["hr_bpm"] = _r(res["hr"], 1)
                rec["sqi"] = _r(res.get("SQI"), 3)
                hrv = res.get("hrv") or {}
                rec["hrv"] = {k: _r(v, 3) for k, v in hrv.items()
                              if isinstance(v, (int, float, np.floating, np.integer))}
                rec["hrv_reliable"] = False
            if len(bvp) > 16:
                bvp = np.asarray(bvp, float)
                f0, _ = _spectral_peak(bvp, model.fps, HR_BAND)
                rec["snr_db"] = _r(_snr_db(bvp, model.fps, f0))
            rec["frames"] = stats
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            rlog.removeHandler(handler)
        if handler.records:
            rec["log"] = handler.records[:20]
        out["models"][name] = rec
    out["total_s"] = round(time.perf_counter() - t0, 2)
    return _jsonable(out)


# --------------------------------------------------------------------------------------------
# video entry point
# --------------------------------------------------------------------------------------------

def _keyframe_check(res: dict, interval_s: float | None) -> None:
    """Phones insert a key (I) frame about every second. Each one re-quantises the picture, which
    adds a small periodic brightness step: a spurious "pulse" at 60 / GOP bpm (60 bpm for a 1 s
    GOP). It matters for face video, where the real pulse is equally tiny."""
    if not interval_s or interval_s <= 0 or res.get("hr_bpm") is None:
        return
    kf_bpm = 60.0 / interval_s
    near = [k for k in (1, 2, 3) if abs(res["hr_bpm"] - k * kf_bpm) <= 3.0]
    res["quality"]["keyframe_rate_bpm"] = _r(kf_bpm, 1)
    # Face: the artefact is as large as the pulse, so a match is never trustworthy. Finger: the pulse
    # is ~10x larger and only a weak (not "good") signal locks onto the key-frame rate.
    if near and (res["mode"] == "face" or res["quality"]["label"] != "good"):
        res["quality"]["keyframe_artifact_suspected"] = True
        res["quality"]["label"] = "poor"  # was one step down: a 150 bpm face read as 60 bpm stayed "fair"
        _warn(res["warnings"], "hr_near_keyframe_rate",
              f"Heart rate {res['hr_bpm']:.0f} bpm is within 3 bpm of {near[0]}x the key-frame rate "
              f"({kf_bpm:.0f}/min); it may be a compression artefact rather than the pulse.")


def _deep_check(res: dict, interval_s: float | None) -> None:
    """The pretrained models lock onto the key-frame rate and disagree on badly compressed video just
    like POS/CHROM do (adversarial CRF 28-35 tests); flag that instead of showing bare numbers."""
    hrs = {k: v["hr_bpm"] for k, v in (res.get("deep") or {}).get("models", {}).items()
           if v.get("hr_bpm") is not None}
    if interval_s and interval_s > 0:
        kf_bpm = 60.0 / interval_s
        near = [k for k, hr in hrs.items() if any(abs(hr - m * kf_bpm) <= 3.0 for m in (1, 2, 3))]
        for k in near:
            res["deep"]["models"][k]["keyframe_artifact_suspected"] = True
        if near:
            _warn(res["warnings"], "deep_hr_near_keyframe_rate",
                  f"Pretrained model HR ({', '.join(near)}) is within 3 bpm of the key-frame rate "
                  f"({kf_bpm:.0f}/min); it may be a compression artefact.")
    if len(hrs) >= 2 and max(hrs.values()) - min(hrs.values()) > 5:
        _warn(res["warnings"], "deep_models_disagree",
              f"The pretrained models disagree by {max(hrs.values()) - min(hrs.values()):.0f} bpm; "
              "treat their heart rates as unreliable.")
    # With a single server-side model (FacePhys may not run here) there is nothing to disagree with,
    # so a pretrained HR also has to agree with the classical POS/CHROM estimate on a usable signal.
    classical = res.get("hr_bpm")
    poor = (res.get("quality") or {}).get("label") == "poor"
    off = [k for k, hr in hrs.items() if classical is None or abs(hr - classical) > 5]
    if hrs and (off or poor):
        why = "the classical signal is poor" if poor else f"it differs from POS/CHROM ({classical} bpm) by more than 5 bpm"
        _warn(res["warnings"], "deep_unconfirmed",
              f"Pretrained model HR ({', '.join(off or hrs)}) is unconfirmed: {why}.")


def analyze_video(video_path: str | Path, mode: str, tier: str = "video", deep: bool = True,
                  roi: tuple[int, int, int, int] | None = None,
                  deep_models: tuple[str, ...] | list[str] = DEEP_MODELS,
                  trace_out: str | Path | None = None) -> dict:
    """extract_trace + analyze_trace; in face mode optionally also open-rppg models (``deep``).

    ``trace_out`` writes the raw per-frame trace as JSON (for re-analysis without the video).
    """
    _check_mode(mode)
    t0 = time.perf_counter()
    tr = extract_trace(video_path, mode, roi=roi)
    t_extract = time.perf_counter() - t0
    if trace_out is not None:
        import json

        Path(trace_out).write_text(json.dumps(_jsonable({
            "mode": mode, "t_ms": np.round(tr["t_s"] * 1000, 3), "r": np.round(tr["r"], 4),
            "g": np.round(tr["g"], 4), "b": np.round(tr["b"], 4), "meta": {"source": "video", "roi": tr["roi"]},
        })))
    res = analyze_trace(tr["t_s"], tr["r"], tr["g"], tr["b"], mode, tier,
                        clipped_fraction={"any": tr["clipped_fraction"], **tr["clipped_fraction_channels"]},
                        extra_warnings=tr["warnings"])
    t_analyze = time.perf_counter() - t0 - t_extract
    res["video"] = _jsonable({
        **tr["video"],
        "fps_nominal": _r(tr["fps_nominal"], 3),
        "fps_measured": _r(tr["fps_measured"], 3),
        "frame_time_jitter_ms": _r(tr["frame_time_jitter_ms"]),
        "n_frames": tr["n_frames"],
        "dropped_frames": tr["dropped_frames"],
        "duplicate_timestamps": tr["duplicate_timestamps"],
        "clipped_fraction": _r(tr["clipped_fraction"], 4),
        "clipped_fraction_channels": {k: _r(v, 4) for k, v in tr["clipped_fraction_channels"].items()},
    })
    res["roi"] = tr["roi"]
    if tr["video"]["codec"] in ("hevc", "h264") and mode == "face":
        _warn(res["warnings"], "compressed_video", f"{tr['video']['codec'].upper()} compression removes "
                                                   "part of the tiny skin-colour pulse; expect several "
                                                   "bpm of error.", "info")
    _keyframe_check(res, tr["video"].get("keyframe_interval_s"))
    timing = {"extract_s": round(t_extract, 2), "analyze_s": round(t_analyze, 2)}
    if mode == "face" and deep:
        status = deep_available()
        if not status["available"]:
            res["deep"] = {"available": False, "reason": status["reason"]}
        else:
            t1 = time.perf_counter()
            res["deep"] = run_deep_models(video_path, roi=roi, models=deep_models)
            timing["deep_s"] = round(time.perf_counter() - t1, 2)
            _deep_check(res, tr["video"].get("keyframe_interval_s"))
    elif mode == "face":
        res["deep"] = {"available": deep_available()["available"], "skipped": True}
    res["timing_s"] = timing
    return res
