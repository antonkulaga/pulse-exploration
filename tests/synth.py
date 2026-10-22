"""Synthetic camera-PPG test material: no real people, only generated pixels.

Videos are rendered with numpy and piped as raw RGB frames into the ffmpeg CLI, which encodes them
the way a phone does (HEVC via libx265 or H.264 via libx264, yuv420p, ~1 s GOP, bitrate capped).

Resolution and bitrate: rendering 1080p frames with per-pixel sensor noise in numpy is slow, so the
default is 640x360. The default bitrate keeps the *bits per pixel* of a phone recording at
20 Mbps 1080p30 (~0.32 bit/pixel, about 2.2 Mbps at 640x360) so the encoder throws away as much
fine detail per pixel as it would on the phone. Pass ``bitrate=20_000_000`` for the literal value.

Ground truth is returned by every generator so tests can compare against it.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PHONE_BITRATE_1080P = 20_000_000  # bit/s at 1920x1080, 30 fps
CODECS = {"hevc": "libx265", "h264": "libx264"}


@dataclass
class Truth:
    hr_bpm: float
    resp_rate_brpm: float
    beat_times_s: np.ndarray
    duration_s: float
    fps: float
    roi: tuple[int, int, int, int] | None = None
    extra: dict = field(default_factory=dict)


def phone_equivalent_bitrate(width: int, height: int, fps: float = 30.0) -> int:
    """Bitrate giving the same bits per pixel as a 20 Mbps 1080p30 phone recording."""
    return int(PHONE_BITRATE_1080P * (width * height) / (1920 * 1080) * (fps / 30.0))


def beat_times(duration_s: float, hr_bpm: float, rng: np.random.Generator,
               resp_rate_brpm: float = 15.0, rsa: float = 0.04, jitter: float = 0.01) -> np.ndarray:
    """Beat onset times with respiratory sinus arrhythmia (IBI modulated at the breathing rate)."""
    mean_ibi = 60.0 / hr_bpm
    f_resp = resp_rate_brpm / 60.0
    t, out = -mean_ibi * rng.uniform(0, 1), []
    while t < duration_s + 2 * mean_ibi:
        out.append(t)
        ibi = mean_ibi * (1 + rsa * np.sin(2 * np.pi * f_resp * t) + jitter * rng.standard_normal())
        t += ibi
    return np.asarray(out)


def _beat_shape(ph: np.ndarray) -> np.ndarray:
    systolic = np.exp(-0.5 * ((ph - 0.20) / 0.075) ** 2)
    diastolic = 0.45 * np.exp(-0.5 * ((ph - 0.52) / 0.10) ** 2)
    return systolic + diastolic


_SHAPE_GRID = _beat_shape(np.linspace(0.0, 1.0, 1001))
_SHAPE_MIN, _SHAPE_MAX = float(_SHAPE_GRID.min()), float(_SHAPE_GRID.max())


def ppg_waveform(t: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Blood-volume pulse in [0, 1]: fast systolic upstroke, dicrotic notch, smaller diastolic wave."""
    t = np.atleast_1d(np.asarray(t, float))
    idx = np.clip(np.searchsorted(onsets, t, side="right") - 1, 0, len(onsets) - 2)
    ibi = onsets[idx + 1] - onsets[idx]
    ph = (t - onsets[idx]) / ibi
    return (_beat_shape(ph) - _SHAPE_MIN) / (_SHAPE_MAX - _SHAPE_MIN)


class _NoiseBank:
    """Per-pixel sensor noise. Drawing fresh Gaussian noise for every frame dominates render time,
    so frames pick one of ``n`` pre-drawn noise fields at random (no periodicity)."""

    def __init__(self, rng: np.random.Generator, h: int, w: int, sigma: float, n: int = 48):
        self.rng = rng
        self.bank = np.clip(np.round(rng.standard_normal((n, h, w, 3), dtype=np.float32) * sigma),
                            -127, 127).astype(np.int8)

    def __call__(self) -> np.ndarray:
        return self.bank[self.rng.integers(len(self.bank))]


def _ffmpeg_writer(path: Path, width: int, height: int, fps: float, codec: str, bitrate: int):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    lib = CODECS[codec]
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", lib, "-b:v", str(bitrate), "-maxrate", str(bitrate), "-bufsize", str(2 * bitrate),
        "-g", str(int(round(fps))), "-pix_fmt", "yuv420p", "-preset", "veryfast",
    ]
    if codec == "hevc":
        cmd += ["-x265-params", "log-level=error", "-tag:v", "hvc1"]
    cmd += ["-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _write(path: Path, frames, width: int, height: int, fps: float, codec: str, bitrate: int | None) -> None:
    bitrate = bitrate or phone_equivalent_bitrate(width, height, fps)
    proc = _ffmpeg_writer(Path(path), width, height, fps, codec, bitrate)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with {rc} writing {path}")


def _textured(rng: np.random.Generator, h: int, w: int, scales=(4, 16, 48), amp=1.0) -> np.ndarray:
    """Static multi-scale texture in roughly [-1, 1] (value noise, bilinear upsampled)."""
    out = np.zeros((h, w), np.float32)
    for s in scales:
        gh, gw = h // s + 2, w // s + 2
        coarse = rng.standard_normal((gh, gw)).astype(np.float32)
        yi = np.linspace(0, gh - 1.001, h)
        xi = np.linspace(0, gw - 1.001, w)
        y0, x0 = yi.astype(int), xi.astype(int)
        fy, fx = (yi - y0)[:, None], (xi - x0)[None, :]
        c = coarse
        up = (c[y0][:, x0] * (1 - fy) * (1 - fx) + c[y0 + 1][:, x0] * fy * (1 - fx)
              + c[y0][:, x0 + 1] * (1 - fy) * fx + c[y0 + 1][:, x0 + 1] * fy * fx)
        out += up
    out /= np.abs(out).max() + 1e-9
    return out * amp


def make_finger_video(path: str | Path, hr_bpm: float = 72.0, duration_s: float = 30.0, fps: float = 30.0,
                      codec: str = "hevc", size: tuple[int, int] = (640, 360), clipped: bool = False,
                      resp_rate_brpm: float = 15.0, bitrate: int | None = None, seed: int = 0) -> Truth:
    """Fingertip over lens + torch: red-dominant frame whose brightness dips with each pulse.

    ``clipped=True`` drives the red channel past 255 so it saturates (the over-exposed case).
    """
    rng = np.random.default_rng(seed)
    w, h = size
    onsets = beat_times(duration_s, hr_bpm, rng, resp_rate_brpm)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    vignette = 1.0 - 0.25 * r2  # torch hot-spot in the middle
    tex = _textured(rng, h, w, scales=(8, 32), amp=0.03)
    base = np.array([300.0 if clipped else 200.0, 38.0, 22.0], np.float32)  # R G B
    ac = np.array([0.015, 0.03, 0.02], np.float32)  # pulsatile modulation depth per channel
    static = (vignette * (1 + tex))[..., None] * base
    noise = _NoiseBank(rng, h, w, sigma=2.0)
    n = int(round(duration_s * fps))
    pulse = ppg_waveform(np.arange(n) / fps, onsets)

    def frames():
        for i in range(n):
            t = i / fps
            resp = 0.004 * np.sin(2 * np.pi * resp_rate_brpm / 60 * t)  # respiratory baseline wander
            gain = ((1 - ac * pulse[i]) * (1 + resp)).astype(np.float32)
            f = static * gain + noise()
            yield np.clip(f, 0, 255).astype(np.uint8)

    _write(Path(path), frames(), w, h, fps, codec, bitrate)
    return Truth(hr_bpm, resp_rate_brpm, onsets, duration_s, fps)


def make_face_video(path: str | Path, hr_bpm: float = 72.0, duration_s: float = 30.0, fps: float = 30.0,
                    codec: str = "hevc", size: tuple[int, int] = (640, 360), amplitude: float = 0.004,
                    skin_rgb: tuple[float, float, float] = (190.0, 140.0, 115.0),
                    resp_rate_brpm: float = 15.0, bitrate: int | None = None, seed: int = 1) -> Truth:
    """A skin-coloured patch on a textured background; the patch colour pulses with the BVP.

    ``amplitude`` is the peak-to-peak relative change of the green channel (0.004 = 0.4 %, i.e.
    about half a grey level on 8-bit video, which is what remote PPG has to work with). The other
    channels follow the blood-volume-pulse signature of skin (de Haan & van Leest 2014,
    Pbv ~ [0.33, 0.77, 0.53]). Slow illumination drift and per-pixel sensor noise are added.
    The returned ``roi`` is an inset of the patch, as (x, y, w, h) in pixels.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    onsets = beat_times(duration_s, hr_bpm, rng, resp_rate_brpm)
    bg = 110 + 70 * _textured(rng, h, w, scales=(3, 9, 27, 81))
    background = np.stack([bg * 0.9, bg, bg * 1.1], axis=-1).astype(np.float32)
    # skin patch, elliptical, centred, with its own fine texture (pores, shading)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, ax, ay = w / 2, h / 2, w * 0.18, h * 0.38
    inside = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
    shade = 1.0 - 0.12 * (((xx - cx) / ax) ** 2) + _textured(rng, h, w, scales=(2, 6), amp=0.04)
    skin = shade[..., None] * np.array(skin_rgb, np.float32)
    static = np.where(inside[..., None], skin, background)
    pbv = np.array([0.33, 0.77, 0.53], np.float32) / 0.77  # relative to green
    roi = (int(cx - ax * 0.6), int(cy - ay * 0.6), int(ax * 1.2), int(ay * 1.2))
    noise = _NoiseBank(rng, h, w, sigma=1.5)
    mask = inside[..., None].astype(np.float32)
    n = int(round(duration_s * fps))
    pulse = ppg_waveform(np.arange(n) / fps, onsets) - 0.5

    def frames():
        for i in range(n):
            t = i / fps
            light = 1 + 0.01 * np.sin(2 * np.pi * t / 23.0)  # slow lighting drift
            gain = (light * (1 + mask * (-amplitude * pbv * pulse[i]))).astype(np.float32)
            f = static * gain + noise()
            yield np.clip(f, 0, 255).astype(np.uint8)

    _write(Path(path), frames(), w, h, fps, codec, bitrate)
    return Truth(hr_bpm, resp_rate_brpm, onsets, duration_s, fps, roi=roi,
                 extra={"amplitude": amplitude, "skin_rgb": skin_rgb})


def make_trace(hr_bpm: float = 72.0, duration_s: float = 40.0, fps: float = 30.0, jitter_ms: float = 3.0,
               seed: int = 2) -> tuple[dict, Truth]:
    """Per-frame RGB means as the browser capture page would send them (fingertip, torch on).

    Frame timestamps have jitter like requestVideoFrameCallback timestamps do.
    """
    rng = np.random.default_rng(seed)
    onsets = beat_times(duration_s, hr_bpm, rng)
    n = int(duration_s * fps)
    t = np.arange(n) / fps + rng.normal(0, jitter_ms / 1000, n)
    t = np.sort(t - t[0])
    p = ppg_waveform(t, onsets)
    r = 200 * (1 - 0.015 * p) + rng.normal(0, 0.05, n)
    g = 38 * (1 - 0.03 * p) + rng.normal(0, 0.05, n)
    b = 22 * (1 - 0.02 * p) + rng.normal(0, 0.05, n)
    body = {"mode": "finger", "t_ms": (t * 1000).round(3).tolist(), "r": r.round(4).tolist(),
            "g": g.round(4).tolist(), "b": b.round(4).tolist(),
            "meta": {"device": "synthetic", "torch": True, "exposure_locked": True}}
    return body, Truth(hr_bpm, 15.0, onsets, duration_s, fps)
