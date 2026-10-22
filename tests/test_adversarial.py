"""Adversarial camera-PPG tests: harder SYNTHETIC videos and traces (no recordings of real people).

Videos are encoded with PyAV (libx264 / libx265, CRF rate control, ~1 s GOP) so frame timestamps can be
set freely (jitter, variable frame rate). Every scenario has a known true HR. A scenario passes when the
result is either accurate (within ``HR_TOL``) or *honest*: it carries a warning / a poor quality label
that tells the user not to trust it. Some scenarios must be accurate (``must_be_accurate``).

Also: an integration app with both routers (camera_api + pages_api), the /capture -> /camera/trace and
/upload -> /camera/video contracts, and the result renderer (common.js) run under Node on real results.

Run: ``uv run pytest tests/test_adversarial.py -q`` (about 2-4 min; ``PULSE_SKIP_DEEP=1`` skips open-rppg).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import synth  # noqa: E402
from pulse_exploration import camera, camera_api, pages_api  # noqa: E402

HR_TOL = 3.0
TB = Fraction(1, 90_000)
NODE = shutil.which("node")
STATIC = pages_api.STATIC

# Codes that tell the user "do not trust this number".
UNTRUSTED = {"low_quality", "hr_disagreement", "hr_near_keyframe_rate", "too_short", "finger_not_detected",
             "no_face_detected", "too_dark"}


# --------------------------------------------------------------------------------------------
# synthetic renderer with free timestamps
# --------------------------------------------------------------------------------------------

def encode(path: Path, times_s: np.ndarray, frame_fn: Callable[[int, float], np.ndarray], size: tuple[int, int],
           codec: str = "hevc", crf: int = 23, fps_nominal: float = 30.0) -> None:
    """Encode frames with presentation times ``times_s`` (need not be uniform)."""
    import av

    w, h = size
    lib = {"hevc": "libx265", "h264": "libx264"}[codec]
    gop = str(int(round(fps_nominal)))
    opts = {"crf": str(crf), "preset": "veryfast", "g": gop}
    if codec == "hevc":
        opts["x265-params"] = f"log-level=error:keyint={gop}:min-keyint={gop}:scenecut=0"
    else:
        opts["sc_threshold"] = "0"
    with av.open(str(path), "w") as out:
        st = out.add_stream(lib, rate=Fraction(round(fps_nominal * 1000), 1000), options=opts)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.time_base = TB
        st.codec_context.time_base = TB
        last = -1
        for i, t in enumerate(times_s):
            fr = av.VideoFrame.from_ndarray(frame_fn(i, float(t)), format="rgb24")
            pts = max(last + 1, int(round(t / TB)))
            fr.pts, fr.time_base, last = pts, TB, pts
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)


def uniform_times(duration_s: float, fps: float) -> np.ndarray:
    return np.arange(int(round(duration_s * fps))) / fps


@dataclass
class Scene:
    """Fingertip or face patch with optional disturbances. All multipliers are functions of time."""
    mode: str
    hr_bpm: float
    duration_s: float = 30.0
    size: tuple[int, int] = (320, 180)
    seed: int = 0
    amplitude: float | None = None  # face: relative green p-p; finger: scales the default AC depth
    clipped_red: float = 0.0  # finger: red base above 255 (0 = 200, the normal case)
    drift: Callable[[float], float] | None = None  # brightness multiplier (auto exposure)
    flicker_hz: float | None = None  # mains light flicker (rolling shutter, short exposure)
    flicker_depth: float = 0.0
    motion: list[tuple[float, float]] = field(default_factory=list)  # (start, end) bursts
    no_pulse: bool = False

    def build(self):
        rng = np.random.default_rng(self.seed)
        w, h = self.size
        onsets = synth.beat_times(self.duration_s + 2, self.hr_bpm, rng)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        noise = synth._NoiseBank(rng, h, w, sigma=2.0 if self.mode == "finger" else 1.5, n=32)
        row_t = (yy[:, 0] / h) * 0.025  # 25 ms rolling-shutter readout, top to bottom
        exposure = 1 / 250.0
        if self.mode == "finger":
            r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            tex = synth._textured(rng, h, w, scales=(8, 32), amp=0.03)
            base = np.array([200.0 + self.clipped_red, 38.0, 22.0], np.float32)
            static = ((1.0 - 0.25 * r2) * (1 + tex))[..., None] * base
            ac = np.array([0.015, 0.03, 0.02], np.float32) * (self.amplitude or 1.0)
            mask = np.ones((h, w, 1), np.float32)
            roi = None
        else:
            bg = 110 + 70 * synth._textured(rng, h, w, scales=(3, 9, 27, 81))
            background = np.stack([bg * 0.9, bg, bg * 1.1], axis=-1).astype(np.float32)
            cx, cy, ax, ay = w / 2, h / 2, w * 0.18, h * 0.38
            inside = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1.0
            shade = 1.0 - 0.12 * (((xx - cx) / ax) ** 2) + synth._textured(rng, h, w, scales=(2, 6), amp=0.04)
            skin = shade[..., None] * np.array([190.0, 140.0, 115.0], np.float32)
            static = np.where(inside[..., None], skin, background)
            ac = (self.amplitude or 0.004) * np.array([0.33, 0.77, 0.53], np.float32) / 0.77
            mask = inside[..., None].astype(np.float32)
            roi = (int(cx - ax * 0.6), int(cy - ay * 0.6), int(ax * 1.2), int(ay * 1.2))

        def frame(i: int, t: float) -> np.ndarray:
            p = 0.0 if self.no_pulse else float(synth.ppg_waveform(np.array([t]), onsets)[0])
            if self.mode == "face":
                p -= 0.5
            gain = 1 - ac * p  # per channel
            light = self.drift(t) if self.drift else 1.0
            img = static
            if any(a <= t < b for a, b in self.motion):
                if self.mode == "finger":  # pressure change: darker, less red
                    light *= 0.75 + 0.1 * np.sin(2 * np.pi * 3 * t)
                    gain = gain * np.array([0.9, 1.15, 1.1], np.float32)
                else:  # head moves sideways
                    img = np.roll(static, int(0.08 * w * np.sin(2 * np.pi * 1.5 * t)), axis=1)
            f = img * (1 + mask * (gain - 1)) * light
            if self.flicker_hz:
                om = 2 * np.pi * self.flicker_hz
                ts = t + row_t
                avg = (np.cos(om * ts) - np.cos(om * (ts + exposure))) / (om * exposure)
                f = f * (1 + self.flicker_depth * avg)[:, None, None].astype(np.float32)
            return np.clip(f + noise(), 0, 255).astype(np.uint8)

        return frame, roi


@dataclass
class Case:
    name: str
    scene: Scene
    fps: float = 30.0
    codec: str = "hevc"
    crf: int = 23
    times: Callable[[float, float, np.random.Generator], np.ndarray] | None = None
    use_roi: bool = True  # face: pass the patch ROI (the synthetic patch is not a face)
    must_be_accurate: bool = False
    must_warn: set[str] = field(default_factory=set)  # at least one of these codes must fire


def jitter_times(sd_ms: float):
    def f(duration, fps, rng):
        t = uniform_times(duration, fps) + rng.normal(0, sd_ms / 1000, int(round(duration * fps)))
        return np.sort(t - t.min())
    return f


def vfr_times(duration, fps, rng):
    """30 fps, dropping to 20 fps in the middle third (low light), plus 2 ms jitter."""
    out, t = [], 0.0
    while t < duration:
        out.append(t + rng.normal(0, 0.002))
        t += 1 / 20 if duration / 3 <= t < 2 * duration / 3 else 1 / fps
    return np.sort(np.asarray(out) - min(out))


def slow_drift(t):  # auto exposure creeping up 25 % over 30 s with a slow wobble
    return 1.0 + 0.25 * t / 30 + 0.03 * np.sin(2 * np.pi * t / 11)


def ae_steps(t):  # auto exposure hunting: 6 % steps every 7 s
    return 1.0 + 0.06 * (int(t / 7) % 2)


CASES = [
    Case("finger_hevc_crf28", Scene("finger", 72), crf=28, must_be_accurate=True),
    Case("finger_hevc_crf35", Scene("finger", 72, seed=1), crf=35),  # 60 bpm key-frame lock, flagged
    Case("face_hevc_crf28", Scene("face", 72, seed=2), crf=28, must_warn={"hr_near_keyframe_rate", "low_quality"}),
    Case("face_hevc_crf35", Scene("face", 72, seed=3), crf=35),
    Case("finger_flicker100_30fps", Scene("finger", 72, seed=4, flicker_hz=100, flicker_depth=0.05),
         must_be_accurate=True),
    # 100 Hz flicker sampled at 25.5 fps (low-light frame rate) aliases to 2 Hz = 120 bpm.
    Case("finger_flicker100_25.5fps", Scene("finger", 72, seed=5, flicker_hz=100, flicker_depth=0.05), fps=25.5),
    Case("face_flicker100_25.5fps", Scene("face", 72, seed=6, flicker_hz=100, flicker_depth=0.05), fps=25.5),
    Case("finger_flicker50_30fps", Scene("finger", 72, seed=7, flicker_hz=50, flicker_depth=0.05)),
    Case("finger_ae_drift", Scene("finger", 72, seed=8, drift=slow_drift), must_warn={"exposure_drift"}),
    Case("finger_ae_steps", Scene("finger", 72, seed=9, drift=ae_steps), must_warn={"exposure_steps"}),
    Case("face_ae_drift", Scene("face", 66, seed=10, drift=slow_drift), must_warn={"exposure_drift"}),
    Case("finger_jitter8ms", Scene("finger", 80, seed=11), times=jitter_times(8), must_be_accurate=True,
         must_warn={"irregular_frame_times"}),
    Case("finger_vfr_30_20", Scene("finger", 80, seed=12), times=vfr_times, must_be_accurate=True),
    Case("finger_motion", Scene("finger", 72, seed=13, motion=[(8, 10), (19, 20.5)]), must_warn=UNTRUSTED),
    Case("face_motion", Scene("face", 72, seed=14, motion=[(8, 10), (19, 20.5)])),
    Case("finger_red_clipped", Scene("finger", 72, seed=15, clipped_red=100), must_be_accurate=True,
         must_warn={"clipped"}),
    Case("finger_red_partly_clipped", Scene("finger", 72, seed=16, clipped_red=70), must_warn={"clipped"}),
    Case("finger_low_perfusion_crf30", Scene("finger", 72, seed=26, amplitude=0.25), crf=30),
    Case("finger_low_perfusion_60bpm_crf30", Scene("finger", 60, seed=27, amplitude=0.25), crf=30),
    Case("finger_25fps", Scene("finger", 72, seed=17), fps=25, must_be_accurate=True),
    Case("finger_60fps", Scene("finger", 72, seed=18), fps=60, must_be_accurate=True),
    Case("face_60fps_h264", Scene("face", 72, seed=19), fps=60, codec="h264"),
    Case("finger_45bpm", Scene("finger", 45, seed=20, duration_s=40), must_be_accurate=True),
    Case("finger_150bpm", Scene("finger", 150, seed=21), must_be_accurate=True),
    Case("face_45bpm", Scene("face", 45, seed=22, duration_s=40)),
    Case("face_150bpm", Scene("face", 150, seed=23)),  # reads 60 (key frames); must be labelled poor
    Case("finger_no_pulse", Scene("finger", 72, seed=24, no_pulse=True), must_warn=UNTRUSTED),
    Case("face_no_pulse_h264", Scene("face", 72, seed=25, no_pulse=True), codec="h264", must_warn=UNTRUSTED),
]
BY_NAME = {c.name: c for c in CASES}


def run_case(case: Case, tmp: Path, deep: bool = False) -> tuple[dict, float]:
    rng = np.random.default_rng(1000 + case.scene.seed)
    times = (case.times or (lambda d, f, r: uniform_times(d, f)))(case.scene.duration_s, case.fps, rng)
    frame_fn, roi = case.scene.build()
    path = tmp / f"{case.name}.mp4"
    encode(path, times, frame_fn, case.scene.size, case.codec, case.crf, case.fps)
    res = camera.analyze_video(path, case.scene.mode, deep=deep, roi=roi if case.use_roi else None)
    return res, case.scene.hr_bpm


def codes(res: dict) -> set[str]:
    return {w["code"] for w in res["warnings"]}


def is_honest(res: dict, true_hr: float) -> tuple[bool, bool]:
    """(accurate, flagged). A result is honest when accurate or flagged."""
    hr = res["hr_bpm"]
    accurate = hr is not None and abs(hr - true_hr) <= HR_TOL
    flagged = hr is None or res["quality"]["label"] == "poor" or bool(codes(res) & UNTRUSTED)
    return accurate, flagged


@pytest.fixture(scope="module")
def tmp_videos(tmp_path_factory):
    return tmp_path_factory.mktemp("adversarial_videos")


@pytest.mark.parametrize("name", [c.name for c in CASES])
def test_adversarial_video(name, tmp_videos):
    case = BY_NAME[name]
    res, true_hr = run_case(case, tmp_videos)
    json.dumps(res, allow_nan=False)
    accurate, flagged = is_honest(res, true_hr)
    summary = (f"{name}: true {true_hr} fft {res['hr_fft_bpm']} peaks {res['hr_peaks_bpm']} "
               f"q={res['quality']['label']} warn={sorted(codes(res))}")
    assert accurate or flagged, "dishonest: " + summary
    if case.must_be_accurate:
        assert accurate, summary
    if case.must_warn:
        assert codes(res) & case.must_warn, summary
    if not accurate and "hr_near_keyframe_rate" in codes(res):
        assert res["quality"]["label"] == "poor", summary
    if accurate and res["quality"]["label"] == "good":
        assert res["hr_peaks_bpm"] is not None and abs(res["hr_peaks_bpm"] - true_hr) <= 5, summary


# --------------------------------------------------------------------------------------------
# traces (tier "capture")
# --------------------------------------------------------------------------------------------

def make_trace(hr=72.0, dur=45.0, fps=30.0, jitter_ms=3.0, seed=0, times=None, red=200.0, ac=1.0,
               drift=None, motion=()):
    rng = np.random.default_rng(seed)
    onsets = synth.beat_times(dur + 2, hr, rng)
    n = int(dur * fps)
    t = times if times is not None else np.sort(np.arange(n) / fps + rng.normal(0, jitter_ms / 1000, n))
    t = t - t[0]
    p = synth.ppg_waveform(t, onsets)
    light = np.array([drift(x) for x in t]) if drift else np.ones_like(t)
    for a, b in motion:
        sel = (t >= a) & (t < b)
        light[sel] *= 0.7 + 0.1 * np.sin(2 * np.pi * 3 * t[sel])
    noise = lambda: rng.normal(0, 0.05, len(t))  # noqa: E731
    r = np.minimum(red * (1 - 0.015 * ac * p) * light + noise(), 255)
    g = 38 * (1 - 0.03 * ac * p) * light + noise()
    b = 22 * (1 - 0.02 * ac * p) * light + noise()
    return t, r, g, b


def _drops(rng, dur):
    tt = np.cumsum(rng.choice([1 / 30] * 3 + [1 / 15], size=int(dur * 40)))
    return tt[tt < dur]


TRACE_CASES = {
    # name: (kwargs, analyze kwargs, true HR, must be accurate, required warning codes)
    "jitter10ms": (dict(jitter_ms=10), {}, 72, True, {"irregular_frame_times"}),
    "dropped_frames": (dict(times=_drops(np.random.default_rng(5), 45)), {}, 72, True, {"irregular_frame_times"}),
    "fps15": (dict(fps=15), {}, 72, True, {"low_fps"}),
    "fps60": (dict(fps=60), {}, 72, True, set()),
    "hr45": (dict(hr=45), {}, 45, True, set()),
    "hr150": (dict(hr=150), {}, 150, True, set()),
    "red_clipped": (dict(red=290), dict(clipped_fraction={"any": 0.9}), 72, True, {"clipped"}),
    "ae_drift": (dict(drift=lambda x: 1 + 0.3 * x / 45), {}, 72, True, {"exposure_drift"}),
    "ae_steps": (dict(drift=lambda x: 1 + 0.06 * (int(x / 7) % 2)), {}, 72, True, {"exposure_steps"}),
    "motion": (dict(motion=[(10, 12), (25, 26)]), {}, 72, False, UNTRUSTED),
    "no_pulse": (dict(ac=0), {}, 72, False, UNTRUSTED),
}


@pytest.mark.parametrize("name", list(TRACE_CASES))
def test_adversarial_trace(name):
    kw, akw, true_hr, must_acc, must_warn = TRACE_CASES[name]
    res = camera.analyze_trace(*make_trace(**kw), "finger", "capture", **akw)
    accurate, flagged = is_honest(res, true_hr)
    summary = f"{name}: fft {res['hr_fft_bpm']} peaks {res['hr_peaks_bpm']} q={res['quality']['label']} {sorted(codes(res))}"
    assert res["tier"] == "capture"
    assert accurate or flagged, "dishonest: " + summary
    if must_acc:
        assert accurate, summary
    if must_warn:
        assert codes(res) & must_warn, summary


def test_capture_any_channel_saturation_counts_as_red_clipping():
    """The capture page only sends the any-channel saturated share; in finger mode that is red."""
    res = camera.analyze_trace(*make_trace(red=240), "finger", "capture", clipped_fraction={"any": 0.4})
    assert res["red_clipped"] and res["primary_method"] == "green" and "clipped" in codes(res)
    assert abs(res["hr_bpm"] - 72) <= HR_TOL
    low = camera.analyze_trace(*make_trace(red=240), "finger", "capture", clipped_fraction={"any": 0.1})
    assert not low["red_clipped"] and "clipped" in codes(low)


def test_hrv_survives_timestamp_noise():
    """Signal sampled uniformly but reported with 10 ms timestamp noise (rVFC callback time)."""
    t, r, g, b = make_trace(hr=66, dur=90, jitter_ms=0, seed=3)
    clean = camera.analyze_trace(t, r, g, b, "finger", "capture")
    noisy = camera.analyze_trace(t + np.random.default_rng(9).normal(0, 0.010, len(t)), r, g, b, "finger", "capture")
    assert "irregular_frame_times" in codes(noisy)
    assert abs(noisy["hrv"]["RMSSD"] - clean["hrv"]["RMSSD"]) < 5


# --------------------------------------------------------------------------------------------
# integration: both routers in one throwaway app (server.py is not touched)
# --------------------------------------------------------------------------------------------

@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(camera_api, "CAMERA_DATA", tmp_path / "camera")
    app = FastAPI()
    app.include_router(camera_api.router)
    app.include_router(pages_api.router)
    return TestClient(app), tmp_path / "camera"


def test_routes_do_not_clash_with_server():
    from pulse_exploration import server

    seen: dict[tuple[str, str], str] = {}
    for src, routes in (("server", server.app.routes), ("camera", camera_api.router.routes),
                        ("pages", pages_api.router.routes)):
        for rt in routes:
            if getattr(rt, "path", None) is None:
                continue  # an included router; its routes are checked through the router itself
            for m in getattr(rt, "methods", None) or {"*"}:
                key = (rt.path, m)
                assert key not in seen, f"{key} defined by {seen.get(key)} and {src}"
                seen[key] = src


def test_pages_and_api_served_together(app_client):
    c, _ = app_client
    for path in ("/", "/upload", "/capture", "/static/common.js", "/static/signal.js"):
        assert c.get(path).status_code == 200, path
    assert c.get("/camera/video").status_code == 405  # POST only
    assert c.get("/static/..%2Fcamera_api.py").status_code == 404


NODE_FRAMES = r"""
// Frames shaped like capture.js onFrame() records them (rVFC metadata, mediaTime already in ms),
// assembled with the page's own buildTrace().
const SIG = require(process.argv[2]);
const [secs, fps, bpm, sat] = process.argv.slice(3).map(Number);
let seed = 7;
const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const frames = [];
for (let i = 0; i < secs * fps; i++) {
  const t = i * 1000 / fps;                       // ms
  const ph = ((t / 1000) * bpm / 60) % 1;
  const p = Math.exp(-0.5 * ((ph - 0.2) / 0.075) ** 2) + 0.45 * Math.exp(-0.5 * ((ph - 0.52) / 0.1) ** 2);
  frames.push({
    mediaTime: 81234.5 + t, captureTime: 123456.7 + t + rnd() * 2, expectedDisplayTime: 123470 + t,
    now: 123475 + t + rnd() * 4, presentedFrames: i,
    r: 200 * (1 - 0.012 * p) + rnd() * 0.1, g: 38 * (1 - 0.025 * p), b: 22, sat: sat,
  });
}
const meta = { page: "capture", tier: "capture", synthetic: true, user_agent: "node", capabilities: { torch: [true] } };
process.stdout.write(JSON.stringify(SIG.buildTrace(frames, "finger", meta)));
"""

NODE_RENDER = r"""
const UI = require(process.argv[2]);
const res = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
process.stdout.write(UI.resultHTML(res));
"""


def _node(script: str, tmp: Path, *args) -> str:
    js = tmp / f"h{abs(hash(script))}.js"
    js.write_text(script)
    r = subprocess.run([NODE, str(js), *map(str, args)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _render(tmp: Path, res: dict) -> str:
    f = tmp / "res.json"
    f.write_text(json.dumps(res))
    return _node(NODE_RENDER, tmp, STATIC / "common.js", f)


def _check_rendered(html: str, res: dict) -> None:
    assert f"tier: {res['tier']}" in html
    if res["hr_bpm"] is not None:
        assert f'class="hr-value">{round(res["hr_bpm"])}<' in html
    else:
        assert "No heart rate" in html
    from html import escape
    for w in res["warnings"]:
        assert escape(w["message"]) in html, w["code"]
    assert f"quality: {res['quality']['label']}" in html
    assert 'data-wave="1"' in html  # waveform{t_s, values} parsed


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("secs,fps,sat", [(45, 30, 0.0), (120, 30, 0.0), (60, 60, 0.0), (45, 30, 0.5)])
def test_capture_page_payload_end_to_end(app_client, tmp_path, secs, fps, sat):
    """capture.js buildTrace() -> POST /camera/trace -> common.js resultHTML(), on the real code."""
    c, store = app_client
    body = json.loads(_node(NODE_FRAMES, tmp_path, STATIC / "signal.js", secs, fps, 75, sat))
    assert set(body) == {"mode", "t_ms", "r", "g", "b", "meta"}
    assert body["meta"]["timestamp_source"] == "mediaTime"
    assert body["t_ms"][1] == pytest.approx(1000 / fps, abs=0.01)  # milliseconds
    r = c.post("/camera/trace", json=body)
    assert r.status_code == 200, r.text[:500]
    res = r.json()
    assert res["tier"] == "capture" and res["mode"] == "finger"
    assert abs(res["hr_bpm"] - 75) <= HR_TOL
    assert res["fps_measured"] == pytest.approx(fps, abs=0.5)
    assert res["meta"]["saturation"]["mean_fraction"] == sat
    assert res["red_clipped"] == (sat >= 0.25)
    assert (res["hrv"] is not None) == (secs >= 60)
    saved = next(store.iterdir())
    assert (saved / "trace.json").exists() and (saved / "result.json").exists()
    _check_rendered(_render(tmp_path, res), res)


def test_trace_endpoint_rejects_bad_payloads(app_client):
    c, _ = app_client
    t, r, g, b = make_trace(dur=30)
    good = {"mode": "finger", "t_ms": (t * 1000).tolist(), "r": r.tolist(), "g": g.tolist(), "b": b.tolist(), "meta": {}}
    assert c.post("/camera/trace", json=good).status_code == 200
    bad = {
        "mismatched_lengths": dict(good, r=good["r"][:-5]),
        "t_in_seconds": dict(good, t_ms=t.tolist()),  # spans 30 ms -> "need 2 s"
        "nan": None,
        "channel_out_of_range": dict(good, g=[300.0] * len(t)),
        "bad_mode": dict(good, mode="ear"),
        "too_few": dict(good, t_ms=good["t_ms"][:5], r=good["r"][:5], g=good["g"][:5], b=good["b"][:5]),
        "missing_field": {k: v for k, v in good.items() if k != "b"},
        "meta_too_big": dict(good, meta={"x": "y" * (camera_api.MAX_META_BYTES + 10)}),
    }
    for name, body in bad.items():
        if name == "nan":
            raw = json.dumps(good).replace(str(good["r"][3]), "NaN", 1)
            resp = c.post("/camera/trace", content=raw, headers={"content-type": "application/json"})
        else:
            resp = c.post("/camera/trace", json=body)
        assert resp.status_code == 422, (name, resp.status_code, resp.text[:200])
        assert "detail" in resp.json()


def test_upload_page_fields_match_video_endpoint(app_client, tmp_path):
    """The multipart field names upload.js sends are the ones /camera/video reads; result renders."""
    c, store = app_client
    js = (STATIC / "upload.js").read_text()
    fields = re.findall(r'form\.append\("([a-z_]+)"', js)
    assert sorted(fields) == ["file", "mode"]
    case = Case("api_finger", Scene("finger", 84, seed=40, duration_s=25))
    frame_fn, _ = case.scene.build()
    path = tmp_path / "rec.mp4"
    encode(path, uniform_times(case.scene.duration_s, 30), frame_fn, case.scene.size)
    with path.open("rb") as fh:
        r = c.post("/camera/video", data={"mode": "finger"},
                   files={"file": ("20261022_101500.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text[:300]
    res = r.json()
    assert res["tier"] == "video" and abs(res["hr_bpm"] - 84) <= HR_TOL
    assert res["upload"]["bytes"] == path.stat().st_size
    assert (store / res["id"] / "video.mp4").exists()
    if NODE:
        _check_rendered(_render(tmp_path, res), res)
    # a phone gallery picker may send no content type at all
    with path.open("rb") as fh:
        r2 = c.post("/camera/video", data={"mode": "face", "deep": "false"}, files={"file": ("clip", fh, "")})
    assert r2.status_code == 200, r2.text[:300]
    assert r2.json()["deep"]["skipped"] is True


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_short_and_error_results_render(app_client, tmp_path):
    c, _ = app_client
    t, r, g, b = make_trace(dur=12)
    res = c.post("/camera/trace", json={"mode": "face", "t_ms": (t * 1000).tolist(), "r": r.tolist(),
                                         "g": g.tolist(), "b": b.tolist(), "meta": {}}).json()
    assert res["hr_bpm"] is None and "too_short" in codes(res)
    _check_rendered(_render(tmp_path, res), res)


@pytest.mark.skipif(os.environ.get("PULSE_SKIP_DEEP") == "1", reason="PULSE_SKIP_DEEP=1")
def test_deep_models_on_compressed_face(tmp_videos):
    """open-rppg on a harsh (CRF 30) face patch: whatever it says, it must not crash and must be reported."""
    case = Case("deep_face_crf30", Scene("face", 72, seed=50), crf=30)
    res, _ = run_case(case, tmp_videos, deep=True)
    json.dumps(res, allow_nan=False)
    models = res["deep"]["models"]
    assert set(models) == set(camera.DEEP_MODELS)
    for rec in models.values():
        assert "hr_bpm" in rec or "error" in rec
        assert rec.get("hrv_reliable") in (None, False)
    hrs = [m["hr_bpm"] for m in models.values() if m.get("hr_bpm") is not None]
    deep_ok = hrs and max(hrs) - min(hrs) <= 5 and all(abs(h - 72) <= HR_TOL for h in hrs)
    assert deep_ok or codes(res) & {"deep_models_disagree", "deep_hr_near_keyframe_rate", "deep_unconfirmed"}, models


def test_deep_check_flags_keyframe_and_disagreement():
    good = {"label": "good"}
    res = {"warnings": [], "hr_bpm": 75.0, "quality": good,
           "deep": {"models": {"A": {"hr_bpm": 60.4}, "B": {"hr_bpm": 75.0}, "C": {"error": "x"}}}}
    camera._deep_check(res, 1.0)
    assert res["deep"]["models"]["A"]["keyframe_artifact_suspected"]
    assert "keyframe_artifact_suspected" not in res["deep"]["models"]["B"]
    assert codes(res) == {"deep_hr_near_keyframe_rate", "deep_models_disagree", "deep_unconfirmed"}
    ok = {"warnings": [], "hr_bpm": 72.5, "quality": good,
          "deep": {"models": {"A": {"hr_bpm": 72.0}, "B": {"hr_bpm": 73.5}}}}
    camera._deep_check(ok, 1.0)
    assert ok["warnings"] == []
    # A single model (the server-side default) is checked against POS/CHROM instead.
    single = {"warnings": [], "hr_bpm": 72.0, "quality": good, "deep": {"models": {"ME-chunk.rlap": {"hr_bpm": 83.0}}}}
    camera._deep_check(single, 1.0)
    assert codes(single) == {"deep_unconfirmed"}
    poor = {"warnings": [], "hr_bpm": 72.0, "quality": {"label": "poor"}, "deep": {"models": {"ME-chunk.rlap": {"hr_bpm": 72.0}}}}
    camera._deep_check(poor, 1.0)
    assert codes(poor) == {"deep_unconfirmed"}


def test_periodic_scene_without_finger_is_poor():
    """What the emulator camera produced: a clean 2 Hz blink in a green-dominant scene. It looks like a
    perfect 120 bpm pulse, but with no fingertip on the lens it must not be reported as good quality."""
    fs, n = 30.0, 750
    t = np.arange(n) / fs
    blink = (np.sin(2 * np.pi * 2.0 * t) > 0).astype(float)
    r, g, b = 166 + 4 * blink, 192 + 4 * blink, 124 + 4 * blink
    res = camera.analyze_trace(t, r, g, b, "finger", tier="capture")
    assert "finger_not_detected" in codes(res)
    assert res["quality"]["label"] == "poor"


def test_face_square_wave_scene_is_poor():
    """The emulator's front camera: a 3 Hz blink swinging ~4 % of the skin level reads as a clean 180 bpm."""
    fs, n = 30.0, 750
    t = np.arange(n) / fs
    blink = (np.sin(2 * np.pi * 3.0 * t) > 0).astype(float)
    for swing in (5.0, 2.4):  # ~3.7 % and the emulator's measured ~1.6 % of the green level
        r, g, b = 121 + 4 * blink, 135 + swing * blink, 75 + 2 * blink
        res = camera.analyze_trace(t, r, g, b, "face", tier="capture")
        assert "implausible_face_amplitude" in codes(res), swing
        assert res["quality"]["label"] == "poor", swing
