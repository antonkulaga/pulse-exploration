"""Camera-PPG backend tests on synthetic videos only (no recordings of real people).

Run: ``uv run pytest -q``. Set ``PULSE_SKIP_DEEP=1`` to skip the open-rppg test (~1 min on CPU).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import synth  # noqa: E402  (tests/ is on sys.path under pytest)
from pulse_exploration import camera, camera_api  # noqa: E402

HR_TOL = 3.0


def codes(res: dict) -> set[str]:
    return {w["code"] for w in res["warnings"]}


def assert_json_safe(res: dict) -> None:
    json.dumps(res, allow_nan=False)


@pytest.fixture(scope="session")
def vids(tmp_path_factory):
    """Render each synthetic video once per session (lazily)."""
    root = tmp_path_factory.mktemp("synthetic_videos")
    cache: dict[str, tuple[Path, synth.Truth]] = {}
    specs = {
        "finger_hevc": (synth.make_finger_video, dict(hr_bpm=72, codec="hevc")),
        "finger_h264": (synth.make_finger_video, dict(hr_bpm=88, codec="h264", seed=3)),
        "finger_clipped": (synth.make_finger_video, dict(hr_bpm=72, codec="hevc", clipped=True, seed=4)),
        "finger_10s": (synth.make_finger_video, dict(hr_bpm=72, duration_s=10, codec="hevc", seed=5)),
        "face_hevc": (synth.make_face_video, dict(hr_bpm=72, codec="hevc")),
        "face_h264": (synth.make_face_video, dict(hr_bpm=66, codec="h264", seed=6)),
    }

    def get(name: str) -> tuple[Path, synth.Truth]:
        if name not in cache:
            fn, kw = specs[name]
            path = root / f"{name}.mp4"
            cache[name] = (path, fn(path, **kw))
        return cache[name]

    return get


# --------------------------------------------------------------------------------------------
# fingertip
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["finger_hevc", "finger_h264"])
def test_finger_hr(vids, name):
    path, truth = vids(name)
    res = camera.analyze_video(path, "finger", deep=False)
    assert_json_safe(res)
    assert res["tier"] == "video" and res["mode"] == "finger"
    assert abs(res["hr_bpm"] - truth.hr_bpm) <= HR_TOL, res["hr_bpm"]
    assert abs(res["hr_peaks_bpm"] - truth.hr_bpm) <= HR_TOL, res["hr_peaks_bpm"]
    assert res["primary_method"] == "red"
    assert res["quality"]["label"] == "good"
    assert "clipped" not in codes(res)
    assert res["hrv"] is None and not res["hrv_reliable"]  # 30 s < 60 s
    assert abs(res["respiration"]["rate_brpm"] - truth.resp_rate_brpm) <= 3
    assert res["video"]["codec"] == ("hevc" if "hevc" in name else "h264")
    assert res["video"]["fps_measured"] == pytest.approx(30, abs=0.1)
    assert len(res["waveform"]["values"]) == len(res["waveform"]["t_s"]) > 100


def test_finger_clipped_warns_and_falls_back_to_green(vids):
    path, truth = vids("finger_clipped")
    res = camera.analyze_video(path, "finger", deep=False)
    assert "clipped" in codes(res)
    assert res["red_clipped"] is True
    assert res["primary_method"] == "green"
    assert res["video"]["clipped_fraction_channels"]["r"] > 0.9
    assert abs(res["hr_bpm"] - truth.hr_bpm) <= HR_TOL


def test_short_video_warns_and_withholds_hr(vids):
    path, _ = vids("finger_10s")
    res = camera.analyze_video(path, "finger", deep=False)
    assert "too_short" in codes(res)
    assert res["hr_bpm"] is None and res["ibi_ms"] is None and res["hrv"] is None
    assert res["duration_s"] == pytest.approx(10, abs=0.1)


def test_timestamps_come_from_container(vids, tmp_path):
    """Drop every 10th frame keeping the original timestamps (VFR): gaps must be seen, HR kept."""
    src, truth = vids("finger_hevc")
    dst = tmp_path / "vfr.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                    "-vf", r"select=mod(n\,10)", "-fps_mode", "vfr", "-c:v", "libx264", "-preset",
                    "ultrafast", "-crf", "12", str(dst)], check=True)
    tr = camera.extract_trace(dst, "finger")
    assert tr["n_frames"] == 810
    assert tr["dropped_frames"] >= 80
    assert tr["fps_measured"] == pytest.approx(27, abs=0.5)
    assert tr["t_s"][-1] == pytest.approx(29.97, abs=0.1)  # not 810 / 30 = 27 s
    res = camera.analyze_trace(tr["t_s"], tr["r"], tr["g"], tr["b"], "finger", "video")
    assert abs(res["hr_bpm"] - truth.hr_bpm) <= HR_TOL


def test_long_finger_trace_reports_hrv():
    body, truth = synth.make_trace(hr_bpm=64, duration_s=90, jitter_ms=2)
    t = np.asarray(body["t_ms"]) / 1000
    res = camera.analyze_trace(t, body["r"], body["g"], body["b"], "finger", "capture")
    assert res["tier"] == "capture"
    assert abs(res["hr_bpm"] - 64) <= HR_TOL
    assert res["hrv"] is not None and res["hrv_reliable"] is True
    ibi = np.diff(truth.beat_times_s) * 1000
    ibi = ibi[(truth.beat_times_s[1:] > 1) & (truth.beat_times_s[1:] < 89)]
    true_rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    assert abs(res["hrv"]["RMSSD"] - true_rmssd) <= 10, (res["hrv"]["RMSSD"], true_rmssd)


def test_analyze_trace_validates():
    with pytest.raises(ValueError):
        camera.analyze_trace([0, 1, 2], [1, 2], [1, 2, 3], [1, 2, 3], "finger", "capture")
    with pytest.raises(ValueError):
        camera.analyze_trace(np.arange(100) / 30, *[np.ones(100)] * 3, "toe", "capture")
    with pytest.raises(ValueError):
        camera.analyze_trace(np.arange(100) / 30, *[np.ones(100)] * 3, "finger", "magic")


# --------------------------------------------------------------------------------------------
# face
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["face_hevc", "face_h264"])
def test_face_hr_with_roi(vids, name):
    path, truth = vids(name)
    res = camera.analyze_video(path, "face", deep=False, roi=truth.roi)
    assert_json_safe(res)
    assert res["tier"] == "video" and res["mode"] == "face"
    assert set(res["methods"]) == {"pos", "chrom"}
    assert abs(res["hr_bpm"] - truth.hr_bpm) <= HR_TOL, res["methods"]
    for m in res["methods"].values():
        assert abs(m["hr_fft_bpm"] - truth.hr_bpm) <= HR_TOL, res["methods"]
    assert res["roi"]["source"] == "override"
    assert res["hrv_reliable"] is False
    assert res["deep"]["skipped"] is True


def test_face_without_face_falls_back_to_centre(vids):
    path, _ = vids("face_hevc")
    tr = camera.extract_trace(path, "face")
    assert tr["roi"]["source"] == "centre_fallback"
    assert {"no_face_detected", "face_detector_unavailable"} & {w["code"] for w in tr["warnings"]}


@pytest.mark.skipif(os.environ.get("PULSE_SKIP_DEEP") == "1", reason="PULSE_SKIP_DEEP=1")
def test_face_deep_models(vids):
    if not camera.deep_available()["available"]:
        pytest.skip("open-rppg not installed")
    path, truth = vids("face_hevc")
    res = camera.analyze_video(path, "face", deep=True, roi=truth.roi)
    deep = res["deep"]
    assert deep["available"] is True, deep
    assert set(deep["models"]) == set(camera.DEEP_MODELS)
    for name, rec in deep["models"].items():
        assert "error" not in rec, (name, rec)
        assert rec["load_s"] >= 0 and rec["infer_s"] >= 0
        assert rec["hrv_reliable"] is False
        print(f"\n{name}: HR {rec['hr_bpm']} (truth {truth.hr_bpm}), SQI {rec['sqi']}, "
              f"load {rec['load_s']} s, inference {rec['infer_s']} s")
        assert abs(rec["hr_bpm"] - truth.hr_bpm) <= 5
    assert_json_safe(res)


# --------------------------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------------------------

@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(camera_api, "CAMERA_DATA", tmp_path / "camera")
    app = FastAPI()
    app.include_router(camera_api.router)
    return TestClient(app), tmp_path / "camera"


def test_api_trace(client):
    c, store = client
    body, _ = synth.make_trace(hr_bpm=75, duration_s=30)
    r = c.post("/camera/trace", json=body)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["tier"] == "capture" and res["mode"] == "finger"
    assert res["meta"] == body["meta"]
    assert abs(res["hr_bpm"] - 75) <= HR_TOL
    (d,) = list(store.iterdir())
    assert d.name.endswith("-finger-trace") and d.name == res["id"]
    assert json.loads((d / "result.json").read_text())["tier"] == "capture"
    assert len(json.loads((d / "trace.json").read_text())["t_ms"]) == len(body["t_ms"])


def test_api_trace_rejects_bad_input(client):
    c, _ = client
    body, _ = synth.make_trace(duration_s=20)
    assert c.post("/camera/trace", json={**body, "r": body["r"][:-5]}).status_code == 422
    assert c.post("/camera/trace", json={**body, "mode": "toe"}).status_code == 422
    assert c.post("/camera/trace", json={**body, "g": [999.0] * len(body["g"])}).status_code == 422
    assert c.post("/camera/trace", content=b"not json",
                  headers={"content-type": "application/json"}).status_code == 422


def test_api_video(client, vids):
    c, store = client
    path, _ = vids("finger_10s")
    with path.open("rb") as fh:
        r = c.post("/camera/video", files={"file": ("clip.mp4", fh, "video/mp4")},
                   data={"mode": "finger", "deep": "false"})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["tier"] == "video" and res["mode"] == "finger"
    assert "too_short" in codes(res)
    (d,) = list(store.iterdir())
    assert d.name.endswith("-finger") and (d / "video.mp4").stat().st_size == path.stat().st_size
    assert json.loads((d / "result.json").read_text())["tier"] == "video"
    assert (d / "trace.json").exists()


def test_api_video_rejects_bad_input(client, vids, monkeypatch):
    c, _ = client
    path, _ = vids("finger_10s")
    data = path.read_bytes()
    assert c.post("/camera/video", files={"file": ("a.mp4", data, "video/mp4")},
                  data={"mode": "toe"}).status_code == 422
    assert c.post("/camera/video", data={"mode": "finger"}).status_code in (415, 422)
    assert c.post("/camera/video", files={"file": ("a.txt", b"hello", "text/plain")},
                  data={"mode": "finger"}).status_code == 415
    r = c.post("/camera/video", files={"file": ("a.mp4", b"\x00" * 4096, "video/mp4")},
               data={"mode": "finger"})
    assert r.status_code == 422  # not decodable
    monkeypatch.setattr(camera_api, "MAX_VIDEO_BYTES", 1000)
    assert c.post("/camera/video", files={"file": ("a.mp4", data, "video/mp4")},
                  data={"mode": "finger"}).status_code == 413
