"""Exploratory PPG age (ppg_age, AI-PPG Age) on synthetic 100 Hz spot runs and fingertip traces (no
recordings of real people). The model is replaced by a stand-in except in the tests marked as needing
the real weights, which are skipped when the weights are not in the local Hugging Face cache (tests
never download). A smoke run over the real 2026-10-22 ppg100 run prints its value when it exists.

Run: ``uv run pytest -q tests/test_ppg_age.py`` (add ``-s`` to see the real-run printout).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_synth as ws

from pulse_exploration import ppg_age as pa
from pulse_exploration import watch_session as w

ROOT = Path(__file__).resolve().parents[1]
REAL_RUN = ROOT / "data" / "runs" / "20261022-015808-record"
PH = np.arange(pa.N_SAMPLES) / pa.N_SAMPLES


def fake_predict(beats: np.ndarray) -> np.ndarray:
    """Stand-in for the model: depends on the beat, so the bootstrap spread is not zero."""
    return 50.0 + 5.0 * np.asarray(beats)[:, 25]


def ukb_like_beat(sys_t: float, dia_t: float, dia_amp: float) -> np.ndarray:
    """A beat shaped like UK Biobank's averaged finger pulse: foot at 0, systolic and diastolic waves."""
    y = (np.exp(-0.5 * ((PH - sys_t) / 0.07) ** 2 * np.where(PH < sys_t, 2.5, 1))
         + dia_amp * np.exp(-0.5 * ((PH - dia_t) / 0.08) ** 2) + 0.6 * np.clip(1 - PH, 0, 1) * (PH > sys_t))
    return pa._z(y)


def sine_wave(ts: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """A pulse with no shape: a sinusoid through the given peaks."""
    return np.cos(2 * np.pi * np.interp(ts, peaks, np.arange(len(peaks), dtype=float)))


def make_spot_run(root: Path, name: str, duration_s: float = 30.0, hr_bpm: float = 65.0,
                  motion: list[tuple[float, float]] | None = None, with_acc: bool = True, wave=ws.ppg_wave,
                  seed: int = 0) -> tuple[Path, ws.WatchTruth, np.ndarray, np.ndarray]:
    """What ``spot ppg100`` uploads: sdk_ppg100 (100 Hz green / IR / red, columns of SamsungSdk.kt) and
    acc. Returns the run, the truth, and the noise-free pulse with its sample times."""
    rng = np.random.default_rng(seed)
    run = root / name
    run.mkdir(parents=True)
    r = ws.beat_times(duration_s, hr_bpm, rng)
    pulse = r + 0.25 + 0.003 * rng.standard_normal(len(r))
    truth = ws.WatchTruth(r, pulse, duration_s, list(motion or []))
    ts = np.arange(0.05, duration_s, 0.01)
    clean = wave(ts, pulse)
    clean = clean - clean.mean()
    art = ws._artefact(ts, truth, rng)
    breath = 0.4 * np.sin(2 * np.pi * 0.25 * ts)
    ms = ws.T0_MS + np.round(ts * 1000).astype(np.int64)
    green = 90000 - 3000 * (clean + breath + art) + 20 * rng.standard_normal(len(ts))
    ir = 470000 - 9000 * (clean + breath + art) + 60 * rng.standard_normal(len(ts))
    red = 440000 - 7000 * (clean + breath + art) + 60 * rng.standard_normal(len(ts))
    z = np.zeros(len(ts), int)
    ws._write_csv(run / "sdk_ppg100.csv", ["timestamp_ms", "wall_ms", "green", "green_status", "ir", "ir_status",
                                           "red", "red_status"],
                  [ms, ms + 8, ws._ints(green), z, ws._ints(ir), z, ws._ints(red), z])
    if with_acc:
        ta = ws._sample_times(duration_s, ws.ACC_FS, rng)
        mm = truth.in_motion(ta)[:, None]
        g = np.array([5.62, 3.49, 4.19]) / np.linalg.norm([5.62, 3.49, 4.19]) * 9.80665
        acc = g + 0.03 * rng.standard_normal((len(ta), 3)) + 2.0 * np.sin(2 * np.pi * 1.8 * ta[:, None]) * mm
        ens, wall = ws._android_clock(ta, rng)
        ws._write_csv(run / "acc.csv", ["event_ns", "wall_ms", "accuracy", "v0", "v1", "v2"],
                      [ens, wall, np.full(len(ta), 3), acc[:, 0], acc[:, 1], acc[:, 2]])
    (run / "meta.txt").write_text(f"model=SM-R965F\nbuild=SYNTHETIC\nsdk_int=36\nstart_wall_ms={ws.T0_MS}\n"
                                  f"end_wall_ms={ws.T0_MS + int(duration_s * 1000)}\nseconds={int(duration_s)}\n")
    return run, truth, ts, clean


def true_beat(ts: np.ndarray, clean: np.ndarray, truth: ws.WatchTruth, a: float, b: float) -> np.ndarray:
    p = truth.pulse_s
    m = (p[:-1] >= a) & (p[1:] <= b)
    return pa.average_beat(ts, clean, p[:-1][m], p[1:][m])["beat"]


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------

def test_net1d_matches_the_published_architecture():
    from pulse_exploration.net1d import Net1D

    model = Net1D(**pa.CONFIG)
    assert sum(p.numel() for p in model.parameters()) == pa.N_PARAMS  # the paper's 389,689
    keys = set(model.state_dict())
    for k in ("first_conv.conv.weight", "first_bn.running_mean", "stage_list.0.block_list.0.bn1.weight",
              "stage_list.3.block_list.1.se_fc2.weight", "dense.bias"):
        assert k in keys


def _real_model_or_skip():
    try:
        return pa.load_model()
    except pa.WeightsMissing:
        pytest.skip("AI-PPG Age weights not in the local Hugging Face cache "
                    "(uv run python -m pulse_exploration.ppg_age --download)")


def test_real_weights_read_pulse_shape():
    _real_model_or_skip()
    young, old = ukb_like_beat(0.15, 0.45, 0.6), ukb_like_beat(0.25, 0.38, 0.1)
    ages = pa.predict_ages(np.stack([young, old]))
    assert np.all(np.isfinite(ages)) and np.all((ages > 20) & (ages < 100))
    # the model reads a distinct, late diastolic wave as younger (paper Fig. 6)
    assert ages[0] < ages[1] - 5
    # a shapeless beat lands where the gate says (60-67 y)
    sine = pa.predict_ages(pa._z(-np.cos(2 * np.pi * PH))[None])[0]
    assert 58 < sine < 70


def test_weights_missing_is_a_skip_reason(tmp_path, monkeypatch):
    run, *_ = make_spot_run(tmp_path, "20261027-100000-spot")

    def missing(_):
        raise pa.WeightsMissing("AI-PPG Age weights are not on this machine: run "
                                "`uv run python -m pulse_exploration.ppg_age --download`")

    monkeypatch.setattr(pa, "predict_ages", missing)
    res = w.analyze_run(run, write=False)
    m = res["metrics"]["ppg_age"]
    assert m["chosen"] is None and m["estimates"] == []
    assert "--download" in m["skipped"][0]["reason"]


# --------------------------------------------------------------------------------------------
# average beat
# --------------------------------------------------------------------------------------------

def test_sinusoid_corr():
    assert pa.sinusoid_corr(np.cos(2 * np.pi * PH + 1.0)) == pytest.approx(1.0, abs=1e-6)
    for beat in (ukb_like_beat(0.15, 0.45, 0.6), ukb_like_beat(0.25, 0.38, 0.1)):
        assert pa.sinusoid_corr(beat) < 0.9


def test_average_beat_recovers_the_pulse_shape(tmp_path):
    run, truth, ts, clean = make_spot_run(tmp_path, "20261027-100001-spot", seed=1)
    est, why = pa.run_estimate(run, predict=fake_predict)
    assert est is not None, why
    beat = np.asarray(est["details"]["beat"])
    assert beat.shape == (pa.N_SAMPLES,)
    assert beat.mean() == pytest.approx(0, abs=1e-3) and beat.std() == pytest.approx(1, abs=1e-3)
    ref = true_beat(ts, clean, truth, 2.0, 28.0)
    assert corr(beat, ref) > 0.98
    assert abs(int(np.argmax(beat)) - int(np.argmax(ref))) <= 3
    assert int(np.argmin(beat[:50])) <= 3  # starts at the foot, like UK Biobank's averaged pulse
    assert est["reliability"] == "experimental"
    assert est["source"] == "sdk_ppg100" and est["feature"] == "ppg100" and est["tier"] == "all"
    gate = est["details"]["gate"]
    assert gate["passed"] and gate["motion_free"] and gate["sinusoid_corr"] < pa.SHAPELESS_CORR
    # about every interval of the recording is averaged
    assert est["n"] == est["value"]["beats_averaged"] >= 0.8 * len(truth.intervals(1.0, 30.0, pulse=True))


def test_every_value_carries_its_band_warning_and_beat_count(tmp_path):
    run, *_ = make_spot_run(tmp_path, "20261027-100002-spot", seed=2)
    est, _ = pa.run_estimate(run, predict=fake_predict)
    v = est["value"]
    assert set(v) == {"ppg_age_y", "band_y", "band_zero_shot_y", "beats_averaged", "resample_sd_y",
                      "absolute_age", "warning"}
    assert v["absolute_age"] is False
    assert v["band_y"][0] == pytest.approx(v["ppg_age_y"] - pa.MAE_HOLDOUT_Y, abs=0.11)
    assert v["band_y"][1] == pytest.approx(v["ppg_age_y"] + pa.MAE_HOLDOUT_Y, abs=0.11)
    assert v["band_zero_shot_y"][1] - v["band_zero_shot_y"][0] == pytest.approx(2 * pa.MAE_ZERO_SHOT_Y, abs=0.2)
    assert v["resample_sd_y"] > 0
    for text in ("not an absolute", "7.6 years", "11.5 years", "38-67", "finger", "wrist green PPG at 100 Hz",
                 "within-person trend"):
        assert text in v["warning"]
    notes = " ".join(est["notes"])
    assert "domain mismatch" in notes and f"{v['beats_averaged']} beats averaged" in notes
    assert est["details"]["model"]["revision"] == pa.HF_REVISION


def test_beats_in_motion_are_left_out(tmp_path):
    run, truth, *_ = make_spot_run(tmp_path, "20261027-100003-spot", motion=[(0.0, 12.0)], seed=3)
    est, _ = pa.run_estimate(run, predict=fake_predict)
    assert est["reliability"] == "experimental"
    # the motion gate pads 1 s around 1 s windows: nothing before 13 s is used
    assert min(a for a, _ in est["details"]["spans_s"]) >= 13.0 - 1e-6
    assert est["n"] <= len(truth.intervals(13.0, 30.0, pulse=True))


def test_all_motion_falls_back_to_unreliable(tmp_path):
    run, *_ = make_spot_run(tmp_path, "20261027-100004-spot", motion=[(k * 4 + 1, k * 4 + 3.5) for k in range(8)],
                            seed=4)
    est, why = pa.run_estimate(run, predict=fake_predict)
    if est is None:  # every beat in motion may also fail the template match
        assert "template-matched" in why or "match their average" in why
        return
    assert est["reliability"] == "unreliable"
    assert not est["details"]["gate"]["motion_free"]
    assert any("motion-free" in r for r in est["details"]["gate"]["reasons"])


def test_no_accelerometer_is_unreliable(tmp_path):
    run, *_ = make_spot_run(tmp_path, "20261027-100005-spot", with_acc=False, seed=5)
    est, _ = pa.run_estimate(run, predict=fake_predict)
    assert est["reliability"] == "unreliable"
    assert "no accelerometer: motion not checked" in est["details"]["gate"]["reasons"]


def test_shapeless_beat_is_unreliable(tmp_path):
    run, *_ = make_spot_run(tmp_path, "20261027-100006-spot", wave=sine_wave, seed=6)
    est, _ = pa.run_estimate(run, predict=fake_predict)
    assert est["reliability"] == "unreliable"
    assert est["details"]["gate"]["sinusoid_corr"] >= pa.SHAPELESS_CORR
    assert any("near-sinusoid" in r for r in est["details"]["gate"]["reasons"])


def test_too_short_has_a_reason(tmp_path):
    run, *_ = make_spot_run(tmp_path, "20261027-100007-spot", duration_s=8.0, seed=7)
    est, why = pa.run_estimate(run, predict=fake_predict)
    assert est is None and why


# --------------------------------------------------------------------------------------------
# watch_session and the fingertip camera
# --------------------------------------------------------------------------------------------

def test_analyze_run_reports_ppg_age(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "predict_ages", fake_predict)
    run, *_ = make_spot_run(tmp_path, "20261027-100008-spot", seed=8)
    res = w.analyze_run(run)
    assert res["analysis_version"] == w.ANALYSIS_VERSION
    m = res["metrics"]["ppg_age"]
    assert m["chosen"]["reliability"] == "experimental" and m["chosen"]["source"] == "sdk_ppg100"
    assert "ppg_age_exploratory" in {x["code"] for x in res["warnings"]}
    assert "ppg_age" not in res["summary"]  # never shown without its band and warning
    assert res["metrics"]["hr"]["chosen"]["source"] == "sdk_ppg100"  # the rest of the analysis is unchanged


def test_fingertip_camera_trace(tmp_path):
    rng = np.random.default_rng(9)
    pulse = ws.beat_times(30.0, 70.0, rng) + 0.25
    t = np.arange(0.0, 30.0, 1 / 29.8) + 0.002 * rng.standard_normal(894)
    wave = ws.ppg_wave(t, pulse)
    trace = {"mode": "finger", "t_ms": (t * 1000).tolist(),
             "r": (200 - 4 * wave + 0.1 * rng.standard_normal(len(t))).tolist(),
             "g": (40 - 0.5 * wave + 0.1 * rng.standard_normal(len(t))).tolist(),
             "b": (30 + 0.1 * rng.standard_normal(len(t))).tolist()}
    est, why = pa.camera_estimate(trace, predict=fake_predict)
    assert est is not None, why
    assert est["source"] == "camera_finger" and est["tier"] == "capture"
    assert "fingertip camera" in est["value"]["warning"]
    tt = np.arange(0, 30, 0.002)
    m = (pulse[:-1] >= 2) & (pulse[1:] <= 28)
    ref = pa.average_beat(tt, ws.ppg_wave(tt, pulse), pulse[:-1][m], pulse[1:][m])["beat"]
    assert corr(np.asarray(est["details"]["beat"]), ref) > 0.97
    assert est["reliability"] == "experimental"  # no accelerometer, but the beats match their average


def test_face_trace_is_refused():
    est, why = pa.camera_estimate({"mode": "face", "t_ms": [], "r": [], "g": [], "b": []})
    assert est is None and "finger" in why


@pytest.mark.skipif(not (REAL_RUN / "sdk_ppg100.csv").is_file(), reason="real run not on this machine")
def test_real_run_smoke():
    _real_model_or_skip()
    est, why = pa.run_estimate(REAL_RUN)
    assert est is not None, why
    v = est["value"]
    print(f"\n{REAL_RUN.name}: PPG age {v['ppg_age_y']} y (band {v['band_y']}), {v['beats_averaged']} beats, "
          f"{est['reliability']}: {est['details']['gate']['reasons']}")
    assert est["reliability"] in ("experimental", "unreliable")
