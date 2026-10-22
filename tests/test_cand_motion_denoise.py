"""Tests for the motion-denoised 25 Hz candidate (pulse_exploration/hrv/cand_motion_denoise.py).

Synthetic tests build a 25 Hz green-PPG stream from known beat times (ADC falls with blood volume,
a dicrotic wave, sensor noise, millisecond timestamps). One test reproduces the hidden sample slips
of the GalaxyPPG stream (a sensor sampling at 40.00 ms read by a 40.15 ms clock that keeps the
latest sample). The last test scores one development participant (P03) and skips without the
dataset. Run: ``uv run pytest -q tests/test_cand_motion_denoise.py`` (a few seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulse_exploration import galaxyppg  # noqa: E402
from pulse_exploration.hrv import bench, time_domain  # noqa: E402
from pulse_exploration.hrv import cand_motion_denoise as cmd  # noqa: E402

needs_data = pytest.mark.skipif(not (galaxyppg.DATASET_DIR / "P03").exists(), reason="GalaxyPPG not unpacked")
T0 = 1_710_000_000_000.0


def beat_times(duration_s: float, seed: int, hr: float = 70.0, rsa_ms: float = 45.0) -> np.ndarray:
    """Beat times (ms from 0) with respiratory sinus arrhythmia and beat-to-beat noise."""
    rng = np.random.default_rng(seed)
    out, t = [], 500.0
    while t < duration_s * 1000 - 500:
        out.append(t)
        t += 60_000 / hr + rsa_ms * np.sin(2 * np.pi * 0.25 * t / 1000) + rng.normal(0, 12)
    return np.array(out)


def blood_volume(tq: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Pulse train: 150 ms raised-cosine upstroke, exponential decay, dicrotic wave at +350 ms."""
    v = np.zeros_like(tq)
    for b in beats:
        d = tq - b
        m = (d > -50) & (d < 1500)
        dd = d[m]
        rise = np.where(dd < 0, 0, np.where(dd < 150, 0.5 * (1 - np.cos(np.pi * dd / 150)), np.exp(-(dd - 150) / 350)))
        v[m] += rise + 0.25 * np.exp(-0.5 * ((dd - 380) / 60) ** 2)
    return v


def stream(beats: np.ndarray, duration_s: float, seed: int, slips: bool = False, acc_amp=None):
    """(ppg_t_ms, ppg, acc_t_ms, acc_xyz). slips=True: content sampled every 40.00 ms on the sensor
    clock, read every 40.15 ms (the latest sample is kept, so one is dropped every ~10.7 s) and
    timestamped at read time, as in GalaxyPPG. acc_amp(t_ms) -> m/s^2 of periodic arm motion."""
    rng = np.random.default_rng(seed + 1)
    n = int(duration_s * 1000 / 40.15)
    read = np.arange(n) * 40.15
    content = np.floor(read / 40.0) * 40.0 if slips else read
    v = blood_volume(content, beats)
    ppg = 2.0e6 - 1.5e4 * v + 800 * np.sin(2 * np.pi * 0.05 * content / 1000) + rng.normal(0, 40, n)
    acc = np.column_stack([rng.normal(0, 0.03, n), rng.normal(0, 0.03, n), 9.81 + rng.normal(0, 0.03, n)])
    if acc_amp is not None:
        amp = acc_amp(read)
        swing = np.sin(2 * np.pi * 2.6 * read / 1000)
        acc += amp[:, None] * np.column_stack([swing, 0.5 * swing, 0.3 * swing])
        ppg += 3.0e3 * amp * swing  # motion artefact in the PPG
    t = T0 + np.rint(read + rng.normal(0, 0.4, n))
    return t, ppg, t.copy(), acc


def kept_rmssd(out: dict) -> tuple[float, int]:
    b, q = out["beats_ms"], out["quality"]
    o = np.argsort(b)
    b, q = b[o], q[o] >= 0.5
    m = time_domain(np.diff(b), q[1:] & q[:-1])
    return m["rmssd"], m["n_ibi"]


def true_rmssd(beats: np.ndarray) -> float:
    return time_domain(np.diff(beats))["rmssd"]


# --------------------------------------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------------------------------------


def test_clean_stream_sub_sample_timing():
    beats = beat_times(180, seed=0)
    out = cmd.MotionDenoiseEstimator().estimate(*stream(beats, 180, seed=0))
    b, q = out["beats_ms"], out["quality"]
    assert len(b) == len(q) == len(out["flags"])
    keep = q >= 0.5
    ref = beats + T0
    m = bench.match_beats(b[keep], ref, tol_ms=50)
    assert len(m["pairs"]) >= 0.97 * len(ref)  # recall
    assert len(m["pairs"]) >= 0.99 * keep.sum()  # precision
    assert np.std(m["resid"]) < 3.0  # far below the 11.5 ms of the 40 ms sample grid
    # intervals: kept RMSSD close to the truth, most intervals kept
    r, n_ibi = kept_rmssd(out)
    assert abs(r - true_rmssd(beats)) < 3.0
    assert n_ibi > 0.85 * (len(beats) - 1)
    # everything is vouched for at rest
    cover = sum(e - s for s, e in out["accept_windows"])
    assert cover > 0.98 * (b.max() - b.min())


def test_hidden_sample_slips_are_voided():
    beats = beat_times(240, seed=1, rsa_ms=20.0)  # RMSSD ~25 ms, as at rest in GalaxyPPG
    args = stream(beats, 240, seed=1, slips=True)
    truth = true_rmssd(beats)
    on = cmd.MotionDenoiseEstimator().estimate(*args)
    off = cmd.MotionDenoiseEstimator(slip="none").estimate(*args)
    err_on, err_off = abs(kept_rmssd(on)[0] - truth), abs(kept_rmssd(off)[0] - truth)
    assert err_off > 3.0  # the slips (one ~40 ms short interval every ~10.7 s) inflate RMSSD
    assert err_on < 0.5 * err_off
    n_slips = 240_000 / 40.0 - 240_000 / 40.15  # ~22 dropped samples
    n_void = int(((on["flags"] & cmd.F_SLIP) > 0).sum())
    assert 0.5 * n_slips < n_void < 1.5 * n_slips
    assert on["quality"][(on["flags"] & cmd.F_SLIP) > 0].max() == 0.0  # placeholders, never used


def test_recording_gap_is_not_an_interval():
    beats = beat_times(120, seed=2)
    t, ppg, at, acc = stream(beats, 120, seed=2)
    cut = (t > T0 + 50_000) & (t < T0 + 62_000)  # 12 s without samples, as in P02
    out = cmd.MotionDenoiseEstimator().estimate(t[~cut], ppg[~cut], at[~cut], acc[~cut])
    b, q = out["beats_ms"], out["quality"]
    o = np.argsort(b)
    b, q = b[o], q[o]
    usable = (q[1:] >= 0.5) & (q[:-1] >= 0.5)
    assert np.diff(b)[usable].max() < 1500
    inside = (b > T0 + 50_000) & (b < T0 + 62_000)
    assert inside.sum() == 1 and q[inside][0] == 0.0 and (out["flags"][o][inside] & cmd.F_GAP).all()


def test_heavy_motion_is_gated_and_light_motion_is_not():
    beats = beat_times(180, seed=3)

    def amp(tr):  # 0-60 s still, 60-100 s jogging-like (14 m/s^2), 100-140 s light (0.15 m/s^2)
        return np.where((tr > 60_000) & (tr < 100_000), 14.0, np.where((tr > 100_000) & (tr < 140_000), 0.15, 0.0))

    out = cmd.MotionDenoiseEstimator().estimate(*stream(beats, 180, seed=3, acc_amp=amp))
    aw = bench._merge(out["accept_windows"])
    assert bench._cover(aw, T0 + 65_000, T0 + 95_000) == 0.0
    assert bench._cover(aw, T0 + 5_000, T0 + 55_000) > 0.99
    assert bench._cover(aw, T0 + 105_000, T0 + 135_000) > 0.99
    b, q = out["beats_ms"], out["quality"]
    assert q[(b > T0 + 65_000) & (b < T0 + 95_000)].max() == 0.0  # beats under heavy motion unusable


def test_helpers():
    # DP tracker skips a dicrotic candidate between two beats and reports a missed beat as k = 2
    tc = np.array([0.0, 8.0, 20.0, 40.0, 60.0, 61.0])
    sc = np.array([1.0, 0.3, 1.0, 1.0, 1.0, 0.2])
    sel, ks = cmd.track_beats(tc, sc, np.full(len(tc), 20.0), cmd.Params())
    assert tc[sel].tolist() == [0.0, 20.0, 40.0, 60.0] and ks[0] == 0 and (ks[1:] == 1).all()
    sel, ks = cmd.track_beats(np.array([0.0, 20.0, 60.0, 80.0]), np.ones(4), np.full(4, 20.0), cmd.Params())
    assert ks.tolist() == [0, 1, 2, 1]
    # slip candidates: an isolated 40 ms short interval, at most one per min gap
    t = np.cumsum(np.r_[0, np.full(30, 800.0)])
    t[10:] -= 40.0
    sj = cmd.slip_candidates(t, np.ones(len(t)), np.ones(len(t) - 1, bool), 30.0, 4000.0)
    assert sj.tolist() == [9]
    # upstroke refinement recovers a sub-sample shift of a band-limited edge
    n = np.arange(200, dtype=float)
    y = np.tanh((n - 100.37) / 2.0)
    assert cmd.refine_upstroke(y, np.array([100]), 10)[0] == pytest.approx(100.37, abs=0.05)


# --------------------------------------------------------------------------------------------------
# one development participant
# --------------------------------------------------------------------------------------------------


@needs_data
def test_dev_participant_p03():
    rows, info = bench.score_participant("P03", cmd.MotionDenoiseEstimator())
    assert info["error"] is None, info["error"]
    w = pl.DataFrame(rows).filter(pl.col("usable"), pl.col("window_s") == 60)
    rest = w.filter(pl.col("group") == "seated-rest")
    assert rest["covered"].mean() > 0.8
    c = rest.filter(pl.col("covered"))
    err = (c["est_rmssd"] - c["ref_rmssd"]).to_numpy()
    assert np.median(np.abs(err)) < 6.0 and abs(err.mean()) < 5.0
    assert not w.filter(pl.col("kind").is_in(["jogging", "running"]))["covered"].any()  # heavy motion gated
    assert info["estimate_s"] < 10
