"""Classical HRV candidate (hrv/cand_classical.py): synthetic 25 Hz watch PPG with known pulse times,
plus one GalaxyPPG development participant (P03) through the bench (skipped when the dataset is not
unpacked). Run: ``uv run pytest -q tests/test_cand_classical.py`` (a few seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import watch_synth as ws  # noqa: E402

from pulse_exploration import galaxyppg  # noqa: E402
from pulse_exploration.hrv import beat_intervals, bench, time_domain  # noqa: E402
from pulse_exploration.hrv.cand_classical import ClassicalEstimator  # noqa: E402

T0_MS = 1_700_000_000_000.0
FS = 24.9  # Galaxy Watch PPG rate
needs_data = pytest.mark.skipif(not (galaxyppg.DATASET_DIR / "P03").exists(), reason="GalaxyPPG not unpacked")


def _synthetic(duration_s: float = 180.0, hr_bpm: float = 68.0, seed: int = 0, motion_s=None, gap_s=None, spike_s=None):
    """Watch-like recording: green ADC that falls with each pulse, ms-rounded 24.9 Hz timestamps,
    sensor noise and baseline wander; 25 Hz accelerometer (gravity + noise, `motion_s` adds 2 Hz
    swings). Returns (ppg_t_ms, ppg, acc_t_ms, acc_xyz, true pulse times in ms)."""
    rng = np.random.default_rng(seed)
    r_s = ws.beat_times(duration_s, hr_bpm, rng)
    pulse_s = r_s + 0.25 + 0.003 * rng.standard_normal(len(r_s))
    t_s = np.arange(int(duration_s * FS)) / FS
    wave = ws.ppg_wave(t_s, pulse_s)
    adc = 2.0e6 - 4000.0 * wave + 40.0 * rng.standard_normal(len(t_s)) + 800.0 * np.sin(2 * np.pi * 0.05 * t_s)
    if spike_s is not None:  # a pressure artefact: one large, fast transient
        adc += 3.0e4 * np.exp(-0.5 * ((t_s - spike_s) / 0.15) ** 2)
    acc = np.c_[0.3 + 0.02 * rng.standard_normal(len(t_s)), 0.1 + 0.02 * rng.standard_normal(len(t_s)), 9.8 + 0.02 * rng.standard_normal(len(t_s))]
    if motion_s is not None:
        m = (t_s >= motion_s[0]) & (t_s < motion_s[1])
        acc[m] += 4.0 * np.sin(2 * np.pi * 2.0 * t_s[m])[:, None] * np.array([1.0, 0.5, 0.8])
        adc[m] += 2.0e4 * np.sin(2 * np.pi * 2.0 * t_s[m])
    keep = np.ones(len(t_s), bool) if gap_s is None else ~((t_s >= gap_s[0]) & (t_s < gap_s[1]))
    t_ms = np.round(T0_MS + 1000.0 * t_s)
    return t_ms[keep], adc[keep], t_ms[keep], acc[keep], T0_MS + 1000.0 * pulse_s


def _kept_rmssd(beats: np.ndarray, keep: np.ndarray, a_ms: float, b_ms: float) -> float:
    m = (beats >= a_ms) & (beats < b_ms)
    ibi, ok = beat_intervals(beats[m], keep[m])
    return time_domain(ibi, ok)["rmssd"]


def test_recovers_beats_and_rmssd_at_rest():
    t, x, at, ax, truth = _synthetic()
    out = ClassicalEstimator().estimate(t, x, at, ax)
    b, q = out["beats_ms"], out["quality"]
    assert np.all(np.diff(b) > 0) and len(q) == len(b) and q.min() >= 0 and q.max() <= 1
    keep = q >= 0.5
    assert keep.mean() > 0.9
    inside = truth[(truth > t[0] + 5000) & (truth < t[-1] - 5000)]
    m = bench.match_beats(b[keep], inside, tol_ms=50)
    assert len(m["pairs"]) / len(inside) > 0.9  # recall of kept beats
    assert np.std(m["resid"]) < 4.0  # sub-sample timing on a 40 ms grid
    # RMSSD of the kept beats vs the true pulse intervals over the same span
    a, z = t[0] + 10_000, t[-1] - 10_000
    true = time_domain(np.diff(truth[(truth >= a) & (truth < z)]))["rmssd"]
    assert abs(_kept_rmssd(b, keep, a, z) - true) < 3.0
    # the whole recording is quiet, so every chunk is accepted
    acc = np.array(out["accept_windows"])
    assert acc[0, 0] <= t[0] and acc[-1, 1] >= t[-1] and np.all(acc[1:, 0] == acc[:-1, 1])


def test_subsample_timing_beats_the_sample_grid():
    t, x, at, ax, truth = _synthetic(seed=1)
    a, z = t[0] + 10_000, t[-1] - 10_000
    true = time_domain(np.diff(truth[(truth >= a) & (truth < z)]))["rmssd"]
    err = {}
    grid = ClassicalEstimator(fiducial="peak", refine="none", align_iters=0)  # beats on the sample grid
    for name, est in {"classical": ClassicalEstimator(), "grid": grid}.items():
        out = est.estimate(t, x, at, ax)
        err[name] = abs(_kept_rmssd(out["beats_ms"], out["quality"] >= 0.5, a, z) - true)
    assert err["classical"] < 2.0 and err["grid"] > err["classical"] + 2.0


def test_motion_chunks_are_rejected():
    t, x, at, ax, _ = _synthetic(motion_s=(60.0, 90.0), seed=2)
    out = ClassicalEstimator().estimate(t, x, at, ax)
    acc = np.array(out["accept_windows"])
    mot0, mot1 = T0_MS + 60_000, T0_MS + 90_000
    overlaps = (acc[:, 1] > mot0) & (acc[:, 0] < mot1)
    assert not overlaps.any()  # every chunk touching the motion is declined
    assert (acc[:, 1] - acc[:, 0]).sum() > 100_000  # the quiet rest stays accepted
    b, q = out["beats_ms"], out["quality"]
    assert np.all(q[(b >= mot0) & (b < mot1)] == 0)
    assert (q[(b < mot0 - 10_000) | (b >= mot1 + 10_000)] >= 0.5).mean() > 0.85


def test_gap_and_artefact_do_not_leak_into_intervals():
    t, x, at, ax, truth = _synthetic(gap_s=(100.0, 103.0), spike_s=50.0, seed=3)
    out = ClassicalEstimator().estimate(t, x, at, ax)
    b, q = out["beats_ms"], out["quality"]
    ibi, ok = beat_intervals(b, q >= 0.5)
    g0, g1 = T0_MS + 100_000, T0_MS + 103_000
    spans_gap = (b[:-1] < g0) & (b[1:] >= g1)
    assert spans_gap.any() and not ok[spans_gap].any()  # the interval across the gap is never used
    assert np.all((ibi[ok] > 600) & (ibi[ok] < 1300))  # kept intervals are physiological at 68 bpm
    # the artefact's neighbourhood is declined, beats a few seconds away are kept
    near = np.abs(b - (T0_MS + 50_000)) < 600
    assert near.any() and np.all(q[near] < 0.5)
    far = (np.abs(b - (T0_MS + 50_000)) > 5000) & (np.abs(b - (g0 + g1) / 2) > 5000) & (b > t[0] + 5000) & (b < t[-1] - 5000)
    assert (q[far] >= 0.5).mean() > 0.85


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        ClassicalEstimator(fiducial="nope")
    with pytest.raises(ValueError):
        ClassicalEstimator(fiducial="mid", refine="parabolic")


@needs_data
def test_development_participant_through_bench():
    rows, info = bench.score_participant("P03", ClassicalEstimator())
    assert info["error"] is None, info["error"]
    w = pl.DataFrame(rows).filter(pl.col("usable"), pl.col("window_s") == 60)
    rest = w.filter(pl.col("group") == "seated-rest")
    assert rest["covered"].mean() > 0.5
    c = rest.filter(pl.col("covered"))
    err = (c["est_rmssd"] - c["ref_rmssd"]).to_numpy()
    assert np.median(np.abs(err)) < 10.0 and err.mean() < 15.0  # naive 25 Hz peaks: bias > +100 ms
    assert c["n_matched_est"].sum() / c["n_est_in"].sum() > 0.9
    # declines locomotion (accelerometer gate) instead of reporting HRV there
    assert not w.filter(pl.col("group") == "locomotion")["covered"].any()
