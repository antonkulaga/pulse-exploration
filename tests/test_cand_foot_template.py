"""Tests for the foot/upstroke + template-matching candidate (pulse_exploration/hrv/cand_foot_template.py).

Synthetic tests build a 25 Hz green-PPG-like signal (ADC falls with the pulse) from known beat
times, optionally losing samples the way the Galaxy Watch does (stamps stay evenly spaced). One
test runs the estimator on GalaxyPPG P02 through the harness and skips when the dataset is not
unpacked. Run: ``uv run pytest -q tests/test_cand_foot_template.py`` (a few seconds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulse_exploration import galaxyppg  # noqa: E402
from pulse_exploration.hrv import beat_intervals, bench, time_domain  # noqa: E402
from pulse_exploration.hrv import cand_foot_template as cft  # noqa: E402

needs_data = pytest.mark.skipif(not (galaxyppg.DATASET_DIR / "P02").exists(), reason="GalaxyPPG not unpacked")


def _beats(duration_s: float, rng: np.random.Generator, rr_ms: float = 850.0, rsa_ms: float = 40.0) -> np.ndarray:
    """Beat (pulse-foot) times in ms: RSA at 0.25 Hz plus beat-to-beat noise (RMSSD ~ 25-35 ms)."""
    t, out = 1000.0, []
    while t < duration_s * 1000 - 1500:
        out.append(t)
        t += rr_ms + rsa_ms * np.sin(2 * np.pi * 0.25 * t / 1000) + rng.normal(0, 12)
    return np.array(out)


def _pulse(tau_s: np.ndarray) -> np.ndarray:
    rise = 0.5 * (1 + np.tanh((tau_s - 0.06) / 0.025))  # steepest upstroke 60 ms after the foot
    decay = np.exp(-np.maximum(tau_s - 0.12, 0) / 0.3)
    notch = 0.15 * np.exp(-(((tau_s - 0.40) / 0.05) ** 2))
    return np.where(tau_s >= -0.1, rise * decay + notch * (tau_s > 0), 0.0)


def _ppg(sample_t_ms: np.ndarray, beats: np.ndarray, rng: np.random.Generator, noise: float = 0.01) -> np.ndarray:
    x = np.zeros(len(sample_t_ms))
    for b in beats:
        m = (sample_t_ms > b - 100) & (sample_t_ms < b + 2500)
        x[m] += _pulse((sample_t_ms[m] - b) / 1000)
    x += 0.3 * np.sin(2 * np.pi * 0.05 * sample_t_ms / 1000) + rng.normal(0, noise, len(x))
    return 2.0e6 - 1.0e4 * x  # green ADC falls as blood volume rises


def _rest_acc(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.c_[rng.normal(0, 0.02, len(t)), rng.normal(0, 0.02, len(t)), 9.81 + rng.normal(0, 0.02, len(t))]


def _match(est: np.ndarray, truth: np.ndarray, tol: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """(estimate - truth) per matched beat after removing the median offset, and truth indices."""
    j = np.clip(np.searchsorted(truth, est), 1, len(truth) - 1)
    near = np.where(np.abs(truth[j - 1] - est) <= np.abs(truth[j] - est), j - 1, j)
    d = est - truth[near]
    d -= np.median(d)
    ok = np.abs(d) < tol
    return d[ok], near[ok]


def _rmssd_kept(out: dict) -> float:
    b, q = out["beats_ms"], out["quality"]
    ibi, ok = beat_intervals(b, q >= 0.5)
    return time_domain(ibi, ok)["rmssd"]


def test_clean_synthetic_timing_is_sub_sample():
    rng = np.random.default_rng(0)
    beats = _beats(180, rng)
    t = np.arange(0, 180_000, 40.0)
    out = cft.FootTemplateEstimator(drops=False).estimate(t, _ppg(t, beats, rng), t, _rest_acc(t, rng))
    assert len(out["beats_ms"]) == len(out["quality"])
    kept = out["beats_ms"][out["quality"] >= 0.5]
    assert len(kept) > 0.95 * len(beats)
    d, near = _match(kept, beats)
    assert len(np.unique(near)) == len(near) > 0.95 * len(beats)
    assert np.std(d) < 3.0  # well under the 40 ms sample grid (uniform quantisation: 11.5 ms SD)
    ref = time_domain(np.diff(beats))["rmssd"]
    assert abs(_rmssd_kept(out) - ref) < 2.0


def _lossy_clock(dur_s: float, rng: np.random.Generator, period: tuple[int, int] = (265, 296)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The watch's sampling: true 40 ms grid, one sample lost every ~280 (jittered), the delivered
    samples stamped evenly. Returns (true times, stamps, lost grid indices) of delivered samples."""
    grid = np.arange(0, dur_s * 1000, 40.0)
    lost = np.cumsum(rng.integers(*period, 80))
    lost = lost[lost < len(grid) - 10]
    keep = np.ones(len(grid), bool)
    keep[lost] = False
    t_true = grid[keep]
    return t_true, np.linspace(t_true[0], t_true[-1], len(t_true)), lost


def test_lost_samples_are_found_and_undone():
    rng = np.random.default_rng(1)
    beats = _beats(300, rng, rsa_ms=15.0)  # strong fast RSA hides losses (see fit_drops)
    t_true, stamps, lost = _lossy_clock(300, rng)
    ppg = _ppg(t_true, beats, rng)
    acc = _rest_acc(stamps, rng)
    off = cft.FootTemplateEstimator(drops=False).estimate(stamps, ppg, stamps, acc)
    est = cft.FootTemplateEstimator()
    on = est.estimate(stamps, ppg, stamps, acc)
    # on the stamp clock the true intervals are what a perfect detector returns
    truth = np.interp(beats, t_true, stamps)
    ref = time_domain(np.diff(truth))["rmssd"]
    true_ref = time_domain(np.diff(beats))["rmssd"]
    assert ref > true_ref + 1.0  # the losses inflate RMSSD on the stamp clock
    for out, name in ((off, "off"), (on, "on")):
        kb = out["beats_ms"][out["quality"] >= 0.5]
        assert len(kb) > 0.9 * len(beats), name
    big_off = np.mean(np.abs(_ibi_err(off, beats, t_true, stamps)) > 20)
    big_on = np.mean(np.abs(_ibi_err(on, beats, t_true, stamps)) > 20)
    assert big_off > 0.04  # ~1 interval in 13 is a loss
    assert big_on < 0.4 * big_off
    assert abs(_rmssd_kept(on) - true_ref) < 0.5 * abs(_rmssd_kept(off) - true_ref)
    assert 0.6 * len(lost) <= est.debug["n_drops"] <= 1.3 * len(lost)


def _ibi_err(out: dict, beats: np.ndarray, t_true: np.ndarray, stamps: np.ndarray) -> np.ndarray:
    """Usable estimated intervals minus the true intervals they span (beats matched on true time)."""
    b, q = out["beats_ms"], out["quality"]
    keep = q >= 0.5
    off = np.median(b[keep] - np.interp(beats[np.searchsorted(beats, b[keep]).clip(1, len(beats) - 1) - 1], t_true, stamps))
    tb = b - off
    e = []
    for i in np.where(keep[:-1] & keep[1:])[0]:
        j0, j1 = (np.argmin(np.abs(np.interp(beats, t_true, stamps) - x)) for x in (tb[i], tb[i + 1]))
        if j1 == j0 + 1:
            e.append((b[i + 1] - b[i]) - (beats[j1] - beats[j0]))
    return np.array(e)


def test_fit_drops_on_a_beat_series():
    rng = np.random.default_rng(2)
    beats = _beats(240, rng, rsa_ms=15.0)
    t_true, stamps, lost = _lossy_clock(240, rng, (250, 271))
    stamped = np.interp(beats, t_true, stamps)  # beat times on the stamp clock
    si = np.interp(beats, t_true, np.arange(len(t_true)))  # position in delivered samples
    shift, n = cft.fit_drops(stamped, np.ones(len(beats), bool), si)
    fixed = stamped + shift
    e_before = np.diff(stamped) - np.diff(beats)
    e_after = np.diff(fixed) - np.diff(beats)
    assert np.mean(e_before < -30) > 0.05  # each loss: one interval ~40 ms short
    assert np.mean(np.abs(e_after) > 20) < 0.3 * np.mean(np.abs(e_before) > 20)
    assert 0.7 * len(lost) <= n <= 1.3 * len(lost)
    assert np.abs(shift).max() <= 40.0  # a zero-mean sawtooth, not a drift


def test_artifact_correction():
    rng = np.random.default_rng(3)
    beats = _beats(120, rng)
    b = beats.copy()
    b[40] += 120.0  # misplaced beat
    extra = (b[60] + b[61]) / 2  # extra beat in the middle of an interval
    b = np.sort(np.r_[np.delete(b, 80), extra])  # missed beat 80
    q, c = np.ones(len(b)), np.ones(len(b))
    bt, q2, _ = cft.artifact_correct(b, q, c)
    kept = bt[q2 > 0]
    assert not np.isclose(kept, extra).any()  # extra beat removed
    assert not np.isclose(kept, beats[40] + 120.0).any()  # misplaced beat rejected
    # the interval across the missed beat is not used: a quality-0 placeholder sits inside it
    gap = (bt > beats[79]) & (bt < beats[81])
    assert gap.any() and (q2[gap] == 0).all()
    ibi, ok = beat_intervals(bt, q2 > 0)
    assert np.abs(ibi[ok] - np.median(ibi[ok])).max() < 250


def test_motion_gate_rejects_beats_during_movement():
    rng = np.random.default_rng(4)
    beats = _beats(120, rng)
    t = np.arange(0, 120_000, 40.0)
    acc = _rest_acc(t, rng)
    move = (t > 50_000) & (t < 70_000)
    acc[move] += rng.normal(0, 3.0, (move.sum(), 3))
    out = cft.FootTemplateEstimator().estimate(t, _ppg(t, beats, rng), t, acc)
    b, q = out["beats_ms"], out["quality"]
    inside = (b > 51_000) & (b < 69_000)
    assert inside.any() and (q[inside] < 0.5).all()
    assert (q[(b < 45_000) | (b > 75_000)] >= 0.5).mean() > 0.9


@needs_data
def test_p02_through_the_harness():
    rows, info = bench.score_participant("P02", cft.FootTemplateEstimator())
    assert info["error"] is None, info["error"]
    rest = [r for r in rows if r["usable"] and r["window_s"] == 60 and r["group"] == "seated-rest"]
    cov = np.mean([r["covered"] for r in rest])
    err = np.array([r["est_rmssd"] - r["ref_rmssd"] for r in rest if r["covered"]])
    assert cov > 0.8
    assert np.median(np.abs(err)) < 8.0  # naive peak picking: ~28 ms at rest
    assert all(not r["covered"] for r in rows if r["usable"] and r["kind"] in ("jogging", "running"))
