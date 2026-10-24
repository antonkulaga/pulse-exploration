"""Beat-interval AF screen: features, segments, persistence, and why it needs its own beat settings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import watch_synth as ws  # noqa: E402

from pulse_exploration.hrv import af_screen  # noqa: E402
from pulse_exploration.hrv.cand_foot_template import FootTemplateEstimator  # noqa: E402


def sinus(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.cumsum(np.full(n, 0.85))
    return 850 + 60 * np.sin(2 * np.pi * 0.25 * t) + 20 * rng.standard_normal(n)


def af(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(750 * rng.lognormal(0, 0.22, n), 330, 1800)


def test_features_separate_sinus_from_af():
    rng = np.random.default_rng(0)
    s, a = sinus(64, rng), af(64, rng)
    assert af_screen.shannon_entropy(np.full(64, 800.0)) == 0.0
    assert af_screen.nrmssd(s) < af_screen.TH_NRMSSD < af_screen.nrmssd(a)
    assert af_screen.shannon_entropy(a) > af_screen.TH_SHANNON


def test_trimming_ignores_a_few_bad_beats_in_sinus_rhythm():
    rng = np.random.default_rng(1)
    s = sinus(64, rng)
    s[[10, 30, 50]] += 400  # three missed-beat-like long intervals
    assert af_screen.nrmssd(s, trim=0) > af_screen.TH_NRMSSD
    assert af_screen.nrmssd(s) < af_screen.TH_NRMSSD


def test_segments_and_persistence():
    rng = np.random.default_rng(2)
    ibi = np.r_[sinus(64 * 3, rng), af(64 * 6, rng)]
    beats = np.r_[0.0, np.cumsum(ibi)]
    segs = af_screen.segments_from_beats(beats)
    assert [s.irregular for s in segs] == [False] * 3 + [True] * 6
    assert len(af_screen.alerts(segs)) == 1
    assert af_screen.alerts(segs[:7]) == []  # only 4 irregular among the first 6+1
    good = np.ones(len(beats), bool)
    good[100] = False  # a rejected beat ends the run, so the first segment restarts after it
    assert len(af_screen.segments_from_beats(beats, good)) < len(segs)


def test_interval_list_respects_adjacency():
    rng = np.random.default_rng(3)
    v = af(130, rng)
    end = np.cumsum(v)
    adj = np.ones(len(v) - 1, bool)
    assert len(af_screen.segments_from_intervals(end, v, np.ones(len(v), bool), adj)) == 2
    adj[50] = False  # a dropped batch between interval 50 and 51
    assert len(af_screen.segments_from_intervals(end, v, np.ones(len(v), bool), adj)) == 1


def _pulse_wave(kind: str, seed: int, dur: float = 200.0):
    rng = np.random.default_rng(seed)
    t, peaks = 0.5, []
    while t < dur:
        peaks.append(t)
        rr = 0.75 * rng.lognormal(0, 0.22) if kind == "af" else 0.85 + 0.06 * np.sin(2 * np.pi * 0.25 * t) + 0.02 * rng.standard_normal()
        t += float(np.clip(rr, 0.33, 1.8))
    ts = np.arange(0, dur, 0.04)
    x = ws.ppg_wave(ts, np.array(peaks) + 0.25) + 0.4 * np.sin(2 * np.pi * 0.2 * ts) + 0.02 * rng.standard_normal(len(ts))
    acc = np.c_[0.02 * rng.standard_normal(len(ts)), 0.02 * rng.standard_normal(len(ts)), 9.81 + 0.02 * rng.standard_normal(len(ts))]
    return ts * 1000, x, acc


def _segments(kind: str, settings: dict):
    t, x, acc = _pulse_wave(kind, 4)
    out = FootTemplateEstimator(invert=False, **settings).estimate(t, x, t, acc)
    return af_screen.segments_from_beats(out["beats_ms"], np.asarray(out["quality"]) >= 0.5)


def test_25hz_pulse_wave_af_is_flagged_with_the_screen_settings():
    segs = _segments("af", af_screen.AF_SETTINGS)
    assert len(segs) >= 2 and all(s.irregular for s in segs)
    assert not any(s.irregular for s in _segments("sinus", af_screen.AF_SETTINGS))


def test_hrv_cleaning_would_hide_af():
    # the HRV settings reject irregular intervals as artefacts, so no 64-beat run survives
    assert _segments("af", {}) == []
