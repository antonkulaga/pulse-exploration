"""HRV benchmark harness tests: GalaxyPPG loader, ECG reference, clock alignment, scoring.

Unit tests run on synthetic beats; the rest use two GalaxyPPG participants (P02, P03) and skip when
the dataset is not unpacked under data/external/galaxyppg/. First run builds the parquet cache
(~10 s), later runs take a few seconds. Run: ``uv run pytest -q tests/test_hrv_bench.py``.
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
from pulse_exploration.hrv.cand_naive import NaiveEstimator  # noqa: E402

UIDS = ("P02", "P03")
needs_data = pytest.mark.skipif(not (galaxyppg.DATASET_DIR / "P02").exists(), reason="GalaxyPPG not unpacked")


# --------------------------------------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------------------------------------


def test_time_domain_known_values():
    ibi = np.array([800.0, 850.0, 800.0, 900.0])
    m = time_domain(ibi)
    assert m["mean_nn"] == pytest.approx(837.5)
    assert m["rmssd"] == pytest.approx(np.sqrt((50**2 + 50**2 + 100**2) / 3))
    assert m["pnn50"] == pytest.approx(100 / 3)
    # an unusable interval breaks the successive differences around it
    m2 = time_domain(ibi, ok=np.array([True, False, True, True]))
    assert m2["n_ibi"] == 3 and m2["n_diff"] == 1 and m2["rmssd"] == pytest.approx(100.0)
    m3 = time_domain(ibi, adjacent=np.array([True, False, True]))
    assert m3["n_diff"] == 2


def test_match_beats_recovers_delay_and_counts():
    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.normal(800, 40, 75))
    est = ref + 120 + rng.normal(0, 8, len(ref))
    est = np.sort(np.r_[est[:-5], ref[10] + 400.0, ref[30] + 380.0])  # 5 misses, 2 spurious beats
    m = bench.match_beats(est, ref, tol_ms=50)
    assert m["delay_ms"] == pytest.approx(120, abs=5)
    assert len(m["pairs"]) == len(ref) - 5
    assert np.abs(m["resid"]).max() < 50
    # pairs are one-to-one
    assert len(np.unique(m["pairs"][:, 1])) == len(m["pairs"])


def test_window_delay_resists_spurious_beats():
    rng = np.random.default_rng(1)
    ref = np.cumsum(rng.normal(700, 30, 90))
    good = ref[::2] + 150 + rng.normal(0, 10, len(ref[::2]))
    junk = rng.uniform(ref[0], ref[-1], 60)  # more junk than half the beats
    robust, plain = bench.window_delay(np.sort(np.r_[good, junk]), ref, 50)
    assert robust == pytest.approx(150, abs=10)
    assert abs(plain - 150) > abs(robust - 150)


# --------------------------------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize("uid", UIDS)
def test_load(uid):
    rec = galaxyppg.load(uid)
    assert rec.has_watch and rec.has_watch_hr
    assert len(rec.sessions) == 20 and rec.sessions[0].name == "adaptation"
    assert {s.kind for s in rec.sessions} >= {"meditation", "rest", "running"}
    t, _ = rec.ppg_arrays()
    assert np.all(np.diff(t) > 0)
    assert 24.0 < (len(t) - 1) / ((t[-1] - t[0]) / 1000) < 25.1
    assert 129.5 < rec.ecg_fs < 131.0
    assert rec.ecg_time_fit["max_abs_residual_ms"] < 5
    assert np.all(np.diff(rec.ecg["t_ms"].to_numpy()) > 0)
    # Polar is in UTC+9 in the files; after the shift it overlaps the watch recording
    assert abs(rec.ecg["t_ms"][0] - t[0]) < 15 * 60_000
    s = galaxyppg.samsung_ibi(rec)
    assert s.height > 1000 and s["ok"].mean() > 0.3
    assert np.all(np.diff(s["t_end_ms"].to_numpy()) > 0)


@needs_data
@pytest.mark.parametrize("uid", UIDS)
def test_reference_agrees_with_polar(uid):
    ref = galaxyppg.reference(uid)
    s = ref.summary
    assert s["nn_frac"] > 0.95
    assert s["nn_polar_confirmed_frac"] > 0.95
    assert s["nn_polar_mae_ms"] < 3.0
    assert 50 < s["hr_median_bpm"] < 110
    assert ref.beats.height == s["n_beats"] and len(ref.nn()) == ref.beats.height - 1


@needs_data
@pytest.mark.parametrize("uid", UIDS)
def test_alignment(uid):
    al = galaxyppg.alignment(uid)
    assert al.coarse_snr > 20
    assert al.n_good_chunks >= al.n_chunks - 1
    assert abs(al.drift_ppm) < 100
    assert al.lobe_ok
    assert al.polarity == -1  # green ADC falls with the pulse upstroke
    # mapping is affine and monotonic
    x = np.array([al.t_ref_ms - 1e6, al.t_ref_ms, al.t_ref_ms + 1e6])
    y = al.to_watch(x)
    assert y[1] - x[1] == pytest.approx(al.offset_ms)
    assert np.all(np.diff(y) > 0)


class Oracle:
    """Reference beats on the watch clock, delayed and jittered: what a perfect detector would give."""

    def __init__(self, uid: str, delay_ms: float = 37.0, jitter_ms: float = 0.0, seed: int = 0, name: str = "oracle"):
        self.t = bench._prepare_ref(uid).t
        self.delay, self.jitter, self.seed, self.name = delay_ms, jitter_ms, seed, name

    def estimate(self, ppg_t_ms, ppg, acc_t_ms, acc_xyz):
        rng = np.random.default_rng(self.seed)
        return {"beats_ms": self.t + self.delay + rng.normal(0, self.jitter, len(self.t)) if self.jitter else self.t + self.delay}


def _rows(uid, est, **cfg):
    rows, info = bench.score_participant(uid, est, bench.BenchConfig(**cfg))
    assert info["error"] is None, info["error"]
    return pl.DataFrame(rows)


@needs_data
@pytest.mark.parametrize("uid", UIDS)
def test_oracle_scores_perfectly(uid):
    w = _rows(uid, Oracle(uid)).filter(pl.col("usable"))
    assert w.height > 30
    assert w["covered"].all()
    assert (w["precision"] > 0.99).all() and (w["recall"] > 0.99).all()
    assert (w["delay_ms"] - 37).abs().max() < 1
    # the oracle also emits the reference's non-NN intervals; usable windows allow <= 10 % of those
    err = (w["est_rmssd"] - w["ref_rmssd"]).abs()
    assert err.median() < 1.0
    dm = (w["est_mean_nn"] - w["ref_mean_nn"]).abs()
    assert dm.median() < 0.5 and dm.max() < 10
    assert w["ibi_mae_ms"].max() < 0.5


@needs_data
def test_intervals_across_ppg_gaps_are_not_used():
    uid = "P02"  # meditation has a 12 s PPG gap inside a 300 s window that keeps > 90 % of samples
    t, _ = galaxyppg.load(uid).ppg_arrays()
    gi = np.where(np.diff(t) > bench.PPG_GAP_MS)[0]
    assert len(gi) and (t[gi + 1] - t[gi]).max() > 10_000

    class GapOracle(Oracle):
        """A detector that finds every beat except those inside PPG gaps (it has no signal there)."""

        def estimate(self, *a):
            b = super().estimate(*a)["beats_ms"]
            inside = np.zeros(len(b), bool)
            for i in gi:
                inside |= (b > t[i]) & (b < t[i + 1])
            return {"beats_ms": b[~inside]}

    plain = _rows(uid, Oracle(uid)).filter(pl.col("usable"))
    gap = _rows(uid, GapOracle(uid)).filter(pl.col("usable"))
    assert gap.filter(pl.col("covered")).height == plain.filter(pl.col("covered")).height
    err = lambda w: (w["est_rmssd"] - w["ref_rmssd"]).abs().max()  # noqa: E731
    assert err(gap) <= err(plain) + 2.0


@needs_data
def test_samsung_cleaning_removes_most_of_the_bias():
    res = bench.run([bench.SamsungIBI(), bench.SamsungIBIClean()], UIDS, workers=1)
    ov = {r["estimator"]: r for r in res.summary().filter(pl.col("window_s") == 60).iter_rows(named=True)}
    raw, clean = ov["samsung_ibi"], ov["samsung_ibi_clean"]
    assert raw["rmssd_bias"] > 15  # unflagged artifacts and lost-sample intervals in the raw list
    assert abs(clean["rmssd_bias"]) < raw["rmssd_bias"] / 2
    assert clean["rmssd_mae_median"] < raw["rmssd_mae_median"]
    assert clean["coverage"] > 0.5


@needs_data
def test_jitter_inflates_rmssd_and_ibi_error():
    uid = UIDS[0]
    w = _rows(uid, Oracle(uid, jitter_ms=10)).filter(pl.col("usable"), pl.col("window_s") == 60)
    # iid timing noise sd s adds ~sqrt(6) s to RMSSD in quadrature and ~2 s / sqrt(pi) to IBI MAE
    assert w["ibi_mae_ms"].median() == pytest.approx(2 * 10 / np.sqrt(np.pi), rel=0.15)
    expect = np.sqrt(w["ref_rmssd"].to_numpy() ** 2 + 6 * 10**2)
    assert np.median(w["est_rmssd"].to_numpy() / expect) == pytest.approx(1.0, abs=0.1)
    assert (w["recall"] > 0.95).all()


@needs_data
def test_declining_and_quality():
    uid = UIDS[0]
    rec = galaxyppg.load(uid)

    class Declines(Oracle):
        def estimate(self, *a):
            out = super().estimate(*a)
            mid = rec.sessions[len(rec.sessions) // 2].start_ms
            return {**out, "accept_windows": [(0, mid)]}

    class AllLowQuality(Oracle):
        def estimate(self, *a):
            out = super().estimate(*a)
            return {**out, "quality": np.zeros(len(out["beats_ms"]))}

    w = _rows(uid, Declines(uid)).filter(pl.col("usable"))
    cov = w["covered"].mean()
    assert 0.2 < cov < 0.8
    assert (w.filter(pl.col("covered"))["end_ms"] <= rec.sessions[len(rec.sessions) // 2].start_ms).all()
    w = _rows(uid, AllLowQuality(uid)).filter(pl.col("usable"))
    assert not w["covered"].any() and (w["n_est_beats"] == 0).all()
    w = _rows(uid, Declines(uid), accept_min_cover=1.0)
    assert w.filter(pl.col("usable"))["covered"].sum() <= cov * w.filter(pl.col("usable")).height


@needs_data
def test_crashing_estimator_is_recorded():
    class Boom:
        name = "boom"

        def estimate(self, *a):
            raise RuntimeError("nope")

    rows, info = bench.score_participant(UIDS[0], Boom())
    assert "nope" in info["error"]
    w = pl.DataFrame(rows)
    assert w.height > 0 and not w["covered"].any() and (w["reason"] == "no_source").all()


@needs_data
def test_naive_and_samsung_end_to_end():
    res = bench.run([NaiveEstimator(), bench.SamsungIBI()], UIDS, workers=1)
    assert res.runs["error"].is_null().all()
    acc = bench.window_accounting(res.windows)
    assert acc.filter(pl.col("reason") == "ok")["n"].sum() > 100
    tabs = res.tables()
    ov = tabs["overall"].filter(pl.col("window_s") == 60)
    naive = ov.filter(pl.col("estimator") == "naive").row(0, named=True)
    sams = ov.filter(pl.col("estimator") == "samsung_ibi").row(0, named=True)
    assert naive["coverage"] > 0.9 and sams["coverage"] > 0.5
    # naive 25 Hz peak picking overestimates RMSSD (cf. the paper: 185.8 vs 41.6 ms with HeartPy)
    assert naive["rmssd_bias"] > 20
    assert naive["precision"] > 0.5 and naive["recall"] > 0.5
    assert np.isnan(sams["precision"])  # interval-only source: no beat matching
    rest = tabs["by_group"].filter(pl.col("window_s") == 60, pl.col("group") == "seated-rest", pl.col("estimator") == "naive")
    assert rest["recall"][0] > 0.7
    assert set(tabs["by_participant"]["uid"]) == set(UIDS)
    md = bench.to_markdown(tabs["overall"], ["estimator", "window_s", *bench.SHOW])
    assert md.count("\n") >= 3
    com = res.summary(common=True)
    assert (com["n_windows"].to_numpy() <= ov["n_windows"].max()).all()
