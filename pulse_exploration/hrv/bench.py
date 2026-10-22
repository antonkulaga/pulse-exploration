"""HRV benchmark on GalaxyPPG: score beat estimators on 25 Hz Galaxy Watch PPG against Polar ECG.

Question: can HRV be computed from the watch's no-setup 25 Hz PPG (tier "25hz"), or only from
SDK-tier sources (Samsung's IBI list, 100 Hz PPG, ECG)?

Plug-in contract (`Estimator`): `estimate(ppg_t_ms, ppg, acc_t_ms, acc_xyz) -> dict` gets one
participant's whole watch recording (watch clock, Unix ms; PPG = raw green ADC of status-0 samples,
-999 blocks removed; ACC in m/s^2, n x 3) and returns
    beats_ms        np.ndarray   beat times on the same clock (any fiducial point)
    quality         np.ndarray | None   per-beat 0..1; beats below `quality_min` are dropped and the
                                        intervals touching them are not used
    accept_windows  list[(start_ms, end_ms)] | None   where the estimator vouches for its output;
                                        None = everywhere. A scoring window counts as accepted when
                                        >= `accept_min_cover` of it is inside these.
The estimator never sees ECG, Polar data, events or the clock alignment.

SDK-tier sources that need other tables implement `RecordingEstimator.estimate_recording(rec)`
and may return intervals instead of beats (`ibi_end_ms`, `ibi_ms`, `ibi_ok`, `ibi_adjacent`);
`SamsungIBI` (GalaxyWatch/HR.csv) is the built-in one. Interval sources get window HRV only.

Scoring, per participant and session (Event.csv, watch clock): non-overlapping 60 s and 300 s
windows anchored at the session start and fully inside it.
- Reference: ECG R-peaks (`galaxyppg.reference`) mapped to the watch clock with
  `galaxyppg.alignment` (a fixed R-triggered-PPG method, independent of every candidate). HRV from
  NN intervals only. A window's reference is used when >= `ref_min_nn_frac` of its intervals are NN
  and >= `ref_min_polar_frac` of those agree with Polar's own RR within 10 ms; windows with
  < `min_ppg_presence` of the expected PPG samples are dropped as recording gaps.
- Estimate: HRV from the kept intervals inside the window, minus intervals that span a PPG gap
  (> `PPG_GAP_MS` without samples); computable when it has
  >= max(`min_ibi`, `min_ibi_frac` x reference NN count) intervals and >= 5 successive differences.
  covered = accepted and computable; coverage = covered / usable windows.
- Beat matching (all windows with beats): constant per-window delay = median of (estimate beat -
  nearest reference beat), re-taken over the differences near their mode so spurious beats do not
  drag it (`window_delay`; the plain median is kept as delay_plain_ms); after removing it,
  one-to-one matching within +-`match_tol_ms`; precision,
  recall, timing SD of matched beats, IBI MAE over consecutive matched pairs whose reference
  interval is NN.
- Summary (covered windows): for RMSSD, SDNN, meanNN: median |error|, median |error| %, bias
  (mean est - ref), Bland-Altman limits (bias +- 1.96 SD), Pearson r; pNN50 median |error|;
  pooled precision/recall/IBI MAE; coverage. By session type, session group, participant, overall.

CLI:
    uv run python -m pulse_exploration.hrv.bench pulse_exploration.hrv.cand_naive:NaiveEstimator \\
        --samsung [--uids P02 P03] [--out data/runs/hrv_bench/naive]
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import polars as pl

from pulse_exploration import galaxyppg
from pulse_exploration.hrv import METRICS, beat_intervals, time_domain

GROUPS = {
    "adaptation": "seated-rest",
    "baseline": "seated-rest",
    "meditation": "seated-rest",
    "rest": "seated-rest",
    "screen-reading": "cognitive-stress",
    "tsst-prep": "cognitive-stress",
    "tsst-speech": "cognitive-stress",
    "ssst-prep": "cognitive-stress",
    "ssst-sing": "cognitive-stress",
    "keyboard-typing": "typing",
    "mobile-typing": "typing",
    "standing": "standing",
    "walking": "locomotion",
    "jogging": "locomotion",
    "running": "locomotion",
}


@runtime_checkable
class Estimator(Protocol):
    name: str

    def estimate(self, ppg_t_ms: np.ndarray, ppg: np.ndarray, acc_t_ms: np.ndarray, acc_xyz: np.ndarray) -> dict: ...


@runtime_checkable
class RecordingEstimator(Protocol):
    name: str

    def estimate_recording(self, rec: galaxyppg.Recording) -> dict | None: ...


@dataclass(frozen=True)
class BenchConfig:
    window_s: tuple[int, ...] = (60, 300)
    match_tol_ms: float = 50.0
    quality_min: float = 0.5
    accept_min_cover: float = 0.9
    min_ibi: int = 10
    min_ibi_frac: float = 0.5
    ref_min_nn_frac: float = 0.9
    ref_min_polar_frac: float = 0.8
    ref_max_edge_ms: float = 3000.0  # first/last reference beat must be this close to the window edges
    min_ppg_presence: float = 0.9


class SamsungIBI:
    """SDK-tier baseline: Samsung's IBI list from GalaxyWatch/HR.csv (ibiStatus == 0), no beat times."""

    name = "samsung_ibi"

    def estimate_recording(self, rec: galaxyppg.Recording) -> dict | None:
        if not rec.has_watch_hr:
            return None
        s = galaxyppg.samsung_ibi(rec)
        return {
            "ibi_end_ms": s["t_end_ms"].to_numpy(),
            "ibi_ms": s["ibi_ms"].to_numpy(),
            "ibi_ok": s["ok"].to_numpy(),
            "ibi_adjacent": s["adjacent_next"].to_numpy()[:-1],
        }


class SamsungIBIClean(SamsungIBI):
    """Samsung's IBI list with the post-processing the 25 Hz candidates get: intervals outside
    300-2000 ms or more than `dev` from the median of `k` neighbours are dropped, and so are
    isolated short intervals (> `slip_ms` below the mean of both adjacent neighbours), which the
    stream's lost samples produce."""

    name = "samsung_ibi_clean"

    def __init__(self, dev: float = 0.2, k: int = 11, slip_ms: float = 30.0):
        self.dev, self.k, self.slip_ms = dev, k, slip_ms

    def estimate_recording(self, rec: galaxyppg.Recording) -> dict | None:
        out = super().estimate_recording(rec)
        if out is None or not len(out["ibi_ms"]):
            return out
        from scipy.ndimage import median_filter

        v, ok, adj = out["ibi_ms"], out["ibi_ok"].copy(), out["ibi_adjacent"]
        idx = np.maximum.accumulate(np.where(ok, np.arange(len(v)), 0))  # carry the last good value
        filled = np.where(ok[idx], v[idx], np.median(v[ok]) if ok.any() else np.median(v))
        med = median_filter(filled, size=self.k, mode="nearest")
        ok &= (v >= 300) & (v <= 2000) & (np.abs(v - med) <= self.dev * med)
        drop = np.zeros(len(v), bool)
        if len(v) > 2:
            mid = slice(1, -1)
            both = ok[mid] & ok[:-2] & ok[2:] & adj[:-1] & adj[1:]
            drop[mid] = both & (v[mid] - 0.5 * (v[:-2] + v[2:]) < -self.slip_ms)
        return {**out, "ibi_ok": ok & ~drop}


# --------------------------------------------------------------------------------------------------
# Window scoring
# --------------------------------------------------------------------------------------------------


def _windows(rec: galaxyppg.Recording, window_s: tuple[int, ...]) -> list[dict]:
    out = []
    for s in rec.sessions:
        for w in window_s:
            n = int((s.end_ms - s.start_ms) // (w * 1000))
            for k in range(n):
                ws = s.start_ms + k * w * 1000
                out.append(
                    {
                        "session": s.name,
                        "kind": s.kind,
                        "group": GROUPS.get(s.kind, "other"),
                        "window_s": w,
                        "k": k,
                        "start_ms": ws,
                        "end_ms": ws + w * 1000,
                    }
                )
    return out


def _merge(intervals: list[tuple[float, float]]) -> np.ndarray:
    if not intervals:
        return np.zeros((0, 2))
    a = np.array(sorted((float(s), float(e)) for s, e in intervals if e > s))
    if not len(a):
        return np.zeros((0, 2))
    merged = [a[0].copy()]
    for s, e in a[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append(np.array([s, e]))
    return np.array(merged)


def _cover(merged: np.ndarray, ws: float, we: float) -> float:
    if not len(merged):
        return 0.0
    ov = np.clip(np.minimum(merged[:, 1], we) - np.maximum(merged[:, 0], ws), 0, None)
    return float(ov.sum() / (we - ws))


def window_delay(est: np.ndarray, ref: np.ndarray, tol_ms: float) -> tuple[float, float]:
    """Constant estimate-minus-reference delay for one window.

    Returns (robust, plain): plain = median over estimate beats of (beat - nearest reference beat).
    Spurious beats spread those differences over +-RR/2 and drag the plain median, so the delay
    used is the same median re-taken twice over the differences within +-2 tol of a start value:
    the centre of the densest 5 ms-binned 50 ms run (the mode)."""
    j = np.clip(np.searchsorted(ref, est), 1, max(len(ref) - 1, 1))
    if len(ref) == 1:
        d = est - ref[0]
    else:
        d = est - np.where(np.abs(ref[j - 1] - est) <= np.abs(ref[j] - est), ref[j - 1], ref[j])
    plain = float(np.median(d))
    lo, hi = np.floor(d.min() / 5) * 5, np.ceil(d.max() / 5) * 5 + 5
    hist, edges = np.histogram(d, bins=np.arange(lo, hi + 5, 5.0))
    dens = np.convolve(hist, np.ones(10), mode="same")
    c = float(edges[int(np.argmax(dens))] + 2.5)
    for _ in range(2):
        near = d[np.abs(d - c) <= 2 * tol_ms]
        if not len(near):
            break
        c = float(np.median(near))
    return c, plain


def match_beats(est: np.ndarray, ref: np.ndarray, tol_ms: float, delay_from: np.ndarray | None = None) -> dict:
    """Constant-delay beat matching of sorted beat times.

    After removing `window_delay` (estimated from est[delay_from] if given), each estimate beat is
    paired with its nearest reference beat if within tol, closest pairs first, one-to-one. Returns
    delay (robust and plain), pairs (est index, ref index) and residuals."""
    if len(est) == 0 or len(ref) == 0:
        return {"delay_ms": math.nan, "delay_plain_ms": math.nan, "pairs": np.zeros((0, 2), int), "resid": np.zeros(0)}

    def nearest(x: np.ndarray) -> np.ndarray:
        if len(ref) == 1:
            return np.zeros(len(x), int)
        j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
        return np.where(np.abs(ref[j - 1] - x) <= np.abs(ref[j] - x), j - 1, j)

    de = est if delay_from is None else est[delay_from]
    if not len(de):
        return {"delay_ms": math.nan, "delay_plain_ms": math.nan, "pairs": np.zeros((0, 2), int), "resid": np.zeros(0)}
    delay, plain = window_delay(de, ref, tol_ms)
    e = est - delay
    j = nearest(e)
    d = e - ref[j]
    cand = np.where(np.abs(d) <= tol_ms)[0]
    cand = cand[np.argsort(np.abs(d[cand]), kind="stable")]
    used_r = np.zeros(len(ref), bool)
    pairs = []
    for i in cand:
        if not used_r[j[i]]:
            used_r[j[i]] = True
            pairs.append((i, j[i]))
    pairs = np.array(sorted(pairs), dtype=int).reshape(-1, 2)
    resid = e[pairs[:, 0]] - ref[pairs[:, 1]] if len(pairs) else np.zeros(0)
    return {"delay_ms": delay, "delay_plain_ms": plain, "pairs": pairs, "resid": resid}


@dataclass
class _Ref:
    t: np.ndarray  # watch clock
    nn: np.ndarray  # per interval
    polar_ok: np.ndarray  # per interval: Polar IBI within 10 ms


def _prepare_ref(uid: str) -> _Ref:
    ref = galaxyppg.reference(uid)
    al = galaxyppg.alignment(uid)
    t = al.to_watch(ref.beats["t_ms"].to_numpy())
    pdiff = ref.beats["polar_diff_ms"].to_numpy()[1:]
    return _Ref(t=t, nn=ref.nn(), polar_ok=np.abs(pdiff) <= galaxyppg.POLAR_CONFIRM_MS)


MATCH_PAD_MS = 1500.0
PPG_GAP_MS = 200.0  # consecutive PPG samples further apart than this mark a recording gap


def _match_window(beats: np.ndarray, keep_all: np.ndarray, ref: _Ref, ws: float, we: float, tol_ms: float) -> dict:
    """Beat-level scores for window [ws, we). Matching runs on sets padded by MATCH_PAD_MS so beats
    near the edges keep their partners; the delay comes from estimate beats inside the window.
    precision = matched / estimate beats whose delay-corrected time is inside; recall = matched /
    reference beats inside; IBI errors over consecutive matched pairs with an NN reference interval
    inside the window."""
    p0, p1 = np.searchsorted(beats, ws - MATCH_PAD_MS), np.searchsorted(beats, we + MATCH_PAD_MS)
    ep = beats[p0:p1][keep_all[p0:p1]]
    r0, r1 = np.searchsorted(ref.t, ws - MATCH_PAD_MS), np.searchsorted(ref.t, we + MATCH_PAD_MS)
    rp = ref.t[r0:r1]
    out: dict[str, Any] = {}
    mt = match_beats(ep, rp, tol_ms, delay_from=(ep >= ws) & (ep < we))
    out["delay_ms"], out["delay_plain_ms"] = mt["delay_ms"], mt["delay_plain_ms"]
    if not np.isfinite(mt["delay_ms"]):
        return out
    e_in = (ep - mt["delay_ms"] >= ws) & (ep - mt["delay_ms"] < we)
    r_in = (rp >= ws) & (rp < we)
    pairs = mt["pairs"]
    pe, pr = (pairs[:, 0], pairs[:, 1]) if len(pairs) else (np.zeros(0, int), np.zeros(0, int))
    out["n_est_in"] = int(e_in.sum())
    out["n_matched_est"] = int(e_in[pe].sum())
    out["n_matched_ref"] = int(r_in[pr].sum())
    if e_in.any() and r_in.any():
        out["precision"] = out["n_matched_est"] / out["n_est_in"]
        out["recall"] = out["n_matched_ref"] / int(r_in.sum())
    res_in = mt["resid"][r_in[pr]] if len(pr) else np.zeros(0)
    if len(res_in) > 2:
        out["timing_sd_ms"] = float(np.std(res_in))
    if len(pairs) > 1:
        cons = (np.diff(pe) == 1) & (np.diff(pr) == 1) & r_in[pr[:-1]] & r_in[pr[1:]]
        cons &= ref.nn[np.minimum(r0 + pr[:-1], len(ref.nn) - 1)]
        if cons.any():
            err = np.abs((ep[pe[1:]] - ep[pe[:-1]])[cons] - (rp[pr[1:]] - rp[pr[:-1]])[cons])
            out.update({"n_ibi_pairs": int(len(err)), "ibi_abs_sum": float(err.sum()), "ibi_mae_ms": float(err.mean())})
    return out


def _score_windows(
    uid: str,
    name: str,
    out: dict | None,
    rec: galaxyppg.Recording,
    ref: _Ref,
    cfg: BenchConfig,
    ppg_t: np.ndarray,
    acc: tuple[np.ndarray, np.ndarray],
) -> list[dict]:
    beats = quality = accept = None
    ibi_src = None
    if out is not None and "beats_ms" in out:
        beats = np.asarray(out["beats_ms"], dtype=np.float64)
        order = np.argsort(beats, kind="stable")
        beats = beats[order]
        q = out.get("quality")
        quality = None if q is None else np.asarray(q, dtype=np.float64)[order]
        aw = out.get("accept_windows")
        accept = None if aw is None else _merge(list(aw))
    elif out is not None and "ibi_ms" in out:
        ibi_src = (
            np.asarray(out["ibi_end_ms"], dtype=np.float64),
            np.asarray(out["ibi_ms"], dtype=np.float64),
            np.asarray(out.get("ibi_ok", np.ones(len(out["ibi_ms"]), bool)), bool),
            np.asarray(out.get("ibi_adjacent", np.ones(max(len(out["ibi_ms"]) - 1, 0), bool)), bool),
        )
        if np.any(np.diff(ibi_src[0]) < 0):
            raise ValueError(f"{name}: ibi_end_ms must be sorted (adjacency refers to stream order)")
        accept = None if out.get("accept_windows") is None else _merge(list(out["accept_windows"]))
    keep_all = None if beats is None else (np.ones(len(beats), bool) if quality is None else quality >= cfg.quality_min)
    # PPG gaps: an interval between two estimate beats that spans one is not a heartbeat interval
    gi = np.where(np.diff(ppg_t) > PPG_GAP_MS)[0]
    gap_s, gap_e = ppg_t[gi], ppg_t[gi + 1]
    acc_t, acc_xyz = acc
    acc_mag = np.linalg.norm(acc_xyz, axis=1) if len(acc_t) else np.zeros(0)

    rows = []
    for w in _windows(rec, cfg.window_s):
        ws, we, W = w["start_ms"], w["end_ms"], w["window_s"]
        row: dict[str, Any] = {"uid": uid, "estimator": name, **w}
        # watch data presence and motion
        n_ppg = int(np.searchsorted(ppg_t, we) - np.searchsorted(ppg_t, ws))
        row["ppg_presence"] = n_ppg / (W * galaxyppg.PPG_FS)
        a0, a1 = np.searchsorted(acc_t, ws), np.searchsorted(acc_t, we)
        row["acc_sd"] = float(acc_mag[a0:a1].std()) if a1 - a0 > 10 else math.nan
        # reference
        i0, i1 = np.searchsorted(ref.t, ws), np.searchsorted(ref.t, we)  # beats i0..i1-1 inside
        rt = ref.t[i0:i1]
        iv = slice(i0, max(i1 - 1, i0))  # intervals between beats inside the window
        nn, pok = ref.nn[iv], ref.polar_ok[iv]
        ibi_ref = np.diff(rt)
        n_int = len(ibi_ref)
        row["ref_nn_frac"] = float(nn.mean()) if n_int else 0.0
        row["ref_polar_frac"] = float(pok[nn].mean()) if nn.any() else 0.0
        edge_ok = n_int > 0 and rt[0] - ws <= cfg.ref_max_edge_ms and we - rt[-1] <= cfg.ref_max_edge_ms
        if out is None:
            reason = "no_source"
        elif row["ppg_presence"] < cfg.min_ppg_presence and beats is not None:
            reason = "watch_gap"
        elif not edge_ok:
            reason = "ref_cover"
        elif row["ref_nn_frac"] < cfg.ref_min_nn_frac:
            reason = "ref_nn"
        elif row["ref_polar_frac"] < cfg.ref_min_polar_frac:
            reason = "ref_polar"
        else:
            reason = "ok"
        row["reason"] = reason
        row["usable"] = reason == "ok"
        rm = time_domain(ibi_ref, nn) if n_int else time_domain(np.zeros(0))
        for m in METRICS:
            row[f"ref_{m}"] = rm[m]
        row["ref_n_nn"] = rm["n_ibi"]
        row["n_ref_beats"] = int(len(rt))
        # estimate
        em = time_domain(np.zeros(0))
        row.update({"accepted": False, "computable": False, "precision": math.nan, "recall": math.nan})
        row.update({"delay_ms": math.nan, "delay_plain_ms": math.nan})
        row.update({"n_est_beats": 0, "n_est_in": 0, "n_matched_est": 0, "n_matched_ref": 0, "timing_sd_ms": math.nan})
        row.update({"ibi_mae_ms": math.nan, "n_ibi_pairs": 0, "ibi_abs_sum": 0.0})
        if out is not None:
            row["accepted"] = True if accept is None else _cover(accept, ws, we) >= cfg.accept_min_cover
            if beats is not None:
                b0, b1 = np.searchsorted(beats, ws), np.searchsorted(beats, we)
                eb, keep = beats[b0:b1], keep_all[b0:b1]
                ibi_e, ok_e = beat_intervals(eb, keep)
                if len(gap_s) and len(eb) > 1:
                    spans = np.searchsorted(gap_s, eb[1:]) - np.searchsorted(gap_e, eb[:-1], side="right")
                    ok_e &= spans <= 0
                em = time_domain(ibi_e, ok_e)
                row["n_est_beats"] = int(keep.sum())
                row.update(_match_window(beats, keep_all, ref, ws, we, cfg.match_tol_ms))
            elif ibi_src is not None:
                te, iv_, ok_, adj_ = ibi_src
                j0, j1 = np.searchsorted(te, ws), np.searchsorted(te, we)
                em = time_domain(iv_[j0:j1], ok_[j0:j1], adj_[j0 : max(j1 - 1, j0)])
            need = max(cfg.min_ibi, cfg.min_ibi_frac * rm["n_ibi"])
            row["computable"] = bool(em["n_ibi"] >= need and em["n_diff"] >= 5)
        for m in METRICS:
            row[f"est_{m}"] = em[m]
        row["est_n_ibi"] = em["n_ibi"]
        row["covered"] = bool(row["accepted"] and row["computable"])
        rows.append(row)
    return rows


def score_participant(uid: str, estimator: Any, cfg: BenchConfig | None = None) -> tuple[list[dict], dict]:
    """Run one estimator on one participant; returns (window rows, run info)."""
    cfg = cfg or BenchConfig()
    rec = galaxyppg.load(uid)
    name = getattr(estimator, "name", type(estimator).__name__)
    info = {"uid": uid, "estimator": name, "estimate_s": 0.0, "error": None}
    if not rec.has_watch or not rec.sessions:
        info["error"] = "no watch data or events"
        return [], info
    ref = _prepare_ref(uid)
    ppg_t, ppg = rec.ppg_arrays()
    acc = rec.acc_arrays()
    t0 = time.perf_counter()
    try:
        if isinstance(estimator, RecordingEstimator) and not isinstance(estimator, Estimator):
            out = estimator.estimate_recording(rec)
        else:
            out = estimator.estimate(ppg_t.copy(), ppg.copy(), acc[0].copy(), acc[1].copy())
    except Exception as exc:  # noqa: BLE001  (a crashing candidate scores as no output)
        out, info["error"] = None, f"{type(exc).__name__}: {exc}"
    info["estimate_s"] = time.perf_counter() - t0
    return _score_windows(uid, name, out, rec, ref, cfg, ppg_t, acc), info


def _job(args: tuple) -> tuple[list[dict], dict]:
    uid, est, cfg = args
    return score_participant(uid, est, cfg)


# --------------------------------------------------------------------------------------------------
# Runner and summaries
# --------------------------------------------------------------------------------------------------


@dataclass
class BenchResult:
    windows: pl.DataFrame
    runs: pl.DataFrame
    config: BenchConfig
    wall_s: float

    def summary(self, by: tuple[str, ...] = ("estimator", "window_s"), common: bool = False) -> pl.DataFrame:
        return summarize(self.windows, by=by, common=common)

    def tables(self, common: bool = False) -> dict[str, pl.DataFrame]:
        return {
            "overall": self.summary(("estimator", "window_s"), common),
            "by_group": self.summary(("estimator", "window_s", "group"), common),
            "by_session": self.summary(("estimator", "window_s", "kind"), common),
            "by_participant": self.summary(("estimator", "window_s", "uid"), common),
        }


def summarize(windows: pl.DataFrame, by: tuple[str, ...] = ("estimator", "window_s"), common: bool = False) -> pl.DataFrame:
    """Summary over usable windows; errors over covered windows. `common` keeps only windows usable
    for every estimator in the table (fair coverage comparison)."""
    w = windows.filter(pl.col("usable"))
    if common:
        n_est = w["estimator"].n_unique()
        keys = ["uid", "session", "window_s", "k"]
        ok = w.group_by(keys).agg(pl.len().alias("n")).filter(pl.col("n") == n_est).select(keys)
        w = w.join(ok, on=keys)
    rows = []
    for key, g in w.group_by(list(by), maintain_order=True):
        c = g.filter(pl.col("covered"))
        r: dict[str, Any] = dict(zip(by, key))
        r["n_windows"] = g.height
        r["n_participants"] = g["uid"].n_unique()
        r["coverage"] = c.height / g.height if g.height else math.nan
        for m in ("rmssd", "sdnn", "mean_nn"):
            est, ref = c[f"est_{m}"].to_numpy(), c[f"ref_{m}"].to_numpy()
            ok = np.isfinite(est) & np.isfinite(ref)
            e = est[ok] - ref[ok]
            r[f"{m}_ref_median"] = float(np.median(ref[ok])) if ok.any() else math.nan
            r[f"{m}_mae_median"] = float(np.median(np.abs(e))) if len(e) else math.nan
            r[f"{m}_ape_median_pct"] = float(np.median(np.abs(e) / ref[ok]) * 100) if len(e) else math.nan
            r[f"{m}_bias"] = float(e.mean()) if len(e) else math.nan
            sd = float(e.std(ddof=1)) if len(e) > 1 else math.nan
            r[f"{m}_loa_low"] = r[f"{m}_bias"] - 1.96 * sd
            r[f"{m}_loa_high"] = r[f"{m}_bias"] + 1.96 * sd
            r[f"{m}_r"] = float(np.corrcoef(est[ok], ref[ok])[0, 1]) if len(e) > 2 and np.std(est[ok]) > 0 else math.nan
        e = (c["est_pnn50"] - c["ref_pnn50"]).drop_nans().drop_nulls().to_numpy()
        r["pnn50_mae_median"] = float(np.median(np.abs(e))) if len(e) else math.nan
        cb = c.filter(pl.col("precision").is_not_nan())  # windows scored at beat level
        n_est, n_ref = cb["n_est_in"].sum(), cb["n_ref_beats"].sum()
        r["precision"] = cb["n_matched_est"].sum() / n_est if n_est else math.nan
        r["recall"] = cb["n_matched_ref"].sum() / n_ref if n_ref else math.nan
        npairs = c["n_ibi_pairs"].sum()
        r["ibi_mae_ms"] = c["ibi_abs_sum"].sum() / npairs if npairs else math.nan
        for col in ("timing_sd_ms", "delay_ms"):
            x = c[col].drop_nans().drop_nulls()
            r[f"{col}_median"] = float(x.median()) if x.len() else math.nan
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).sort(list(by)) if rows else pl.DataFrame()


def window_accounting(windows: pl.DataFrame) -> pl.DataFrame:
    """Why windows were (not) usable, per estimator and window length."""
    return windows.group_by("estimator", "window_s", "reason").agg(pl.len().alias("n")).sort("estimator", "window_s", "reason")


def run(
    estimators: Any,
    uids: list[str] | tuple[str, ...] | None = None,
    cfg: BenchConfig | None = None,
    workers: int | None = None,
) -> BenchResult:
    """Score one or more estimators on participants `uids` (default: all with watch data)."""
    cfg = cfg or BenchConfig()
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    ests = list(estimators) if isinstance(estimators, (list, tuple)) else [estimators]
    uids = list(uids or [u for u in galaxyppg.UIDS if u != "P01"])
    t0 = time.perf_counter()
    jobs = [(u, e, cfg) for e in ests for u in uids]
    if workers == 1 or len(jobs) == 1:
        results = [_job(j) for j in jobs]
    else:
        galaxyppg.prepare_all(uids, workers=workers)  # build caches once, in parallel
        with ProcessPoolExecutor(max_workers=workers or min(len(jobs), os.cpu_count() or 4), mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_job, jobs))
    rows = [r for rs, _ in results for r in rs]
    runs = pl.DataFrame([info for _, info in results], schema_overrides={"error": pl.String})
    return BenchResult(windows=pl.DataFrame(rows, infer_schema_length=None), runs=runs, config=cfg, wall_s=time.perf_counter() - t0)


SHOW = [
    "n_windows",
    "coverage",
    "rmssd_ref_median",
    "rmssd_mae_median",
    "rmssd_ape_median_pct",
    "rmssd_bias",
    "rmssd_loa_low",
    "rmssd_loa_high",
    "sdnn_mae_median",
    "sdnn_bias",
    "sdnn_loa_low",
    "sdnn_loa_high",
    "mean_nn_mae_median",
    "precision",
    "recall",
    "ibi_mae_ms",
]


def to_markdown(df: pl.DataFrame, cols: list[str] | None = None, digits: int = 2) -> str:
    if df.is_empty():
        return "(empty)"
    cols = [c for c in (cols or df.columns) if c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.select(cols).iter_rows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append("" if math.isnan(v) else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _load_estimator(spec: str) -> Any:
    mod, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(mod), attr or "Estimator")
    return obj() if isinstance(obj, type) else obj


def main(argv: list[str]) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("estimators", nargs="*", help="module:Class, e.g. pulse_exploration.hrv.cand_naive:NaiveEstimator")
    ap.add_argument("--samsung", action="store_true", help="also score Samsung's IBI list (SDK tier)")
    ap.add_argument("--uids", nargs="*", default=None)
    ap.add_argument("--windows", nargs="*", type=int, default=[60, 300])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--common", action="store_true", help="summaries over windows usable for every estimator")
    ap.add_argument("--out", type=Path, default=None, help="directory for windows.parquet + summary CSVs (keep under data/)")
    a = ap.parse_args(argv)
    ests = [_load_estimator(s) for s in a.estimators] + ([SamsungIBI()] if a.samsung else [])
    if not ests:
        ap.error("no estimators")
    res = run(ests, a.uids, BenchConfig(window_s=tuple(a.windows)), a.workers)
    tabs = res.tables(common=a.common)
    for name in ("overall", "by_group", "by_session"):
        print(f"\n## {name}\n")
        print(to_markdown(tabs[name], [c for c in tabs[name].columns if c in ("estimator", "window_s", "group", "kind")] + SHOW))
    print("\n## windows\n")
    print(to_markdown(window_accounting(res.windows)))
    errs = res.runs.filter(pl.col("error").is_not_null())
    if errs.height:
        print("\n## errors\n")
        print(to_markdown(errs))
    est_s = res.runs.group_by("estimator").agg(pl.col("estimate_s").sum())
    print(f"\nwall {res.wall_s:.1f} s; estimator CPU time: " + ", ".join(f"{e} {s:.1f} s" for e, s in est_s.iter_rows()))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        res.windows.write_parquet(a.out / "windows.parquet")
        res.runs.write_csv(a.out / "runs.csv")
        for name, t in tabs.items():
            t.write_csv(a.out / f"summary_{name}.csv")
        (a.out / "config.json").write_text(json.dumps(asdict(res.config), indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
