"""HRV from Galaxy Watch PPG: shared time-domain metrics.

- `bench.py`: the benchmark harness (GalaxyPPG, ECG reference, scoring, summary tables).
- `cand_*.py`: candidate beat estimators that plug into the harness (`Estimator` protocol).

Conventions: times in ms, intervals in ms. An interval list comes with `ok` (the interval is usable)
and, optionally, `adjacent` (interval j and j + 1 are consecutive heart beats, so their difference
counts for RMSSD/pNN50). Intervals built from one beat list are always adjacent.
"""

from __future__ import annotations

import numpy as np

METRICS = ("mean_nn", "sdnn", "rmssd", "pnn50")


def beat_intervals(beats_ms: np.ndarray, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Intervals between consecutive beats and whether both ends are kept."""
    b = np.asarray(beats_ms, dtype=np.float64)
    ibi = np.diff(b)
    if keep is None:
        return ibi, np.ones(len(ibi), bool)
    k = np.asarray(keep, bool)
    return ibi, k[1:] & k[:-1]


def time_domain(ibi_ms: np.ndarray, ok: np.ndarray | None = None, adjacent: np.ndarray | None = None) -> dict[str, float]:
    """meanNN, SDNN (ddof=1), RMSSD, pNN50 (%) of the usable intervals.

    Successive differences use only pairs (j, j + 1) where both intervals are usable and, if
    `adjacent` (length n - 1) is given, marked adjacent. NaN where there is too little data."""
    ibi = np.asarray(ibi_ms, dtype=np.float64)
    ok = np.ones(len(ibi), bool) if ok is None else np.asarray(ok, bool)
    x = ibi[ok]
    pair = ok[1:] & ok[:-1]
    if adjacent is not None:
        pair &= np.asarray(adjacent, bool)
    d = np.diff(ibi)[pair]
    nan = float("nan")
    return {
        "mean_nn": float(x.mean()) if len(x) else nan,
        "sdnn": float(x.std(ddof=1)) if len(x) > 1 else nan,
        "rmssd": float(np.sqrt(np.mean(d * d))) if len(d) else nan,
        "pnn50": float(100.0 * np.mean(np.abs(d) > 50.0)) if len(d) else nan,
        "n_ibi": int(len(x)),
        "n_diff": int(len(d)),
    }
