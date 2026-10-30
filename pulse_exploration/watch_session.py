"""Laptop-side analysis of one watch run directory (SHARED CONTRACT, capabilities schema 1).

A run directory ``data/runs/<yyyyMMdd-HHmmss>-<kind>/`` holds ``capabilities.json``, one CSV per
source alias (columns as written by ``watch/probe/.../Sources.kt``), the ``<kind>.tsv`` summary and
``meta.txt``. ``analyze_run(run_dir)`` reads it, reports every estimate it can compute for each
metric plus the chosen one, writes ``analysis.json`` into the run directory and returns it.
Older runs without ``capabilities.json`` get capabilities inferred from the CSV files present.

Tiers: ``"25hz"`` = plain Android SensorManager (hr_raw 25 Hz PPG, hr, acc, gyro; no user setup);
``"all"`` = Samsung Health Sensor SDK allowed (Health Platform developer mode or partner approval).

Every estimate is ``{value, source, feature, tier, reliability, n, notes, details}``; reliability is
``"reference" | "good" | "experimental" | "unreliable"``. Metrics:

* ``hr`` (bpm): Samsung's algorithm (sdk_hr, hr; 1 Hz), ECG R-peaks, and PPG beats + spectrum
  (sdk_ppg100 green, sdk_ppg green, hr_raw AC channel v5).
* ``hrv`` (``{rmssd_ms, sdnn_ms, pnn50, mean_nn_ms, n_beats, window_s}``) from, in priority order,
  ECG R-peaks (reference with >= 20 s lead-on), 25 Hz SDK green PPG, Samsung IBI lists, 100 Hz
  on-demand PPG (good only when the quality gate passes) and 25 Hz hr_raw. 25 Hz PPG uses
  foot_template (``hrv/cand_foot_template.py``, see docs/hrv-25hz-benchmark.md) in ~60 s windows
  that keep at least half their expected intervals; sdk_ppg is good with >= 54 s of such windows,
  hr_raw experimental at best until it is checked against the watch ECG.
* ``motion``, ``on_wrist``, ``skin_temp``, ``spo2`` summaries.
* ``ppg_age`` (only in runs with sdk_ppg100, and only when the AI-PPG Age weights are already on this
  machine): an exploratory PPG age from the averaged motion-free beat, experimental at best, never an
  absolute age; its value carries the error band and the warning (pulse_exploration/ppg_age.py).

The chosen estimate is the first one in the metric's priority order that is not unreliable (the
first one overall when all are). Successive differences (RMSSD, pNN50) only use intervals that
are adjacent in time, never pairs across a rejected beat or a gap.

PPG beats for HR (25 and 100 Hz) and 100 Hz HRV: uniform grid, 0.5-3.5 Hz band-pass for detection and 0.5-8 Hz for
timing, polyphase upsampling to ~200 Hz, coarse peaks, then template matching (normalised
cross-correlation with the median beat, parabolic sub-sample refinement, two passes), NeuroKit2
``signal_fixpeaks`` (Kubios) for missed / extra / ectopic beats. Beats it moved or inserted are not
used. The quality gate uses the accelerometer (1 s windows whose per-axis RMS deviation exceeds
``MOTION_RMS`` are motion, padded by ``MOTION_PAD_S``) and each beat's template correlation.

Research prototype, not a medical device.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal as sps

from pulse_exploration.hrv.cand_foot_template import FS as FT_FS
from pulse_exploration.hrv.cand_foot_template import FootTemplateEstimator
from pulse_exploration.watch_raw import OFFSET, read_csv

# 0.2: suspected dropped Samsung IBIs, no "good" without motion check
# 0.3: 25 Hz PPG HRV by foot_template; Samsung IBIs cleaned of lost-sample intervals; sdk_ppg before sdk_hr
# 0.4: optional exploratory ppg_age metric (AI-PPG Age) for runs with sdk_ppg100
ANALYSIS_VERSION = "watch-session-0.4"
ANALYSIS_FILE = "analysis.json"
CAPABILITIES_FILE = "capabilities.json"
CAPABILITIES_SCHEMA = 1
RELIABILITY = ("reference", "good", "experimental", "unreliable")

ANDROID = ("hr_raw", "hr", "acc", "gyro")
FEATURES = ("ppg", "hr", "ibi", "motion", "skin_temp", "ppg100", "ecg", "spo2")
RESOLUTION: dict[str, dict[str, list[str]]] = {
    "all": {"ppg": ["sdk_ppg", "hr_raw"], "hr": ["sdk_hr", "hr"], "ibi": ["sdk_hr"], "motion": ["acc", "gyro"],
            "skin_temp": ["sdk_skin"], "ppg100": ["sdk_ppg100", "acc"], "ecg": ["sdk_ecg"], "spo2": ["sdk_spo2"]},
    "25hz": {"ppg": ["hr_raw"], "hr": ["hr"], "ibi": [], "motion": ["acc", "gyro"],
             "skin_temp": [], "ppg100": [], "ecg": [], "spo2": []},
}
COMPANIONS = {"ppg100": {"acc"}}  # recorded with the feature, but not the feature's signal
SDK_TYPES = {"sdk_ppg": "PPG_CONTINUOUS", "sdk_hr": "HEART_RATE_CONTINUOUS", "sdk_acc": "ACCELEROMETER_CONTINUOUS",
             "sdk_skin": "SKIN_TEMPERATURE_CONTINUOUS", "sdk_eda": "EDA_CONTINUOUS", "sdk_ppg100": "PPG_ON_DEMAND",
             "sdk_ecg": "ECG_ON_DEMAND", "sdk_spo2": "SPO2_ON_DEMAND"}
POLICY_REASON = "needs Samsung Health Platform developer mode"

HR_PRIORITY = ("sdk_hr", "hr", "sdk_ecg", "sdk_ppg100", "sdk_ppg", "hr_raw")
HRV_PRIORITY = ("sdk_ecg", "sdk_ppg", "sdk_hr", "sdk_ppg100", "hr_raw")  # sdk_ppg beat cleaned IBIs on GalaxyPPG
PPG_SOURCES = ("sdk_ppg100", "sdk_ppg", "hr_raw")

MIN_RUN_S = 20.0  # shorter runs get a "too_short" warning
SHORT_HRV_S = 60.0  # HRV windows below this are ultra-short-term
NN_RANGE_MS = (300.0, 2000.0)
LOCAL_DEV = 0.25  # an interval further than this share from its local median is implausible
MIN_DIFFS = 5  # fewest successive differences for an HRV value at all
# accelerometer motion gate
MOTION_WINDOW_S = 1.0
# m/s^2, RMS per-axis deviation in a 1 s window; a still wrist reads ~0.06 (SM-R965F). On GalaxyPPG
# P02-P06 it keeps 92 % of seated-rest and 96 % of stress-task seconds and no walking or running
MOTION_RMS = 0.5
MOTION_PAD_S = 1.0
# PPG beat quality gate
FS_UP = 200.0
BEAT_CORR = 0.8  # beats matching the template worse than this are not used
GATE_CORR = 0.9  # median template correlation of the beats used
GATE_MIN_S = 20.0  # seconds of clean, motion-free intervals
GATE_MIN_NN = 20
GATE_MIN_SEGMENT_S = 10.0  # longest contiguous clean stretch
GATE_MAX_ARTIFACT = 0.2  # share of motion-free beats that fixpeaks had to move, insert or remove
# HRV from 25 Hz PPG (sdk_ppg, hr_raw): foot_template from the GalaxyPPG benchmark
# (docs/hrv-25hz-benchmark.md: RMSSD median error ~3 ms on 60 s windows at rest), with its own beat
# shape, accelerometer and artifact checks instead of the gate above
HRV25_SOURCES = ("sdk_ppg", "hr_raw")
HRV25_WINDOW_S = 60.0  # the window length the benchmark validated
HRV25_MIN_KEPT = 0.5  # a window counts when this share of its expected intervals survive ...
HRV25_MIN_IBI = 10  # ... and at least this many
HRV25_BEST = {"sdk_ppg": "good", "hr_raw": "experimental"}  # hr_raw: not yet checked against the watch ECG
# m/s^2, per-axis accelerometer deviation over 2 s: catches wrist turns the magnitude gate misses.
# Chosen on GalaxyPPG P02-P06 (95 % of seated-rest seconds stay below it); held out it costs 2 points
# of coverage at the same error (3.2 vs 3.1 ms)
HRV25_AXIS_MAX = 1.0
HRV25_HIGH_RMSSD_MS = 50.0  # above this the lost-sample correction also trims genuine short intervals
DISAGREE_RMSSD_MS = 10.0  # two good HRV sources further apart than this get a warning
# ECG
ECG_FS = 500.0
ECG_REFERENCE_S = 20.0
ECG_MIN_S = 10.0
# Samsung IBI: rows arrive at 1 Hz; a longer gap between rows means a row (and its beats) is missing
IBI_ROW_GAP_S = 1.5
IBI_DROP_SHARE = 0.5
# cleaning validated on GalaxyPPG (bench.SamsungIBIClean): outliers against the median of 11 ...
IBI_LOCAL_DEV = 0.2
IBI_LOCAL_HALF = 5
IBI_SLIP_MS = 30.0  # ... and isolated short intervals: the PPG stream silently loses a sample
# Samsung lists a beat ~6 s after it happened: 5.2-7.0 s against sdk_ppg beats on SM-R965F (one 30 s
# run), 6.3-8.5 s against ECG on GalaxyPPG (Watch5); the first rows of a run list beats from before it
IBI_LAG_S = 6.0


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """numpy -> Python, NaN / inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _r(x: Any, nd: int = 2) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _warn(warns: list, code: str, message: str, severity: str = "warning") -> None:
    if not any(w["code"] == code and w["message"] == message for w in warns):
        warns.append({"code": code, "severity": severity, "message": message})


def _tier_of(alias: str) -> str:
    return "all" if alias.startswith("sdk_") else "25hz"


def _col(df: pl.DataFrame, name: str) -> np.ndarray:
    """A column as float64 (missing column or nulls -> NaN)."""
    if name not in df.columns:
        return np.full(df.height, np.nan)
    return df[name].cast(pl.Float64, strict=False).fill_null(float("nan")).to_numpy().astype(np.float64)


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray:
    hi = min(hi, 0.45 * fs)
    sos = sps.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _parabolic(y: np.ndarray, i: int) -> float:
    """Sub-sample offset of the extremum at index i."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    d = a - 2 * b + c
    return 0.0 if d == 0 or not np.isfinite(d) else float(np.clip(0.5 * (a - c) / d, -0.5, 0.5))


def _spectral_peak(x: np.ndarray, fs: float, band=(0.6, 3.5)) -> float | None:
    """Dominant frequency (Hz) in the heart-rate band, Welch with 0.005 Hz zero-padded bins."""
    if len(x) < 4 * fs or not np.all(np.isfinite(x)) or np.std(x) == 0:
        return None
    nper = int(min(len(x), 20 * fs))
    nfft = int(2 ** math.ceil(math.log2(max(nper, fs / 0.005))))
    f, p = sps.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, nfft=nfft, detrend="constant")
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or p[m].sum() <= 0:
        return None
    fm, pm = f[m], p[m]
    i = int(np.argmax(pm))
    return float(fm[i] + _parabolic(pm, i) * (f[1] - f[0]))


def _local_median(x: np.ndarray, half: int = 3) -> np.ndarray:
    return np.array([np.median(x[max(0, i - half):i + half + 1]) for i in range(len(x))]) if len(x) else x


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if not len(mask):
        return []
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _merge_spans(a: np.ndarray, b: np.ndarray, tol: float = 1e-6) -> list[list[float]]:
    spans: list[list[float]] = []
    for s, e in zip(a, b):
        if spans and s <= spans[-1][1] + tol:
            spans[-1][1] = max(spans[-1][1], float(e))
        else:
            spans.append([float(s), float(e)])
    return [[round(s, 2), round(e, 2)] for s, e in spans]


def _estimate(value, source: str, feature: str, reliability: str, n: int, notes: list[str],
              details: dict | None = None) -> dict:
    assert reliability in RELIABILITY
    return {"value": value, "source": source, "feature": feature, "tier": _tier_of(source),
            "reliability": reliability, "n": int(n), "notes": notes, "details": details or {}}


def _hrv_values(segments: list[np.ndarray]) -> dict | None:
    """Time-domain HRV from contiguous NN segments (ms); differences never cross segments."""
    segments = [s for s in segments if len(s)]
    if not segments:
        return None
    nn = np.concatenate(segments)
    diffs = np.concatenate([np.diff(s) for s in segments])
    if len(diffs) < MIN_DIFFS or len(nn) < 2:
        return None
    return {"rmssd_ms": _r(np.sqrt(np.mean(diffs ** 2))), "sdnn_ms": _r(np.std(nn, ddof=1)),
            "pnn50": _r(100 * np.mean(np.abs(diffs) > 50)), "mean_nn_ms": _r(np.mean(nn)),
            "n_beats": int(len(nn) + len(segments)), "window_s": _r(nn.sum() / 1000, 1)}


def _segments(nn: np.ndarray, ok: np.ndarray) -> list[np.ndarray]:
    """Contiguous runs of accepted, adjacent intervals."""
    return [nn[s:e] for s, e in _runs(ok)]


# --------------------------------------------------------------------------------------------
# reading a run
# --------------------------------------------------------------------------------------------

def _read_kv(path: Path) -> dict[str, str]:
    out = {}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    if not lines:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t") + [""] * (len(head) - len(ln.split("\t"))))) for ln in lines[1:]]


def read_summary(run: Path) -> list[dict[str, str]]:
    """Rows of ``<kind>.tsv`` (tier, alias, available, registered, events, rate_hz, ...)."""
    kind = run.name.rsplit("-", 1)[-1]
    candidates = [run / f"{kind}.tsv"] + sorted(p for p in run.glob("*.tsv") if p.name != f"{kind}.tsv")
    for p in candidates:
        rows = _read_tsv(p)
        if rows and {"tier", "alias"} <= set(rows[0]):
            return rows
    return []


def load_tables(run: Path, warns: list | None = None) -> dict[str, pl.DataFrame]:
    """Every non-empty source CSV, keyed by alias."""
    out = {}
    for p in sorted(run.glob("*.csv")):
        try:
            df = read_csv(p)
        except Exception as exc:  # malformed upload
            if warns is not None:
                _warn(warns, "unreadable_csv", f"{p.name}: {type(exc).__name__}: {exc}")
            continue
        if df is not None:
            out[p.stem] = df
    return out


# --------------------------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------------------------

def _unavailable_reason(feature: str, tier: str, sdk: dict, errors: dict[str, str]) -> str:
    def why(alias: str) -> str:
        return POLICY_REASON if "SDK_POLICY_ERROR" in errors[alias] else f"{alias}: {errors[alias]}"

    own = [a for a in RESOLUTION[tier][feature] if a not in COMPANIONS.get(feature, ())]
    sdk_alias = next((a for a in RESOLUTION["all"][feature] if a.startswith("sdk_")), None)
    if own:  # the tier offers this feature, it just did not deliver
        for alias in own:
            if errors.get(alias):
                return why(alias)
        if sdk_alias in own and sdk.get("supported") and SDK_TYPES[sdk_alias] not in sdk["supported"]:
            return "not supported on this watch"
        return "not recorded in this run"
    # SDK-only feature in tier 25hz
    reasons = [POLICY_REASON] if sdk.get("policy") == "denied" else []
    if sdk_alias and errors.get(sdk_alias) and why(sdk_alias) not in reasons:
        reasons.append(why(sdk_alias))
    if sdk_alias and sdk.get("supported") and SDK_TYPES[sdk_alias] not in sdk["supported"]:
        reasons.append("not supported on this watch")
    return "; ".join(reasons) or "needs the Samsung Health Sensor SDK (tier all)"


def resolve_features(tier: str, sdk: dict, recorded: set[str], errors: dict[str, str] | None = None) -> dict:
    """Contract feature resolution for the aliases actually recorded."""
    errors = errors or {}
    out = {}
    for f in FEATURES:
        cands = RESOLUTION[tier][f]
        have = [a for a in cands if a in recorded]
        if any(a not in COMPANIONS.get(f, ()) for a in have):
            out[f] = {"available": True, "tier": _tier_of(have[0]), "sources": have, "reason": ""}
        else:
            out[f] = {"available": False, "tier": None, "sources": [],
                      "reason": _unavailable_reason(f, tier, sdk, errors)}
    return out


def infer_capabilities(run: Path, tables: dict[str, pl.DataFrame], summary: list[dict]) -> dict:
    """capabilities.json for older runs, from the CSVs present, the summary and sdk.tsv."""
    recorded = set(tables)
    errors = {r["alias"]: r.get("error", "") for r in summary if r.get("error")}
    sdk_rows = _read_tsv(run / "sdk.tsv")
    sdk_tsv = sdk_rows[0] if sdk_rows else {}
    has_sdk_data = any(a.startswith("sdk_") for a in recorded)
    tier = "all" if has_sdk_data else "25hz"
    if has_sdk_data:
        policy = "allowed"
    elif any("SDK_POLICY_ERROR" in e for e in errors.values()):
        policy = "denied"
    else:
        policy = "unknown"
    sdk = {
        "installed": bool(sdk_tsv) or has_sdk_data,
        "connected": sdk_tsv.get("connected", "").lower() == "true" or has_sdk_data,
        "policy": policy,
        "error": sdk_tsv.get("error", "") or next((e for a, e in errors.items() if a.startswith("sdk_")), ""),
        "version": sdk_tsv.get("version", ""),
        "supported": [s for s in sdk_tsv.get("supported", "").split(",") if s],
    }
    listed = {r["alias"]: r.get("available", "").lower() == "true" for r in summary}
    android = {a: a in recorded or listed.get(a, False) for a in ANDROID}
    return {"schema": CAPABILITIES_SCHEMA, "tier": tier, "sdk": sdk, "android": android,
            "features": resolve_features(tier, sdk, recorded, errors)}


# --------------------------------------------------------------------------------------------
# clock and motion
# --------------------------------------------------------------------------------------------

class _Clock:
    """Seconds from the run start. Android rows carry event_ns (boot clock, mapped to the wall clock
    with the 10th percentile of wall_ms - event_ns); SDK rows carry timestamp_ms (wall clock)."""

    def __init__(self, meta: dict[str, str], tables: dict[str, pl.DataFrame]):
        starts = []
        for df in tables.values():
            col = "timestamp_ms" if "timestamp_ms" in df.columns else "wall_ms"
            v = _col(df, col)
            if np.isfinite(v).any():
                starts.append(float(np.nanmin(v)))
        try:
            self.t0_ms = float(meta["start_wall_ms"])
        except (KeyError, ValueError):
            self.t0_ms = min(starts) if starts else 0.0

    def times(self, df: pl.DataFrame) -> np.ndarray:
        if "event_ns" in df.columns:
            ms = _col(df, "event_ns") / 1e6
            lag = _col(df, "wall_ms") - ms
            offset = float(np.nanpercentile(lag, 10)) if np.isfinite(lag).any() else -ms[0]
            return (ms + offset - self.t0_ms) / 1000
        return (_spread_duplicates(_col(df, "timestamp_ms")) - self.t0_ms) / 1000


def _spread_duplicates(ts: np.ndarray) -> np.ndarray:
    """Per-sample times when the SDK stamps several samples with the same millisecond."""
    if len(ts) < 3 or not np.all(np.isfinite(ts)) or np.all(np.diff(ts) > 0):
        return ts
    idx = np.arange(len(ts))
    first = np.r_[True, np.diff(ts) != 0]
    ai, at = idx[first], ts[first]
    if len(ai) < 2:
        return ts
    step = float(np.median(np.diff(at) / np.diff(ai)))
    out = np.interp(idx, ai, at)
    tail = idx > ai[-1]
    out[tail] = at[-1] + (idx[tail] - ai[-1]) * step
    return out


@dataclass
class _Motion:
    source: str | None
    t0: float = 0.0
    rms: np.ndarray | None = None  # per 1 s window, m/s^2
    moving: np.ndarray | None = None  # before padding
    still: np.ndarray | None = None  # after padding; windows with too few samples are not still
    rate_hz: float | None = None

    @property
    def checked(self) -> bool:
        return self.source is not None

    def still_between(self, a, b) -> np.ndarray:
        """True where [a, b] (run seconds) lies wholly inside still windows (always True without
        an accelerometer: callers check ``checked``)."""
        a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
        if not self.checked:
            return np.ones(len(a), bool)
        n = len(self.still)
        i0 = np.floor((a - self.t0) / MOTION_WINDOW_S).astype(int)
        i1 = np.floor((b - self.t0) / MOTION_WINDOW_S).astype(int)
        inside = (i0 >= 0) & (i1 < n) & (i1 >= i0)
        cum = np.r_[0, np.cumsum(~self.still)]
        return inside & (cum[np.clip(i1, 0, n - 1) + 1] - cum[np.clip(i0, 0, n - 1)] == 0)


def _accelerometer(tables: dict[str, pl.DataFrame], clock: _Clock) -> tuple[str, np.ndarray, np.ndarray] | None:
    """(source, run seconds, m/s^2 n x 3), finite rows only; Android acc preferred."""
    if "acc" in tables:
        src, df = "acc", tables["acc"]
        a = np.column_stack([_col(df, c) for c in ("v0", "v1", "v2")])
    elif "sdk_acc" in tables:  # raw counts: scale so the median magnitude is 1 g
        src, df = "sdk_acc", tables["sdk_acc"]
        a = np.column_stack([_col(df, c) for c in ("x", "y", "z")])
        mag = np.nanmedian(np.linalg.norm(a, axis=1))
        a = a * (9.80665 / mag) if mag > 0 else a
    else:
        return None
    t = clock.times(df)
    ok = np.isfinite(t) & np.all(np.isfinite(a), axis=1)
    return src, t[ok], a[ok]


def _motion(tables: dict[str, pl.DataFrame], clock: _Clock) -> _Motion:
    acc = _accelerometer(tables, clock)
    if acc is None:
        return _Motion(None)
    src, t, a = acc
    if len(t) < 20 or t.max() - t.min() < 2 * MOTION_WINDOW_S:
        return _Motion(None)
    order = np.argsort(t, kind="stable")
    t, a = t[order], a[order]
    t0 = float(t[0])
    w = np.floor((t - t0) / MOTION_WINDOW_S).astype(int)
    n = int(w.max()) + 1
    cnt = np.bincount(w, minlength=n).astype(float)
    var = np.zeros(n)
    for k in range(3):
        s1 = np.bincount(w, a[:, k], minlength=n)
        s2 = np.bincount(w, a[:, k] ** 2, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            var += np.where(cnt > 0, s2 / np.maximum(cnt, 1) - (s1 / np.maximum(cnt, 1)) ** 2, 0)
    rms = np.sqrt(np.clip(var, 0, None))
    dur = t[-1] - t0
    rate = (len(t) - 1) / dur if dur > 0 else None
    enough = cnt >= max(3, 0.3 * (rate or 0) * MOTION_WINDOW_S)
    moving = (rms > MOTION_RMS) & enough
    pad = int(math.ceil(MOTION_PAD_S / MOTION_WINDOW_S))
    padded = np.convolve(moving.astype(int), np.ones(2 * pad + 1, int), mode="same") > 0
    return _Motion(src, t0, rms, moving, ~padded & enough, rate)


def _motion_estimate(m: _Motion, tables: dict[str, pl.DataFrame]) -> dict:
    segs = [[round(m.t0 + s * MOTION_WINDOW_S, 1), round(m.t0 + e * MOTION_WINDOW_S, 1)] for s, e in _runs(m.moving)]
    value = {"still_share": _r(np.mean(m.still), 3), "motion_s": _r(m.moving.sum() * MOTION_WINDOW_S, 1),
             "duration_s": _r(len(m.rms) * MOTION_WINDOW_S, 1), "rms_median_ms2": _r(np.median(m.rms), 3),
             "rms_max_ms2": _r(np.max(m.rms), 3), "motion_segments_s": segs}
    details = {"rate_hz": _r(m.rate_hz), "window_s": MOTION_WINDOW_S, "threshold_ms2": MOTION_RMS,
               "pad_s": MOTION_PAD_S}
    if "gyro" in tables:
        g = np.column_stack([_col(tables["gyro"], c) for c in ("v0", "v1", "v2")])
        g = g[np.all(np.isfinite(g), axis=1)]
        if len(g):
            details["gyro_rms_dps"] = _r(np.degrees(np.sqrt(np.mean(np.sum(g ** 2, axis=1)))), 2)
    notes = ["1 s windows with per-axis RMS deviation above the threshold are motion"]
    if m.source == "sdk_acc":
        notes.append("Samsung raw accelerometer counts, scaled so the median magnitude is 1 g")
    return _estimate(value, m.source, "motion", "good", int(len(m.rms)), notes, details)


# --------------------------------------------------------------------------------------------
# PPG beats
# --------------------------------------------------------------------------------------------

@dataclass
class _Beats:
    source: str
    fs: float  # measured native rate
    t: np.ndarray  # beat times after artifact correction (run s)
    corr: np.ndarray  # template correlation, NaN for beats fixpeaks moved or inserted
    det: np.ndarray  # detection signal on the native grid
    grid: np.ndarray
    grid_valid: np.ndarray
    n_detected: int
    original_t: np.ndarray  # beats before artifact correction
    original_kept: np.ndarray  # whether each original beat survived unchanged

    def valid_between(self, a, b) -> np.ndarray:
        dt = self.grid[1] - self.grid[0]
        n = len(self.grid)
        i0 = np.floor((np.asarray(a) - self.grid[0]) / dt).astype(int)
        i1 = np.ceil((np.asarray(b) - self.grid[0]) / dt).astype(int)
        inside = (i0 >= 0) & (i1 < n)
        cum = np.r_[0, np.cumsum(~self.grid_valid)]
        return inside & (cum[np.clip(i1, 0, n - 1) + 1] - cum[np.clip(i0, 0, n - 1)] == 0)


def _ppg_input(alias: str, df: pl.DataFrame, clock: _Clock) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(t, upright pulse signal, sample valid) for one PPG source."""
    t = clock.times(df)
    notes = []
    if alias == "hr_raw":
        valid = _col(df, "v0") == 2
        light = -_col(df, "v1")  # raw light level drops as blood volume rises
        ac = _col(df, "v5") - OFFSET
        sign = -1.0  # SM-R965F: v5 follows the raw light level
        m = valid & np.isfinite(light) & np.isfinite(ac)
        if m.sum() > 250 and np.std(light[m]) > 0 and np.std(ac[m]) > 0:
            dt = float(np.median(np.diff(t[m])))
            try:
                c = np.corrcoef(_bandpass(light[m] - light[m].mean(), 1 / dt, 0.5, 3.5),
                                _bandpass(ac[m] - ac[m].mean(), 1 / dt, 0.5, 3.5))[0, 1]
                sign = 1.0 if c >= 0 else -1.0
            except ValueError:
                pass
        notes.append(f"hr_raw AC channel v5 - 2^21, sign {'+' if sign > 0 else '-'} (matched to inverted v1); "
                     "only on-wrist samples (v0 == 2)")
        return t, sign * ac, valid, notes
    green = _col(df, "green")
    status = _col(df, "green_status")
    valid = np.isfinite(green) & ((status == 0) | ~np.isfinite(status))
    notes.append(f"{alias} green channel, inverted; samples with green_status != 0 dropped; "
                 "sample times from timestamp_ms")
    return t, -green, valid, notes


def _coarse_peaks(x: np.ndarray, fs: float, period: float) -> np.ndarray:
    med = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med)) or np.std(x)
    distance = min(0.45 * period, 0.4)  # never hides beats below 150 bpm, whatever the prior
    idx, _ = sps.find_peaks(x, distance=max(1, int(distance * fs)), prominence=0.4 * scale)
    return idx


def _still_f0(det: np.ndarray, grid: np.ndarray, grid_valid: np.ndarray, fs: float, motion: _Motion) -> float | None:
    """Spectral HR (Hz) of the longest motion-free, on-wrist stretch of >= 10 s (else of everything)."""
    still = motion.still_between(grid, grid) & grid_valid
    runs = [r for r in _runs(still) if (r[1] - r[0]) / fs >= 10]
    if runs:
        s, e = max(runs, key=lambda r: r[1] - r[0])
        return _spectral_peak(det[s:e], fs)
    return _spectral_peak(det, fs)


def _template_refine(x: np.ndarray, idx: np.ndarray, fs: float, period: float,
                     passes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Beat positions (fractional samples) aligned to the median beat by normalised
    cross-correlation with parabolic refinement, and each beat's correlation."""
    pre, post = int(round(0.3 * period * fs)), int(round(0.5 * period * fs))
    length, maxlag = pre + post, max(1, int(round(0.12 * period * fs)))
    pos = idx.astype(float)
    corr = np.full(len(idx), np.nan)
    for _ in range(passes):
        segs = [x[c - pre:c + post] for c in np.round(pos).astype(int) if c - pre >= 0 and c + post <= len(x)]
        if len(segs) < 3:
            break
        tmpl = np.median(np.asarray(segs), axis=0)
        tmpl = tmpl - tmpl.mean()
        norm = np.linalg.norm(tmpl)
        if norm == 0:
            break
        tmpl /= norm
        new = pos.copy()
        for j, c in enumerate(np.round(pos).astype(int)):
            s, e = c - pre - maxlag, c + post + maxlag
            if s < 0 or e > len(x):
                corr[j] = np.nan
                continue
            win = sliding_window_view(x[s:e], length)
            win = win - win.mean(axis=1, keepdims=True)
            nrm = np.linalg.norm(win, axis=1)
            cc = np.where(nrm > 0, win @ tmpl / np.where(nrm > 0, nrm, 1), -1.0)
            k = int(np.argmax(cc))
            new[j] = c - maxlag + k + _parabolic(cc, k)
            corr[j] = cc[k]
        pos = new
    return pos, corr


def _fixpeaks(times: np.ndarray) -> np.ndarray:
    """NeuroKit2 Kubios artifact correction (missed / extra / ectopic / long-short beats)."""
    if len(times) < 10:
        return times
    import neurokit2 as nk

    ms = np.round((times - times[0]) * 1000).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, fixed = nk.signal_fixpeaks(ms, sampling_rate=1000, iterative=True, method="Kubios")
    return times[0] + np.asarray(fixed, float) / 1000


def _ppg_beats(alias: str, t: np.ndarray, x: np.ndarray, valid: np.ndarray, motion: _Motion) -> _Beats | None:
    ok = np.isfinite(t) & np.isfinite(x)
    t, x, valid = t[ok], x[ok], valid[ok]
    order = np.argsort(t, kind="stable")
    t, x, valid = t[order], x[order], valid[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, x, valid = t[keep], x[keep], valid[keep]
    if valid.sum() < 10:
        return None
    first, last = np.flatnonzero(valid)[[0, -1]]
    t, x, valid = t[first:last + 1], x[first:last + 1], valid[first:last + 1]
    if t[-1] - t[0] < 10:
        return None
    dt = float(np.median(np.diff(t)))
    fs = 1 / dt
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1) * dt
    xg = np.interp(grid, t[valid], x[valid])
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.where(grid - t[j - 1] <= t[j] - grid, j - 1, j)
    gv = valid[near] & (np.abs(t[near] - grid) <= 1.5 * dt)
    xg = xg - xg.mean()
    det = _bandpass(xg, fs, 0.5, 3.5)
    tim = _bandpass(xg, fs, 0.5, 8.0)
    up = max(1, int(round(FS_UP / fs)))
    fsu = fs * up
    det_u, tim_u = sps.resample_poly(det, up, 1), sps.resample_poly(tim, up, 1)
    f0 = _still_f0(det, grid, gv, fs, motion)
    period = 1 / f0 if f0 else 0.85
    coarse = _coarse_peaks(det_u, fsu, period)
    if len(coarse) < 6:
        return None
    period = float(np.clip(np.median(np.diff(coarse)) / fsu, 0.3, 2.0))
    pos, corr = _template_refine(tim_u, coarse, fsu, period)
    times = grid[0] + pos / fsu
    # two detections that converged onto one beat: keep the better match
    keep = np.ones(len(times), bool)
    for i in range(1, len(times)):
        if times[i] - times[i - 1] < 0.3:
            worse = i if np.nan_to_num(corr[i], nan=-2) < np.nan_to_num(corr[i - 1], nan=-2) else i - 1
            keep[worse] = False
    times, corr = times[keep], corr[keep]
    fixed = _fixpeaks(times)
    # map corrected beats back to the measured ones; moved or inserted beats get no correlation
    k = np.clip(np.searchsorted(times, fixed), 1, len(times) - 1) if len(times) > 1 else np.zeros(len(fixed), int)
    nearest = np.where(np.abs(fixed - times[k - 1]) <= np.abs(fixed - times[k]), k - 1, k)
    matched = np.abs(fixed - times[nearest]) <= 0.02
    out_t = np.where(matched, times[nearest], fixed)
    out_c = np.where(matched, corr[nearest], np.nan)
    kept = np.zeros(len(times), bool)
    kept[nearest[matched]] = True
    return _Beats(alias, fs, out_t, out_c, det, grid, gv, len(times), times, kept)


def _ppg_intervals(b: _Beats, motion: _Motion) -> dict:
    """Interval bookkeeping: quality-accepted, motion-free accepted, and the gate."""
    a, e = b.t[:-1], b.t[1:]
    nn = (e - a) * 1000
    good_beat = np.isfinite(b.corr) & (b.corr >= BEAT_CORR)
    quality = good_beat[:-1] & good_beat[1:] & (nn >= NN_RANGE_MS[0]) & (nn <= NN_RANGE_MS[1])
    quality &= b.valid_between(a, e)
    if quality.any():
        med = _local_median(nn[quality])
        dev = np.zeros(len(nn), bool)
        dev[np.flatnonzero(quality)] = np.abs(nn[quality] - med) > LOCAL_DEV * med
        quality &= ~dev
    still = motion.still_between(a, e)
    ok = quality & still
    segs = _segments(nn, ok)
    clean_s = float(nn[ok].sum() / 1000)
    longest = max((float(s.sum() / 1000) for s in segs), default=0.0)
    used = np.zeros(len(b.t), bool)
    used[:-1] |= ok
    used[1:] |= ok
    med_corr = float(np.median(b.corr[used])) if used.any() else None
    # artifact share among beats in motion-free time
    still_orig = motion.still_between(b.original_t, b.original_t)
    still_fixed = motion.still_between(b.t, b.t)
    n_still = int(still_orig.sum())
    n_art = int(max((~b.original_kept & still_orig).sum(), (~np.isfinite(b.corr) & still_fixed).sum()))
    art_share = n_art / n_still if n_still else None
    total_s = float(b.grid[-1] - b.grid[0])
    reasons = []
    if not motion.checked:
        reasons.append("no accelerometer: motion not checked")
    if clean_s < GATE_MIN_S:
        reasons.append(f"only {clean_s:.0f} s of clean motion-free intervals (needs {GATE_MIN_S:.0f})")
    if int(ok.sum()) < GATE_MIN_NN:
        reasons.append(f"only {int(ok.sum())} clean intervals (needs {GATE_MIN_NN})")
    if longest < GATE_MIN_SEGMENT_S:
        reasons.append(f"longest clean stretch {longest:.0f} s (needs {GATE_MIN_SEGMENT_S:.0f})")
    if med_corr is not None and med_corr < GATE_CORR:
        reasons.append(f"median beat template correlation {med_corr:.2f} (needs {GATE_CORR})")
    if art_share is not None and art_share > GATE_MAX_ARTIFACT:
        reasons.append(f"{art_share:.0%} of motion-free beats needed artifact correction")
    gate = {"passed": not reasons, "reasons": reasons, "motion_checked": motion.checked,
            "clean_s": _r(clean_s, 1), "longest_clean_s": _r(longest, 1), "median_corr": _r(med_corr, 3),
            "artifact_share": _r(art_share, 3), "beats_detected": b.n_detected, "beats": int(len(b.t)),
            "intervals": int(len(nn)), "rejected_quality": int((~quality).sum()),
            "rejected_motion": int((quality & ~still).sum()), "recording_s": _r(total_s, 1)}
    return {"a": a, "e": e, "nn": nn, "ok": ok, "quality": quality, "segments": segs, "gate": gate}


def _ppg_hr(b: _Beats, iv: dict, motion: _Motion) -> dict | None:
    nn, ok = iv["nn"], iv["ok"]
    f0 = _still_f0(b.det, b.grid, b.grid_valid, b.fs, motion)
    spec_bpm = f0 * 60 if f0 else None
    notes = [f"PPG beats at {b.fs:.1f} Hz (template-matched) and Welch spectrum"]
    if not motion.checked:
        notes.append("no accelerometer: motion not checked")
    if ok.sum() >= 8:
        peaks_bpm, n, basis = 60000 / np.median(nn[ok]), int(ok.sum()), "motion-free clean intervals"
    elif iv["quality"].sum() >= 5:
        peaks_bpm, n, basis = 60000 / np.median(nn[iv["quality"]]), int(iv["quality"].sum()), "all clean intervals"
        notes.append("too few motion-free beats: peaks HR uses intervals during motion")
    else:
        peaks_bpm, n, basis = None, 0, None
    value = peaks_bpm if peaks_bpm is not None else spec_bpm
    if value is None:
        return None
    agree = peaks_bpm is not None and spec_bpm is not None and abs(peaks_bpm - spec_bpm) <= 5
    med_corr = iv["gate"]["median_corr"]
    if basis == "motion-free clean intervals" and agree and (med_corr or 0) >= BEAT_CORR and motion.checked:
        rel = "good"
    elif peaks_bpm is not None and (agree or basis == "motion-free clean intervals"):
        rel = "experimental"
    else:
        rel = "unreliable"
    if not agree:
        notes.append("peak and spectrum HR disagree by more than 5 bpm" if peaks_bpm and spec_bpm
                     else "only one of peaks / spectrum available")
    feature = "ppg100" if b.source == "sdk_ppg100" else "ppg"
    return _estimate(_r(value, 1), b.source, feature, rel, n, notes,
                     {"peaks_bpm": _r(peaks_bpm, 1), "spectrum_bpm": _r(spec_bpm, 1), "peaks_basis": basis,
                      "fs_hz": _r(b.fs), "median_corr": med_corr})


def _ppg_hrv(b: _Beats, iv: dict) -> tuple[dict | None, str | None]:
    gate = iv["gate"]
    vals = _hrv_values(iv["segments"])
    if vals is None:
        return None, "too few motion-free clean beats for HRV"
    hundred = b.source == "sdk_ppg100"
    rel = ("good" if hundred else "experimental") if gate["passed"] else "unreliable"
    notes = [(f"{'100' if hundred else '25'} Hz PPG pulse intervals (pulse-rate variability), template-matched "
              "sub-sample beat times, NeuroKit2 signal_fixpeaks (Kubios); only motion-free windows "
              "and beats matching the template")]
    if not hundred:
        notes.append("25 Hz PPG: experimental at best")
    if not gate["passed"]:
        notes.append("quality gate failed: " + "; ".join(gate["reasons"]))
    if (vals["window_s"] or 0) < SHORT_HRV_S:
        notes.append(f"ultra-short window ({vals['window_s']} s < {SHORT_HRV_S:.0f} s)")
    feature = "ppg100" if hundred else "ppg"
    spans = _merge_spans(iv["a"][iv["ok"]], iv["e"][iv["ok"]])
    return _estimate(vals, b.source, feature, rel, int(iv["ok"].sum()), notes,
                     {"gate": gate, "spans_s": spans, "fs_hz": _r(b.fs)}), None


def _acc_25hz(acc: tuple[str, np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    """Accelerometer as 40 ms bin means (ms, n x 3): foot_template's motion gate assumes 25 Hz."""
    if acc is None or not len(acc[1]):
        return np.zeros(0), np.zeros((0, 3))
    _, t, a = acc
    order = np.argsort(t, kind="stable")
    t, a = t[order], a[order]
    b = np.floor(t * FT_FS).astype(np.int64)
    k = b - b[0]
    cnt = np.bincount(k)
    has = cnt > 0
    xyz = np.column_stack([np.bincount(k, a[:, i])[has] / cnt[has] for i in range(3)])
    return (np.flatnonzero(has) + b[0] + 0.5) / FT_FS * 1000, xyz


@dataclass
class _Beats25:
    t: np.ndarray  # beat times, s from the run start
    good: np.ndarray  # beats foot_template vouches for
    t_first: float  # first and last PPG sample, s (on wrist or not)
    t_last: float
    fs_hz: float  # on-wrist sample rate
    motion_checked: bool
    lost_samples: int  # silently lost samples foot_template corrected for


def _hrv25_beats(t: np.ndarray, x: np.ndarray, valid: np.ndarray,
                 acc25: tuple[np.ndarray, np.ndarray]) -> tuple[_Beats25 | None, str | None]:
    """foot_template beats of a 25 Hz PPG stream (on-wrist samples only)."""
    m = np.isfinite(t) & np.isfinite(x)
    if not m.any():
        return None, "no 25 Hz PPG samples"
    t_first, t_last = float(t[m].min()), float(t[m].max())
    m &= valid
    t, x = t[m], x[m]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]
    if len(t) < 2 or t[-1] - t[0] < 10:
        return None, "under 10 s of on-wrist 25 Hz PPG"
    acc_t, acc_xyz = acc25
    est = FootTemplateEstimator(invert=False, axis_max=HRV25_AXIS_MAX)  # _ppg_input already turned the pulse upright
    out = est.estimate(t * 1000, x, acc_t, acc_xyz)
    good = np.asarray(out["quality"], float) >= 0.5
    if good.sum() < 6:
        return None, f"foot_template kept {int(good.sum())} beats"
    return _Beats25(t=np.asarray(out["beats_ms"], float) / 1000, good=good, t_first=t_first, t_last=t_last,
                    fs_hz=(len(t) - 1) / (t[-1] - t[0]),
                    motion_checked=len(acc_t) >= 4 * FT_FS,  # foot_template ignores shorter accelerometer data
                    lost_samples=int(est.debug.get("n_drops", 0))), None


def _ppg_hrv25(alias: str, t: np.ndarray, x: np.ndarray, valid: np.ndarray,
               acc25: tuple[np.ndarray, np.ndarray]) -> tuple[dict | None, str | None]:
    """HRV from 25 Hz PPG with foot_template. The run is cut into windows of about HRV25_WINDOW_S;
    a window counts when enough of its expected intervals survive the estimator's own beat-shape,
    motion and artifact checks, and HRV uses the intervals of the windows that count. Windows span
    the whole recording, so off-wrist time counts against them."""
    beats, why = _hrv25_beats(t, x, valid, acc25)
    if beats is None:
        return None, why
    bt, good, t_first, t_last, motion_checked = beats.t, beats.good, beats.t_first, beats.t_last, beats.motion_checked
    a, e = bt[:-1], bt[1:]
    nn = (e - a) * 1000
    ok = good[:-1] & good[1:]  # rejected beats, and the placeholder after each recording gap, break intervals
    med = float(np.median(nn[ok])) if ok.any() else math.nan
    t0, t1 = t_first, t_last
    n_win = max(1, round((t1 - t0) / HRV25_WINDOW_S))
    width = (t1 - t0) / n_win
    widx = np.clip(((a - t0) // width).astype(int), 0, n_win - 1)
    use = np.zeros(len(nn), bool)
    windows = []
    for w in range(n_win):
        inside = ok & (widx == w)
        expected = width * 1000 / med if math.isfinite(med) else 0.0
        n_ibi = int(inside.sum())
        accepted = (n_ibi >= max(HRV25_MIN_IBI, HRV25_MIN_KEPT * expected)
                    and int((inside[1:] & inside[:-1]).sum()) >= MIN_DIFFS)
        if accepted:
            use |= inside
        windows.append({"start_s": _r(t0 + w * width, 1), "end_s": _r(t0 + (w + 1) * width, 1),
                        "intervals": n_ibi, "expected": _r(expected, 1), "accepted": bool(accepted)})
    accepted_s = width * sum(w["accepted"] for w in windows)
    reasons = []
    if not motion_checked:
        reasons.append("no accelerometer: motion not checked")
    if not use.any():
        reasons.append(f"no {width:.0f} s window kept {HRV25_MIN_KEPT:.0%} of its expected intervals "
                       f"(and >= {HRV25_MIN_IBI})")
    passed = not reasons
    vals = _hrv_values(_segments(nn, use if use.any() else ok))
    if vals is None:
        return None, "too few clean beats for HRV"
    if not passed:
        rel = "unreliable"
    elif accepted_s < 0.9 * HRV25_WINDOW_S:
        rel = "experimental"  # shorter than the window the benchmark validated
    else:
        rel = HRV25_BEST[alias]
    notes = ["25 Hz PPG pulse intervals (pulse-rate variability) by foot_template: steepest upstroke on a "
             "1 kHz spline, aligned to the running median beat; beats with a poor shape, in motion "
             "(accelerometer) or flagged as artifacts are not used; silently lost samples are corrected",
             f"GalaxyPPG benchmark (Galaxy Watch5 SDK green PPG vs chest ECG): RMSSD median error ~3 ms on "
             f"{HRV25_WINDOW_S:.0f} s windows at rest (docs/hrv-25hz-benchmark.md)"]
    if HRV25_BEST[alias] != "good":
        notes.append(f"{alias} has not been checked against the watch ECG yet: {HRV25_BEST[alias]} at best")
    if passed and accepted_s < 0.9 * HRV25_WINDOW_S:
        notes.append(f"accepted windows cover {accepted_s:.0f} s, under the validated {HRV25_WINDOW_S:.0f} s")
    if not passed:
        notes.append("quality gate failed: " + "; ".join(reasons))
    if (vals["window_s"] or 0) < SHORT_HRV_S:
        notes.append(f"ultra-short window ({vals['window_s']} s < {SHORT_HRV_S:.0f} s)")
    if (vals["rmssd_ms"] or 0) > HRV25_HIGH_RMSSD_MS:
        notes.append(f"RMSSD above {HRV25_HIGH_RMSSD_MS:.0f} ms: the lost-sample correction may also trim genuine "
                     "short intervals, so the value may be a few ms low")
    used = use if use.any() else ok
    gate = {"passed": passed, "reasons": reasons, "motion_checked": motion_checked, "method": "foot_template",
            "windows": windows, "accepted_s": _r(accepted_s, 1), "beats_kept": int(good.sum()),
            "beats_rejected": int((~good).sum()), "intervals_used": int(used.sum()),
            "lost_samples_corrected": beats.lost_samples}
    return _estimate(vals, alias, "ppg", rel, int(used.sum()), notes,
                     {"gate": gate, "spans_s": _merge_spans(a[used], e[used]),
                      "fs_hz": _r(beats.fs_hz)}), None


# --------------------------------------------------------------------------------------------
# Samsung HR / IBI, ECG, skin temperature, SpO2, on-wrist
# --------------------------------------------------------------------------------------------

def _samsung_hr(df: pl.DataFrame) -> dict | None:
    hr, st = _col(df, "hr"), _col(df, "hr_status")
    ok = np.isfinite(hr) & (hr > 0) & (st == 1)
    if not ok.any():
        return None
    v = hr[ok]
    return _estimate(_r(np.median(v), 1), "sdk_hr", "hr", "good", int(ok.sum()),
                     ["Samsung HEART_RATE_CONTINUOUS (1 Hz), median of rows with hr_status == 1"],
                     {"min": _r(v.min(), 1), "max": _r(v.max(), 1), "rows": df.height})


def _android_hr(df: pl.DataFrame) -> dict | None:
    hr, acc = _col(df, "v0"), _col(df, "accuracy")
    ok = np.isfinite(hr) & (hr > 0) & (acc >= 2)
    rel = "good"
    notes = ["Android TYPE_HEART_RATE (Samsung algorithm, 1 Hz), median of rows with accuracy >= 2"]
    if not ok.any():
        ok = np.isfinite(hr) & (hr > 0) & (acc >= 1)
        rel, notes = "experimental", ["Android TYPE_HEART_RATE, only low-accuracy rows"]
    if not ok.any():
        return None
    v = hr[ok]
    return _estimate(_r(np.median(v), 1), "hr", "hr", rel, int(ok.sum()), notes,
                     {"min": _r(v.min(), 1), "max": _r(v.max(), 1), "rows": df.height})


def _split_list(v: Any) -> list[float]:
    if v is None:
        return []
    out = []
    for s in str(v).split("|"):
        s = s.strip()
        if s:
            try:
                out.append(float(s))
            except ValueError:
                out.append(float("nan"))
    return out


def _samsung_ibi(df: pl.DataFrame, clock: _Clock, motion: _Motion) -> tuple[dict | None, str | None]:
    T = clock.times(df)
    order = np.argsort(T, kind="stable")
    ibis = df["ibi_ms"].to_list() if "ibi_ms" in df.columns else [None] * df.height
    stats = df["ibi_status"].to_list() if "ibi_status" in df.columns else [None] * df.height
    vals, sts, row, first_in_row = [], [], [], []
    row_sum = np.zeros(df.height)
    for r in order:
        v, s = _split_list(ibis[r]), _split_list(stats[r])
        s = s + [float("nan")] * (len(v) - len(s))
        row_sum[r] = np.nansum(v) / 1000 if v else 0.0
        for k, x in enumerate(v):
            vals.append(x)
            sts.append(s[k])
            row.append(r)
            first_in_row.append(k == 0)
    if not vals:
        return None, "no IBI values in sdk_hr"
    vals, sts, row, first_in_row = map(np.asarray, (vals, sts, row, first_in_row))
    # a row more than IBI_ROW_GAP_S after the previous one means rows (and beats) are missing
    rank = np.empty(df.height, int)
    rank[order] = np.arange(df.height)
    Ts = T[order]
    row_gap = np.r_[False, np.diff(Ts) > IBI_ROW_GAP_S]
    status_ok = sts == 0
    plaus = np.isfinite(vals) & (vals >= NN_RANGE_MS[0]) & (vals <= NN_RANGE_MS[1])
    # Beats dropped inside the row sequence (no row missing): the time elapsed since the first row
    # minus the sum of the listed intervals jitters within about one row period (beat phase vs row
    # time; a front-loaded batch makes it fall), but rises for good by one interval when a beat is
    # missing. A rise above its running maximum by more than IBI_DROP_SHARE of the median interval
    # means a beat went missing after the row that set that maximum: adjacency is broken before
    # every row since then (the drop cannot be pinned to one row).
    pos = np.isfinite(vals) & (vals > 0)
    listed = np.bincount(rank[row[pos]], vals[pos], minlength=df.height) / 1000
    drift = (Ts - Ts[0]) - np.cumsum(listed)
    med_ibi = float(np.median(vals[plaus])) / 1000 if plaus.any() else 1.0
    dropped_before = np.zeros(df.height, bool)
    drop_events, top, top_at = 0, drift[0], 0
    for k in range(1, df.height):
        if drift[k] > top + IBI_DROP_SHARE * med_ibi and not row_gap[k]:  # a missing row explains it
            dropped_before[top_at + 1:k + 1] = True
            drop_events += 1
        if drift[k] >= top:
            top, top_at = drift[k], k
    gap_before = row_gap | dropped_before
    # adjacency: consecutive in the list, and no missing row in between
    adjacent = np.r_[True, ~(first_in_row[1:] & gap_before[rank[row[1:]]])]
    ok = status_ok & plaus
    if ok.any():
        med = _local_median(vals[ok], half=IBI_LOCAL_HALF)
        dev = np.zeros(len(vals), bool)
        dev[np.flatnonzero(ok)] = np.abs(vals[ok] - med) > IBI_LOCAL_DEV * med
        plaus &= ~dev
        ok &= ~dev
    # a lost PPG sample shortens one interval by a sample period: drop isolated short intervals
    slip = np.zeros(len(vals), bool)
    if len(vals) > 2:
        both = ok[1:-1] & ok[:-2] & ok[2:] & adjacent[1:-1] & adjacent[2:]
        slip[1:-1] = both & (vals[1:-1] - 0.5 * (vals[:-2] + vals[2:]) < -IBI_SLIP_MS)
    ok &= ~slip
    # the row lists beats from about IBI_LAG_S earlier: check motion where they happened
    still = motion.still_between(T[row] - IBI_LAG_S - row_sum[row], T[row] - IBI_LAG_S)
    ok_all = ok & still
    breaks = ~ok_all | ~adjacent
    # build segments: an interval starts a new segment when it is not adjacent to the previous one
    segs, cur = [], []
    for i in range(len(vals)):
        if not ok_all[i]:
            if cur:
                segs.append(np.asarray(cur))
            cur = []
            continue
        if cur and not adjacent[i]:
            segs.append(np.asarray(cur))
            cur = []
        cur.append(vals[i])
    if cur:
        segs.append(np.asarray(cur))
    hv = _hrv_values(segs)
    if hv is None:
        return None, f"too few valid Samsung IBIs ({int(ok_all.sum())})"
    n = int(ok_all.sum())
    n_diffs = sum(max(0, len(s) - 1) for s in segs)
    rel = "good" if n >= 20 and n_diffs >= 10 and motion.checked else "experimental"
    notes = [("Samsung HEART_RATE_CONTINUOUS IBI lists: intervals carry no beat times, some beats are "
              "dropped between batches; successive differences only between intervals listed next to "
              "each other with no missing row and no suspected dropped beat (elapsed time not covered "
              "by the listed intervals) in between"),
             f"dropped intervals with ibi_status != 0, outside 300-2000 ms, > {IBI_LOCAL_DEV:.0%} from the "
             f"median of {2 * IBI_LOCAL_HALF + 1}, or > {IBI_SLIP_MS:.0f} ms shorter than both neighbours' mean "
             "(a silently lost PPG sample)",
             f"motion checked {IBI_LAG_S} s before each row, where Samsung's listed beats happened"]
    if not motion.checked:
        notes.append("no accelerometer: motion not checked")
    if (hv["window_s"] or 0) < SHORT_HRV_S:
        notes.append(f"ultra-short window ({hv['window_s']} s < {SHORT_HRV_S:.0f} s)")
    n_status_ok = int((status_ok & np.isfinite(vals)).sum())
    if (hv["rmssd_ms"] or 0) > HRV25_HIGH_RMSSD_MS or slip.sum() > 0.1 * max(n_status_ok, 1):
        notes.append(f"{int(slip.sum())} of {n_status_ok} intervals dropped as lost-sample errors: at high HRV that "
                     "rule also removes genuine short intervals, so RMSSD may be low")
    details = {"intervals": int(len(vals)), "dropped_status": int((~status_ok).sum()),
               "dropped_implausible": int((status_ok & ~plaus).sum()), "dropped_lost_sample": int(slip.sum()),
               "dropped_motion": int((ok & ~still).sum()), "row_gaps": int(row_gap.sum()),
               "suspected_dropped_beats": int(drop_events), "rows_after_suspected_drop": int(dropped_before.sum()),
               "segments": len(segs), "successive_diffs": int(n_diffs), "breaks": int(breaks.sum())}
    return _estimate(hv, "sdk_hr", "ibi", rel, n, notes, details), None


def _ecg_times(df: pl.DataFrame, clock: _Clock) -> np.ndarray:
    """Seconds from the run start for each ECG sample, from the sample count.

    The watch stamps each 10-sample bundle with about its delivery time, not its sampling time: on
    SM-R965F the stamps ran at 215-835 Hz for the first ~12 s while the electrode settled (up to 2.8 s
    late) and jitter ~2 ms after that, while the samples themselves are evenly spaced (501.3 Hz by the
    watch clock). So bundles are numbered by the 8-bit ``sequence`` of their first sample (a jump is a
    lost bundle) and a line through the earliest stamps, which latency only ever delays, maps that
    number to the clock. Falls back to the stamps when there is no usable sequence."""
    seq = _col(df, "sequence")
    ts = _col(df, "timestamp_ms")
    head = np.flatnonzero(seq >= 0)
    if len(head) < 10 or not np.isfinite(ts[head]).all():
        return clock.times(df)
    k = np.r_[0, np.cumsum((np.diff(seq[head]).astype(np.int64) - 1) % 256 + 1)].astype(float)
    size = float(np.median(np.diff(head)))
    rows = np.arange(len(seq))
    owner = np.maximum(np.searchsorted(head, rows, side="right") - 1, 0)
    n = k[owner] + (rows - head[owner]) / size  # in bundles
    y = ts[head]
    p = np.polyfit(k, y, 1)
    for _ in range(5):
        early = y - np.polyval(p, k) <= np.median(y - np.polyval(p, k))
        p = np.polyfit(k[early], y[early], 1)
    return (np.polyval(p, n) - clock.t0_ms) / 1000


@dataclass
class _EcgPeaks:
    t: np.ndarray  # R-peak times, s from the run start
    ok: np.ndarray  # per interval: lead-on throughout, plausible and near its local median
    lead_on_s: float
    detected: int  # peaks NeuroKit2 found, before the edge, lead-off and height checks
    inverted: bool


def _ecg_peaks(df: pl.DataFrame, clock: _Clock) -> tuple[_EcgPeaks | None, str | None]:
    """R-peaks of the Samsung ECG, or the reason there are too few."""
    import neurokit2 as nk

    t = _ecg_times(df, clock)
    v = _col(df, "ecg_mv")
    lead = _col(df, "lead_off")
    ok = np.isfinite(t) & np.isfinite(v)
    t, v, lead = t[ok], v[ok], lead[ok]
    order = np.argsort(t, kind="stable")
    t, v, lead = t[order], v[order], lead[order]
    if len(t) < 2 or t[-1] - t[0] < 5:
        return None, "less than 5 s of ECG"
    # the watch flags lead-off (0 on, 5 off) on the first sample of each 10-sample bundle and writes -1
    # on the other nine, which take their bundle's flag
    head = np.flatnonzero(lead >= 0)
    if not len(head):
        return None, "no lead-off flags in the ECG"
    on = lead[head[np.maximum(np.searchsorted(head, np.arange(len(lead)), side="right") - 1, 0)]] == 0
    grid = np.arange(t[0], t[-1], 1 / ECG_FS)
    sig = np.interp(grid, t, v)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    near = np.where(grid - t[j - 1] <= t[j] - grid, j - 1, j)
    gon = on[near] & (np.abs(t[near] - grid) <= 0.01)
    lead_on_s = float(gon.sum() / ECG_FS)
    if lead_on_s < 5:
        return None, f"only {lead_on_s:.1f} s of lead-on ECG"
    sig = np.where(gon, sig, np.median(sig[gon]) if gon.any() else 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = nk.ecg_clean(sig, sampling_rate=int(ECG_FS))
        clean, inverted = nk.ecg_invert(clean, sampling_rate=int(ECG_FS))
        _, info = nk.ecg_peaks(clean, sampling_rate=int(ECG_FS), correct_artifacts=False)
    peaks = np.asarray(info["ECG_R_Peaks"], int)
    n_detected = len(peaks)
    # drop detections next to a record edge or lead-off (filter transients) and low ones (T waves, noise)
    margin = int(0.25 * ECG_FS)
    near_off = np.convolve((~gon).astype(int), np.ones(2 * margin + 1, int), mode="same") > 0
    near_off[:margin] = near_off[-margin:] = True
    peaks = peaks[~near_off[peaks]]
    if len(peaks):
        amp = clean[peaks]
        peaks = peaks[amp >= 0.4 * np.median(amp)]
    if len(peaks) < 4:
        return None, "fewer than 4 R-peaks"
    rt = grid[0] + np.array([p + _parabolic(clean, p) for p in peaks]) / ECG_FS
    a, e = rt[:-1], rt[1:]
    nn = (e - a) * 1000
    cum = np.r_[0, np.cumsum(~gon)]
    i0 = np.clip(np.floor((a - grid[0]) * ECG_FS).astype(int), 0, len(grid) - 1)
    i1 = np.clip(np.ceil((e - grid[0]) * ECG_FS).astype(int), 0, len(grid) - 1)
    okv = (cum[i1 + 1] - cum[i0] == 0) & (nn >= NN_RANGE_MS[0]) & (nn <= NN_RANGE_MS[1])
    if okv.any():
        med = _local_median(nn[okv])
        dev = np.zeros(len(nn), bool)
        dev[np.flatnonzero(okv)] = np.abs(nn[okv] - med) > LOCAL_DEV * med
        okv &= ~dev
    if okv.sum() < 3:
        return None, "too few clean RR intervals"
    return _EcgPeaks(t=rt, ok=okv, lead_on_s=lead_on_s, detected=n_detected, inverted=bool(inverted)), None


def _ecg(df: pl.DataFrame, clock: _Clock) -> tuple[dict | None, dict | None, str | None]:
    """(HR estimate, HRV estimate, reason when nothing could be computed)."""
    p, why = _ecg_peaks(df, clock)
    if p is None:
        return None, None, why
    rt, okv, lead_on_s, inverted = p.t, p.ok, p.lead_on_s, p.inverted
    a, e = rt[:-1], rt[1:]
    nn = (e - a) * 1000
    rel = "reference" if lead_on_s >= ECG_REFERENCE_S else ("good" if lead_on_s >= ECG_MIN_S else "unreliable")
    base_notes = [(f"Samsung ECG_ON_DEMAND at {ECG_FS:.0f} Hz (timestamp_ms), NeuroKit2 ecg_clean + "
                   "ecg_peaks, parabolic sub-sample R-peaks; lead-off samples, peaks within 0.25 s of an edge "
                   "or lead-off, low peaks and intervals > 25 % from the local median excluded")]
    if rel != "reference":
        base_notes.append(f"only {lead_on_s:.0f} s lead-on (reference needs {ECG_REFERENCE_S:.0f} s)")
    if inverted:
        base_notes.append("ECG polarity was inverted")
    details = {"lead_on_s": _r(lead_on_s, 1), "r_peaks": int(len(rt)), "r_peaks_detected": p.detected,
               "inverted": bool(inverted),
               "nn_rejected": int((~okv).sum()), "spans_s": _merge_spans(a[okv], e[okv])}
    hr = _estimate(_r(60000 / np.median(nn[okv]), 1), "sdk_ecg", "ecg", rel, int(okv.sum()), base_notes, details)
    hv = _hrv_values(_segments(nn, okv))
    hrv = None
    if hv is not None:
        notes = list(base_notes)
        if (hv["window_s"] or 0) < SHORT_HRV_S:
            notes.append(f"ultra-short window ({hv['window_s']} s < {SHORT_HRV_S:.0f} s)")
        hrv = _estimate(hv, "sdk_ecg", "ecg", rel, int(okv.sum()), notes, details)
    return hr, hrv, None


def _skin(df: pl.DataFrame) -> dict | None:
    obj, amb, st = _col(df, "object_c"), _col(df, "ambient_c"), _col(df, "status")
    ok = np.isfinite(obj) & ((st == 0) | ~np.isfinite(st))
    if not ok.any():
        return None
    value = {"object_c_median": _r(np.median(obj[ok])), "object_c_min": _r(obj[ok].min()),
             "object_c_max": _r(obj[ok].max()),
             "ambient_c_median": _r(np.nanmedian(amb[ok])) if np.isfinite(amb[ok]).any() else None}
    return _estimate(value, "sdk_skin", "skin_temp", "good" if ok.sum() >= 3 else "experimental", int(ok.sum()),
                     ["Samsung SKIN_TEMPERATURE_CONTINUOUS, rows with status == 0; object = skin surface"],
                     {"rows": df.height})


def _spo2(df: pl.DataFrame) -> tuple[dict | None, str | None]:
    sp, st, hr, flag = _col(df, "spo2"), _col(df, "status"), _col(df, "hr"), _col(df, "accuracy_flag")
    done = np.flatnonzero((st == 2) & np.isfinite(sp) & (sp > 0))
    counts = {str(int(s)): int((st == s).sum()) for s in np.unique(st[np.isfinite(st)])}
    if not len(done):
        return None, f"no completed SpO2 measurement (status counts {counts})"
    i = done[-1]
    return _estimate(_r(sp[i], 1), "sdk_spo2", "spo2", "good", int(len(done)),
                     ["Samsung SPO2_ON_DEMAND, last row with status == 2 (measurement completed)"],
                     {"hr": _r(hr[i], 1), "accuracy_flag": _r(flag[i], 0), "status_counts": counts}), None


def _on_wrist(df: pl.DataFrame) -> dict | None:
    s = _col(df, "v0")
    s = s[np.isfinite(s)]
    if not len(s):
        return None
    return _estimate(_r(np.mean(s == 2), 3), "hr_raw", "ppg", "good", int(len(s)),
                     ["share of hr_raw samples with status v0 == 2 (on the wrist)"],
                     {"off_wrist_samples": int((s == 1).sum())})


# --------------------------------------------------------------------------------------------
# run analysis
# --------------------------------------------------------------------------------------------

def _choose(estimates: list[dict], priority: tuple[str, ...]) -> dict | None:
    """First estimate in priority order that is not unreliable (else the first one)."""
    rank = {s: i for i, s in enumerate(priority)}
    ordered = sorted(estimates, key=lambda e: (e["reliability"] == "unreliable", rank.get(e["source"], 99)))
    return ordered[0] if ordered else None


def _metric(estimates: list[dict], priority: tuple[str, ...], skipped: list[dict]) -> dict:
    rank = {s: i for i, s in enumerate(priority)}
    estimates = sorted(estimates, key=lambda e: rank.get(e["source"], 99))
    return {"chosen": _choose(estimates, priority), "estimates": estimates, "skipped": skipped}


def _safe(skipped: list, source: str, fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # one bad source must not sink the whole analysis
        skipped.append({"source": source, "reason": f"error: {type(exc).__name__}: {exc}"})
        return None


def analyze_run(run_dir: str | Path, write: bool = True) -> dict:
    """Analyse one run directory; writes ``analysis.json`` there (unless ``write=False``)."""
    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f"no run directory {run}")
    warns: list[dict] = []
    tables = load_tables(run, warns)
    summary = read_summary(run)
    meta = _read_kv(run / "meta.txt")

    caps_path = run / CAPABILITIES_FILE
    caps, caps_source = None, "inferred"
    if caps_path.is_file():
        try:
            caps = json.loads(caps_path.read_text())
            caps_source = "file"
            if caps.get("schema") != CAPABILITIES_SCHEMA:
                _warn(warns, "capabilities_schema", f"capabilities.json schema {caps.get('schema')!r}, "
                      f"expected {CAPABILITIES_SCHEMA}")
        except (ValueError, OSError) as exc:
            _warn(warns, "capabilities_unreadable", f"capabilities.json: {exc}")
            caps = None
    if caps is None:
        caps = infer_capabilities(run, tables, summary)
        _warn(warns, "capabilities_inferred", "no capabilities.json: tier and sources inferred from the "
              "files present", "info")
    tier = caps.get("tier")
    if tier not in RESOLUTION:
        tier = "all" if any(a.startswith("sdk_") for a in tables) else "25hz"

    clock = _Clock(meta, tables)
    motion = _motion(tables, clock)
    sources = {}
    for alias, df in tables.items():
        tt = clock.times(df) if ("event_ns" in df.columns or "timestamp_ms" in df.columns) else np.array([])
        tt = tt[np.isfinite(tt)]
        dur = float(tt.max() - tt.min()) if len(tt) > 1 else 0.0
        sources[alias] = {"rows": df.height, "tier": _tier_of(alias), "duration_s": _r(dur, 1),
                          "rate_hz": _r((len(tt) - 1) / dur) if dur > 0 else None}
    try:
        duration = float(meta["seconds"])
    except (KeyError, ValueError):
        duration = max((s["duration_s"] or 0 for s in sources.values()), default=0.0)

    hr_est, hrv_est, hr_skip, hrv_skip = [], [], [], []
    other: dict[str, tuple[list, list]] = {k: ([], []) for k in ("motion", "on_wrist", "skin_temp", "spo2")}

    def keep(store: tuple[list, list], source: str, est: dict | None, reason: str) -> None:
        if est:
            store[0].append(est)
        elif not any(sk["source"] == source for sk in store[1]):  # _safe may have logged an error
            store[1].append({"source": source, "reason": reason})

    hr, hrv = (hr_est, hr_skip), (hrv_est, hrv_skip)
    if "sdk_hr" in tables:
        keep(hr, "sdk_hr", _safe(hr_skip, "sdk_hr", _samsung_hr, tables["sdk_hr"]), "no rows with hr_status == 1")
        if res := _safe(hrv_skip, "sdk_hr", _samsung_ibi, tables["sdk_hr"], clock, motion):
            keep(hrv, "sdk_hr", res[0], res[1])
    if "hr" in tables:
        keep(hr, "hr", _safe(hr_skip, "hr", _android_hr, tables["hr"]), "no rows with a heart rate")
    if "sdk_ecg" in tables and (res := _safe(hrv_skip, "sdk_ecg", _ecg, tables["sdk_ecg"], clock)):
        hr_e, hrv_e, why = res
        keep(hr, "sdk_ecg", hr_e, why or "too few RR intervals")
        keep(hrv, "sdk_ecg", hrv_e, why or "too few RR intervals")

    acc25 = _acc_25hz(_accelerometer(tables, clock))

    def run_ppg(alias: str):
        t, x, valid, notes = _ppg_input(alias, tables[alias], clock)
        beats = _ppg_beats(alias, t, x, valid, motion)
        h = v = None
        hr_why = hrv_why = "no usable PPG (under 10 s on-wrist or under 6 beats)"
        if beats is not None:
            iv = _ppg_intervals(beats, motion)
            h, hr_why = _ppg_hr(beats, iv, motion), "no heart rate from PPG"
            if alias not in HRV25_SOURCES:
                v, hrv_why = _ppg_hrv(beats, iv)
        if alias in HRV25_SOURCES:
            v, hrv_why = _ppg_hrv25(alias, t, x, valid, acc25)
        for est in (h, v):
            if est:
                est["notes"] = notes + est["notes"]
        return h, v, hr_why, hrv_why or "no HRV from PPG"

    for alias in PPG_SOURCES:
        if alias not in tables:
            continue
        failed: list[dict] = []
        res = _safe(failed, alias, run_ppg, alias)
        if res is None:  # raised: report it under both metrics
            hr_skip.extend(failed)
            hrv_skip.extend(failed)
            continue
        h, v, hr_why, hrv_why = res
        keep(hr, alias, h, hr_why)
        keep(hrv, alias, v, hrv_why)

    if motion.checked:
        other["motion"][0].append(_motion_estimate(motion, tables))
    else:
        other["motion"][1].append({"source": "acc", "reason": "no accelerometer recorded"})
    if "hr_raw" in tables:
        keep(other["on_wrist"], "hr_raw", _safe(other["on_wrist"][1], "hr_raw", _on_wrist, tables["hr_raw"]),
             "no hr_raw status values")
    else:
        other["on_wrist"][1].append({"source": "hr_raw", "reason": "hr_raw not recorded"})
    if "sdk_skin" in tables:
        keep(other["skin_temp"], "sdk_skin", _safe(other["skin_temp"][1], "sdk_skin", _skin, tables["sdk_skin"]),
             "no rows with status == 0")
    if "sdk_spo2" in tables and (res := _safe(other["spo2"][1], "sdk_spo2", _spo2, tables["sdk_spo2"])):
        keep(other["spo2"], "sdk_spo2", res[0], res[1])

    metrics = {"hr": _metric(hr_est, HR_PRIORITY, hr_skip), "hrv": _metric(hrv_est, HRV_PRIORITY, hrv_skip)}
    for k, (est, sk) in other.items():
        metrics[k] = _metric(est, (est[0]["source"],) if est else (), sk)
    if "sdk_ppg100" in tables:  # exploratory; analysis never downloads the weights
        from pulse_exploration import ppg_age

        age: tuple[list, list] = ([], [])
        if res := _safe(age[1], "sdk_ppg100", ppg_age.watch_estimate, "sdk_ppg100", tables["sdk_ppg100"], clock,
                        motion):
            keep(age, "sdk_ppg100", res[0], res[1])
        metrics["ppg_age"] = _metric(age[0], ("sdk_ppg100",), age[1])
        if pa := metrics["ppg_age"]["chosen"]:
            v = pa["value"]
            _warn(warns, "ppg_age_exploratory", f"PPG age {v['ppg_age_y']} y (band {v['band_y'][0]}-"
                  f"{v['band_y'][1]} y, {v['beats_averaged']} beats, {pa['reliability']}) is exploratory, not an "
                  "absolute age: a within-person trend at best", "info")

    # warnings
    if not tables:
        _warn(warns, "no_data", "no source delivered any data")
    if 0 < duration < MIN_RUN_S:
        _warn(warns, "too_short",
              f"run is {duration:.0f} s; HR needs >= {MIN_RUN_S:.0f} s, HRV >= {SHORT_HRV_S:.0f} s")
    ow = metrics["on_wrist"]["chosen"]
    if ow and ow["value"] is not None and ow["value"] < 0.95:
        _warn(warns, "off_wrist", f"hr_raw reports off-wrist for {1 - ow['value']:.0%} of samples")
    mo = metrics["motion"]["chosen"]
    if mo and (mo["value"]["still_share"] or 0) < 0.8:
        _warn(warns, "motion", f"motion in {mo['value']['motion_s']} s of the run "
              f"(still share {mo['value']['still_share']:.0%}); beats in motion windows are not used for HRV")
    for r in summary:
        err = (r.get("error") or "").strip()
        if r.get("tier") == "blocked":  # reserved for Samsung's own apps; expected to fail
            if err:
                _warn(warns, "blocked_source", f"{r.get('alias')}: {err}", "info")
        elif err:
            msg = f"{r.get('alias')}: {err}"
            if "SDK_POLICY_ERROR" in err:
                msg += f" ({POLICY_REASON})"
            _warn(warns, "source_error", msg)
        elif r.get("registered", "").lower() == "true" and r.get("events", "") in ("0", ""):
            _warn(warns, "no_events", f"{r.get('alias')}: registered but delivered no events")
    hv = metrics["hrv"]["chosen"]
    if hv is None:
        _warn(warns, "no_hrv", "no HRV estimate: " + "; ".join(f"{s['source']}: {s['reason']}" for s in hrv_skip)
              if hrv_skip else "no HRV estimate: no beat-to-beat source recorded", "info")
    else:
        if hv["reliability"] == "unreliable":
            _warn(warns, "hrv_unreliable", f"HRV from {hv['source']} is unreliable: "
                  + "; ".join(n for n in hv["notes"] if "gate" in n or "lead-on" in n))
        if (hv["value"]["window_s"] or 0) < SHORT_HRV_S:
            _warn(warns, "hrv_ultra_short", f"HRV window is {hv['value']['window_s']} s (< {SHORT_HRV_S:.0f} s): "
                  "ultra-short-term values", "info")
        # PPG and Samsung's IBIs come from the same sensor: when both are usable they should agree
        usable = {e["source"]: e["value"]["rmssd_ms"] for e in metrics["hrv"]["estimates"]
                  if e["reliability"] != "unreliable" and e["value"]["rmssd_ms"] is not None}
        pair = [s for s in ("sdk_ppg", "hr_raw") if s in usable]
        if "sdk_hr" in usable and pair and abs(usable["sdk_hr"] - usable[pair[0]]) > DISAGREE_RMSSD_MS:
            _warn(warns, "hrv_sources_disagree", f"RMSSD {usable[pair[0]]} ms from {pair[0]} vs {usable['sdk_hr']} "
                  f"ms from Samsung's IBIs (> {DISAGREE_RMSSD_MS:.0f} ms apart): treat both with caution", "info")
    if metrics["hr"]["chosen"] is None:
        _warn(warns, "no_hr", "no heart-rate estimate")

    feature_used = {e["feature"] for m in metrics.values() for e in m["estimates"]}
    hr_c, hrv_c = metrics["hr"]["chosen"], metrics["hrv"]["chosen"]
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "run": run.name,
        "kind": run.name.rsplit("-", 1)[-1] if "-" in run.name else None,
        "tier": tier,
        "capabilities_source": caps_source,
        "capabilities": caps,
        "meta": meta,
        "duration_s": _r(duration, 1),
        "sources": sources,
        "features_used": [f for f in FEATURES if f in feature_used],
        "summary": {
            "hr_bpm": hr_c["value"] if hr_c else None, "hr_source": hr_c["source"] if hr_c else None,
            "hr_reliability": hr_c["reliability"] if hr_c else None,
            "rmssd_ms": hrv_c["value"]["rmssd_ms"] if hrv_c else None,
            "sdnn_ms": hrv_c["value"]["sdnn_ms"] if hrv_c else None,
            "hrv_source": hrv_c["source"] if hrv_c else None,
            "hrv_reliability": hrv_c["reliability"] if hrv_c else None,
            "still_share": mo["value"]["still_share"] if mo else None,
            "on_wrist_share": ow["value"] if ow else None,
        },
        "metrics": metrics,
        "recorder": [{k: r.get(k, "") for k in ("tier", "alias", "registered", "events", "rate_hz", "error")}
                     for r in summary],
        "warnings": warns,
        "library": {"neurokit2": importlib.metadata.version("neurokit2")},
    }
    result = _jsonable(result)
    if write:
        tmp = run / (ANALYSIS_FILE + ".part")
        tmp.write_text(json.dumps(result, indent=1, allow_nan=False))
        tmp.replace(run / ANALYSIS_FILE)
    return result


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m pulse_exploration.watch_session <run_dir> [--no-write]", file=sys.stderr)
        return 2
    res = analyze_run(Path(argv[0]), write="--no-write" not in argv)
    print(json.dumps({"run": res["run"], "tier": res["tier"], "summary": res["summary"],
                      "warnings": res["warnings"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
