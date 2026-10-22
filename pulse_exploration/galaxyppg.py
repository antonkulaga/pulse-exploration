"""GalaxyPPG dataset: loader, ECG reference beats and watch-to-Polar clock alignment.

Dataset: Park, Zheng & Lee, "GalaxyPPG", Scientific Data 12:892 (2025), Zenodo
10.5281/zenodo.14635823, CC-BY-4.0. 24 participants wore a Galaxy Watch 5 (25 Hz green PPG,
25 Hz accelerometer, Samsung HR/IBI at 1 Hz), a Polar H10 chest strap (ECG ~130 Hz, RR) and an
Empatica E4 (ignored here) through a ~1 h protocol of rest, stress, typing and locomotion sessions.
Known gaps: P01 has no watch data or events, P07/P08 have no watch HR, P02 has no skin temperature,
P15 loses ~10 min of watch data.

Clocks:
- Watch tables (`timestamp`) and Event.csv are Unix ms on the watch/phone clock; scoring windows
  live on this clock.
- Polar tables use `phoneTimestamp` in UTC+9 (README); we subtract 9 h. Polar Sensor Logger derives
  phoneTimestamp from the sensor clock with a fixed offset (residual < 1.5 ms), so the "smooth" ECG
  time is a linear fit of phone time on sensorTimestamp. Watch and Polar clocks still differ by up
  to ~6 s and drift by -15..-55 ppm, which `alignment()` estimates.
- Event.csv is assumed to be on the watch/phone clock; a few seconds either way do not matter for
  60 s windows.

Usage:
    rec = load("P02")                  # parsed tables, cached as parquet
    ref = reference("P02")             # ECG R-peaks (Polar clock) + quality + Polar IBI cross-check
    al = alignment("P02")              # watch = polar + offset (+ drift)
    uv run python -m pulse_exploration.galaxyppg [P02 ...]   # per-participant reference/alignment table
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from scipy import signal, stats

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = Path(os.environ.get("GALAXYPPG_DIR", ROOT / "data/external/galaxyppg/unpacked/Dataset"))
CACHE_DIR = Path(os.environ.get("GALAXYPPG_CACHE", ROOT / "data/cache/galaxyppg"))
CACHE_VERSION = "v1"  # raw parsed tables
REFERENCE_VERSION = 3  # bump when the reference method or its outputs change
ALIGNMENT_VERSION = 5
UIDS = tuple(f"P{i:02d}" for i in range(1, 25))
POLAR_UTC_SHIFT_MS = 9 * 3600 * 1000  # Polar phoneTimestamp is UTC+9
PPG_FS = 25.0
SAMSUNG_IBI_LAG_MS = 7400.0  # Samsung's IBI list trails the beats by this much (see samsung_ibi)


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    name: str
    start_ms: int
    end_ms: int

    @property
    def kind(self) -> str:
        """Session type without the repeat number: 'meditation-2' -> 'meditation', 'rest-3' -> 'rest'."""
        head, _, tail = self.name.rpartition("-")
        return head if head and tail.isdigit() else self.name


@dataclass
class Recording:
    """One participant. Times are Unix ms; watch tables on the watch clock, Polar on the Polar clock.

    ppg:       t_ms i64, value f64 (green ADC), status i32 (0 normal, -999 higher-priority sensor)
    acc:       t_ms i64, x/y/z f64 (m/s^2, same timestamps as ppg)
    hr:        t_ms i64, received_ms i64, hr i32, status i32, ibi list[i32], ibi_status list[i32],
               batch i32 (rows delivered together; a batch's IBIs sit on its first row). File order;
               t_ms is not monotonic where a batch overlaps the live rows before it
    ecg:       t_ms f64 (smoothed, Polar clock), phone_ms i64, sensor_ns i64, uv f64
    polar_ibi: t_ms i64, ibi_ms i32 (Polar's own RR; timestamps are reconstructed, +-0.3 s)
    polar_hr:  t_ms i64, hr i32
    skin_temp: t_ms i64, ambient f64, object f64, status i32
    """

    uid: str
    meta: dict
    sessions: list[Session]
    ppg: pl.DataFrame
    acc: pl.DataFrame
    hr: pl.DataFrame
    ecg: pl.DataFrame
    polar_ibi: pl.DataFrame
    polar_hr: pl.DataFrame
    skin_temp: pl.DataFrame
    ecg_fs: float
    ecg_time_fit: dict

    @property
    def has_watch(self) -> bool:
        return self.ppg.height > 0

    @property
    def has_watch_hr(self) -> bool:
        return self.hr.height > 0

    def ppg_arrays(self, status_ok: tuple[int, ...] = (0, 500)) -> tuple[np.ndarray, np.ndarray]:
        """(t_ms, value) of PPG samples whose status is in `status_ok` (drops -999 blocks)."""
        d = self.ppg.filter(pl.col("status").is_in(list(status_ok)))
        return d["t_ms"].to_numpy().astype(np.float64), d["value"].to_numpy().astype(np.float64)

    def acc_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.acc["t_ms"].to_numpy().astype(np.float64), self.acc.select("x", "y", "z").to_numpy()


_EMPTY = {
    "ppg": {"t_ms": pl.Int64, "value": pl.Float64, "status": pl.Int32},
    "acc": {"t_ms": pl.Int64, "x": pl.Float64, "y": pl.Float64, "z": pl.Float64},
    "hr": {
        "t_ms": pl.Int64,
        "received_ms": pl.Int64,
        "hr": pl.Int32,
        "status": pl.Int32,
        "ibi": pl.List(pl.Int32),
        "ibi_status": pl.List(pl.Int32),
        "batch": pl.Int32,
    },
    "skin_temp": {"t_ms": pl.Int64, "ambient": pl.Float64, "object": pl.Float64, "status": pl.Int32},
}
TABLES = ("ppg", "acc", "hr", "ecg", "polar_ibi", "polar_hr", "skin_temp", "sessions")


def _read(path: Path, name: str, select: dict[str, tuple[str, pl.DataType]]) -> pl.DataFrame:
    """CSV -> renamed/cast columns, or an empty frame of the same schema when the file is missing."""
    if not path.exists():
        return pl.DataFrame(schema=_EMPTY.get(name, {new: dt for _, (new, dt) in select.items()}))
    df = pl.read_csv(path, infer_schema_length=10000)
    return df.select([pl.col(src).cast(dt).alias(new) for src, (new, dt) in select.items()])


def _ecg_time(phone_ms: np.ndarray, sensor_ns: np.ndarray) -> tuple[np.ndarray, dict]:
    """Smooth monotonic Polar-clock time: phone_ms ~ a_seg + b * sensor_ms, one slope, one intercept
    per segment (segments split at sensor gaps > 1 s; GalaxyPPG has none). Outliers (> 20 ms) are
    dropped from the fit twice."""
    s_ms = (sensor_ns - sensor_ns[0]) / 1e6
    seg = np.r_[0, np.cumsum(np.diff(s_ms) > 1000.0)]
    keep = np.ones(len(s_ms), bool)
    for _ in range(3):
        b = np.polyfit(s_ms[keep], phone_ms[keep], 1)[0]
        a = np.array([np.median((phone_ms - b * s_ms)[keep & (seg == k)]) for k in range(seg.max() + 1)])
        res = phone_ms - (a[seg] + b * s_ms)
        keep = np.abs(res) < 20.0
    t = a[seg] + b * s_ms
    dt = np.diff(s_ms)
    fs = 1000.0 / np.median(dt)
    info = {
        "slope": float(b),
        "drift_ppm_vs_phone": float((b - 1) * 1e6),
        "segments": int(seg.max() + 1),
        "residual_ms_p01_p99": [float(np.percentile(res, 1)), float(np.percentile(res, 99))],
        "max_abs_residual_ms": float(np.abs(res).max()),
        "fs_hz": float(fs),
    }
    return t, info


def _parse(uid: str) -> dict:
    d = DATASET_DIR / uid
    if not d.exists():
        raise FileNotFoundError(f"{d} not found (set GALAXYPPG_DIR)")
    gw, polar = d / "GalaxyWatch", d / "PolarH10"
    i64, i32, f64 = pl.Int64, pl.Int32, pl.Float64
    ppg = _read(gw / "PPG.csv", "ppg", {"timestamp": ("t_ms", i64), "ppg": ("value", f64), "status": ("status", i32)})
    acc = _read(gw / "ACC.csv", "acc", {"timestamp": ("t_ms", i64), "x": ("x", f64), "y": ("y", f64), "z": ("z", f64)})
    hr_path = gw / "HR.csv"
    if hr_path.exists():
        hr = pl.read_csv(hr_path).select(
            pl.col("timestamp").cast(i64).alias("t_ms"),
            pl.col("dataReceived").cast(i64).alias("received_ms"),
            pl.col("hr").cast(i32),
            pl.col("hrStatus").cast(i32).alias("status"),
            pl.col("ibi").str.json_decode(pl.List(i32)),
            pl.col("ibiStatus").str.json_decode(pl.List(i32)).alias("ibi_status"),
        )
        hr = hr.with_columns(
            (pl.col("received_ms") != pl.col("received_ms").shift(1)).fill_null(True).cum_sum().cast(i32).alias("batch") - 1
        )
    else:
        hr = pl.DataFrame(schema=_EMPTY["hr"])
    skin = _read(
        gw / "SkinTemp.csv",
        "skin_temp",
        {"timestamp": ("t_ms", i64), "ambientTemp": ("ambient", f64), "objectTemp": ("object", f64), "status": ("status", i32)},
    )
    for name, df in (("ppg", ppg), ("acc", acc)):  # HR rows may overlap at batch boundaries (P12, P15, P17)
        if df.height and not (df["t_ms"].diff().drop_nulls() > 0).all():
            raise ValueError(f"{uid} {name}: timestamps not strictly increasing")

    raw = pl.read_csv(polar / "ECG.csv")
    phone = raw["phoneTimestamp"].to_numpy().astype(np.float64) - POLAR_UTC_SHIFT_MS
    t, fit = _ecg_time(phone, raw["sensorTimestamp"].to_numpy())
    ecg = pl.DataFrame(
        {
            "t_ms": t,
            "phone_ms": phone.astype(np.int64),
            "sensor_ns": raw["sensorTimestamp"].cast(i64),
            "uv": raw["ecg"].cast(f64),
        }
    )
    pibi = pl.read_csv(polar / "IBI.csv").select(
        (pl.col("phoneTimestamp").cast(i64) - POLAR_UTC_SHIFT_MS).alias("t_ms"), pl.col("duration").cast(i32).alias("ibi_ms")
    )
    phr = pl.read_csv(polar / "HR.csv").select(
        (pl.col("phoneTimestamp").cast(i64) - POLAR_UTC_SHIFT_MS).alias("t_ms"), pl.col("hr").cast(i32)
    )

    sessions = []
    ev_path = d / "Event.csv"
    if ev_path.exists():
        ev = pl.read_csv(ev_path)
        open_: dict[str, int] = {}
        for ts, name, status in ev.iter_rows():
            if status == "ENTER":
                open_[name] = int(ts)
            elif status == "EXIT" and name in open_:
                sessions.append({"name": name, "start_ms": open_.pop(name), "end_ms": int(ts)})
    sess = pl.DataFrame(sessions, schema={"name": pl.String, "start_ms": i64, "end_ms": i64})

    meta_df = pl.read_csv(DATASET_DIR / "Meta.csv", infer_schema_length=0).filter(pl.col("UID") == uid)
    meta = meta_df.row(0, named=True) if meta_df.height else {"UID": uid}
    meta = {k: (int(v) if isinstance(v, str) and v.isdigit() else v) for k, v in meta.items()}
    return {
        "ppg": ppg,
        "acc": acc,
        "hr": hr,
        "ecg": ecg,
        "polar_ibi": pibi,
        "polar_hr": phr,
        "skin_temp": skin,
        "sessions": sess,
        "info": {"uid": uid, "meta": meta, "ecg_fs": fit["fs_hz"], "ecg_time_fit": fit},
    }


def _cache_dir(uid: str) -> Path:
    return CACHE_DIR / CACHE_VERSION / uid


def _write_atomic(target: Path, tables: dict[str, pl.DataFrame], info: dict) -> None:
    """Write tables + info.json into a temp dir, then rename it over `target` (safe across processes)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    for name, df in tables.items():
        df.write_parquet(tmp / f"{name}.parquet")
    (tmp / "info.json").write_text(json.dumps(info, indent=1, default=float))
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    try:
        os.replace(tmp, target)
    except OSError:  # another process won the race
        shutil.rmtree(tmp, ignore_errors=True)


def load(uid: str, refresh: bool = False) -> Recording:
    """Parsed participant `uid` ('P01'..'P24'); parquet cache under data/cache/galaxyppg/<version>/."""
    cdir = _cache_dir(uid) / "raw"
    if refresh or not (cdir / "info.json").exists():
        parsed = _parse(uid)
        info = parsed.pop("info")
        _write_atomic(cdir, parsed, info)
    info = json.loads((cdir / "info.json").read_text())
    t = {name: pl.read_parquet(cdir / f"{name}.parquet") for name in TABLES}
    sessions = [Session(*row) for row in t.pop("sessions").iter_rows()]
    return Recording(uid=uid, meta=info["meta"], sessions=sessions, ecg_fs=info["ecg_fs"], ecg_time_fit=info["ecg_time_fit"], **t)


def samsung_ibi(rec: Recording) -> pl.DataFrame:
    """Samsung's own IBI list (SDK tier) as one interval per row, on the watch clock.

    Columns: t_end_ms f64 (approximate end beat time), ibi_ms f64, ok bool (ibiStatus == 0),
    adjacent_next bool (this interval and the next one are consecutive beats).

    A batch's IBIs sit on its first row and cover the batch's span [first row - 1 s, last row]. When
    they sum to at least the span (live 1 s rows) the last beat is put at the last row's time and the
    rest cumulated backwards; when they sum to less (dropped beats inside a long batch) the
    positions are stretched over the span. Positions are only used to put intervals into windows;
    the IBI values are untouched. Beats placed before the end of the previous batch (overlapping
    batches) are dropped. Two intervals are adjacent when consecutive in the stream and their placed
    times agree with the second interval to within 1.5 s (allows the 1 s row granularity).
    Samsung reports each beat ~7 s after it happens: matching blocks of IBIs to the ECG RR sequence
    puts the list SAMSUNG_IBI_LAG_MS late (dev P02-P06 median; 6.3-8.5 s per participant), so all
    positions are moved back by that constant."""
    schema = {"t_end_ms": pl.Float64, "ibi_ms": pl.Float64, "ok": pl.Boolean, "adjacent_next": pl.Boolean}
    if not rec.has_watch_hr:
        return pl.DataFrame(schema=schema)
    ends, vals, oks = [], [], []
    last_end = -np.inf
    for _, grp in rec.hr.group_by("batch", maintain_order=True):
        ibi = [v for lst in grp["ibi"] for v in lst]
        st = [v for lst in grp["ibi_status"] for v in lst]
        if not ibi:
            continue
        ibi_a = np.asarray(ibi, dtype=np.float64)
        t_rows = grp["t_ms"].to_numpy().astype(np.float64)
        t0, t1 = t_rows.min() - 1000.0, t_rows.max()
        cum = np.cumsum(ibi_a)
        if cum[-1] >= t1 - t0:
            e = t1 - (cum[-1] - cum)
        else:
            e = t0 + cum * (t1 - t0) / cum[-1]
        keep = e > last_end
        ends.append(e[keep])
        vals.append(ibi_a[keep])
        oks.append(np.asarray(st)[keep] == 0)
        if keep.any():
            last_end = e[keep].max()
    if not ends:
        return pl.DataFrame(schema=schema)
    e, v, ok = np.concatenate(ends) - SAMSUNG_IBI_LAG_MS, np.concatenate(vals), np.concatenate(oks)
    adj = np.r_[np.abs(np.diff(e) - v[1:]) < 1500.0, False]
    return pl.DataFrame({"t_end_ms": e, "ibi_ms": v, "ok": ok, "adjacent_next": adj})


# --------------------------------------------------------------------------------------------------
# Reference beats from the Polar H10 ECG
# --------------------------------------------------------------------------------------------------

QRS_WIN_MS = (-80.0, 120.0)  # beat snippet for the local template match
TEMPLATE_BEATS = 31  # running-median template over this many neighbouring beats
MIN_TEMPLATE_CORR = 0.6  # beats below this correlate poorly with their neighbours -> invalid
ZHAO_SEG_S = 10.0  # NeuroKit2 zhao2018 quality is computed on 10 s segments
NN_RANGE_MS = (300.0, 2000.0)
NN_MAX_LOCAL_DEV = 0.3  # an NN interval stays within +-30 % of the median of its 11 neighbours
POLAR_CONFIRM_MS = 10.0  # an NN interval is "Polar-confirmed" when Polar's IBI is within this


@dataclass
class Reference:
    """R-peaks on the Polar clock, one row per beat:
    t_ms f64            sub-sample (parabolic) R-peak time
    valid bool          not moved/added by Kubios, template_corr >= 0.6, 10 s segment not
                        "Unacceptable" by NeuroKit2 zhao2018
    corrected bool      moved or inserted by Kubios artifact correction
    template_corr f64   correlation with the running median of 31 neighbouring beats
    zhao str            zhao2018 label of the beat's 10 s segment
    nn bool             the interval ending at this beat is an NN interval (see _nn)
    polar_diff_ms f64   Polar IBI minus that interval (NaN where Polar has no matching interval)"""

    uid: str
    beats: pl.DataFrame
    summary: dict

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.beats["t_ms"].to_numpy(), self.beats["valid"].to_numpy()

    def nn(self) -> np.ndarray:
        """bool per interval (len n_beats - 1): interval i (beat i -> i + 1) is NN."""
        return self.beats["nn"].to_numpy()[1:]


def _refine_peaks(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Local maximum within +-2 samples, then a parabolic fit through it: fractional sample index."""
    n = len(x)
    idx = np.asarray(idx, dtype=np.int64)
    win = np.clip(idx[:, None] + np.arange(-2, 3)[None, :], 0, n - 1)
    m = win[np.arange(len(idx)), np.argmax(x[win], axis=1)]
    m = np.clip(m, 1, n - 2)
    y0, y1, y2 = x[m - 1], x[m], x[m + 1]
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(den < 0, 0.5 * (y0 - y2) / den, 0.0)
    return m + np.clip(delta, -0.5, 0.5)


def _template_corr(x: np.ndarray, idx: np.ndarray, fs: float) -> np.ndarray:
    """Correlation of each beat with the running median of its TEMPLATE_BEATS neighbours."""
    from scipy.ndimage import median_filter

    lo, hi = int(round(QRS_WIN_MS[0] * fs / 1000)), int(round(QRS_WIN_MS[1] * fs / 1000))
    win = np.clip(idx[:, None] + np.arange(lo, hi + 1)[None, :], 0, len(x) - 1)
    snip = x[win]
    tmpl = median_filter(snip, size=(TEMPLATE_BEATS, 1), mode="nearest")
    a = snip - snip.mean(1, keepdims=True)
    b = tmpl - tmpl.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, (a * b).sum(1) / den, 0.0)


def _zhao_segments(x: np.ndarray, fs: float) -> np.ndarray:
    """NeuroKit2 zhao2018 (fuzzy) label for each 10 s segment; the tail joins the last segment."""
    import neurokit2 as nk

    n = int(ZHAO_SEG_S * fs)
    starts = list(range(0, max(len(x) - n, 0) + 1, n)) or [0]
    labels = []
    for k, s in enumerate(starts):
        e = len(x) if k == len(starts) - 1 else s + n
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels.append(str(nk.ecg_quality(x[s:e], sampling_rate=fs, method="zhao2018", approach="fuzzy")))
        except Exception:  # noqa: BLE001  (NeuroKit raises on segments without detectable beats)
            labels.append("Unacceptable")
    return np.array(labels)


def _nn(t: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intervals between consecutive beats (ms) and whether each is an NN interval: both ends
    valid, inside NN_RANGE_MS and within NN_MAX_LOCAL_DEV of the median of its 11 neighbours (catches
    missed/extra beats that Kubios leaves when a whole stretch is noisy)."""
    from scipy.ndimage import median_filter

    ibi = np.diff(t)
    ok = valid[1:] & valid[:-1] & (ibi >= NN_RANGE_MS[0]) & (ibi <= NN_RANGE_MS[1])
    x = np.where(ok, ibi, np.nan)
    # NaN-robust running median: fill gaps with the previous good value before filtering
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = x[idx] if ok.any() else ibi
    filled = np.where(np.isfinite(filled), filled, np.nanmedian(x) if ok.any() else 800.0)
    med = median_filter(filled, size=11, mode="nearest")
    ok &= np.abs(ibi - med) <= NN_MAX_LOCAL_DEV * med
    return ibi, ok


POLAR_BLOCK = 16  # Polar IBIs per matching block
POLAR_SEARCH_BEATS = 150  # +- beats searched around the timestamp-based guess
POLAR_BLOCK_MAX_MEDIAN_MS = 4.0  # a block matches when the median |difference| is below this


def _polar_crosscheck(t_ref: np.ndarray, pibi: pl.DataFrame) -> tuple[np.ndarray, dict]:
    """Match PolarH10/IBI.csv (Polar's on-device RR, 1 ms resolution) to our R-R intervals.

    Polar IBI timestamps are reconstructed by the logger (+-0.3 s jitter, re-anchored every few
    beats, and they lose time where Polar drops beats), so they only give a coarse guess. Durations
    fix the pairing: each block of 16 consecutive Polar IBIs is slid over our intervals within
    +-150 beats of that guess and accepted at the shift with the smallest median |difference| if
    that median is < 4 ms. Returns, per our interval (ending at beat i + 1), Polar minus ours in ms
    (NaN where no accepted block covers it), and a summary."""
    ibi = np.diff(t_ref)
    out = np.full(len(ibi), np.nan)
    p_dur = pibi["ibi_ms"].to_numpy().astype(float)
    p_end = pibi["t_ms"].to_numpy().astype(float) + p_dur
    if len(p_dur) < POLAR_BLOCK or len(ibi) < POLAR_BLOCK:
        return out, {"polar_n": int(len(p_dur))}
    k0 = np.clip(np.searchsorted(t_ref, p_end), 1, len(t_ref) - 1) - 1  # ibi index near each Polar end
    starts = list(range(0, len(p_dur) - POLAR_BLOCK + 1, POLAR_BLOCK))
    if starts[-1] != len(p_dur) - POLAR_BLOCK:
        starts.append(len(p_dur) - POLAR_BLOCK)
    matched = np.zeros(len(p_dur), bool)
    offs = np.arange(POLAR_BLOCK)
    for m0 in starts:
        blk = p_dur[m0 : m0 + POLAR_BLOCK]
        cand = k0[m0] + np.arange(-POLAR_SEARCH_BEATS, POLAR_SEARCH_BEATS + 1)
        cand = cand[(cand >= 0) & (cand + POLAR_BLOCK <= len(ibi))]
        if not len(cand):
            continue
        cost = np.median(np.abs(ibi[cand[:, None] + offs[None, :]] - blk[None, :]), axis=1)
        j = int(np.argmin(cost))
        if cost[j] < POLAR_BLOCK_MAX_MEDIAN_MS:
            idx = cand[j] + offs
            out[idx] = blk - ibi[idx]
            matched[m0 : m0 + POLAR_BLOCK] = True
    in_range = (p_end >= t_ref[0]) & (p_end <= t_ref[-1])
    return out, {
        "polar_n": int(in_range.sum()),
        "polar_matched_frac": float(matched[in_range].mean()) if in_range.any() else 0.0,
        "polar_artifact_ibis": int(((p_dur < 250) | (p_dur > 2500)).sum()),
    }


def _compute_reference(rec: Recording) -> tuple[pl.DataFrame, dict]:
    import neurokit2 as nk

    fs = rec.ecg_fs
    t_ecg = rec.ecg["t_ms"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = nk.ecg_clean(rec.ecg["uv"].to_numpy(), sampling_rate=fs, method="neurokit")
        _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit", correct_artifacts=False)
        raw = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
        art, fixed = nk.signal_fixpeaks(raw, sampling_rate=fs, method="Kubios", iterative=True)
    fixed = np.unique(np.asarray(fixed, dtype=np.int64))
    corrected = ~np.isin(fixed, raw)  # moved (ectopic/long-short) or inserted (missed) by Kubios
    pos = _refine_peaks(clean, fixed).astype(float)
    pos[corrected] = fixed[corrected]  # interpolated positions are not refined against the ECG
    t = np.interp(pos, np.arange(len(t_ecg)), t_ecg)
    corr = _template_corr(clean, fixed, fs)
    zhao = _zhao_segments(clean, fs)
    seg_of_beat = np.minimum((fixed / (ZHAO_SEG_S * fs)).astype(int), len(zhao) - 1)
    beat_zhao = zhao[seg_of_beat]
    valid = (~corrected) & (corr >= MIN_TEMPLATE_CORR) & (beat_zhao != "Unacceptable")
    polar_diff, polar_summary = _polar_crosscheck(t, rec.polar_ibi)
    ibi, nn_ok = _nn(t, valid)
    beats = pl.DataFrame(
        {
            "t_ms": t,
            "valid": valid,
            "corrected": corrected,
            "template_corr": corr,
            "zhao": beat_zhao.astype(str),
            "nn": np.r_[False, nn_ok],  # interval ending at this beat is an NN interval
            "polar_diff_ms": np.r_[np.nan, polar_diff],  # interval ending at this beat, Polar - ours
        }
    )
    pd_nn = polar_diff[nn_ok]
    pd_m = pd_nn[np.isfinite(pd_nn)]
    summary = {
        "uid": rec.uid,
        "ecg_fs_hz": fs,
        "duration_min": float((t_ecg[-1] - t_ecg[0]) / 60000),
        "n_beats": int(len(t)),
        "n_raw_peaks": int(len(raw)),
        "kubios_ectopic": int(len(art.get("ectopic", []))),
        "kubios_missed": int(len(art.get("missed", []))),
        "kubios_extra": int(len(art.get("extra", []))),
        "kubios_longshort": int(len(art.get("longshort", []))),
        "n_corrected": int(corrected.sum()),
        "n_low_template_corr": int((corr < MIN_TEMPLATE_CORR).sum()),
        "zhao_segments": {k: int(v) for k, v in zip(*np.unique(zhao, return_counts=True))},
        "invalid_beat_frac": float(1 - valid.mean()),
        "nn_frac": float(nn_ok.mean()),
        "nn_polar_matched_frac": float(np.isfinite(pd_nn).mean()),
        "nn_polar_confirmed_frac": float((np.abs(pd_nn) <= POLAR_CONFIRM_MS).mean()),
        "nn_polar_mae_ms": float(np.mean(np.abs(pd_m))) if len(pd_m) else None,
        "nn_polar_bias_ms": float(np.mean(pd_m)) if len(pd_m) else None,
        "nn_polar_within_5ms_frac": float(np.mean(np.abs(pd_m) <= 5)) if len(pd_m) else None,
        "hr_median_bpm": float(60000 / np.median(ibi[nn_ok])),
        "rmssd_whole_ms": float(np.sqrt(np.mean(np.diff(ibi)[nn_ok[1:] & nn_ok[:-1]] ** 2))),
    }
    summary.update(polar_summary)
    return beats, summary


def reference(uid: str, refresh: bool = False) -> Reference:
    """ECG reference beats for `uid` (cached)."""
    cdir = _cache_dir(uid) / "reference"
    info = cdir / "info.json"
    if refresh or not info.exists() or json.loads(info.read_text()).get("version") != REFERENCE_VERSION:
        beats, summary = _compute_reference(load(uid))
        _write_atomic(cdir, {"beats": beats}, {"version": REFERENCE_VERSION, **summary})
    summary = json.loads(info.read_text())
    summary.pop("version", None)
    return Reference(uid, pl.read_parquet(cdir / "beats.parquet"), summary)


# --------------------------------------------------------------------------------------------------
# Clock alignment: watch vs Polar
# --------------------------------------------------------------------------------------------------

ALIGN_BAND_HZ = (0.5, 5.0)
ALIGN_MAX_LAG_MS = 240_000.0  # coarse search range
ALIGN_FINE_HALF_MS = 400.0  # fine search +-this around the coarse lag (< half the shortest RR)
ALIGN_CHUNK_MS = 300_000.0
ALIGN_MIN_CHUNK_SNR = 4.0
STRATA_TOL_MS = 150.0


@dataclass
class Alignment:
    """watch_ms = polar_ms + offset_ms + drift_ppm * 1e-6 * (polar_ms - t_ref_ms).

    Method (fixed, independent of any candidate estimator): R-peak-triggered average of the
    band-passed (0.5-5 Hz, zero-phase) first derivative of the 25 Hz watch PPG. The lag that
    maximises it is where, on the watch clock, the steepest systolic upstroke follows an R-peak.
    So offset_ms = clock offset + pulse arrival time to the maximum upslope (~150-300 ms at the
    wrist); the two cannot be separated with this data, and the scorer re-estimates a per-window
    delay anyway. Coarse lag: FFT cross-correlation over +-240 s; fine lag and drift: per 5-min chunk
    (+-400 ms around the coarse lag, 2 ms steps), Theil-Sen line through chunks with SNR >= 4 (the
    drift term also absorbs slow pulse-arrival-time trends). Lobe check (`lobe_ok`): split by RR, the
    chosen lobe must move less than the neighbouring-beat lobes at +-RR. hr_xcorr_offset_ms (Samsung 1 Hz HR vs ECG HR) is reported but
    unreliable: Samsung's HR lags by 4-27 s and is broad, so it only bounds the coarse search."""

    uid: str
    offset_ms: float
    drift_ppm: float
    t_ref_ms: float
    polarity: int  # +1: raw ADC rises with the pulse; -1: ADC falls (upstroke = most negative slope)
    coarse_lag_ms: float
    coarse_snr: float
    coarse_second_ratio: float  # < ~0.7 means the coarse lag is unambiguous
    n_chunks: int
    n_good_chunks: int
    chunk_resid_mad_ms: float | None
    drift_ci_ppm: list[float] | None
    hr_xcorr_offset_ms: float | None
    hr_xcorr_r: float | None
    strata_lag_ms: list[float] | None  # chosen lobe's lag (relative to coarse) for fast / slow beats
    strata_shift_ms: list[float] | None  # slow - fast lag of the k-1, chosen and k+1 lobes
    chunks: list[dict] = field(default_factory=list)

    @property
    def lobe_ok(self) -> bool:
        """The chosen lobe moves least between fast and slow beats (neighbour-beat lobes move by the
        RR difference), and stays within STRATA_TOL_MS of the coarse lag in both."""
        if not self.strata_shift_ms or not self.strata_lag_ms:
            return False
        a = np.abs(np.nan_to_num(np.array(self.strata_shift_ms), nan=np.inf))
        return bool(a[1] < a[0] and a[1] < a[2] and all(abs(x) <= STRATA_TOL_MS for x in self.strata_lag_ms))

    def to_watch(self, t_polar_ms: np.ndarray) -> np.ndarray:
        t = np.asarray(t_polar_ms, dtype=np.float64)
        return t + self.offset_ms + self.drift_ppm * 1e-6 * (t - self.t_ref_ms)


def _ppg_slope(rec: Recording) -> tuple[np.ndarray, np.ndarray]:
    """Uniform 25 Hz grid (watch clock) and the robust-normalised band-passed PPG derivative, zero in
    gaps (> 200 ms) and outside status 0."""
    t, v = rec.ppg_arrays()
    grid = np.arange(t[0], t[-1], 1000.0 / PPG_FS)
    y = np.interp(grid, t, v)
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    gap = (t[j] - t[j - 1]) > 200.0
    sos = signal.butter(2, [ALIGN_BAND_HZ[0], ALIGN_BAND_HZ[1]], "bandpass", fs=PPG_FS, output="sos")
    y = signal.sosfiltfilt(sos, y - np.median(y))
    d = np.gradient(y)
    # robust scale in 10 s blocks so motion bursts do not dominate the average
    blk = int(10 * PPG_FS)
    for s in range(0, len(d), blk):
        seg = d[s : s + blk]
        mad = np.median(np.abs(seg - np.median(seg))) + 1e-12
        d[s : s + blk] = np.clip(seg / (1.4826 * mad), -5, 5)
    d[gap] = 0.0
    return grid, d


def _triggered(grid: np.ndarray, d: np.ndarray, t_beats: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Mean of d(t_beat + lag) over beats inside the grid, for each lag."""
    out = np.empty(len(lags))
    for i, L in enumerate(lags):
        x = t_beats + L
        x = x[(x > grid[0]) & (x < grid[-1])]
        out[i] = np.interp(x, grid, d).mean() if len(x) else np.nan
    return out


def _hr_xcorr(rec: Recording, t_r: np.ndarray, valid: np.ndarray) -> tuple[float | None, float | None]:
    hr = rec.hr.filter((pl.col("status") == 1) & (pl.col("hr") > 0))
    if hr.height < 300:
        return None, None
    wt, wh = hr["t_ms"].to_numpy().astype(float), hr["hr"].to_numpy().astype(float)
    ok = valid[1:] & valid[:-1]
    tb, hb = t_r[1:][ok], 60000.0 / np.diff(t_r)[ok]
    lags = np.arange(-ALIGN_MAX_LAG_MS, ALIGN_MAX_LAG_MS + 1, 250.0)
    best = (None, -np.inf)
    for L in lags:
        he = np.interp(wt - L, tb, hb, left=np.nan, right=np.nan)
        m = np.isfinite(he)
        if m.sum() < 300:
            continue
        r = np.corrcoef(he[m], wh[m])[0, 1]
        if r > best[1]:
            best = (float(L), float(r))
    return best


def _compute_alignment(rec: Recording, ref: Reference) -> Alignment:
    t_r, valid = ref.arrays()
    tv = t_r[valid]
    grid, d = _ppg_slope(rec)
    # coarse: correlate d with an impulse train at R-peaks placed on the same grid (lag in samples)
    step = 1000.0 / PPG_FS
    pos = (tv - grid[0]) / step
    max_lag = int(ALIGN_MAX_LAG_MS / step)
    n = len(grid)
    imp = np.zeros(n + 2 * max_lag + 2)
    p = pos + max_lag
    keep = (p >= 0) & (p < len(imp) - 1)
    i0 = np.floor(p[keep]).astype(int)
    w = p[keep] - i0
    np.add.at(imp, i0, 1 - w)
    np.add.at(imp, i0 + 1, w)
    # c[L] = sum_k d(pos_k + L): correlate d (length n) against the padded impulse train
    c = signal.correlate(d, imp, mode="full", method="fft")
    lag_samples = np.arange(-(len(imp) - 1), n) + max_lag
    sel = np.abs(lag_samples) <= max_lag
    c, lag_samples = c[sel], lag_samples[sel]
    k = int(np.argmax(np.abs(c)))
    polarity = int(np.sign(c[k])) or 1
    coarse = float(lag_samples[k] * step)
    far = np.abs(lag_samples * step - coarse) > 3000
    snr = float(np.abs(c[k]) / (1.4826 * np.median(np.abs(c[far] - np.median(c[far]))) + 1e-12))
    # ambiguity: strongest same-polarity peak more than 300 ms away, relative to the chosen one
    away = np.abs(lag_samples * step - coarse) > 300
    second = float(np.max(polarity * c[away]) / (polarity * c[k])) if away.any() else 0.0

    # fine + drift per chunk
    lags = np.arange(coarse - ALIGN_FINE_HALF_MS, coarse + ALIGN_FINE_HALF_MS + 1e-9, 2.0)
    chunks = []
    for s in np.arange(tv[0], tv[-1], ALIGN_CHUNK_MS):
        tb = tv[(tv >= s) & (tv < s + ALIGN_CHUNK_MS)]
        if len(tb) < 100:
            continue
        prof = polarity * _triggered(grid, d, tb, lags)
        if not np.isfinite(prof).any():
            continue
        kk = int(np.nanargmax(prof))
        # null: the mean of n samples of a unit-scale signal at unrelated times has sd ~ sd(d) / sqrt(n)
        m = (grid >= s) & (grid < s + ALIGN_CHUNK_MS)
        noise = np.std(d[m]) / np.sqrt(len(tb)) if m.any() else np.inf
        chunks.append(
            {
                "t_mid_ms": float(np.median(tb)),
                "lag_ms": float(lags[kk]),
                "peak": float(prof[kk]),
                "snr": float(prof[kk] / (noise + 1e-12)),
                "edge": bool(kk in (0, len(lags) - 1)),
                "n_beats": int(len(tb)),
            }
        )
    good = [c_ for c_ in chunks if c_["snr"] >= ALIGN_MIN_CHUNK_SNR and not c_["edge"]]
    t_ref = float(np.median(tv))
    if len(good) >= 4:
        x = np.array([g["t_mid_ms"] for g in good]) - t_ref
        y = np.array([g["lag_ms"] for g in good])
        slope, icpt, lo, hi = stats.theilslopes(y, x)
        offset, drift = float(icpt), float(slope * 1e6)
        resid = y - (icpt + slope * x)
        mad, ci = float(1.4826 * np.median(np.abs(resid - np.median(resid)))), [float(lo * 1e6), float(hi * 1e6)]
    else:
        prof = polarity * _triggered(grid, d, tv, lags)
        offset, drift, mad, ci = float(lags[int(np.nanargmax(prof))]), 0.0, None, None
    # lobe check: split beats by preceding RR (fastest / slowest third). The pulse of beat k stays at
    # the same lag in both (up to a pulse-arrival-time change); the lobes of beats k-1 / k+1 sit at
    # -RR / +RR and move by the RR difference. The chosen lobe must move least of the three.
    from scipy.signal import find_peaks

    rr = np.r_[np.nan, np.diff(t_r)][valid]
    rr_med = float(np.nanmedian(rr))
    q = np.nanpercentile(rr, [33, 67])
    wide = np.arange(coarse - 2000.0, coarse + 2000.0 + 1e-9, 4.0)
    pos = []
    for sel in (rr < q[0], rr > q[1]):
        prof = polarity * _triggered(grid, d, tv[sel], wide)
        prof = np.nan_to_num(prof, nan=float(np.nanmin(prof)))
        pk, _ = find_peaks(prof, distance=int(0.25 * rr_med / 4.0), prominence=0.1 * float(np.ptp(prof)))
        lag_pk = wide[pk] - coarse
        pos.append([float(lag_pk[np.argmin(np.abs(lag_pk - c_))]) if len(pk) else np.nan for c_ in (-rr_med, 0.0, rr_med)])
    fast, slow = np.array(pos[0]), np.array(pos[1])
    strata_shift = (slow - fast).tolist()  # [k-1 lobe, chosen lobe, k+1 lobe]
    strata = [float(fast[1]), float(slow[1])]
    hr_off, hr_r = _hr_xcorr(rec, t_r, valid)
    return Alignment(
        uid=rec.uid,
        offset_ms=offset,
        drift_ppm=drift,
        t_ref_ms=t_ref,
        polarity=polarity,
        coarse_lag_ms=coarse,
        coarse_snr=snr,
        coarse_second_ratio=second,
        n_chunks=len(chunks),
        n_good_chunks=len(good),
        chunk_resid_mad_ms=mad,
        drift_ci_ppm=ci,
        hr_xcorr_offset_ms=hr_off,
        hr_xcorr_r=hr_r,
        strata_lag_ms=strata,
        strata_shift_ms=strata_shift,
        chunks=chunks,
    )


def alignment(uid: str, refresh: bool = False) -> Alignment:
    """Watch-vs-Polar clock alignment for `uid` (cached as JSON). Raises if there is no watch PPG."""
    path = _cache_dir(uid) / "alignment.json"
    if refresh or not path.exists() or json.loads(path.read_text()).get("version") != ALIGNMENT_VERSION:
        rec = load(uid)
        if not rec.has_watch:
            raise ValueError(f"{uid}: no Galaxy Watch data")
        al = _compute_alignment(rec, reference(uid))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"version": ALIGNMENT_VERSION, **asdict(al)}, indent=1))
        os.replace(tmp, path)
    d = json.loads(path.read_text())
    d.pop("version", None)
    return Alignment(**d)


def prepare(uid: str, refresh: bool = False) -> dict:
    """Load + reference + alignment for one participant; returns a flat summary row."""
    rec = load(uid, refresh=refresh)
    ref = reference(uid, refresh=refresh)
    row = {k: v for k, v in ref.summary.items() if not isinstance(v, dict)}
    row["zhao_unacceptable_segments"] = ref.summary["zhao_segments"].get("Unacceptable", 0)
    row["has_watch"], row["has_watch_hr"] = rec.has_watch, rec.has_watch_hr
    if rec.has_watch:
        al = alignment(uid, refresh=refresh)
        row.update(
            {
                "offset_ms": al.offset_ms,
                "drift_ppm": al.drift_ppm,
                "align_snr": al.coarse_snr,
                "align_second_ratio": al.coarse_second_ratio,
                "good_chunks": f"{al.n_good_chunks}/{al.n_chunks}",
                "chunk_resid_mad_ms": al.chunk_resid_mad_ms,
                "hr_xcorr_offset_ms": al.hr_xcorr_offset_ms,
                "polarity": al.polarity,
                "strata_lag_ms": ",".join(f"{x:+.0f}" for x in al.strata_lag_ms or []),
                "strata_shift_ms": ",".join(f"{x:+.0f}" for x in al.strata_shift_ms or []),
                "lobe_ok": al.lobe_ok,
            }
        )
    return row


def prepare_all(uids: tuple[str, ...] | list[str] = UIDS, refresh: bool = False, workers: int | None = None) -> pl.DataFrame:
    """prepare() for many participants in parallel; one summary row each."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    uids = list(uids)
    if workers == 1 or len(uids) == 1:
        rows = [prepare(u, refresh) for u in uids]
    else:
        with ProcessPoolExecutor(max_workers=workers or min(len(uids), os.cpu_count() or 4), mp_context=mp.get_context("spawn")) as ex:
            rows = list(ex.map(prepare, uids, [refresh] * len(uids)))
    return pl.DataFrame(rows, infer_schema_length=None)


if __name__ == "__main__":
    import time

    t0 = time.time()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    df = prepare_all(args or UIDS, refresh="--refresh" in sys.argv)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(df)
    print(f"{time.time() - t0:.1f} s")
