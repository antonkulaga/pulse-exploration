"""E0: format stress tests, a fair wrist check and a polarity detector, with frozen encoders and existing probes.

From the generalization strategy (`docs/ppg-generalization-strategy.md`). Nothing is trained; the question is
how far the existing readouts move when only the input format changes (E0a), how much of the wrist failure is
device, state or site (E0b), and whether polarity can be detected instead of hard-coded per source (E0c).

**E0a windows** (:func:`build_stress`). The same 30 s windows as the stored VitalDB variant, for test people,
rebuilt from the raw 500 Hz PLETH so each perturbation acts where a device would:

1. The raw track is gap-filled (as kron-ppg does), resampled to 25 Hz *without* band-pass (polyphase
   anti-aliasing only) and cut with 10 s of lead-in and 2 s of tail.
2. A perturbation is applied to that unfiltered 25 Hz signal.
3. The band-pass follows: zero-phase (the reference) or causal (``causal``). Lead-in and tail are dropped,
   and the window is robust-scaled.

``stored`` is the window exactly as kron-ppg prepared it (band-pass at 500 Hz before resampling); comparing
``ref`` with it shows what filtering at 25 Hz alone changes.

**E0b windows** (:func:`build_galaxy`). From GalaxyPPG:

- Galaxy Watch5 green PPG, inverted, placed on a uniform 25 Hz grid by its own timestamps.
- Empatica E4 BVP (64 Hz), aligned to the watch per person by cross-correlation and decimated to 25 Hz.
- 30 s windows on the same seconds, in the rest sessions (``baseline``, ``rest-1..3``) and in all sessions;
  both devices go through the reference preparation.

**E0c** (:func:`polarity`). The rule is upstroke shorter than downstroke on the average beat (rise time
under half the beat period), with skewness as the tie-breaker. It runs on every window as given and flipped,
and is scored against the known orientation.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import butter, find_peaks, resample_poly, sosfilt, sosfiltfilt
from scipy.stats import skew

FS = 25.0
SECONDS = 30.0
LEAD_S, TAIL_S = 10.0, 2.0
BAND = (0.5, 8.0)
HR_RANGE_BPM = (30.0, 200.0)
PERTURBATIONS = ("ref", "causal", "lowpass4", "lowpass6", "invert", "drop", "resample64", "resample20",
                 "interp64", "gainstep", "noise20db", "noise10db", "wander", "phase")
REST_SESSIONS = ("baseline", "rest-1", "rest-2", "rest-3")


# ---- the preparation, as kron_ppg.signal does it -------------------------------------------------
def _sos(fs: float):
    return butter(2, list(BAND), btype="bandpass", fs=fs, output="sos")


def bandpass_zero(x: np.ndarray, fs: float) -> np.ndarray:
    sos = _sos(fs)
    padlen = min(x.shape[-1] - 1, 3 * (2 * len(sos) + 1) * 4)
    return sosfiltfilt(sos, x, axis=-1, padlen=padlen)


def bandpass_causal(x: np.ndarray, fs: float) -> np.ndarray:
    return sosfilt(_sos(fs), x, axis=-1)


def resample(x: np.ndarray, fs: float, target: float) -> np.ndarray:
    if abs(fs - target) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    r = Fraction(target / fs).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), r.numerator, r.denominator, axis=-1)


def robust_scale(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    return (x - med) / max((q75 - q25) / 1.349, 1e-9)


def fill_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if not bad.any():
        return x, np.ones(len(x), bool)
    idx = np.arange(len(x))
    good = ~bad
    if good.sum() < 2:
        return np.zeros_like(x), np.zeros(len(x), bool)
    filled = np.interp(idx, idx[good], x[good])
    valid = good.copy()
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a <= max_gap and a > 0 and b < len(x):
            valid[a:b] = True
    filled[~valid] = 0.0
    return filled, valid


def periodicity(seg: np.ndarray, fs: float = FS) -> tuple[float, float]:
    """``(hr_bpm, periodicity)``: kron-ppg's autocorrelation gate statistic."""
    x = seg - seg.mean()
    lo = int(fs * 60.0 / HR_RANGE_BPM[1])
    hi = min(int(fs * 60.0 / HR_RANGE_BPM[0]), len(x) // 2)
    spec = np.fft.rfft(x, n=2 * len(x))
    ac = np.fft.irfft(spec * np.conj(spec))[:len(x)]
    ac = ac / max(ac[0], 1e-12)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return 60.0 * fs / lag, float(ac[lag])


# ---- perturbations on the unfiltered 25 Hz signal ------------------------------------------------
def _frac_delay(x: np.ndarray, d: float) -> np.ndarray:
    n = len(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * f * d), n=n)


def perturb_raw(seg: np.ndarray, name: str, rng: np.random.Generator, ref_power: float) -> np.ndarray:
    """``seg`` is unfiltered 25 Hz signal with lead-in; the window proper is its last 30 s + tail."""
    n = len(seg)
    lead = int(LEAD_S * FS)
    size = int(SECONDS * FS)
    if name in ("ref", "causal"):
        return seg
    if name.startswith("lowpass"):
        sos = butter(4, float(name.removeprefix("lowpass")), btype="lowpass", fs=FS, output="sos")
        return sosfilt(sos, seg)
    if name == "invert":
        return -seg
    if name == "drop":  # one sample lost every 250-300 (10-12 s), timestamps kept even: the rest shifts left
        keep = np.ones(n, bool)
        t = int(rng.integers(0, 250))
        while t < n:
            keep[t] = False
            t += int(rng.integers(250, 301))
        kept = seg[keep]
        return np.concatenate([kept, np.full(n - len(kept), kept[-1])])
    if name == "resample64":
        return resample(resample(seg, FS, 64.0), 64.0, FS)[:n]
    if name == "resample20":
        return resample(resample(seg, FS, 20.0), 20.0, FS)[:n]
    if name == "interp64":
        t25 = np.arange(n) / FS
        t64 = np.arange(int(n * 64 / FS)) / 64.0
        return np.interp(t25, t64, np.interp(t64, t25, seg))
    if name == "gainstep":  # an AGC / LED-current step half-way through the window
        y = seg.copy()
        y[lead + size // 2:] *= 1.5
        return y
    if name.startswith("noise"):
        snr_db = float(name.removeprefix("noise").removesuffix("db"))
        return seg + rng.normal(size=n) * np.sqrt(ref_power / 10 ** (snr_db / 10))
    if name == "wander":
        t = np.arange(n) / FS
        q75, q25 = np.percentile(seg[lead:lead + size], [75, 25])
        return seg + (q75 - q25) * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 2 * np.pi))
    if name == "phase":
        return _frac_delay(seg, 0.5)
    raise ValueError(name)


def finish(seg: np.ndarray, causal: bool) -> np.ndarray:
    """Band-pass (zero-phase, or causal for ``causal``), drop lead-in and tail, robust-scale."""
    y = bandpass_causal(seg, FS) if causal else bandpass_zero(seg, FS)
    lead = int(LEAD_S * FS)
    return robust_scale(y[lead:lead + int(SECONDS * FS)])


# ---- E0a: VitalDB test windows ------------------------------------------------------------------
def _read_track(path: Path) -> np.ndarray:
    df = pl.read_csv(path, schema_overrides={"SNUADC/PLETH": pl.Float64})
    return df.get_column("SNUADC/PLETH").fill_null(np.nan).to_numpy().astype(np.float64)


def _stress_recording(args) -> list[dict]:
    path, recording, items, seed = args
    raw = _read_track(path)
    fs = 500.0
    x, valid = fill_gaps(raw, int(round(0.5 * fs)))
    if valid.sum() < 2 * fs:
        return []
    x = x - np.median(x[valid])
    x[~valid] = 0.0
    stored_like = resample(bandpass_zero(x, fs), fs, FS)  # kron-ppg's order: band-pass, then resample
    x25 = resample(x, fs, FS)                              # the device's view: no band-pass yet
    lead, size, tail = int(LEAD_S * FS), int(SECONDS * FS), int(TAIL_S * FS)
    out = []
    for it in items:
        s = int(round(it["start_s"] * FS))
        if s - lead < 0 or s + size + tail > len(x25):
            continue
        seg = x25[s - lead:s + size + tail]
        ref_power = float(np.var(bandpass_zero(seg, FS)[lead:lead + size]))
        rng = np.random.default_rng(seed + s)
        windows = {"stored_rebuilt": robust_scale(stored_like[s:s + size])}
        for p in PERTURBATIONS:
            windows[p] = finish(perturb_raw(seg, p, rng, ref_power), causal=(p == "causal"))
        for p, w in windows.items():
            hr, per = periodicity(w)
            out.append({"perturbation": p, "subject": it["subject"], "recording": recording,
                        "start_s": it["start_s"], "hr_bpm": hr, "periodicity": per,
                        "signal": w.astype(np.float32)})
    return out


def _write_variant(folder: Path, rows: list[dict], subjects: pl.DataFrame, source: str, note: str) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    size = int(SECONDS * FS)
    table = pl.DataFrame({
        "subject": [r["subject"] for r in rows], "recording": [r["recording"] for r in rows],
        "start_s": [float(r["start_s"]) for r in rows], "hr_bpm": pl.Series([r["hr_bpm"] for r in rows], dtype=pl.Float32),
        "periodicity": pl.Series([r["periodicity"] for r in rows], dtype=pl.Float32),
        "signal": pl.Series(np.stack([r["signal"] for r in rows]).astype(np.float16)).cast(pl.Array(pl.Float16, size)),
    })
    table.write_parquet(folder / "windows.parquet", compression="zstd")
    subjects.write_parquet(folder / "subjects.parquet")
    (folder / "info.json").write_text(json.dumps({"source": source, "sample_rate": FS, "seconds": SECONDS,
                                                  "windows": len(table), "note": note}, indent=2) + "\n")
    return len(table)


def build_stress(variant: Path, pleth: Path, out: Path, per_subject: int = 24, workers: int = 8,
                 split: str = "test", seed: int = 0) -> dict:
    from .data import load_variant

    v = load_variant(variant, per_subject)
    rows = v.rows.filter(pl.col("split") == split)
    keep = rows.with_row_index("i")
    stored = {(r["recording"], r["start_s"]): v.x[i] for r, i in
              zip(rows.iter_rows(named=True), v.rows.with_row_index("i").filter(pl.col("split") == split)["i"])}
    tasks = []
    for (recording,), g in keep.group_by(["recording"], maintain_order=True):
        path = pleth / f"{int(recording):04d}.csv.gz"
        tasks.append((path, recording, g.select("subject", "start_s").to_dicts(), seed))
    rows_by_p: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(_stress_recording, tasks, chunksize=2):
            for r in res:
                rows_by_p.setdefault(r["perturbation"], []).append(r)
    subjects = pl.read_parquet(variant / "subjects.parquet")
    counts = {}
    for p, rs in rows_by_p.items():
        counts[p] = _write_variant(out / p, rs, subjects, v.source, f"E0a {p} from {variant}")
    # the stored windows themselves, for the same (recording, start) pairs
    st = [{"subject": r["subject"], "recording": r["recording"], "start_s": r["start_s"],
           "signal": stored[(r["recording"], r["start_s"])].astype(np.float32)} for r in rows_by_p.get("ref", [])]
    for r in st:
        r["hr_bpm"], r["periodicity"] = periodicity(r["signal"])
    counts["stored"] = _write_variant(out / "stored", st, subjects, v.source, f"E0a stored windows of {variant}")
    agree = np.array([np.corrcoef(a["signal"], b["signal"])[0, 1] for a, b in zip(st, rows_by_p.get("ref", []))])
    info = {"variant": str(variant), "split": split, "per_subject": per_subject, "windows": counts,
            "recordings": len(tasks), "ref_vs_stored_r_median": float(np.median(agree)) if len(agree) else None}
    (out / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


# ---- E0b: GalaxyPPG, watch and E4 on the same seconds ------------------------------------------
def _xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """Lag (samples) that best aligns ``b`` to ``a`` (b shifted later by the lag)."""
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    best, lag = -np.inf, 0
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            c = np.dot(a[k:], b[:n - k])
        else:
            c = np.dot(a[:n + k], b[-k:])
        if c > best:
            best, lag = c, k
    return lag


def _galaxy_person(args) -> tuple[list[dict], dict]:
    root, uid = args
    d = Path(root) / uid
    if not (d / "GalaxyWatch" / "PPG.csv").exists() or not (d / "E4" / "BVP.csv").exists() or not (d / "Event.csv").exists():
        return [], {"uid": uid, "skipped": "missing files"}
    w = pl.read_csv(d / "GalaxyWatch" / "PPG.csv").select("timestamp", "ppg").drop_nulls()
    e = pl.read_csv(d / "E4" / "BVP.csv").select("timestamp", "value").drop_nulls()
    ev = pl.read_csv(d / "Event.csv")
    tw = w["timestamp"].to_numpy() / 1000.0
    xw = -w["ppg"].to_numpy().astype(np.float64)       # inverted: systolic peak up
    te = e["timestamp"].to_numpy() / 1e6
    xe = e["value"].to_numpy().astype(np.float64)
    t0, t1 = max(tw[0], te[0]), min(tw[-1], te[-1])
    if t1 - t0 < 120:
        return [], {"uid": uid, "skipped": "no overlap"}
    grid = np.arange(t0, t1, 1 / FS)
    w25 = np.interp(grid, tw, xw)
    e64 = np.interp(np.arange(te[0], te[-1], 1 / 64.0), te, xe)
    e25_full = resample(e64, 64.0, FS)
    te25 = te[0] + np.arange(len(e25_full)) / FS
    e25 = np.interp(grid, te25, e25_full)
    # clock offset of the E4 against the watch, from the band-passed pulse over the whole overlap
    lag = _xcorr_lag(bandpass_zero(w25, FS), bandpass_zero(e25, FS), int(20 * FS))
    e25 = np.interp(grid, te25 + lag / FS, e25_full)
    sessions = []
    names = ev["session"].to_list()
    stamps = ev["timestamp"].to_numpy() / 1000.0
    status = ev["status"].to_list()
    for i, (nm, st) in enumerate(zip(names, status)):
        if st != "ENTER":
            continue
        end = next((stamps[j] for j in range(i + 1, len(names)) if names[j] == nm and status[j] == "EXIT"), None)
        if end is not None:
            sessions.append((nm, stamps[i], end))
    size, lead, tail = int(SECONDS * FS), int(LEAD_S * FS), int(TAIL_S * FS)
    rows = []
    for nm, a, b in sessions:
        s = int(np.ceil((a - t0) * FS))
        while s + size <= int((b - t0) * FS):
            if s - lead >= 0 and s + size + tail <= len(grid):
                ww = finish(w25[s - lead:s + size + tail], False)
                we = finish(e25[s - lead:s + size + tail], False)
                hw, pw = periodicity(ww)
                he, pe = periodicity(we)
                for dev, sig, hr, per in (("watch", ww, hw, pw), ("e4", we, he, pe)):
                    rows.append({"device": dev, "session": nm, "rest": nm in REST_SESSIONS, "subject": uid,
                                 "recording": uid, "start_s": round(float(grid[s] - t0), 2), "hr_bpm": hr,
                                 "periodicity": per, "signal": sig.astype(np.float32), "pair_ok": pw >= 0.35 and pe >= 0.35})
            s += size
    return rows, {"uid": uid, "e4_lag_s": lag / FS, "sessions": len(sessions), "windows": len(rows) // 2}


def build_galaxy(root: Path, out: Path, workers: int = 8) -> dict:
    """``root`` is GalaxyPPG's unpacked ``Dataset`` folder."""
    meta = pl.read_csv(root / "Meta.csv", infer_schema_length=0)
    uids = meta["UID"].to_list()
    rows, info = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r, i in pool.map(_galaxy_person, [(str(root), u) for u in uids]):
            rows += r
            info.append(i)
    subjects = meta.select(pl.col("UID").alias("subject"), pl.col("AGE").cast(pl.Float64).alias("age"),
                           pl.col("GENDER").alias("sex"))
    counts = {}
    paired = {(r["subject"], r["start_s"]) for r in rows if r["pair_ok"]}
    for name, keep in (("rest", lambda r: r["rest"]), ("all", lambda r: True)):
        for dev in ("watch", "e4"):
            sel = [r for r in rows if r["device"] == dev and keep(r) and (r["subject"], r["start_s"]) in paired]
            if sel:
                counts[f"galaxy-{name}-{dev}"] = _write_variant(out / f"galaxy-{name}-{dev}", sel, subjects, "galaxyppg",
                                                                f"E0b {name} sessions, {dev}, windows passing on both devices")
    res = {"people": info, "windows": counts, "rest_sessions": list(REST_SESSIONS)}
    (out / "build-galaxy.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


# ---- E0c: polarity --------------------------------------------------------------------------------
def polarity_up(seg: np.ndarray, fs: float = FS) -> bool | None:
    """True when the window's systolic upstroke points up; None when no beats are found."""
    hr, _ = periodicity(seg, fs)
    period = 60.0 / hr * fs
    peaks, _ = find_peaks(seg, distance=max(1, int(0.6 * period)), prominence=0.3 * np.std(seg))
    if len(peaks) < 3:
        return None
    rises = []
    for p in peaks:
        a = max(0, int(p - 0.8 * period))
        if p - a < 2:
            continue
        foot = a + int(np.argmin(seg[a:p]))
        rises.append(p - foot)
    if not rises:
        return None
    ratio = float(np.median(rises)) / period
    if abs(ratio - 0.5) < 0.05:
        return bool(skew(seg) > 0)
    return ratio < 0.5


def polarity(windows: dict[str, np.ndarray], up: dict[str, bool], groups: dict[str, np.ndarray] | None = None) -> dict:
    """Accuracy of :func:`polarity_up` per source on every window as given and flipped, and of a majority
    vote over each person's windows (polarity is a property of a device and placement, not of a window)."""
    out = {}
    for name, x in windows.items():
        res = {"windows": int(len(x))}
        for flip in (False, True):
            truth = up[name] != flip
            calls = np.array([polarity_up(-w if flip else w) for w in x], dtype=object)
            ok = np.array([c is not None for c in calls])
            hit = np.array([c == truth for c in calls[ok]])
            res[f"{'flipped' if flip else 'as_given'}_window_accuracy"] = round(float(hit.mean()), 4) if ok.any() else None
            res[f"{'flipped' if flip else 'as_given'}_undecided"] = int((~ok).sum())
            if groups is not None:
                g = groups[name]
                votes = {}
                for k, c in zip(g[ok], calls[ok]):
                    votes.setdefault(k, []).append(c == truth)
                res[f"{'flipped' if flip else 'as_given'}_person_accuracy"] = round(float(np.mean([np.mean(v) > 0.5 for v in votes.values()])), 4)
                res["people"] = len(votes)
        out[name] = res
    return out


# ---- scoring ----------------------------------------------------------------------------------------
def _person_scores(windows: pl.DataFrame, models: dict, features: str) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame]:
    from .probe import apply_scores, subject_features
    table, x = subject_features(windows, features)
    return apply_scores(table, x, models), x, table


def _embed_matrix(windows: pl.DataFrame, features: str) -> np.ndarray:
    if features == "hr":
        return windows.select("hr_bpm", "periodicity").to_numpy().astype(np.float64)
    return windows.get_column("emb").to_numpy().astype(np.float64)


def score_encoder(name: str, clean: Path, emb: Path, cases: Path | None, features: str = "emb") -> dict:
    """E0a and E0b for one encoder. ``clean`` holds its VitalDB embeddings (the probe's training set);
    ``emb`` its embeddings of the E0 windows (``vitaldb-<perturbation>``, ``galaxyppg-galaxy-*``)."""
    from .labels import label_table
    from .probe import _auc, fit_scores, subject_features

    stem = "vitaldb-fs25-w30s"
    clean_w = pl.read_parquet(clean / f"{stem}.parquet")
    labels = label_table(pl.read_parquet(clean / f"{stem}.subjects.parquet"), cases)
    table, x = subject_features(clean_w, features)
    _, _, models = fit_scores(table, x, labels)
    scaler = models["age"][0]
    out: dict = {"encoder": name, "stress": {}, "galaxy": {}}

    def load(stem_: str) -> pl.DataFrame | None:
        f = emb / f"{stem_}.parquet"
        return pl.read_parquet(f) if f.exists() else None

    ref_w = load("vitaldb-ref")
    if ref_w is not None:
        ref_s, ref_x, _ = _person_scores(ref_w, models, features)
        lab = ref_s.select("key").join(labels, on="key", how="left", maintain_order="left")
        zr = scaler.transform(ref_x)
        rng = np.random.default_rng(0)
        i, j = rng.integers(0, len(zr), 4000), rng.integers(0, len(zr), 4000)
        between = float(np.median(np.linalg.norm(zr[i] - zr[j], axis=1)[i != j]))
        # window-level distances need window-level features; the hr baseline only has person-level ones
        ref_win = scaler.transform(_embed_matrix(ref_w, features)) if features == "emb" else None
        ref_key = ref_w.select("key", "start_s").with_row_index("ri")

        def readout(scores: pl.DataFrame) -> dict:
            r = {}
            for t in ("ckd", "asa_ge3", "preop_htn"):
                if f"score_{t}" in scores.columns:
                    r[t] = _auc(lab.get_column(t).to_numpy(), scores.get_column(f"score_{t}").to_numpy())
            if "score_bmi" in scores.columns:
                y, sc = lab.get_column("bmi").to_numpy(), scores.get_column("score_bmi").to_numpy()
                ok = np.isfinite(y) & np.isfinite(sc)
                r["bmi_r2"] = float(np.corrcoef(y[ok], sc[ok])[0, 1] ** 2)
            age = lab.get_column("age").to_numpy()
            r["age_mae"] = float(np.nanmean(np.abs(scores.get_column("score_age").to_numpy() - age)))
            return r

        base = readout(ref_s)
        for p in ("stored", "stored_rebuilt") + PERTURBATIONS:
            w = load(f"vitaldb-{p}")
            if w is None:
                continue
            s_, x_, _ = _person_scores(w, models, features)
            j_ = ref_s.select("key", "score_age").join(s_.select("key", "score_age"), on="key", suffix="_p")
            d = (j_["score_age_p"] - j_["score_age"]).to_numpy()
            s_ = ref_s.select("key").join(s_, on="key", how="left", maintain_order="left")
            r = readout(s_)
            if ref_win is not None:
                m = w.select("key", "start_s").with_row_index("pi").join(ref_key, on=["key", "start_s"])
                zp = scaler.transform(_embed_matrix(w, features))
                dist = np.linalg.norm(zp[m["pi"].to_numpy()] - ref_win[m["ri"].to_numpy()], axis=1)
            else:
                dist = np.array([np.nan])
            out["stress"][p] = {
                "people": int(len(d)), "mean_abs_delta_age": round(float(np.mean(np.abs(d))), 3),
                "mean_delta_age": round(float(np.mean(d)), 3), "age_mae": round(r["age_mae"], 3),
                **{f"delta_auc_{t}": round(r[t] - base[t], 4) for t in ("ckd", "asa_ge3", "preop_htn") if t in r},
                **({"delta_r2_bmi": round(r["bmi_r2"] - base["bmi_r2"], 4)} if "bmi_r2" in r else {}),
                "sensitivity_ratio": round(float(np.median(dist)) / between, 4),
            }

    # E0b
    young = None
    if ref_w is not None:
        lab_ref = ref_s.select("key", "score_age").join(labels.select("key", "age"), on="key")
        young = lab_ref.filter(pl.col("age") < 30)
        out["galaxy"]["vitaldb_test_under30"] = {
            "people": len(young), "mean_age": round(float(young["age"].mean()), 2) if len(young) else None,
            "mean_predicted": round(float(young["score_age"].mean()), 2) if len(young) else None}
    per_dev = {}
    for name_ in ("galaxy-rest-watch", "galaxy-rest-e4", "galaxy-all-watch", "galaxy-all-e4"):
        w = load(f"galaxyppg-{name_}")
        if w is None:
            continue
        s_, x_, t_ = _person_scores(w, models, features)
        glab = label_table(pl.read_parquet(emb / f"galaxyppg-{name_}.subjects.parquet"), None)
        j_ = s_.select("key", "score_age").join(glab.select("key", "age"), on="key")
        per_dev[name_] = (j_, scaler.transform(x_), t_.get_column("key").to_numpy())
        out["galaxy"][name_] = {"people": len(j_), "mean_age": round(float(j_["age"].mean()), 2),
                                "mean_predicted": round(float(j_["score_age"].mean()), 2),
                                "mae": round(float((j_["score_age"] - j_["age"]).abs().mean()), 2)}
    if "galaxy-rest-watch" in per_dev and "galaxy-rest-e4" in per_dev:
        a, za, ka = per_dev["galaxy-rest-watch"]
        b, zb, kb = per_dev["galaxy-rest-e4"]
        j_ = a.join(b, on="key", suffix="_e4")
        out["galaxy"]["rest_watch_vs_e4"] = {"people": len(j_), "icc_predicted_age": round(_icc(j_["score_age"].to_numpy(), j_["score_age_e4"].to_numpy()), 4),
                                             "mean_difference": round(float((j_["score_age"] - j_["score_age_e4"]).mean()), 2)}
        common = [k for k in ka if k in set(kb)]
        ia = {k: i for i, k in enumerate(ka)}
        ib = {k: i for i, k in enumerate(kb)}
        same = [_cos(za[ia[k]], zb[ib[k]]) for k in common]
        other = [_cos(za[ia[k]], zb[ib[l]]) for k in common for l in common if k != l]
        out["galaxy"]["rest_watch_vs_e4"].update(cosine_same_person=round(float(np.mean(same)), 4),
                                                 cosine_other_person=round(float(np.mean(other)), 4))
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _icc(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1), absolute agreement, two raters."""
    x = np.column_stack([a, b])
    n, k = x.shape
    grand = x.mean()
    msr = k * np.sum((x.mean(1) - grand) ** 2) / (n - 1)
    msc = n * np.sum((x.mean(0) - grand) ** 2) / (k - 1)
    sse = np.sum((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def summarize(results: Path) -> str:
    rows = [json.loads(f.read_text()) for f in sorted(results.glob("*.json"))]
    rows = [r for r in rows if "encoder" in r]
    lines = ["## E0a: format stress test (VitalDB test people; probes fitted on clean train embeddings, not refitted)", "",
             "Δ is against `ref` (the same windows, zero-phase band-pass at 25 Hz). Flag: mean |Δ age| > 2 y or "
             "sensitivity ratio > 0.3.", "",
             "| encoder | perturbation | mean abs Δ age | mean Δ age | Δ AUC CKD | Δ AUC ASA≥3 | Δ AUC HTN | Δ R² BMI | sensitivity ratio |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for p, m in r["stress"].items():
            flag = " ⚑" if m["mean_abs_delta_age"] > 2 or (m["sensitivity_ratio"] or 0) > 0.3 else ""
            lines.append(f"| {r['encoder']} | {p}{flag} | {m['mean_abs_delta_age']:.2f} | {m['mean_delta_age']:+.2f} | "
                         f"{m.get('delta_auc_ckd', float('nan')):+.3f} | {m.get('delta_auc_asa_ge3', float('nan')):+.3f} | "
                         f"{m.get('delta_auc_preop_htn', float('nan')):+.3f} | {m.get('delta_r2_bmi', float('nan')):+.3f} | "
                         f"{m['sensitivity_ratio']:.3f} |")
    lines += ["", "## E0b: GalaxyPPG at rest, Galaxy Watch5 vs Empatica E4 on the same seconds", "",
              "| encoder | VitalDB test <30: true / predicted | rest watch: predicted (MAE) | rest E4: predicted (MAE) "
              "| all watch: predicted | ICC watch vs E4 | cosine same / other person |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        g = r["galaxy"]
        yv = g.get("vitaldb_test_under30", {})
        rw, re_, aw = g.get("galaxy-rest-watch", {}), g.get("galaxy-rest-e4", {}), g.get("galaxy-all-watch", {})
        c = g.get("rest_watch_vs_e4", {})
        lines.append(f"| {r['encoder']} | {yv.get('mean_age', float('nan')):.1f} / {yv.get('mean_predicted', float('nan')):.1f} (n={yv.get('people', 0)}) "
                     f"| {rw.get('mean_predicted', float('nan')):.1f} ({rw.get('mae', float('nan')):.1f}) "
                     f"| {re_.get('mean_predicted', float('nan')):.1f} ({re_.get('mae', float('nan')):.1f}) "
                     f"| {aw.get('mean_predicted', float('nan')):.1f} | {c.get('icc_predicted_age', float('nan')):.2f} "
                     f"| {c.get('cosine_same_person', float('nan')):.2f} / {c.get('cosine_other_person', float('nan')):.2f} |")
    pol = results / "polarity.json"
    if pol.exists():
        lines += ["", "## E0c: polarity detector (rise time < half the beat period; skewness tie-break)", "",
                  "| source | windows | as given | flipped | person vote (as given / flipped) |", "|---|---|---|---|---|"]
        for name, m in json.loads(pol.read_text()).items():
            lines.append(f"| {name} | {m['windows']} | {m['as_given_window_accuracy']} | {m['flipped_window_accuracy']} "
                         f"| {m.get('as_given_person_accuracy')} / {m.get('flipped_person_accuracy')} |")
    return "\n".join(lines) + "\n"


def run_polarity(stress_windows: Path, galaxy_windows: Path, bidmc: Path | None, out: Path) -> dict:
    """E0c over the stored VitalDB test windows, BIDMC, and GalaxyPPG's watch (inverted: up) and E4 windows.

    The E4's true orientation is not documented, so it is taken from its correlation with the inverted
    watch signal over the same seconds (positive: up).
    """
    from .data import load_variant
    windows, up, groups = {}, {}, {}
    for name, folder, is_up in (("vitaldb (stored)", stress_windows / "stored", True),
                                ("bidmc", bidmc, True),
                                ("galaxy watch (inverted)", galaxy_windows / "galaxy-all-watch", True)):
        if folder is None or not (folder / "windows.parquet").exists():
            continue
        v = load_variant(folder, 0)
        windows[name], up[name], groups[name] = v.x, is_up, v.rows.get_column("key").to_numpy()
    e4 = galaxy_windows / "galaxy-all-e4"
    if e4.exists() and "galaxy watch (inverted)" in windows:
        v = load_variant(e4, 0)
        w = load_variant(galaxy_windows / "galaxy-all-watch", 0)
        r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(v.x, w.x)])
        windows["galaxy e4"], up["galaxy e4"], groups["galaxy e4"] = v.x, bool(np.nanmedian(r) > 0), v.rows.get_column("key").to_numpy()
        e4_corr = float(np.nanmedian(r))
    else:
        e4_corr = None
    res = polarity(windows, up, groups)
    res["_e4_truth"] = {"median_r_with_inverted_watch": e4_corr, "up": up.get("galaxy e4")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "polarity.json").write_text(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=2) + "\n")
    (out / "polarity-e4-truth.json").write_text(json.dumps(res["_e4_truth"], indent=2) + "\n")
    return res


def main(argv: list[str] | None = None) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="fm_baselines.e0")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build-stress")
    a.add_argument("variant", type=Path)
    a.add_argument("--pleth", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--per-subject", type=int, default=24)
    a.add_argument("--workers", type=int, default=8)
    b = sub.add_parser("build-galaxy")
    b.add_argument("root", type=Path, help="GalaxyPPG's unpacked Dataset folder")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--workers", type=int, default=8)
    c = sub.add_parser("polarity")
    c.add_argument("--stress", type=Path, required=True)
    c.add_argument("--galaxy", type=Path, required=True)
    c.add_argument("--bidmc", type=Path, default=None)
    c.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score", help="E0a + E0b for every encoder with embeddings under --emb")
    s.add_argument("--clean", type=Path, required=True, help="folder of <encoder>/vitaldb-fs25-w30s.parquet")
    s.add_argument("--emb", type=Path, required=True, help="folder of <encoder>/ embeddings of the E0 windows")
    s.add_argument("--cases", type=Path, default=None)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--encoders", default="", help="comma-separated subset")
    m = sub.add_parser("summary")
    m.add_argument("results", type=Path)
    x = p.parse_args(argv)
    if x.cmd == "build-stress":
        print(json.dumps(build_stress(x.variant, x.pleth, x.out, x.per_subject, x.workers), indent=2))
    elif x.cmd == "build-galaxy":
        print(json.dumps(build_galaxy(x.root, x.out, x.workers), indent=2))
    elif x.cmd == "polarity":
        print(json.dumps(run_polarity(x.stress, x.galaxy, x.bidmc, x.out), indent=2, default=str))
    elif x.cmd == "score":
        from threadpoolctl import threadpool_limits
        x.out.mkdir(parents=True, exist_ok=True)
        wanted = {e for e in x.encoders.split(",") if e}
        for enc in sorted(d.name for d in x.emb.iterdir() if d.is_dir()):
            if wanted and enc not in wanted:
                continue
            clean = x.clean / enc
            if not (clean / "vitaldb-fs25-w30s.parquet").exists():
                print(f"skip {enc}: no clean embeddings in {clean}")
                continue
            with threadpool_limits(4):
                r = score_encoder(enc, clean, x.emb / enc, x.cases)
            (x.out / f"{enc}.json").write_text(json.dumps(r, indent=2) + "\n")
            print(json.dumps({"encoder": enc, "stress": {k: v["mean_abs_delta_age"] for k, v in r["stress"].items()}}), flush=True)
        # the no-model baseline: heart rate and periodicity of the same windows
        first = next((d for d in sorted(x.emb.iterdir()) if d.is_dir() and (x.clean / d.name / "vitaldb-fs25-w30s.parquet").exists()), None)
        if first is not None and (not wanted or "hr" in wanted):
            r = score_encoder("hr", x.clean / first.name, first, x.cases, features="hr")
            (x.out / "hr.json").write_text(json.dumps(r, indent=2) + "\n")
        text = summarize(x.out)
        (x.out / "summary.md").write_text(text)
        print(text)
    else:
        print(summarize(x.results))


if __name__ == "__main__":
    main()
