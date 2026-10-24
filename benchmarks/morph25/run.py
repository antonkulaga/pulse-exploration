"""Does pulse shape survive 25 Hz? Pulse-shape indices from 25 Hz versus 64 Hz wrist PPG on GalaxyPPG.

The Galaxy Watch stream is 25 Hz, so GalaxyPPG's Empatica E4 BVP (64 Hz, same protocol, other wrist)
is the reference. Its BVP is already band-limited below ~10 Hz by Empatica, which is also where
wrist green PPG keeps >99.9 % of its pulse power (our watch's 100 Hz SDK green: 12.5-50 Hz holds
~3e-4 of the 0.8-3 Hz band power), so this tests reconstruction from 25 Hz samples, not content
the E4 never had.

Per participant, seated-rest sessions are cut into 60 s windows. In each window an average beat is
built (beats aligned at the steepest upstroke, normalised, kept when they correlate >= 0.9 with the
window's median beat) from:
- ref: 64 Hz upsampled to 1 kHz with a band-limited (polyphase) filter, beats found on itself
- spline / bandlimited: the signal decimated to 25 Hz (anti-aliased, random sample phase) and
  rebuilt at 1 kHz by cubic spline or band-limited upsampling, beats found on the rebuilt signal
- oracle: like spline, but aligned at the reference's beat times (isolates sampling from timing)
Indices on the average beat: crest time (upstroke to systolic peak), pulse width at half height,
and the second-derivative ratios b/a and d/a. For each index the table gives the agreement with
the reference and the error relative to the spread between people (below ~0.3 the index keeps
people apart about as well as at 64 Hz).

    uv run python benchmarks/morph25/run.py [--out data/benchmarks/morph25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import interpolate, signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pulse_exploration import galaxyppg  # noqa: E402
from pulse_exploration.hrv.bench import GROUPS  # noqa: E402

FS_E4 = 64.0
FS_HI = 1000.0
WIN_S = 60.0
PRE_MS, POST_MS = 150, 600  # average-beat span around the upstroke
CORR_MIN = 0.9
MIN_BEATS = 30
INDICES = ("crest_ms", "width50_ms", "b_a", "d_a")


def load_e4(uid: str) -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_csv(galaxyppg.DATASET_DIR / uid / "E4" / "BVP.csv")
    return df["timestamp"].to_numpy() / 1000.0, df["value"].to_numpy().astype(float)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 2) -> np.ndarray:
    sos = signal.butter(order, (lo, hi), "bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def upstrokes(y: np.ndarray, fs: float) -> np.ndarray:
    """Sample indices of the steepest upstroke of each beat (detection on a 0.5-8 Hz copy)."""
    d = np.gradient(bandpass(y, fs, 0.5, 8.0))
    pk, prop = signal.find_peaks(d, distance=int(0.33 * fs), height=0.0)
    if len(pk) < 3:
        return pk
    h = prop["peak_heights"]
    return pk[h >= 0.4 * np.median(h)]


def average_beat(y: np.ndarray, fs: float, idx: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Mean of normalised beats around the upstroke indices that match the median beat."""
    pre, post = int(PRE_MS * fs / 1000), int(POST_MS * fs / 1000)
    segs = [y[i - pre:i + post] for i in idx if i - pre >= 0 and i + post <= len(y)]
    if len(segs) < MIN_BEATS:
        return None, len(segs)
    s = np.asarray(segs)
    s = (s - s.min(axis=1, keepdims=True)) / np.ptp(s, axis=1, keepdims=True)
    med = np.median(s, axis=0)
    r = np.array([np.corrcoef(b, med)[0, 1] for b in s])
    good = s[r >= CORR_MIN]
    if len(good) < MIN_BEATS:
        return None, len(good)
    return good.mean(axis=0), len(good)


def indices(beat: np.ndarray, fs: float) -> dict[str, float]:
    """Crest time, width at half height and SDPPG b/a, d/a of an average beat sampled at fs."""
    nan = float("nan")
    out = dict.fromkeys(INDICES, nan)
    f = int(PRE_MS * fs / 1000)
    lo = int(np.argmin(beat[: f + 1]))  # foot: minimum before the upstroke
    pk = f + int(np.argmax(beat[f:f + int(0.4 * fs)]))
    out["crest_ms"] = (pk - lo) / fs * 1000
    amp = beat[pk] - beat[lo]
    half = beat[lo] + amp / 2
    up = lo + int(np.argmax(beat[lo:pk + 1] >= half))
    after = np.where(beat[pk:] < half)[0]
    if len(after):
        out["width50_ms"] = (pk + after[0] - up) / fs * 1000
    w = int(0.031 * fs) | 1
    sd = signal.savgol_filter(beat, w, 3, deriv=2)
    # a: largest SDPPG maximum in early systole, then alternating extrema b, c, d, e
    seg = slice(max(lo - int(0.05 * fs), 0), pk)
    if seg.stop - seg.start < 3:
        return out
    a_i = seg.start + int(np.argmax(sd[seg]))
    ext = signal.argrelextrema(sd, np.greater)[0], signal.argrelextrema(sd, np.less)[0]
    seq, kind, cur = [], "min", a_i
    for _ in range(4):  # b (min), c (max), d (min), e (max)
        pool = ext[1] if kind == "min" else ext[0]
        nxt = pool[(pool > cur) & (pool < cur + int(0.45 * fs))]
        if not len(nxt):
            break
        cur = int(nxt[0])
        seq.append(cur)
        kind = "max" if kind == "min" else "min"
    a = sd[a_i]
    if a > 0 and len(seq) >= 1:
        out["b_a"] = sd[seq[0]] / a
    if a > 0 and len(seq) >= 3:
        out["d_a"] = sd[seq[2]] / a
    return out


def windows_for(uid: str, rng: np.random.Generator) -> list[dict]:
    rec = galaxyppg.load(uid)
    t, x = load_e4(uid)
    rows = []
    for s in rec.sessions:
        if GROUPS.get(s.kind) != "seated-rest":
            continue
        for k in range(int((s.end_ms - s.start_ms - 10_000) // (WIN_S * 1000))):
            a = s.start_ms + 5_000 + k * WIN_S * 1000
            m = (t >= a) & (t < a + WIN_S * 1000)
            if m.sum() < 0.95 * WIN_S * FS_E4:
                continue
            xw = x[m] - x[m].mean()
            # reference: band-limited 64 -> 1000 Hz
            ref = bandpass(signal.resample_poly(xw, 125, 8), FS_HI, 0.4, 12.0)
            # 25 Hz: anti-aliased decimation of a randomly phased 1 kHz copy
            ph = int(rng.integers(0, 40))
            s25 = signal.resample_poly(signal.resample_poly(xw, 125, 8)[ph:], 1, 40)
            t25 = (ph + np.arange(len(s25)) * 40) / FS_HI
            th = np.arange(len(ref)) / FS_HI
            th = th[(th >= t25[0]) & (th <= t25[-1])]
            spl = bandpass(interpolate.CubicSpline(t25, s25)(th), FS_HI, 0.4, 12.0)
            bl = signal.resample_poly(s25, 40, 1)[: len(th)]
            bl = bandpass(np.pad(bl, (0, len(th) - len(bl))) if len(bl) < len(th) else bl, FS_HI, 0.4, 12.0)
            ref_c = ref[int(th[0] * FS_HI): int(th[0] * FS_HI) + len(th)]
            idx_ref = upstrokes(ref_c, FS_HI)
            variants = {
                "ref": (ref_c, idx_ref),
                "spline": (spl, upstrokes(spl, FS_HI)),
                "bandlimited": (bl, upstrokes(bl, FS_HI)),
                "oracle": (spl, idx_ref),
            }
            for name, (y, idx) in variants.items():
                beat, n = average_beat(y, FS_HI, idx)
                row = {"uid": uid, "session": s.name, "k": k, "variant": name, "beats": n}
                row.update(indices(beat, FS_HI) if beat is not None else dict.fromkeys(INDICES, float("nan")))
                rows.append(row)
    return rows


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    ref = df.filter(pl.col("variant") == "ref").drop("variant", "beats")
    out = []
    for v in ("spline", "bandlimited", "oracle"):
        cmp = df.filter(pl.col("variant") == v).drop("variant", "beats").join(ref, on=["uid", "session", "k"], suffix="_ref")
        for ix in INDICES:
            c = cmp.select("uid", pl.col(ix).alias("v"), pl.col(f"{ix}_ref").alias("r")).drop_nans().drop_nulls()
            if c.height < 10:
                continue
            d = (c["v"] - c["r"]).to_numpy()
            between = c.group_by("uid").agg(pl.col("r").mean())["r"].std()
            within = c.group_by("uid").agg(pl.col("r").std())["r"].median()
            out.append({"variant": v, "index": ix, "windows": c.height, "people": c["uid"].n_unique(),
                        "ref_mean": c["r"].mean(), "r": float(np.corrcoef(c["v"], c["r"])[0, 1]),
                        "bias": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                        "between_people_sd": between, "within_person_sd": within,
                        "error_vs_between": float(d.std(ddof=1)) / between if between else float("nan")})
    return pl.DataFrame(out)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/morph25"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    rows = []
    for uid in [u for u in galaxyppg.UIDS if u != "P01"]:
        if (galaxyppg.DATASET_DIR / uid / "E4" / "BVP.csv").exists():
            rows.extend(windows_for(uid, rng))
    df = pl.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out / "windows.parquet")
    s = summarize(df)
    s.write_csv(a.out / "summary.csv")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    usable = df.filter(pl.col("variant") == "ref", pl.col("crest_ms").is_not_nan())
    print(f"{usable['uid'].n_unique()} people, {usable.height} reference windows with >= {MIN_BEATS} good beats")
    print(s.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main(sys.argv[1:])
