"""AF screen sensitivity and specificity on the MIT-BIH AF and normal-sinus-rhythm databases.

Beat times come from the databases' own annotations: the AF database's QRS detections (`.qrsc`,
manually corrected, where they exist, else `.qrs`) with rhythm labels from `.atr`, and the
normal-sinus-rhythm database's beat annotations (`.atr`, ectopic beats included, as a watch would
see them). Records 04936 and 05091 are excluded for incorrect rhythm annotations, as in Lee et al.
2013. Data: PhysioNet afdb 1.0.0 and nsrdb 1.0.0 (Open Data Commons Attribution 1.0), annotation and
header files only, in data/external/physionet/.

64-beat segments are labelled AF when every beat lies in an AFIB episode and sinus when every beat
lies in an N episode (AF database) or anywhere in the normal-rhythm database; mixed segments and
atrial flutter are left out. Each rule is also run after adding Gaussian jitter to the beat times
(5 and 10 ms SD) to mimic beat timing from 25 Hz PPG, whose interval error on GalaxyPPG was 7-8 ms.

    uv run python benchmarks/afdb/run.py [--out data/benchmarks/afdb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pulse_exploration.hrv import af_screen  # noqa: E402

ROOT = Path("data/external/physionet")
EXCLUDE = {"04936", "05091"}
BEATS = set("NLRBAaJSVrFejnE/fQ?")
RULES = {"Lee 2013 (untrimmed)": 0, "trimmed (ours)": af_screen.TRIM_DIFFS}
JITTER_MS = (0.0, 5.0, 10.0)


def afdb_records(root: Path) -> list[dict]:
    import wfdb

    out = []
    for rec in (root / "afdb" / "RECORDS").read_text().split():
        if rec in EXCLUDE:
            continue
        base = str(root / "afdb" / rec)
        fs = wfdb.rdheader(base).fs
        ext = "qrsc" if (root / "afdb" / f"{rec}.qrsc").exists() else "qrs"
        beats = wfdb.rdann(base, ext).sample / fs * 1000
        atr = wfdb.rdann(base, "atr")
        starts = atr.sample / fs * 1000
        labels = [a.strip().lstrip("(") for a in atr.aux_note]
        # rhythm at each beat: the last rhythm change at or before it
        idx = np.searchsorted(starts, beats, side="right") - 1
        rhythm = np.array([labels[i] if i >= 0 else "?" for i in idx])
        out.append({"db": "afdb", "record": rec, "beats_ms": beats, "rhythm": rhythm, "beat_source": ext})
    return out


def nsrdb_records(root: Path) -> list[dict]:
    import wfdb

    out = []
    for rec in (root / "nsrdb" / "RECORDS").read_text().split():
        base = str(root / "nsrdb" / rec)
        if not Path(base + ".atr").exists():
            continue
        fs = wfdb.rdheader(base).fs
        ann = wfdb.rdann(base, "atr")
        keep = np.array([s in BEATS for s in ann.symbol])
        beats = ann.sample[keep] / fs * 1000
        out.append({"db": "nsrdb", "record": rec, "beats_ms": beats, "rhythm": np.full(len(beats), "N"), "beat_source": "atr"})
    return out


def segments(rec: dict, jitter_ms: float, trim: int, rng: np.random.Generator) -> list[dict]:
    b = rec["beats_ms"] + (rng.normal(0, jitter_ms, len(rec["beats_ms"])) if jitter_ms else 0)
    b = np.sort(b)
    ibi = np.diff(b)
    rows = []
    n = af_screen.SEGMENT
    for k in range(0, len(ibi) - n + 1, n):
        seg = ibi[k:k + n]
        if np.any(seg <= 0) or np.any(seg > 3000):  # annotation gaps
            continue
        rh = set(rec["rhythm"][k:k + n + 1])
        label = "af" if rh == {"AFIB"} else "sinus" if rh == {"N"} else None
        if label is None:
            continue
        r, s = af_screen.nrmssd(seg, trim=trim), af_screen.shannon_entropy(seg)
        rows.append({"db": rec["db"], "record": rec["record"], "t_ms": float(b[k]), "end_ms": float(b[k + n]),
                     "label": label, "nrmssd": r, "shannon": s, "hr_bpm": 60000 / float(seg.mean()),
                     "irregular": bool(r >= af_screen.TH_NRMSSD and s >= af_screen.TH_SHANNON)})
    return rows


def alerts_by_record(df: pl.DataFrame) -> pl.DataFrame:
    """Persistence alerts per record, split into alerts during AF (most of the alert's segments are
    AF) and false alerts (most are sinus); an alert window can start in the sinus segment just before
    an AF episode, so its first segment alone does not decide."""
    out = []
    for (rule, jit, db, rec), g in df.group_by("rule", "jitter_ms", "db", "record", maintain_order=True):
        g = g.sort("t_ms")
        segs = [af_screen.Segment(r["t_ms"], r["end_ms"], r["nrmssd"], r["shannon"], 0.0, r["irregular"]) for r in g.iter_rows(named=True)]
        al = af_screen.alerts(segs)
        is_af = (g["label"] == "af").to_numpy()
        t0 = g["t_ms"].to_numpy()
        dur = (g["end_ms"] - g["t_ms"]).to_numpy()
        covered = np.zeros(len(g), bool)
        n_af = n_sinus = 0
        for a, b in al:
            inside = (t0 >= a) & (t0 < b)
            covered |= inside
            if is_af[inside].mean() >= 0.5:
                n_af += 1
            else:
                n_sinus += 1
        out.append({"rule": rule, "jitter_ms": jit, "db": db, "record": rec, "alerts_in_af": n_af, "alerts_in_sinus": n_sinus,
                    "af_hours_in_alerts": float(dur[covered & is_af].sum()) / 3.6e6,
                    "af_hours": float(g.filter(pl.col("label") == "af").select((pl.col("end_ms") - pl.col("t_ms")).sum()).item() or 0) / 3.6e6,
                    "sinus_hours": float(g.filter(pl.col("label") == "sinus").select((pl.col("end_ms") - pl.col("t_ms")).sum()).item() or 0) / 3.6e6})
    return pl.DataFrame(out)


def loro_threshold(seg: pl.DataFrame, grid: np.ndarray) -> tuple[pl.DataFrame, list[float]]:
    """Leave one record out: for each record, pick the normalised-RMSSD threshold that maximises
    sensitivity + mean(specificity in AF patients' sinus rhythm, specificity in healthy records)
    on all other records, and apply it to the held-out record. Returns its segments with the
    out-of-sample `irregular` flag, and the thresholds chosen."""
    r = seg["nrmssd"].to_numpy()
    sh = seg["shannon"].to_numpy() >= af_screen.TH_SHANNON
    lab = seg["label"].to_numpy()
    db = seg["db"].to_numpy()
    rec = seg["record"].to_numpy()
    flag = np.zeros(len(seg), bool)
    chosen = []
    for held in np.unique(rec):
        tr = rec != held
        best, best_th = -1.0, float(grid[0])
        for th in grid:
            irr = (r >= th) & sh
            af, sa, sn = tr & (lab == "af"), tr & (lab == "sinus") & (db == "afdb"), tr & (db == "nsrdb")
            score = irr[af].mean() + 0.5 * ((~irr[sa]).mean() + (~irr[sn]).mean())
            if score > best:
                best, best_th = score, float(th)
        chosen.append(best_th)
        te = ~tr
        flag[te] = (r[te] >= best_th) & sh[te]
    return seg.with_columns(irregular=pl.Series(flag)), chosen


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/afdb"))
    a = ap.parse_args(argv)
    recs = afdb_records(ROOT) + nsrdb_records(ROOT)
    rows = []
    for rule, trim in RULES.items():
        for jit in JITTER_MS:
            rng = np.random.default_rng(0)
            for rec in recs:
                for r in segments(rec, jit, trim, rng):
                    rows.append({"rule": rule, "jitter_ms": jit, **r})
    df = pl.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out / "segments.parquet")
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(220)
    print(f"{df.filter(pl.col('rule') == next(iter(RULES)), pl.col('jitter_ms') == 0)['record'].n_unique()} records")

    def rate(label: str, db: str | None = None):
        f = pl.col("label") == label
        if db:
            f = f & (pl.col("db") == db)
        return f

    summary = df.group_by("rule", "jitter_ms").agg(
        af_segments=rate("af").sum(),
        sensitivity=(pl.col("irregular") & rate("af")).sum() / rate("af").sum(),
        specificity_afdb_sinus=(~pl.col("irregular") & rate("sinus", "afdb")).sum() / rate("sinus", "afdb").sum(),
        specificity_nsrdb=(~pl.col("irregular") & rate("sinus", "nsrdb")).sum() / rate("sinus", "nsrdb").sum(),
        nsrdb_segments=rate("sinus", "nsrdb").sum(),
    ).sort("rule", "jitter_ms").with_columns(pl.col(pl.Float64).round(4))
    print("\nPer 64-beat segment")
    print(summary)
    bands = df.filter(pl.col("label") == "af", pl.col("jitter_ms") == 0).with_columns(
        band=pl.when(pl.col("hr_bpm") < 60).then(pl.lit("<60")).when(pl.col("hr_bpm") < 90).then(pl.lit("60-90"))
        .when(pl.col("hr_bpm") < 120).then(pl.lit("90-120")).otherwise(pl.lit(">=120"))
    ).group_by("rule", "band").agg(segments=pl.len(), sensitivity=pl.col("irregular").mean().round(4)).sort("rule", "band")
    print("\nAF sensitivity by heart rate (no jitter)")
    print(bands)
    sweep = []
    base = df.filter(pl.col("rule") == "trimmed (ours)", pl.col("jitter_ms") == 5.0)
    for th in (0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13):
        irr = (pl.col("nrmssd") >= th) & (pl.col("shannon") >= af_screen.TH_SHANNON)
        r = base.select(
            sens=(irr & (pl.col("label") == "af")).sum() / (pl.col("label") == "af").sum(),
            spec_afdb=(~irr & (pl.col("label") == "sinus") & (pl.col("db") == "afdb")).sum() / ((pl.col("label") == "sinus") & (pl.col("db") == "afdb")).sum(),
            spec_nsrdb=(~irr & (pl.col("db") == "nsrdb")).sum() / (pl.col("db") == "nsrdb").sum(),
        ).row(0, named=True)
        sweep.append({"threshold": th, **{k: round(v, 4) for k, v in r.items()}})
    print("\nTrimmed rule, 5 ms jitter: threshold sweep on normalised RMSSD")
    print(pl.DataFrame(sweep))
    grid = np.round(np.arange(0.05, 0.1301, 0.005), 3)
    loro_rows, loro_th = [], {}
    for jit in JITTER_MS:
        seg = df.filter(pl.col("rule") == "trimmed (ours)", pl.col("jitter_ms") == jit)
        out, chosen = loro_threshold(seg, grid)
        loro_th[jit] = chosen
        loro_rows.append(out.with_columns(rule=pl.lit("trimmed, threshold left one record out")))
    loro = pl.concat(loro_rows)
    loro_sum = loro.group_by("rule", "jitter_ms").agg(
        sensitivity=(pl.col("irregular") & rate("af")).sum() / rate("af").sum(),
        specificity_afdb_sinus=(~pl.col("irregular") & rate("sinus", "afdb")).sum() / rate("sinus", "afdb").sum(),
        specificity_nsrdb=(~pl.col("irregular") & rate("sinus", "nsrdb")).sum() / rate("sinus", "nsrdb").sum(),
    ).sort("jitter_ms").with_columns(pl.col(pl.Float64).round(4))
    print("\nTrimmed rule with the threshold chosen on the other records (leave one record out)")
    print(loro_sum)
    for jit, ch in loro_th.items():
        print(f"  jitter {jit:g} ms: thresholds chosen {min(ch):.3f}–{max(ch):.3f}, median {np.median(ch):.3f}")
    loro_sum.write_csv(a.out / "loro.csv")
    df = pl.concat([df, loro.select(df.columns)])
    al = alerts_by_record(df)
    al_sum = al.group_by("rule", "jitter_ms", "db").agg(
        records=pl.len(), af_hours=pl.col("af_hours").sum().round(1), sinus_hours=pl.col("sinus_hours").sum().round(1),
        records_with_af=(pl.col("af_hours") > 0).sum(),
        records_af_alerted=((pl.col("af_hours") > 0) & (pl.col("alerts_in_af") > 0)).sum(),
        af_time_in_alerts=(pl.col("af_hours_in_alerts").sum() / pl.col("af_hours").sum()).round(3),
        false_alerts=pl.col("alerts_in_sinus").sum(),
    ).with_columns(false_alerts_per_24h=(pl.col("false_alerts") / pl.col("sinus_hours") * 24).round(2)).sort("rule", "jitter_ms", "db")
    print("\nPersistence alerts (5 of 6 segments)")
    print(al_sum)
    summary.write_csv(a.out / "summary.csv")
    bands.write_csv(a.out / "by_rate.csv")
    pl.DataFrame(sweep).write_csv(a.out / "sweep.csv")
    al_sum.write_csv(a.out / "alerts.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
