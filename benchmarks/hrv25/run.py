"""Score every 25 Hz HRV candidate and Samsung's IBI list (raw and cleaned) on GalaxyPPG, dev and held-out split.

Candidates were tuned on the development set P02-P06 only; P07-P24 is the held-out set. Everything
below is regenerated from the dataset and the frozen candidate defaults:

    uv run python benchmarks/hrv25/run.py [--out data/benchmarks/hrv25] [--workers N]

Output (gitignored, under data/): windows.parquet (one row per estimator x window), summary CSVs
per split (overall, by group, by session, by participant; *_common = only windows every estimator
could be scored on), runs.csv, and PNG plots (Bland-Altman of RMSSD per method, RMSSD error and
coverage by activity group). The markdown tables printed at the end feed docs/hrv-25hz-benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from pulse_exploration import galaxyppg  # noqa: E402
from pulse_exploration.hrv import bench  # noqa: E402
from pulse_exploration.hrv.cand_classical import ClassicalEstimator  # noqa: E402
from pulse_exploration.hrv.cand_foot_template import FootTemplateEstimator  # noqa: E402
from pulse_exploration.hrv.cand_motion_denoise import MotionDenoiseEstimator  # noqa: E402
from pulse_exploration.hrv.cand_naive import NaiveEstimator  # noqa: E402

DEV = tuple(f"P{i:02d}" for i in range(2, 7))
HELD_OUT = tuple(f"P{i:02d}" for i in range(7, 25))
ORDER = ["naive", "classical", "foot_template", "motion_denoise", "samsung_ibi", "samsung_ibi_clean"]
GROUP_ORDER = ["seated-rest", "cognitive-stress", "standing", "typing", "locomotion"]
WITHIN_MS = 10.0  # "usable" RMSSD: within +-10 ms of the ECG reference


def estimators() -> list:
    return [NaiveEstimator(), ClassicalEstimator(), FootTemplateEstimator(), MotionDenoiseEstimator(), bench.SamsungIBI(), bench.SamsungIBIClean()]


def within_table(w: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Share of covered windows with |RMSSD error| <= WITHIN_MS, and the yield: the share of all
    usable windows that are covered and that accurate."""
    u = w.filter(pl.col("usable")).with_columns(
        ok=(pl.col("covered") & ((pl.col("est_rmssd") - pl.col("ref_rmssd")).abs() <= WITHIN_MS)).fill_null(False)
    )
    return (
        u.group_by(by)
        .agg(
            n_usable=pl.len(),
            n_covered=pl.col("covered").sum(),
            within=pl.col("ok").sum() / pl.col("covered").sum(),
            yield_=pl.col("ok").mean(),
        )
        .rename({"yield_": "yield"})
        .sort(by)
    )


def _order(df: pl.DataFrame) -> pl.DataFrame:
    if "estimator" not in df.columns:
        return df
    keys = [pl.col("estimator").replace_strict({e: i for i, e in enumerate(ORDER)}, default=99)]
    rest = [c for c in ("window_s", "group", "kind", "uid") if c in df.columns]
    return df.sort(keys + [pl.col(c) for c in rest])


def plot_bland_altman(w: pl.DataFrame, path: Path, title: str) -> None:
    c = w.filter(pl.col("usable"), pl.col("covered"))
    ests = [e for e in ORDER if e in c["estimator"].unique().to_list()]
    fig, axes = plt.subplots(1, len(ests), figsize=(4 * len(ests), 4), sharey=False)
    for ax, e in zip(np.atleast_1d(axes), ests):
        g = c.filter(pl.col("estimator") == e)
        est, ref = g["est_rmssd"].to_numpy(), g["ref_rmssd"].to_numpy()
        ok = np.isfinite(est) & np.isfinite(ref)
        est, ref = est[ok], ref[ok]
        if not len(est):
            ax.set_title(f"{e}: no windows")
            continue
        mean, diff = (est + ref) / 2, est - ref
        bias, sd = diff.mean(), diff.std(ddof=1)
        ax.scatter(mean, diff, s=6, alpha=0.5)
        for y, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
            ax.axhline(y, color="k", ls=ls, lw=0.8)
        ax.axhspan(-WITHIN_MS, WITHIN_MS, color="tab:green", alpha=0.08)
        ax.set_title(f"{e} (n={len(est)})\nbias {bias:+.1f}, LoA {bias - 1.96 * sd:+.0f}…{bias + 1.96 * sd:+.0f} ms", fontsize=9)
        ax.set_xlabel("mean of est and ECG RMSSD (ms)")
        lim = max(30.0, float(np.percentile(np.abs(diff), 98)) * 1.2)
        ax.set_ylim(-lim, lim)
    np.atleast_1d(axes)[0].set_ylabel("est − ECG RMSSD (ms)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_by_group(t: pl.DataFrame, wt: pl.DataFrame, path: Path, title: str) -> None:
    ests = [e for e in ORDER if e in t["estimator"].unique().to_list()]
    groups = [g for g in GROUP_ORDER if g in t["group"].unique().to_list()]
    x = np.arange(len(groups))
    wd = 0.8 / len(ests)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    for i, e in enumerate(ests):
        r = {row["group"]: row for row in t.filter(pl.col("estimator") == e).iter_rows(named=True)}
        q = {row["group"]: row for row in wt.filter(pl.col("estimator") == e).iter_rows(named=True)}
        err = [r[g]["rmssd_mae_median"] if g in r else np.nan for g in groups]
        yld = [q[g]["yield"] if g in q else np.nan for g in groups]
        a1.bar(x + i * wd, err, wd, label=e)
        a2.bar(x + i * wd, yld, wd, label=e)
    a1.set_yscale("log")
    a1.axhline(WITHIN_MS, color="k", lw=0.8, ls="--")
    a1.set_ylabel("RMSSD median |error| (ms, log)")
    a2.set_ylabel(f"share of usable windows covered and within ±{WITHIN_MS:.0f} ms")
    a2.set_ylim(0, 1)
    for a in (a1, a2):
        a.set_xticks(x + 0.4 - wd / 2, groups, rotation=15)
    a2.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/benchmarks/hrv25"))
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    res = bench.run(estimators(), DEV + HELD_OUT, bench.BenchConfig(), a.workers)
    w = res.windows.with_columns(split=pl.when(pl.col("uid").is_in(DEV)).then(pl.lit("dev")).otherwise(pl.lit("held_out")))
    w.write_parquet(a.out / "windows.parquet")
    res.runs.write_csv(a.out / "runs.csv")
    (a.out / "config.json").write_text(json.dumps({**asdict(res.config), "dev": DEV, "held_out": HELD_OUT}, indent=1))

    show = ["n_windows", "n_participants", "coverage", "rmssd_ref_median", "rmssd_mae_median", "rmssd_bias",
            "rmssd_loa_low", "rmssd_loa_high", "sdnn_mae_median", "sdnn_bias", "precision", "recall", "ibi_mae_ms"]
    for split in ("dev", "held_out"):
        ws = w.filter(pl.col("split") == split)
        for common in (False, True):
            suf = "_common" if common else ""
            tabs = {
                "overall": bench.summarize(ws, ("estimator", "window_s"), common),
                "by_group": bench.summarize(ws, ("estimator", "window_s", "group"), common),
                "by_session": bench.summarize(ws, ("estimator", "window_s", "kind"), common),
                "by_participant": bench.summarize(ws, ("estimator", "window_s", "uid"), common),
            }
            for name, t in tabs.items():
                _order(t).write_csv(a.out / f"{split}_{name}{suf}.csv")
            if split == "held_out":
                for name in ("overall", "by_group"):
                    print(f"\n## {split} {name}{suf}\n")
                    t = _order(tabs[name])
                    print(bench.to_markdown(t, [c for c in ("estimator", "window_s", "group") if c in t.columns] + show, digits=1))
        wt_all = within_table(ws, ["estimator", "window_s"])
        wt_grp = within_table(ws, ["estimator", "window_s", "group"])
        _order(wt_all).write_csv(a.out / f"{split}_within.csv")
        _order(wt_grp).write_csv(a.out / f"{split}_within_by_group.csv")
        print(f"\n## {split}: share within ±{WITHIN_MS:.0f} ms\n")
        print(bench.to_markdown(_order(wt_all), digits=2))
        print(bench.to_markdown(_order(wt_grp.filter(pl.col("window_s") == 60)), digits=2))
        for W in (60, 300):
            plot_bland_altman(ws.filter(pl.col("window_s") == W), a.out / f"{split}_bland_altman_{W}s.png",
                              f"RMSSD vs ECG, {split} ({', '.join(DEV) if split == 'dev' else 'P07–P24'}), {W} s windows")
        plot_by_group(
            bench.summarize(ws.filter(pl.col("window_s") == 60), ("estimator", "window_s", "group")),
            wt_grp.filter(pl.col("window_s") == 60),
            a.out / f"{split}_by_group_60s.png",
            f"RMSSD error and yield by activity, {split}, 60 s windows",
        )

    print("\n## windows (held out)\n")
    print(bench.to_markdown(bench.window_accounting(w.filter(pl.col("split") == "held_out"))))
    errs = res.runs.filter(pl.col("error").is_not_null())
    if errs.height:
        print("\n## errors\n")
        print(bench.to_markdown(errs))
    est_s = res.runs.group_by("estimator").agg(pl.col("estimate_s").sum())
    print(f"\nwall {res.wall_s:.1f} s; estimator CPU: " + ", ".join(f"{e} {s:.1f} s" for e, s in est_s.iter_rows()))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    assert set(DEV + HELD_OUT) == set(galaxyppg.UIDS) - {"P01"}
    main(sys.argv[1:])
