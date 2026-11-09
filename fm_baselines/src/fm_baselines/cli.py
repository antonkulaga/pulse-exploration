"""``fm-baselines embed | probe | kron-eval | summary``.

    fm-baselines embed pulseppg <kron-fm>/data/interim/ppg/vitaldb/fs25-w30s ... --out <runs>/emb
    fm-baselines probe <runs>/emb --cases <kron-fm>/data/input/ppg/vitaldb/cases.csv.gz --out <runs>/probe
    fm-baselines kron-eval <kron-ppg run>... --subjects <variant>/subjects.parquet --cases ... --out <runs>/probe
    fm-baselines summary <runs>/probe                          # markdown tables to stdout

KronLSTM encoders are embedded by ``python -m fm_baselines.kron_embed`` in kron-fm's environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl

TRAIN = "vitaldb"
EXTERNAL = ("bidmc", "galaxyppg")


def _probe_task(task: tuple) -> dict:
    """One embedding set's probes in a worker process, with one BLAS thread so workers do not compete."""
    from threadpoolctl import threadpool_limits

    from .probe import probe_dir
    model_dir, stem, externals, out, cases, features, name = task
    with threadpool_limits(1):
        m = probe_dir(model_dir, stem, externals, out, cases, features=features, name=name)
    return {"model": m["model"], "train": stem, "age_test_mae": m.get("age", {}).get("test_mae")}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fm-baselines")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="frozen foundation-model embeddings of prepared windows")
    e.add_argument("model", choices=["pulseppg", "papagei_s", "anyppg"])
    e.add_argument("variants", nargs="+", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--per-subject", type=int, default=24, help="windows per subject, spread in time (0 = all)")
    e.add_argument("--device", default="cuda")
    e.add_argument("--batch", type=int, default=512)
    pr = sub.add_parser("probe", help="linear probes of every embedding set in a folder, plus the hr baseline")
    pr.add_argument("emb", type=Path)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--cases", type=Path, default=None, help="VitalDB cases.csv.gz for the lab traits")
    pr.add_argument("--models", default="", help="comma-separated subset of the folder's models")
    pr.add_argument("--workers", type=int, default=0, help="parallel probes (default: the job's CPUs)")
    k = sub.add_parser("kron-eval", help="score kron-ppg's own trait and clock heads like the probes")
    k.add_argument("runs", nargs="+", type=Path)
    k.add_argument("--subjects", type=Path, required=True, help="VitalDB subjects.parquet of the variant")
    k.add_argument("--cases", type=Path, default=None)
    k.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary", help="markdown tables of all results")
    s.add_argument("results", type=Path)
    a = p.parse_args(argv)

    if a.cmd == "embed":
        from .embed import embed_variants
        embed_variants(a.model, a.variants, a.out, a.per_subject, a.device, a.batch)
    elif a.cmd == "probe":
        from concurrent.futures import ProcessPoolExecutor

        from .probe import summarize
        wanted = {m for m in a.models.split(",") if m}
        tasks, done_hr = [], set()
        for model_dir in sorted(d for d in a.emb.iterdir() if d.is_dir()):
            if wanted and model_dir.name not in wanted:
                continue
            for train in sorted(model_dir.glob(f"{TRAIN}-*.json")):
                stem = train.stem
                variant = stem.removeprefix(f"{TRAIN}-")
                externals = [f"{x}-{variant}" for x in EXTERNAL]
                tasks.append((model_dir, stem, externals, a.out / model_dir.name / stem, a.cases, "emb", None))
                if variant not in done_hr:  # the no-model baseline, once per rate
                    tasks.append((model_dir, stem, externals, a.out / "hr" / stem, a.cases, "hr", "hr"))
                    done_hr.add(variant)
        workers = a.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks) or 1)) as pool:
            for m in pool.map(_probe_task, tasks):
                print(json.dumps(m), flush=True)
        text = summarize(a.out)
        (a.out / "summary.md").write_text(text)
        print(text)
    elif a.cmd == "kron-eval":
        from .labels import label_table
        from .probe import kron_head_eval, summarize
        subjects = pl.read_parquet(a.subjects)
        if "key" not in subjects.columns:
            subjects = subjects.with_columns((pl.lit(f"{TRAIN}/") + pl.col("subject").cast(pl.String)).alias("key"))
        labels = label_table(subjects, a.cases)
        for run in a.runs:
            m = kron_head_eval(run, labels, a.out / f"kron-head-{run.name}" / f"{TRAIN}-fs25-w30s")
            print(json.dumps({"run": str(run), "age": m.get("age", {}).get("test_mae")}), flush=True)
        text = summarize(a.out)
        (a.out / "summary.md").write_text(text)
        print(text)
    else:
        from .probe import summarize
        print(summarize(a.results))


if __name__ == "__main__":
    main()
