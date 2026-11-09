"""``fm-baselines embed | probe | summary``.

    fm-baselines embed pulseppg <kron-fm>/data/interim/ppg/vitaldb/fs25-w30s ... --out <runs>/emb
    fm-baselines probe <runs>/emb --out <runs>/probe          # every model, every rate, plus the hr baseline
    fm-baselines summary <runs>/probe                          # markdown table to stdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import LOADERS

TRAIN = "vitaldb"
EXTERNAL = ("bidmc", "galaxyppg")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fm-baselines")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="frozen-encoder embeddings of prepared windows")
    e.add_argument("model", choices=sorted(LOADERS))
    e.add_argument("variants", nargs="+", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--per-subject", type=int, default=24, help="windows per subject, spread in time (0 = all)")
    e.add_argument("--device", default="cuda")
    e.add_argument("--batch", type=int, default=512)
    pr = sub.add_parser("probe", help="linear probes of every model and rate in an embedding folder")
    pr.add_argument("emb", type=Path)
    pr.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summary", help="markdown table of probe results")
    s.add_argument("results", type=Path)
    a = p.parse_args(argv)

    if a.cmd == "embed":
        from .embed import embed_variants
        embed_variants(a.model, a.variants, a.out, a.per_subject, a.device, a.batch)
    elif a.cmd == "probe":
        from .probe import run, summarize
        done_hr = set()
        for model_dir in sorted(d for d in a.emb.iterdir() if d.is_dir()):
            for train in sorted(model_dir.glob(f"{TRAIN}-*.json")):
                stem = train.stem
                variant = stem.removeprefix(f"{TRAIN}-")
                externals = [f"{x}-{variant}" for x in EXTERNAL]
                m = run(model_dir, stem, externals, a.out / model_dir.name / stem)
                print(json.dumps({"model": model_dir.name, "train": stem, "age": m["age"]}), flush=True)
                if variant not in done_hr:  # the no-model baseline, once per rate
                    run(model_dir, stem, externals, a.out / "hr" / stem, features="hr")
                    done_hr.add(variant)
        text = summarize(a.out)
        (a.out / "summary.md").write_text(text)
        print(text)
    else:
        from .probe import summarize
        print(summarize(a.results))


if __name__ == "__main__":
    main()
