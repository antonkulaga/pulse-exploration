"""Frozen-encoder embeddings of prepared windows.

Writes ``<out>/<model>/<source>-<variant>.parquet`` (one row per window: ``source``, ``subject``, ``key``,
``split``, ``recording``, ``start_s``, ``hr_bpm``, ``periodicity``, ``emb``), the variant's subject table
beside it as ``<source>-<variant>.subjects.parquet``, and ``<source>-<variant>.json`` with counts and timing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .data import load_variant
from .models import LOADERS


def embed_variants(model: str, folders: list[Path], out: Path, per_subject: int, device: str,
                   batch: int = 512) -> list[dict]:
    encoder = LOADERS[model]()
    encoder.net.to(device)
    target = out / model
    target.mkdir(parents=True, exist_ok=True)
    done = []
    for folder in folders:
        t0 = time.time()
        v = load_variant(folder, per_subject)
        t_load = time.time() - t0
        emb = encoder.embed(v.x, v.sample_rate, device, batch)
        if not np.isfinite(emb).all():
            raise RuntimeError(f"{model} gave non-finite embeddings on {folder}")
        name = f"{v.source}-{Path(folder).name}"
        table = v.rows.with_columns(pl.Series("emb", emb).cast(pl.Array(pl.Float32, emb.shape[1])))
        table.write_parquet(target / f"{name}.parquet", compression="zstd")
        v.subjects.write_parquet(target / f"{name}.subjects.parquet")
        info = {
            "model": model, "variant": str(folder), "source": v.source, "sample_rate": v.sample_rate,
            "model_rate": encoder.fs, "segment_s": encoder.segment_s, "windows": len(v.rows),
            "subjects": v.rows.get_column("key").n_unique(), "per_subject": per_subject, "dim": int(emb.shape[1]),
            "load_s": round(t_load, 1), "embed_s": round(time.time() - t0 - t_load, 1),
            "device": torch.cuda.get_device_name() if device.startswith("cuda") else device,
            "splits": dict(v.rows.group_by("split").agg(pl.col("key").n_unique()).iter_rows()),
        }
        (target / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps(info), flush=True)
        done.append(info)
    return done
