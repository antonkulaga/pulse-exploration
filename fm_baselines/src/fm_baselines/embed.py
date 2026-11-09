"""Frozen-encoder embeddings of prepared windows.

Writes ``<out>/<model>/<source>-<variant>.parquet`` (one row per window: ``source``, ``subject``, ``key``,
``split``, ``recording``, ``start_s``, ``hr_bpm``, ``periodicity``, ``emb``), the variant's subject table
beside it as ``<source>-<variant>.subjects.parquet``, and ``<source>-<variant>.json`` with counts and timing.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .data import Variant, load_variant


def write_embeddings(target: Path, folder: Path, v: Variant, emb: np.ndarray, info: dict) -> dict:
    if not np.isfinite(emb).all():
        raise RuntimeError(f"non-finite embeddings on {folder}")
    target.mkdir(parents=True, exist_ok=True)
    name = f"{v.source}-{Path(folder).name}"
    v.rows.with_columns(pl.Series("emb", emb.astype(np.float32)).cast(pl.Array(pl.Float32, emb.shape[1]))) \
        .write_parquet(target / f"{name}.parquet", compression="zstd")
    v.subjects.write_parquet(target / f"{name}.subjects.parquet")
    info = {**info, "variant": str(folder), "source": v.source, "sample_rate": v.sample_rate,
            "windows": len(v.rows), "subjects": v.rows.get_column("key").n_unique(), "dim": int(emb.shape[1]),
            "splits": dict(v.rows.group_by("split").agg(pl.col("key").n_unique()).iter_rows())}
    (target / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)
    return info


def embed_with(name: str, fn: Callable[[np.ndarray, float], np.ndarray], folders: list[Path], out: Path,
               per_subject: int, device: str, extra: dict | None = None) -> list[dict]:
    done = []
    for folder in folders:
        t0 = time.time()
        v = load_variant(folder, per_subject)
        t_load = time.time() - t0
        emb = fn(v.x, v.sample_rate)
        done.append(write_embeddings(out / name, folder, v, emb, {
            "model": name, "per_subject": per_subject, "load_s": round(t_load, 1),
            "embed_s": round(time.time() - t0 - t_load, 1),
            "device": torch.cuda.get_device_name() if device.startswith("cuda") else device, **(extra or {})}))
    return done


def embed_variants(model: str, folders: list[Path], out: Path, per_subject: int, device: str,
                   batch: int = 512) -> list[dict]:
    from .models import LOADERS

    encoder = LOADERS[model]()
    encoder.net.to(device)
    return embed_with(model, lambda x, fs: encoder.embed(x, fs, device, batch), folders, out, per_subject, device,
                      {"model_rate": encoder.fs, "segment_s": encoder.segment_s})
