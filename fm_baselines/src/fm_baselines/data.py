"""Read kron-ppg's prepared windows and reproduce its person-level split.

A prepared variant (``kron-ppg prepare``, in kron-fm) is a folder
``data/interim/ppg/<source>/fs<rate>-w30s/`` with

* ``windows.parquet``: ``subject``, ``recording``, ``start_s``, ``hr_bpm``, ``periodicity`` and ``signal``,
  a robust-scaled 30 s window band-passed to 0.5-8 Hz, stored as ``Array(Float16, L)``;
* ``subjects.parquet``: ``subject``, ``age``, ``sex`` and the source's flags (VitalDB: ``preop_htn``,
  ``preop_dm``, ``asa``, ``death_inhosp``);
* ``info.json``: ``source``, ``sample_rate``, ``seconds`` and counts.

The split must match kron-ppg's exactly, or the baselines are not comparable: kron-ppg keys a subject
``<source>/<subject>`` and hashes ``kron-ppg/<key>`` with SHA-256 (``kron_ppg.data.subject_split``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


def kron_split(key: str, val: float = 0.15, test: float = 0.15) -> str:
    """``train``, ``val`` or ``test`` for a ``<source>/<subject>`` key, as ``kron_ppg.data.subject_split``."""
    u = int(hashlib.sha256(f"kron-ppg/{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else ("val" if u < test + val else "train")


@dataclass
class Variant:
    """Selected windows of one prepared variant: signals as float32 ``(n, L)`` and one row per window."""

    source: str
    sample_rate: float
    seconds: float
    rows: pl.DataFrame  # source, subject, key, split, recording, start_s, hr_bpm, periodicity
    x: np.ndarray
    subjects: pl.DataFrame


def pick_rows(table: pl.DataFrame, per_subject: int) -> np.ndarray:
    """Row numbers of at most ``per_subject`` windows per subject, spread evenly over its windows in time."""
    if per_subject <= 0:
        return np.arange(len(table))
    groups = (table.with_row_index("row")
              .sort(["subject", "recording", "start_s"])
              .group_by("subject", maintain_order=True)
              .agg(pl.col("row")))
    picked = []
    for rows in groups.get_column("row").to_list():
        n, m = len(rows), min(len(rows), per_subject)
        picked.extend(rows[i] for i in (np.arange(m) * n) // m)
    return np.sort(np.asarray(picked, dtype=np.int64))


def load_variant(folder: str | Path, per_subject: int = 24) -> Variant:
    folder = Path(folder)
    info = json.loads((folder / "info.json").read_text())
    source = str(info["source"])
    meta = pl.read_parquet(folder / "windows.parquet", columns=["subject", "recording", "start_s"])
    rows = pick_rows(meta, per_subject)
    picked = (pl.scan_parquet(folder / "windows.parquet")
              .with_row_index("row")
              .join(pl.LazyFrame({"row": pl.Series(rows, dtype=pl.UInt32)}), on="row", how="semi")
              .select("row", pl.col("subject").cast(pl.String), pl.col("recording").cast(pl.String), "start_s",
                      "hr_bpm", "periodicity", "signal")
              .collect())
    length = int(round(float(info["seconds"]) * float(info["sample_rate"])))
    x = picked.get_column("signal").to_numpy().astype(np.float32).reshape(len(picked), length)
    table = picked.drop("signal").with_columns(
        pl.lit(source).alias("source"),
        (pl.lit(source + "/") + pl.col("subject")).alias("key"),
    )
    table = table.with_columns(pl.col("key").map_elements(kron_split, return_dtype=pl.String).alias("split"))
    subjects = pl.read_parquet(folder / "subjects.parquet").with_columns(
        pl.col("subject").cast(pl.String),
        (pl.lit(source + "/") + pl.col("subject").cast(pl.String)).alias("key"))
    return Variant(source, float(info["sample_rate"]), float(info["seconds"]),
                   table.select("source", "subject", "key", "split", "recording", "start_s", "hr_bpm",
                                "periodicity"), x, subjects)
