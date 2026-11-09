import json

import numpy as np
import polars as pl
import pytest

from fm_baselines.data import kron_split, load_variant, pick_rows
from fm_baselines.probe import run


def test_split_matches_kron_ppg():
    # values from kron_ppg.data.subject_split(keys, "kron-ppg") in kron-fm b0c2f40
    want = {"vitaldb/1": "test", "vitaldb/2": "train", "vitaldb/5": "test", "vitaldb/7": "val",
            "vitaldb/9": "test", "vitaldb/2487": "train", "bidmc/01": "train", "galaxyppg/P01": "train"}
    assert {k: kron_split(k) for k in want} == want


def test_pick_rows_spreads_and_caps():
    table = pl.DataFrame({"subject": ["a"] * 10 + ["b"] * 2, "recording": ["r"] * 12,
                          "start_s": [float(i) for i in range(10)] + [0.0, 30.0]})
    rows = pick_rows(table, 4)
    assert rows.tolist() == [0, 2, 5, 7, 10, 11]
    assert pick_rows(table, 0).tolist() == list(range(12))


def _variant(folder, source, n_subjects, fs=25.0, seed=0, flags=True):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    length = int(30 * fs)
    subjects, rows, sig = [], [], []
    for i in range(n_subjects):
        age = float(rng.uniform(20, 80))
        s = {"subject": str(i), "age": age, "sex": "M" if i % 2 else "F"}
        if flags:
            s |= {"preop_htn": int(age + rng.normal(0, 10) > 55), "preop_dm": int(rng.random() < 0.3),
                  "asa": int(1 + (age > 60) + rng.integers(0, 2))}
        subjects.append(s)
        for k in range(5):
            rows.append({"subject": str(i), "recording": str(i), "start_s": 30.0 * k,
                         "hr_bpm": 60 + age / 4 + rng.normal(), "periodicity": 0.8})
            t = np.arange(length) / fs
            sig.append(np.sin(2 * np.pi * (1 + age / 100) * t) + 0.1 * rng.normal(size=length))
    pl.DataFrame(rows).with_columns(
        pl.Series("signal", np.array(sig, np.float16)).cast(pl.Array(pl.Float16, length))
    ).write_parquet(folder / "windows.parquet")
    pl.DataFrame(subjects).write_parquet(folder / "subjects.parquet")
    (folder / "info.json").write_text(json.dumps({"source": source, "sample_rate": fs, "seconds": 30.0}))


def test_load_and_probe_on_synthetic(tmp_path):
    _variant(tmp_path / "vitaldb" / "fs25-w30s", "vitaldb", 300)
    _variant(tmp_path / "bidmc" / "fs25-w30s", "bidmc", 20, seed=1, flags=False)
    emb = tmp_path / "emb" / "fake"
    emb.mkdir(parents=True)
    for source in ("vitaldb", "bidmc"):
        v = load_variant(tmp_path / source / "fs25-w30s", per_subject=3)
        assert v.x.shape[1] == 750 and v.rows.group_by("key").len().get_column("len").max() == 3
        # a fake embedding: the dominant frequency and noise, so age is linearly readable
        f = np.abs(np.fft.rfft(v.x, axis=1))[:, 20:60]
        e = np.column_stack([f, np.random.default_rng(0).normal(size=(len(v.x), 8))]).astype(np.float32)
        stem = f"{source}-fs25-w30s"
        v.rows.with_columns(pl.Series("emb", e).cast(pl.Array(pl.Float32, e.shape[1]))).write_parquet(emb / f"{stem}.parquet")
        v.subjects.write_parquet(emb / f"{stem}.subjects.parquet")
    m = run(emb, "vitaldb-fs25-w30s", ["bidmc-fs25-w30s"], tmp_path / "probe")
    assert m["age"]["test_mae"] < m["age"]["test_mae_median_baseline"]
    assert set(m["flags"]) == {"preop_htn", "preop_dm", "asa_ge3", "sex_male"}
    assert m["external"]["bidmc-fs25-w30s"]["with_age"] == 20
    h = run(emb, "vitaldb-fs25-w30s", ["bidmc-fs25-w30s"], tmp_path / "hr", features="hr")
    assert h["age"]["test_mae"] == pytest.approx(h["age"]["test_mae"])
