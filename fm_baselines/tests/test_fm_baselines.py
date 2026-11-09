import json

import numpy as np
import polars as pl
import pytest

from fm_baselines.data import kron_split, load_variant, pick_rows
from fm_baselines.labels import egfr_ckd_epi_2021, label_table
from fm_baselines.probe import evaluate, kron_head_scores, probe_dir, summarize


def test_split_matches_kron_ppg():
    # values from kron_ppg.data.subject_split(keys, "kron-ppg") in kron-fm b0c2f40
    want = {"vitaldb/1": "test", "vitaldb/2": "train", "vitaldb/5": "test", "vitaldb/7": "val",
            "vitaldb/9": "test", "vitaldb/2487": "train", "bidmc/01": "train", "galaxyppg/P01": "train"}
    assert {k: kron_split(k) for k in want} == want


def test_pick_rows_spreads_and_caps():
    table = pl.DataFrame({"subject": ["a"] * 10 + ["b"] * 2, "recording": ["r"] * 12,
                          "start_s": [float(i) for i in range(10)] + [0.0, 30.0]})
    assert pick_rows(table, 4).tolist() == [0, 2, 5, 7, 10, 11]
    assert pick_rows(table, 0).tolist() == list(range(12))


def test_egfr_ckd_epi_2021():
    e = egfr_ckd_epi_2021(np.array([1.0, 0.7, np.nan]), np.array([60.0, 60.0, 60.0]), np.array([1.0, 0.0, 1.0]))
    assert e[0] == pytest.approx(86.2, abs=0.2)
    assert e[1] == pytest.approx(99.0, abs=0.2)
    assert np.isnan(e[2])


def test_label_table_joins_cases(tmp_path):
    subjects = pl.DataFrame({"subject": ["1", "2", "3"], "key": ["vitaldb/1", "vitaldb/2", "vitaldb/3"],
                             "age": [70.0, 40.0, 55.0], "sex": ["M", "F", "F"], "bmi": [25.0, 22.0, 30.0],
                             "asa": [3, 1, 2], "preop_htn": [1, 0, 0], "preop_dm": [0, 0, 1], "caseid": [11, 12, 13]})
    pl.DataFrame({"caseid": [11, 12, 13], "preop_cr": [2.0, 0.6, None], "preop_alb": [3.0, 4.5, 4.0],
                  "preop_hb": [12.5, 11.0, 13.0], "preop_gluc": [140, 90, 200],
                  "preop_pft": ["Mild obstructive", "Normal", None]}).write_csv(tmp_path / "cases.csv.gz",
                                                                               compression="gzip")
    lab = label_table(subjects, tmp_path / "cases.csv.gz")
    assert lab.get_column("ckd").to_list()[:2] == [1.0, 0.0] and np.isnan(lab["ckd"][2])
    assert lab.get_column("anemia").to_list() == [1.0, 1.0, 0.0]
    assert lab.get_column("low_albumin").to_list() == [1.0, 0.0, 0.0]
    assert lab.get_column("pft_abnormal").to_list()[:2] == [1.0, 0.0] and np.isnan(lab["pft_abnormal"][2])
    assert lab.get_column("asa_ge3").to_list() == [1.0, 0.0, 0.0]
    assert lab.get_column("sex_male").to_list() == [1.0, 0.0, 0.0]


def _variant(folder, source, n_subjects, fs=25.0, seed=0, flags=True):
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    length = int(30 * fs)
    subjects, rows, sig = [], [], []
    for i in range(n_subjects):
        age = float(rng.uniform(20, 80))
        s = {"subject": str(i), "age": age, "sex": "M" if i % 2 else "F", "bmi": 20 + rng.normal(0, 3)}
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
    _variant(tmp_path / "vitaldb" / "fs25-w30s", "vitaldb", 400)
    _variant(tmp_path / "bidmc" / "fs25-w30s", "bidmc", 20, seed=1, flags=False)
    emb = tmp_path / "emb" / "fake"
    emb.mkdir(parents=True)
    for source in ("vitaldb", "bidmc"):
        v = load_variant(tmp_path / source / "fs25-w30s", per_subject=3)
        assert v.x.shape[1] == 750 and v.rows.group_by("key").len().get_column("len").max() == 3
        # a fake embedding: the spectrum around the pulse rate, so age is linearly readable
        f = np.abs(np.fft.rfft(v.x, axis=1))[:, 20:60]
        e = np.column_stack([f, np.random.default_rng(0).normal(size=(len(v.x), 8))]).astype(np.float32)
        stem = f"{source}-fs25-w30s"
        v.rows.with_columns(pl.Series("emb", e).cast(pl.Array(pl.Float32, e.shape[1]))).write_parquet(emb / f"{stem}.parquet")
        v.subjects.write_parquet(emb / f"{stem}.subjects.parquet")
    m = probe_dir(emb, "vitaldb-fs25-w30s", ["bidmc-fs25-w30s"], tmp_path / "probe" / "fake" / "vitaldb-fs25-w30s", None)
    assert m["age"]["test_mae"] < m["age"]["test_mae_median_baseline"]
    assert {"preop_htn", "asa_ge3", "sex_male", "bmi"} <= set(m["traits"])
    assert m["external"]["bidmc-fs25-w30s"]["with_age"] == 20
    probe_dir(emb, "vitaldb-fs25-w30s", ["bidmc-fs25-w30s"], tmp_path / "probe" / "hr" / "vitaldb-fs25-w30s", None,
              features="hr", name="hr")
    text = summarize(tmp_path / "probe")
    assert "| fake | fs25 |" in text and "| hr | fs25 |" in text


def test_kron_head_scores_and_evaluate(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    age = rng.uniform(20, 80, n)
    keys = [f"vitaldb/{i}" for i in range(n)]
    split = np.where(np.arange(n) % 2, "val", "test")
    htn = (age + rng.normal(0, 10, n) > 55).astype(float)
    pl.DataFrame({"split": split, "subject": keys, "age": age, "age_predicted": age + rng.normal(0, 5, n),
                  "preop_htn": htn, "preop_htn_predicted": htn + rng.normal(0, 1, n)}).write_parquet(tmp_path / "predictions.parquet")
    s = kron_head_scores(tmp_path)
    assert set(s.columns) >= {"key", "split", "score_age", "score_preop_htn"}
    labels = pl.DataFrame({"key": keys, "age": age, "sex_male": (np.arange(n) % 3 == 0).astype(float),
                           "preop_htn": htn})
    m = evaluate(s, labels)
    assert m["age"]["test_mae"] < m["age"]["test_mae_median_baseline"]
    assert m["traits"]["preop_htn"]["delta"] > 0
