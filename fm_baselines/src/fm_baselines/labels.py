"""Per-person targets for aging and health, from a prepared variant's subject table and VitalDB's case table.

kron-ppg's subject table carries ``age``, ``sex``, ``bmi``, ``asa``, ``preop_htn``, ``preop_dm`` and ``caseid``.
VitalDB's ``cases.csv.gz`` (in kron-fm's ``data/input/ppg/vitaldb``) adds the preoperative labs and lung
function, joined on ``caseid``. Targets:

| name | kind | from | why it matters for aging |
|---|---|---|---|
| ``age`` | regression | age | the clock |
| ``bmi`` | regression | bmi | adiposity |
| ``egfr`` | regression | CKD-EPI 2021 from creatinine, age, sex | kidney function falls with age |
| ``albumin`` | regression | preop_alb, g/dL | frailty and nutrition; a PhenoAge input |
| ``hemoglobin`` | regression | preop_hb, g/dL | anemia of aging |
| ``glucose`` | regression | preop_gluc, mg/dL (not always fasting) | glycaemia; a PhenoAge input |
| ``preop_htn`` | binary | preop_htn | hypertension |
| ``preop_dm`` | binary | preop_dm | diabetes |
| ``asa_ge3`` | binary | asa >= 3 | severe systemic disease |
| ``ckd`` | binary | eGFR < 60 | chronic kidney disease |
| ``anemia`` | binary | Hb < 13 (men), < 12 (women) | WHO anemia |
| ``low_albumin`` | binary | albumin < 3.5 g/dL | hypoalbuminemia |
| ``pft_abnormal`` | binary | preop_pft not "Normal" | obstructive or restrictive lung function |
| ``sex_male`` | binary | sex | a control the literature says PPG reads well |

Atrial fibrillation on the preoperative ECG (16 of 6,388 cases) and in-hospital death (57) are too rare
for a test split of about 900 people, so they are left out.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

REGRESSION = ("age", "bmi", "egfr", "albumin", "hemoglobin", "glucose")
BINARY = ("preop_htn", "preop_dm", "asa_ge3", "ckd", "anemia", "low_albumin", "pft_abnormal", "sex_male")
TRAITS = REGRESSION + BINARY


def egfr_ckd_epi_2021(cr: np.ndarray, age: np.ndarray, male: np.ndarray) -> np.ndarray:
    """eGFR in mL/min/1.73 m2 (Inker et al., NEJM 2021, race-free), NaN where an input is missing."""
    kappa = np.where(male == 1, 0.9, 0.7)
    alpha = np.where(male == 1, -0.302, -0.241)
    ratio = cr / kappa
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200 * 0.9938 ** age
             * np.where(male == 1, 1.0, 1.012))
    return np.where(np.isfinite(cr) & (cr > 0) & np.isfinite(age) & np.isfinite(male), e, np.nan)


def _num(df: pl.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return df.get_column(col).cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()


def _flag(values: np.ndarray, test) -> np.ndarray:
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    out[ok] = test(values[ok]).astype(float)
    return out


def label_table(subjects: pl.DataFrame, cases: Path | None = None) -> pl.DataFrame:
    """``key`` plus one Float64 column per trait (NaN when unknown) for every subject of a variant."""
    df = subjects
    if cases is not None and Path(cases).exists() and "caseid" in df.columns:
        extra = (pl.read_csv(cases, infer_schema_length=10_000)
                 .select("caseid", "preop_cr", "preop_alb", "preop_hb", "preop_gluc", "preop_pft")
                 .with_columns(pl.col("caseid").cast(pl.Int64)))
        df = df.with_columns(pl.col("caseid").cast(pl.Int64, strict=False)).join(extra, on="caseid", how="left")
    age = _num(df, "age")
    sex = (df.get_column("sex").cast(pl.String).str.strip_chars().str.to_lowercase()
           if "sex" in df.columns else pl.Series([None] * len(df), dtype=pl.String))
    male = np.where(sex.is_in(["m", "male"]).to_numpy(), 1.0, np.where(sex.is_in(["f", "female"]).to_numpy(), 0.0,
                                                                        np.nan))
    egfr = egfr_ckd_epi_2021(_num(df, "preop_cr"), age, male)
    alb, hb = _num(df, "preop_alb"), _num(df, "preop_hb")
    pft = (df.get_column("preop_pft").cast(pl.String).str.strip_chars() if "preop_pft" in df.columns
           else pl.Series([None] * len(df), dtype=pl.String))
    pft_abnormal = np.where(pft.is_null().to_numpy() | (pft == "").fill_null(True).to_numpy(), np.nan,
                            (pft != "Normal").fill_null(False).to_numpy().astype(float))
    anemia = np.where(np.isfinite(hb) & np.isfinite(male), (hb < np.where(male == 1, 13.0, 12.0)).astype(float),
                      np.nan)
    cols = {
        "age": age, "bmi": _num(df, "bmi"), "egfr": egfr, "albumin": alb, "hemoglobin": hb,
        "glucose": _num(df, "preop_gluc"),
        "preop_htn": _flag(_num(df, "preop_htn"), lambda v: v >= 1),
        "preop_dm": _flag(_num(df, "preop_dm"), lambda v: v >= 1),
        "asa_ge3": _flag(_num(df, "asa"), lambda v: v >= 3),
        "ckd": _flag(egfr, lambda v: v < 60), "anemia": anemia,
        "low_albumin": _flag(alb, lambda v: v < 3.5), "pft_abnormal": pft_abnormal, "sex_male": male,
    }
    return pl.DataFrame({"key": df.get_column("key").cast(pl.String), **{k: pl.Series(k, v, dtype=pl.Float64)
                                                                          for k, v in cols.items()}})
