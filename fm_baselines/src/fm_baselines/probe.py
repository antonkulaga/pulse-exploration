"""Linear probes on subject-level mean embeddings, on kron-ppg's person-level split.

For each embedding set (model x rate) trained on ``vitaldb``:

* **age**: ridge regression, alpha chosen on validation MAE, fitted on train. Reported per person: MAE and
  Pearson r on test, against predicting the training median; then on the external sets (BIDMC, GalaxyPPG)
  with the VitalDB fit unchanged.
* **age gap**: the prediction minus its validation-fitted line on age (the bias correction kron-ppg uses),
  and the AUC of that gap for each flag on test, beside age's own AUC. This is the outcome check: a clock
  is worth having only if its gap carries risk.
* **flags** (``preop_htn``, ``preop_dm``, ``asa_ge3``, ``sex_male``): logistic regression, C chosen on
  validation AUC, with three feature sets: ``age+sex`` alone, the embedding alone, and embedding+age+sex.
  The paired bootstrap of (embedding+age+sex) - (age+sex) on test says whether PPG adds anything.

A feature set named ``hr`` (mean and SD of each window's heart rate and periodicity) is the no-model
baseline, built from the same windows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

FLAGS = {
    "preop_htn": lambda s: _num(s, "preop_htn") >= 1,
    "preop_dm": lambda s: _num(s, "preop_dm") >= 1,
    "asa_ge3": lambda s: _num(s, "asa") >= 3,
    "sex_male": lambda s: _sex(s),
}
ALPHAS = np.logspace(-1, 5, 13)
CS = np.logspace(-4, 1, 11)
BOOT = 1000


def _num(s: pl.DataFrame, col: str) -> np.ndarray:
    if col not in s.columns:
        return np.full(len(s), np.nan)
    return s.get_column(col).cast(pl.Float64, strict=False).fill_null(np.nan).to_numpy()


def _sex(s: pl.DataFrame) -> np.ndarray:
    if "sex" not in s.columns:
        return np.full(len(s), np.nan)
    m = s.get_column("sex").cast(pl.String).str.to_lowercase()
    return np.where(m.is_in(["m", "male"]).to_numpy(), 1.0, np.where(m.is_in(["f", "female"]).to_numpy(), 0.0,
                                                                       np.nan))


def subject_table(windows: pl.DataFrame, subjects: pl.DataFrame, features: str) -> tuple[pl.DataFrame, np.ndarray]:
    """One row per subject (key, split, age, flags as 0/1/NaN, n windows) and its feature matrix."""
    keys = windows.get_column("key").to_numpy()
    uniq, first, inv, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    if features == "hr":
        v = windows.select("hr_bpm", "periodicity").to_numpy().astype(np.float64)
        mean = np.zeros((len(uniq), 2))
        np.add.at(mean, inv, v)
        mean /= counts[:, None]
        sq = np.zeros((len(uniq), 2))
        np.add.at(sq, inv, (v - mean[inv]) ** 2)
        x = np.column_stack([mean, np.sqrt(sq / np.maximum(counts - 1, 1)[:, None])])
    else:
        emb = windows.get_column("emb").to_numpy().astype(np.float64)
        x = np.zeros((len(uniq), emb.shape[1]))
        np.add.at(x, inv, emb)
        x /= counts[:, None]
    agg = pl.DataFrame({"key": uniq, "split": windows.get_column("split").to_numpy()[first],
                        "n_windows": counts})
    table = agg.join(subjects.drop("subject"), on="key", how="left", maintain_order="left")
    age = _num(table, "age")
    flags = {name: f(table).astype(np.float64) for name, f in FLAGS.items()}
    table = table.select("key", "split", "n_windows").with_columns(
        pl.Series("age", age), *[pl.Series(k, v) for k, v in flags.items()])
    return table, x


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _boot_ci(y: np.ndarray, *scores: np.ndarray, seed: int = 0) -> list[float]:
    """95% bootstrap interval of AUC(scores[0]), or of AUC(scores[0]) - AUC(scores[1]) when two are given."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y) & np.all([np.isfinite(s) for s in scores], axis=0)
    y = y[ok]
    scores = [s[ok] for s in scores]
    vals = []
    for _ in range(BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        a = roc_auc_score(y[i], scores[0][i])
        vals.append(a - roc_auc_score(y[i], scores[1][i]) if len(scores) > 1 else a)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else []


def _fit_ridge(x, y, tr, va):
    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(x[tr], y[tr])
        mae = float(np.mean(np.abs(m.predict(x[va]) - y[va])))
        if best is None or mae < best[0]:
            best = (mae, a, m)
    return best[2], float(best[1])


def _fit_logistic(x, y, tr, va):
    best = None
    for c in CS:
        m = LogisticRegression(C=c, max_iter=5000).fit(x[tr], y[tr])
        auc = _auc(y[va], m.decision_function(x[va]))
        if best is None or auc > best[0]:
            best = (auc, c, m)
    return best[2], float(best[1])


def _r(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def probe(train: tuple[pl.DataFrame, np.ndarray], externals: dict[str, tuple[pl.DataFrame, np.ndarray]]) -> tuple[dict, pl.DataFrame]:
    table, x = train
    split = table.get_column("split").to_numpy()
    age = table.get_column("age").to_numpy()
    scaler = StandardScaler().fit(x[split == "train"])
    z = scaler.transform(x)
    out: dict = {"subjects": {s: int((split == s).sum()) for s in ("train", "val", "test")}}

    # age
    has = np.isfinite(age)
    tr, va, te = (has & (split == "train")), (has & (split == "val")), (has & (split == "test"))
    ridge, alpha = _fit_ridge(z, age, tr, va)
    pred = ridge.predict(z)
    median = float(np.median(age[tr]))
    slope, intercept = np.polyfit(age[va], pred[va], 1)
    gap = pred - (intercept + slope * age)
    out["age"] = {
        "alpha": alpha,
        **{f"{name}_mae": round(float(np.mean(np.abs(pred[m] - age[m]))), 3) for name, m in
           (("train", tr), ("val", va), ("test", te))},
        "test_r": round(_r(pred[te], age[te]), 4),
        "test_mae_median_baseline": round(float(np.mean(np.abs(median - age[te]))), 3),
        "gap_bias_fit_on_val": {"slope": round(float(slope), 4), "intercept": round(float(intercept), 3)},
        "test_gap_r_with_age": round(_r(gap[te], age[te]), 4),
    }
    preds = {"pred_age": pred, "age_gap": gap}

    # flags: age+sex, embedding, embedding+age+sex; and the age gap's own AUC
    sex = table.get_column("sex_male").to_numpy()
    out["flags"] = {}
    for flag in FLAGS:
        y = table.get_column(flag).to_numpy()
        ok = np.isfinite(y) & has & (np.isfinite(sex) | (flag == "sex_male"))
        if ok.sum() == 0 or len(np.unique(y[ok & (split == "train")])) < 2:
            continue
        base = np.column_stack([age] if flag == "sex_male" else [age, np.nan_to_num(sex, nan=0.5)])
        base = np.nan_to_num((base - base[tr].mean(0)) / base[tr].std(0))  # rows without age are never scored
        res = {"n_test": int((ok & (split == "test")).sum()), "positives_test": int(np.nansum(y[ok & (split == "test")]))}
        scores = {}
        for name, feats in (("age_sex", base), ("ppg", z), ("ppg_age_sex", np.column_stack([z, base]))):
            m, c = _fit_logistic(feats, y, ok & (split == "train"), ok & (split == "val"))
            scores[name] = np.where(ok, m.decision_function(feats), np.nan)
            t = ok & (split == "test")
            res[name] = {"C": c, "test_auc": round(_auc(y[t], scores[name][t]), 4),
                         "test_auc_ci": _boot_ci(y[t], scores[name][t])}
        t = ok & (split == "test")
        res["delta_ppg_age_sex_minus_age_sex"] = {
            "test": round(res["ppg_age_sex"]["test_auc"] - res["age_sex"]["test_auc"], 4),
            "ci": _boot_ci(y[t], scores["ppg_age_sex"][t], scores["age_sex"][t])}
        if flag != "sex_male":
            res["age_alone_test_auc"] = round(_auc(y[t], age[t]), 4)
            res["age_gap_test_auc"] = round(_auc(y[t], gap[t]), 4)
            res["age_gap_test_auc_ci"] = _boot_ci(y[t], gap[t])
        out["flags"][flag] = res
        preds[f"p_{flag}"] = scores["ppg_age_sex"]

    # external sets with the VitalDB fit unchanged
    out["external"] = {}
    sex_model = None
    if "sex_male" in out["flags"]:
        y = table.get_column("sex_male").to_numpy()
        ok = np.isfinite(y)
        sex_model, _ = _fit_logistic(z, y, ok & (split == "train"), ok & (split == "val"))
    for name, (etable, ex) in externals.items():
        ez = scaler.transform(ex)
        eage = etable.get_column("age").to_numpy()
        ep = ridge.predict(ez)
        m = np.isfinite(eage)
        res = {"subjects": int(len(etable)), "with_age": int(m.sum())}
        if m.sum():
            res.update(mae=round(float(np.mean(np.abs(ep[m] - eage[m]))), 3), r=round(_r(ep[m], eage[m]), 4),
                       mae_median_baseline=round(float(np.mean(np.abs(median - eage[m]))), 3),
                       mean_pred=round(float(ep[m].mean()), 2), mean_age=round(float(eage[m].mean()), 2))
        esex = etable.get_column("sex_male").to_numpy()
        if sex_model is not None and np.isfinite(esex).sum():
            res["sex_auc"] = round(_auc(esex, sex_model.decision_function(ez)), 4)
        out["external"][name] = res

    predictions = table.with_columns(*[pl.Series(k, v) for k, v in preds.items()])
    return out, predictions


def run(emb_dir: Path, train: str, externals: list[str], out: Path, features: str = "emb") -> dict:
    """``train`` and ``externals`` are file stems in ``emb_dir`` such as ``vitaldb-fs25-w30s``."""
    def load(stem: str):
        windows = pl.read_parquet(emb_dir / f"{stem}.parquet")
        subjects = pl.read_parquet(emb_dir / f"{stem}.subjects.parquet")
        return subject_table(windows, subjects, features)

    metrics, predictions = probe(load(train), {s: load(s) for s in externals if (emb_dir / f"{s}.parquet").exists()})
    metrics = {"model": emb_dir.name if features == "emb" else "hr", "train": train, **metrics}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    predictions.write_parquet(out / "predictions.parquet")
    return metrics


def summarize(results: Path) -> str:
    """A markdown table over every ``<results>/<model>/<train>/metrics.json``."""
    rows = []
    for f in sorted(results.glob("*/*/metrics.json")):
        m = json.loads(f.read_text())
        a = m["age"]
        fl = m.get("flags", {})
        ext = m.get("external", {})
        bidmc = next((v for k, v in ext.items() if k.startswith("bidmc")), {})
        galaxy = next((v for k, v in ext.items() if k.startswith("galaxyppg")), {})

        def cell(flag, key):
            v = fl.get(flag, {})
            if key == "gap":
                return f"{v.get('age_gap_test_auc', float('nan')):.2f}" if v else "–"
            d = v.get("delta_ppg_age_sex_minus_age_sex", {})
            ci = d.get("ci") or [float("nan")] * 2
            return f"{v['ppg']['test_auc']:.2f} / {d.get('test', float('nan')):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if v else "–"

        rows.append(
            f"| {m['model']} | {m['train'].split('-')[1]} | {a['test_mae']:.2f} ({a['test_mae_median_baseline']:.2f}) "
            f"| {a['test_r']:.2f} | {bidmc.get('mae', float('nan')):.2f} / {bidmc.get('r', float('nan')):.2f} "
            f"| {galaxy.get('mean_pred', float('nan')):.1f} | {cell('sex_male', 'auc')} "
            f"| {cell('preop_htn', 'auc')} | {cell('preop_dm', 'auc')} | {cell('asa_ge3', 'auc')} "
            f"| {cell('preop_htn', 'gap')} / {cell('preop_dm', 'gap')} / {cell('asa_ge3', 'gap')} |")
    head = ("| model | rate | test age MAE (median baseline) | test r | BIDMC MAE / r | GalaxyPPG mean predicted age "
            "| sex AUC / Δ over age | HTN AUC / Δ over age+sex [95% CI] | DM | ASA≥3 | age-gap AUC HTN / DM / ASA |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|")
    return head + "\n" + "\n".join(rows) + "\n"
