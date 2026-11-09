"""From embeddings to per-person trait scores, and one evaluation for every model.

**Scores** (:func:`fit_scores`). A person's feature is the mean of their window embeddings. For each trait in
:mod:`fm_baselines.labels`, a linear probe is trained on VitalDB *train* people:

- ridge for numbers, logistic regression for flags;
- the penalty is chosen on *validation* people.

Its output on everyone is the PPG score. The ``hr`` feature set (mean and SD of each window's heart rate and
periodicity) is the no-model baseline.

**Evaluation** (:func:`evaluate`). The same function judges any table of scores: frozen foundation models,
frozen KronLSTM encoders, and kron-ppg's own trained heads. It never looks at train people, so a head that
overfits its training people is judged fairly.

- **age**: test MAE and r against the validation median. The age gap is the prediction minus its line on
  age fitted on validation people. The gap's AUC is reported for each flag on test people, beside age's own.
- **every other trait**: does the PPG score say more than age and sex?
  - Two models are fitted on validation people and scored on test people: age+sex, and age+sex+score.
    They are logistic for flags (AUC) and linear for numbers (R²).
  - The paired bootstrap of the difference has a 95% interval. An interval above 0 means the PPG adds
    something.
  - The score alone is reported too (AUC or r).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .labels import BINARY, REGRESSION, TRAITS

ALPHAS = np.logspace(-1, 5, 13)
CS = np.logspace(-4, 1, 11)
BOOT = 1000
MIN_POSITIVES = 20


# ---- features ------------------------------------------------------------------------------------
def subject_features(windows: pl.DataFrame, features: str = "emb") -> tuple[pl.DataFrame, np.ndarray]:
    """One row per person (``key``, ``split``, ``n_windows``) and the mean of their windows' features."""
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
    table = pl.DataFrame({"key": uniq.astype(str), "split": windows.get_column("split").to_numpy()[first],
                          "n_windows": counts})
    return table, x


# ---- stage 1: scores -----------------------------------------------------------------------------
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(s)
    if len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def fit_scores(table: pl.DataFrame, x: np.ndarray, labels: pl.DataFrame,
               traits: tuple[str, ...] = TRAITS) -> tuple[pl.DataFrame, dict, dict]:
    """PPG scores for every person and trait, the chosen penalties, and the fitted (scaler, model) per trait."""
    lab = table.select("key").join(labels, on="key", how="left", maintain_order="left")
    split = table.get_column("split").to_numpy()
    scaler = StandardScaler().fit(x[split == "train"])
    z = scaler.transform(x)
    scores, chosen, models = {}, {}, {}
    for t in traits:
        y = lab.get_column(t).to_numpy() if t in lab.columns else np.full(len(lab), np.nan)
        ok = np.isfinite(y)
        tr, va = ok & (split == "train"), ok & (split == "val")
        if tr.sum() < 50 or va.sum() < 20:
            continue
        if t in BINARY:
            if min(y[tr].sum(), (1 - y[tr]).sum()) < MIN_POSITIVES or len(np.unique(y[va])) < 2:
                continue
            best = None
            for c in CS:
                m = LogisticRegression(C=c, max_iter=5000).fit(z[tr], y[tr])
                auc = _auc(y[va], m.decision_function(z[va]))
                if best is None or auc > best[0]:
                    best = (auc, c, m)
            scores[t], chosen[t], models[t] = best[2].decision_function(z), {"C": float(best[1])}, best[2]
        else:
            best = None
            for a in ALPHAS:
                m = Ridge(alpha=a).fit(z[tr], y[tr])
                mae = float(np.mean(np.abs(m.predict(z[va]) - y[va])))
                if best is None or mae < best[0]:
                    best = (mae, a, m)
            scores[t], chosen[t], models[t] = best[2].predict(z), {"alpha": float(best[1])}, best[2]
    out = table.select("key", "split", "n_windows").with_columns(
        *[pl.Series(f"score_{t}", v, dtype=pl.Float64) for t, v in scores.items()])
    return out, chosen, {t: (scaler, m) for t, m in models.items()}


def apply_scores(table: pl.DataFrame, x: np.ndarray, models: dict) -> pl.DataFrame:
    """Scores of people outside VitalDB with the VitalDB fits unchanged."""
    cols = []
    for t, (scaler, m) in models.items():
        z = scaler.transform(x)
        cols.append(pl.Series(f"score_{t}", m.decision_function(z) if t in BINARY else m.predict(z),
                              dtype=pl.Float64))
    return table.select("key", "split", "n_windows").with_columns(*cols)


# ---- stage 2: evaluation -------------------------------------------------------------------------
def _boot(fn, n: int, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT):
        v = fn(rng.integers(0, n, n))
        if np.isfinite(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)] if vals else []


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def _r(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def evaluate(scores: pl.DataFrame, labels: pl.DataFrame) -> dict:
    """Test-set metrics of every ``score_<trait>`` column; fits only on validation people."""
    df = scores.join(labels, on="key", how="left", maintain_order="left")
    split = df.get_column("split").to_numpy()
    age = df.get_column("age").to_numpy()
    male = df.get_column("sex_male").to_numpy()
    va_all, te_all = split == "val", split == "test"
    out: dict = {"people": {"val": int(va_all.sum()), "test": int(te_all.sum())}, "traits": {}}

    if "score_age" in df.columns:
        pred = df.get_column("score_age").to_numpy()
        va, te = va_all & np.isfinite(age), te_all & np.isfinite(age)
        median = float(np.median(age[va]))
        slope, intercept = np.polyfit(age[va], pred[va], 1)
        gap = pred - (intercept + slope * age)
        out["age"] = {
            "test_people": int(te.sum()),
            "test_mae": round(float(np.mean(np.abs(pred[te] - age[te]))), 3),
            "test_r": round(_r(pred[te], age[te]), 4),
            "test_mae_median_baseline": round(float(np.mean(np.abs(median - age[te]))), 3),
            "val_mae": round(float(np.mean(np.abs(pred[va] - age[va]))), 3),
            "gap_line_on_val": {"slope": round(float(slope), 4), "intercept": round(float(intercept), 3)},
            "test_gap_r_with_age": round(_r(gap[te], age[te]), 4),
            "gap_auc": {},
        }
        for t in BINARY:
            if t == "sex_male" or t not in df.columns:
                continue
            y = df.get_column(t).to_numpy()
            m = te & np.isfinite(y)
            if m.sum() and len(np.unique(y[m])) == 2:
                yy, gg = y[m], gap[m]
                out["age"]["gap_auc"][t] = {
                    "auc_gap": round(_auc(yy, gg), 4),
                    "ci": _boot(lambda i: _auc(yy[i], gg[i]), len(yy)),
                    "auc_age": round(_auc(yy, age[m]), 4), "positives": int(yy.sum()), "n": int(m.sum())}

    for t in TRAITS:
        col = f"score_{t}"
        if t == "age" or col not in df.columns or t not in df.columns:
            continue
        y = df.get_column(t).to_numpy()
        s = df.get_column(col).to_numpy()
        base_cols = [age] if t == "sex_male" else [age, male]
        base = np.column_stack(base_cols)
        ok = np.isfinite(y) & np.isfinite(s) & np.all(np.isfinite(base), axis=1)
        va, te = ok & va_all, ok & te_all
        if va.sum() < 20 or te.sum() < 20:
            continue
        mu, sd = base[va].mean(0), base[va].std(0) + 1e-9
        b = (base - mu) / sd
        sz = (s - s[va].mean()) / (s[va].std() + 1e-9)
        full = np.column_stack([b, sz])
        res: dict = {"test_n": int(te.sum())}
        if t in BINARY:
            if len(np.unique(y[va])) < 2 or len(np.unique(y[te])) < 2:
                continue
            res["test_positives"] = int(y[te].sum())
            m0 = LogisticRegression(C=1.0, max_iter=5000).fit(b[va], y[va])
            m1 = LogisticRegression(C=1.0, max_iter=5000).fit(full[va], y[va])
            p0, p1 = m0.decision_function(b[te]), m1.decision_function(full[te])
            yy, ss = y[te], s[te]
            res.update(
                auc_ppg=round(_auc(yy, ss), 4), auc_ppg_ci=_boot(lambda i: _auc(yy[i], ss[i]), len(yy)),
                auc_base=round(_auc(yy, p0), 4), auc_base_plus_ppg=round(_auc(yy, p1), 4),
                delta=round(_auc(yy, p1) - _auc(yy, p0), 4),
                delta_ci=_boot(lambda i: _auc(yy[i], p1[i]) - _auc(yy[i], p0[i]), len(yy)),
                ppg_weight=round(float(m1.coef_[0, -1]), 4))
        else:
            m0 = LinearRegression().fit(b[va], y[va])
            m1 = LinearRegression().fit(full[va], y[va])
            p0, p1 = m0.predict(b[te]), m1.predict(full[te])
            yy, ss = y[te], s[te]
            res.update(
                r_ppg=round(_r(yy, ss), 4), mae_ppg=round(float(np.mean(np.abs(yy - ss))), 3),
                r2_base=round(_r2(yy, p0), 4), r2_base_plus_ppg=round(_r2(yy, p1), 4),
                delta=round(_r2(yy, p1) - _r2(yy, p0), 4),
                delta_ci=_boot(lambda i: _r2(yy[i], p1[i]) - _r2(yy[i], p0[i]), len(yy)))
        out["traits"][t] = res
    return out


def evaluate_external(scores: pl.DataFrame, labels: pl.DataFrame) -> dict:
    """Age and sex on another source, with the VitalDB fits unchanged."""
    df = scores.join(labels, on="key", how="left", maintain_order="left")
    out = {"people": len(df)}
    if "score_age" in df.columns:
        age, pred = df.get_column("age").to_numpy(), df.get_column("score_age").to_numpy()
        m = np.isfinite(age)
        if m.sum():
            out.update(with_age=int(m.sum()), mae=round(float(np.mean(np.abs(pred[m] - age[m]))), 3),
                       r=round(_r(pred[m], age[m]), 4), mean_pred=round(float(pred[m].mean()), 2),
                       mean_age=round(float(age[m].mean()), 2))
    if "score_sex_male" in df.columns:
        out["sex_auc"] = round(_auc(df.get_column("sex_male").to_numpy(), df.get_column("score_sex_male").to_numpy()), 4)
    return out


# ---- runs over folders ---------------------------------------------------------------------------
def probe_dir(emb_dir: Path, train: str, externals: list[str], out: Path, cases: Path | None,
              features: str = "emb", name: str | None = None) -> dict:
    """Stage 1 + 2 for one embedding set: ``<emb_dir>/<train>.parquet`` and its external sets."""
    from .labels import label_table

    def load(stem: str):
        windows = pl.read_parquet(emb_dir / f"{stem}.parquet")
        labels = label_table(pl.read_parquet(emb_dir / f"{stem}.subjects.parquet"), cases)
        table, x = subject_features(windows, features)
        return table, x, labels

    table, x, labels = load(train)
    scores, chosen, models = fit_scores(table, x, labels)
    metrics = {"model": name or emb_dir.name, "train": train, "penalties": chosen, **evaluate(scores, labels),
               "external": {}}
    all_scores = [scores]
    for stem in externals:
        if (emb_dir / f"{stem}.parquet").exists():
            et, ex, el = load(stem)
            es = apply_scores(et, ex, {t: models[t] for t in ("age", "sex_male") if t in models})
            metrics["external"][stem] = evaluate_external(es, el)
            all_scores.append(es.with_columns(pl.lit("external").alias("split")))
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pl.concat(all_scores, how="diagonal").write_parquet(out / "scores.parquet")
    return metrics


def kron_head_scores(run: Path) -> pl.DataFrame:
    """kron-ppg's own predictions (a ``traits`` or ``clock`` run) as a score table keyed like ours."""
    if (run / "predictions.parquet").exists():
        p = pl.read_parquet(run / "predictions.parquet")
    else:
        p = pl.read_csv(run / "predictions.csv", schema_overrides={"subject": pl.String})
    p = p.rename({"subject": "key"}).with_columns(pl.col("key").cast(pl.String))
    if "predicted" in p.columns:  # an age clock
        cols = [pl.col("predicted").cast(pl.Float64).alias("score_age")]
    else:
        cols = [pl.col(c).cast(pl.Float64).alias("score_" + c.removesuffix("_predicted"))
                for c in p.columns if c.endswith("_predicted")]
    n = pl.col("windows") if "windows" in p.columns else pl.lit(None, dtype=pl.Int64)
    return p.select("key", "split", n.alias("n_windows"), *cols).filter(pl.col("split").is_in(["val", "test"]))


def kron_head_eval(run: Path, labels: pl.DataFrame, out: Path) -> dict:
    scores = kron_head_scores(run)
    metrics = {"model": f"kron-head:{run.name}", "train": "vitaldb-fs25-w30s", **evaluate(scores, labels)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    scores.write_parquet(out / "scores.parquet")
    return metrics


# ---- summary -------------------------------------------------------------------------------------
def _fmt_delta(r: dict) -> str:
    ci = r.get("delta_ci") or [float("nan")] * 2
    mark = " **" if ci and ci[0] > 0 else ""
    return f"{r['delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]{mark}"


def summarize(results: Path) -> str:
    """Markdown tables over every ``<results>/<model>/<train>/metrics.json``."""
    runs = [json.loads(f.read_text()) for f in sorted(results.glob("*/*/metrics.json"))]
    runs = [m for m in runs if "traits" in m]
    lines = ["## Age", "",
             "| model | rate | test MAE (median baseline) | test r | BIDMC MAE / r | GalaxyPPG mean predicted (true 20–29) |",
             "|---|---|---|---|---|---|"]
    for m in runs:
        a = m.get("age")
        if not a:
            continue
        ext = m.get("external", {})
        bidmc = next((v for k, v in ext.items() if k.startswith("bidmc")), {})
        gal = next((v for k, v in ext.items() if k.startswith("galaxyppg")), {})
        lines.append(f"| {m['model']} | {m['train'].split('-')[1]} | {a['test_mae']:.2f} ({a['test_mae_median_baseline']:.2f}) "
                     f"| {a['test_r']:.2f} | {bidmc.get('mae', float('nan')):.2f} / {bidmc.get('r', float('nan')):.2f} "
                     f"| {gal.get('mean_pred', float('nan')):.1f} |")
    flags = [t for t in BINARY if t != "sex_male"]
    lines += ["", "## Age gap: AUC of the bias-corrected gap on test people (age's own AUC in brackets)", "",
              "| model | rate | " + " | ".join(flags) + " |", "|---|---|" + "---|" * len(flags)]
    for m in runs:
        g = m.get("age", {}).get("gap_auc", {})
        if g:
            lines.append(f"| {m['model']} | {m['train'].split('-')[1]} | " + " | ".join(
                f"{g[t]['auc_gap']:.2f} ({g[t]['auc_age']:.2f})" if t in g else "–" for t in flags) + " |")
    for kind, traits, head in (("Flags", BINARY, "PPG AUC / Δ AUC over age+sex [95% CI]"),
                               ("Numbers", tuple(t for t in REGRESSION if t != "age"), "PPG r / Δ R² over age+sex [95% CI]")):
        lines += ["", f"## {kind}: {head}; ** = interval above 0", "",
                  "| model | rate | " + " | ".join(traits) + " |", "|---|---|" + "---|" * len(traits)]
        for m in runs:
            tr = m["traits"]
            if not any(t in tr for t in traits):
                continue
            cells = []
            for t in traits:
                r = tr.get(t)
                if not r:
                    cells.append("–")
                elif "auc_ppg" in r:
                    cells.append(f"{r['auc_ppg']:.2f} / {_fmt_delta(r)}")
                else:
                    cells.append(f"{r['r_ppg']:.2f} / {_fmt_delta(r)}")
            lines.append(f"| {m['model']} | {m['train'].split('-')[1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
