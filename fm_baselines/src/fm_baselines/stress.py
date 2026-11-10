"""Format stress tests: how far each encoder's readings move when the input format changes.

Probes are fitted on clean training embeddings (the same fits as the main probe run). Validation and test
people's windows are then perturbed, put back through the canonical preparation kron-ppg uses (band-pass
0.5-8 Hz, Butterworth order 2, zero phase; robust scaling per window), embedded and scored. A perturbation
the preparation removes cannot hurt; the ones it leaves in are what a watch would show the model:

| name | what it simulates |
|---|---|
| ``clean`` | the windows as prepared |
| ``flip`` | inverted polarity (GalaxyPPG's watch PPG must be inverted; a reader that misses it) |
| ``lowpass4``, ``lowpass2.5`` | a device or a wrist that damps the pulse above 4 or 2.5 Hz |
| ``rate20`` | a 20 Hz sensor: down to 20 Hz and back to 25 |
| ``lost_samples`` | ``hr_raw``: one sample in every 10-12 s silently lost, timestamps kept even |
| ``noise10db`` | white noise at 10 dB SNR |
| ``motion`` | 1-3 Hz bursts twice the pulse's amplitude over a fifth of the window |

Per encoder and perturbation it reports the full evaluation (:func:`fm_baselines.probe.evaluate`) plus the
stability of each person's age reading against clean: the mean shift, the SD of the shift, and r.

    python -m fm_baselines.stress pulseppg --variant <vitaldb fs25-w30s> --clean-emb <emb>/pulseppg --out <dir>
    python -m fm_baselines.stress kronlstm_pretrain --kron <checkpoint.pt> ...   # in kron-fm's environment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import butter, resample_poly, sosfiltfilt

PERTURBATIONS = ("clean", "flip", "lowpass4", "lowpass2.5", "rate20", "lost_samples", "noise10db", "motion")


def canonical(x: np.ndarray, fs: float) -> np.ndarray:
    """kron-ppg's window preparation: band-pass 0.5-8 Hz (order 2, zero phase), robust scale per window."""
    sos = butter(2, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    padlen = min(x.shape[-1] - 1, 3 * (2 * len(sos) + 1) * 4)
    y = sosfiltfilt(sos, x, axis=-1, padlen=padlen)
    med = np.median(y, axis=-1, keepdims=True)
    q75, q25 = np.percentile(y, [75, 25], axis=-1, keepdims=True)
    return ((y - med) / ((q75 - q25) / 1.349 + 1e-6)).astype(np.float32)


def perturb(x: np.ndarray, fs: float, name: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, length = x.shape
    if name == "clean":
        return x
    if name == "flip":
        y = -x
    elif name.startswith("lowpass"):
        sos = butter(4, float(name.removeprefix("lowpass")), btype="lowpass", fs=fs, output="sos")
        y = sosfiltfilt(sos, x, axis=-1)
    elif name == "rate20":
        y = resample_poly(resample_poly(x, 4, 5, axis=1), 5, 4, axis=1)[:, :length]
    elif name == "lost_samples":
        y = np.empty_like(x)
        for i in range(n):
            keep = np.ones(length, bool)
            t = int(rng.integers(0, int(10 * fs)))
            while t < length:
                keep[t] = False
                t += int(rng.uniform(10, 12) * fs)
            kept = x[i, keep]
            y[i] = np.concatenate([kept, np.full(length - len(kept), kept[-1])])
    elif name == "noise10db":
        power = x.var(axis=1, keepdims=True)
        y = x + rng.normal(size=x.shape) * np.sqrt(power / 10)
    elif name == "motion":
        y = x.copy()
        t = np.arange(length) / fs
        span = length // 5
        amp = 2 * x.std(axis=1)
        for i in range(n):
            start = int(rng.integers(0, length - span))
            f = rng.uniform(1, 3)
            y[i, start:start + span] += amp[i] * np.sin(2 * np.pi * f * t[:span] + rng.uniform(0, 2 * np.pi))
    else:
        raise ValueError(name)
    return canonical(y, fs)


def run(name: str, embed, variant: Path, clean_emb: Path, out: Path, cases: Path | None, per_subject: int) -> dict:
    from .data import load_variant
    from .labels import label_table
    from .probe import apply_scores, evaluate, fit_scores, subject_features

    stem = f"vitaldb-{variant.name}"
    clean_windows = pl.read_parquet(clean_emb / f"{stem}.parquet")
    labels = label_table(pl.read_parquet(clean_emb / f"{stem}.subjects.parquet"), cases)
    table, x = subject_features(clean_windows)
    _, _, models = fit_scores(table, x, labels)

    v = load_variant(variant, per_subject)
    held = v.rows.with_row_index("i").filter(pl.col("split").is_in(["val", "test"]))
    idx = held.get_column("i").to_numpy()
    rows = held.drop("i")
    results, clean_age = {}, None
    for p in PERTURBATIONS:
        xs = perturb(v.x[idx], v.sample_rate, p)
        emb = embed(xs, v.sample_rate)
        w = rows.with_columns(pl.Series("emb", emb.astype(np.float32)).cast(pl.Array(pl.Float32, emb.shape[1])))
        t, f = subject_features(w)
        scores = apply_scores(t, f, models)
        m = evaluate(scores, labels)
        age = scores.select("key", "score_age")
        if p == "clean":
            clean_age = age
        else:
            j = clean_age.join(age, on="key", suffix="_p")
            d = (j["score_age_p"] - j["score_age"]).to_numpy()
            m["age_stability"] = {"mean_shift": round(float(d.mean()), 3), "sd_shift": round(float(d.std()), 3),
                                  "r_with_clean": round(float(np.corrcoef(j["score_age"], j["score_age_p"])[0, 1]), 4)}
        results[p] = m
        print(json.dumps({"encoder": name, "perturbation": p, "age_mae": m.get("age", {}).get("test_mae"),
                          **m.get("age_stability", {})}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"encoder": name, "variant": str(variant), "results": results},
                                                 indent=2) + "\n")
    return results


def summarize(folder: Path) -> str:
    """Markdown: per encoder, per perturbation, test age MAE and stability, and three flags' ΔAUC."""
    lines = ["| encoder | perturbation | age MAE | shift (SD) | r with clean | Δ HTN | Δ CKD | Δ ASA≥3 |",
             "|---|---|---|---|---|---|---|---|"]
    for f in sorted(folder.glob("*.json")):
        d = json.loads(f.read_text())
        for p, m in d["results"].items():
            s = m.get("age_stability", {})
            tr = m.get("traits", {})

            def delta(t):
                return f"{tr[t]['delta']:+.3f}" if t in tr else "–"
            shift = f"{s['mean_shift']:+.1f} ({s['sd_shift']:.1f})" if s else "–"
            lines.append(f"| {d['encoder']} | {p} | {m['age']['test_mae']:.2f} | {shift} | {s.get('r_with_clean', 1.0):.2f} "
                         f"| {delta('preop_htn')} | {delta('ckd')} | {delta('asa_ge3')} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    import torch

    p = argparse.ArgumentParser(prog="fm_baselines.stress")
    p.add_argument("encoder", help="pulseppg, papagei_s, anyppg, or a name for --kron")
    p.add_argument("--variant", type=Path, required=True, help="the VitalDB variant the clean embeddings came from")
    p.add_argument("--clean-emb", type=Path, required=True, help="<emb>/<encoder>, with vitaldb-<variant>.parquet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=None)
    p.add_argument("--kron", type=Path, default=None, help="kron-ppg checkpoint (run in kron-fm's environment)")
    p.add_argument("--random", action="store_true", help="with --kron: untrained twin")
    p.add_argument("--per-subject", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--summary", action="store_true", help="only print the summary of --out")
    a = p.parse_args(argv)
    if a.summary:
        print(summarize(a.out))
        return
    dev = torch.device(a.device)
    if a.kron:
        from kron_ppg.model import KronPPG, KronPPGConfig, patchify
        from kron_ppg.train import load_encoder
        if a.random:
            torch.manual_seed(0)
            enc = KronPPG(KronPPGConfig(**torch.load(a.kron, map_location="cpu", weights_only=True)["config"])).to(dev)
        else:
            enc = load_encoder(a.kron, dev)
        enc.eval()

        @torch.no_grad()
        def embed(x, fs):
            out = []
            for i in range(0, len(x), 256):
                t = torch.from_numpy(np.ascontiguousarray(x[i:i + 256], dtype=np.float32)).to(dev)
                patches, ok = patchify(t, enc.config.patch)
                out.append(enc.embedding(enc.encode(patches, ok), ok).float().cpu().numpy())
            return np.concatenate(out)
    else:
        from .models import LOADERS
        e = LOADERS[a.encoder]()
        e.net.to(dev)

        def embed(x, fs):
            return e.embed(x, fs, a.device)
    run(a.encoder, embed, a.variant, a.clean_emb, a.out, a.cases, a.per_subject)


if __name__ == "__main__":
    main()
