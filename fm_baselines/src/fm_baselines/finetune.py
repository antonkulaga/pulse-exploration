"""Fine-tune a foundation model on VitalDB with one multi-task head: the fair counterpart of kron-ppg's encoder,
which saw VitalDB during pretraining.

- **Head.** One MLP on the encoder's pooled embedding (the mean over a window's segments) predicts every
  target in :mod:`fm_baselines.labels`. Numbers are learned in units of their training SD (masked L1),
  flags as logits (masked BCE), with each target weighted equally.
- **Training.** Windows are drawn from train people only, weighted so each person counts equally. AdamW,
  encoder 1e-4 and head 1e-3, cosine schedule with warm-up, bf16 autocast
  (fp16 with loss scaling on cards without bf16, such as the 2080 Ti). The step with the lowest
  validation loss is kept.
- **Scoring.** Each validation and test person's score per target is the mean head output over their
  windows. :func:`fm_baselines.probe.evaluate` judges those scores exactly as it judges the linear probes
  and kron-ppg's heads.

Defaults: 6,000 steps of 128 windows; 60 windows per training person; 24 per held-out person.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from .data import load_variant
from .labels import BINARY, TRAITS, label_table
from .models import LOADERS
from .probe import evaluate, evaluate_external


def _targets(rows: pl.DataFrame, labels: pl.DataFrame) -> np.ndarray:
    lab = rows.select("key").join(labels, on="key", how="left", maintain_order="left")
    return np.column_stack([lab.get_column(t).to_numpy() if t in lab.columns else np.full(len(lab), np.nan)
                            for t in TRAITS]).astype(np.float32)


class Head(nn.Module):
    def __init__(self, dim: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fm_baselines.finetune")
    p.add_argument("model", choices=sorted(LOADERS))
    p.add_argument("--variant", type=Path, required=True, help="VitalDB fs25-w30s variant folder")
    p.add_argument("--external", type=Path, nargs="*", default=[], help="other sources' variants (age, sex)")
    p.add_argument("--cases", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True, help="folder for metrics.json, scores.parquet, log.jsonl")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--train-per-subject", type=int, default=60)
    p.add_argument("--eval-per-subject", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    enc = LOADERS[a.model]()
    enc.net.to(dev)
    head = Head(enc.dim, len(TRAITS)).to(dev)
    bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and not bf16)
    is_bin = torch.tensor([t in BINARY for t in TRAITS], device=dev)

    t0 = time.time()
    train_v = load_variant(a.variant, a.train_per_subject)
    eval_v = load_variant(a.variant, a.eval_per_subject)
    labels = label_table(train_v.subjects, a.cases)
    tr = np.flatnonzero(train_v.rows.get_column("split").to_numpy() == "train")
    y_tr = _targets(train_v.rows[tr], labels)
    mu = np.nanmean(y_tr, axis=0)
    sd = np.nanstd(y_tr, axis=0) + 1e-6
    mu[[TRAITS.index(t) for t in BINARY]] = 0.0
    sd[[TRAITS.index(t) for t in BINARY]] = 1.0
    keys = train_v.rows.get_column("key").to_numpy()[tr]
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inv]
    weight /= weight.sum()
    held = eval_v.rows.with_row_index("i").filter(pl.col("split").is_in(["val", "test"]))
    val_rows = held.filter(pl.col("split") == "val")
    vi = val_rows.get_column("i").to_numpy()
    y_val = (_targets(val_rows, labels) - mu) / sd
    print(json.dumps({"model": a.model, "train_windows": len(tr), "train_people": len(counts),
                      "eval_windows": len(held), "load_s": round(time.time() - t0, 1)}), flush=True)

    def forward(x: np.ndarray) -> torch.Tensor:
        s = enc.segments(x, train_v.sample_rate)
        n, k, seg = s.shape
        t = torch.from_numpy(s.reshape(n * k, 1, seg)).to(dev)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
            e = enc.forward(t).float().reshape(n, k, -1).mean(1)
            return head(e).float()

    def loss_fn(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ok = torch.isfinite(y)
        yz = torch.nan_to_num(y)
        reg = F.l1_loss(out, yz, reduction="none")
        bce = F.binary_cross_entropy_with_logits(out, yz.clamp(0, 1), reduction="none")
        per = torch.where(is_bin, bce, reg) * ok
        return (per.sum(0) / ok.sum(0).clamp_min(1)).mean()

    @torch.no_grad()
    def predict(idx: np.ndarray, v) -> np.ndarray:
        enc.net.eval(), head.eval()
        out = [forward(v.x[idx[i:i + 512]]).cpu().numpy() for i in range(0, len(idx), 512)]
        enc.net.train(), head.train()
        return np.concatenate(out) * sd + mu

    opt = torch.optim.AdamW([{"params": enc.net.parameters(), "lr": a.lr},
                             {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    warm = max(1, a.steps // 20)
    best, best_step = math.inf, -1
    enc.net.train(), head.train()
    log = (a.out / "log.jsonl").open("w")
    for step in range(1, a.steps + 1):
        f = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (a.steps - warm)))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * f
        b = rng.choice(len(tr), a.batch, p=weight)
        y = torch.from_numpy((y_tr[b] - mu) / sd).to(dev)
        loss = loss_fn(forward(train_v.x[tr[b]]), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(enc.net.parameters()) + list(head.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % a.eval_every == 0 or step == a.steps:
            pv = (predict(vi, eval_v) - mu) / sd
            vl = float(loss_fn(torch.from_numpy(pv).float().to(dev), torch.from_numpy(y_val).to(dev)).item())
            rec = {"step": step, "train_loss": round(float(loss.item()), 4), "val_loss": round(vl, 4),
                   "elapsed_s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps(rec), flush=True)
            if vl < best:
                best, best_step = vl, step
                torch.save({"encoder": enc.net.state_dict(), "head": head.state_dict(), "mu": mu, "sd": sd,
                            "traits": list(TRAITS)}, a.out / "checkpoint.pt")
    log.close()

    ck = torch.load(a.out / "checkpoint.pt", map_location=dev, weights_only=False)
    enc.net.load_state_dict(ck["encoder"])
    head.load_state_dict(ck["head"])

    def person_scores(v, idx: np.ndarray, rows: pl.DataFrame) -> pl.DataFrame:
        pred = predict(idx, v)
        frame = rows.select("key", "split").with_columns(
            *[pl.Series(f"score_{t}", pred[:, j].astype(np.float64)) for j, t in enumerate(TRAITS)])
        return frame.group_by("key", maintain_order=True).agg(pl.col("split").first(), pl.len().alias("n_windows"),
                                                              *[pl.col(f"score_{t}").mean() for t in TRAITS])

    scores = person_scores(eval_v, held.get_column("i").to_numpy(), held)
    metrics = {"model": f"{a.model}_finetune", "train": f"vitaldb-{a.variant.name}", "best_step": best_step,
               "best_val_loss": round(best, 4), "steps": a.steps, **evaluate(scores, labels), "external": {}}
    all_scores = [scores]
    for ext in a.external:
        ev = load_variant(ext, a.eval_per_subject)
        es = person_scores(ev, np.arange(len(ev.rows)), ev.rows).with_columns(pl.lit("external").alias("split"))
        metrics["external"][f"{ev.source}-{ext.name}"] = evaluate_external(es, label_table(ev.subjects, None))
        all_scores.append(es)
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pl.concat(all_scores, how="diagonal").write_parquet(a.out / "scores.parquet")
    print(json.dumps({"model": metrics["model"], "best_step": best_step, "age": metrics.get("age", {})}), flush=True)


if __name__ == "__main__":
    main()
