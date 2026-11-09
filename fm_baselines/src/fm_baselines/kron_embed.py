"""Embeddings from kron-ppg's KronLSTM encoders, in the same format as the foundation models'.

It runs in kron-fm's environment, because the KronLSTM kernels live there. ``fm_baselines`` goes on
``PYTHONPATH``; it needs only numpy and polars from that environment:

    cd <kron-fm> && PYTHONPATH=<pulse-exploration>/fm_baselines/src \\
        uv run --no-sync python -m fm_baselines.kron_embed <checkpoint.pt> <name> <variant>... --out <emb>

The embedding is the encoder's own pooled one: the mean final hidden state over a 30 s window's valid
patches (``KronPPG.embedding``), which is also what kron-ppg's age and trait heads read. ``--random``
builds an untrained encoder with the checkpoint's configuration: the architecture without what training
taught it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .embed import embed_with


def main(argv: list[str] | None = None) -> None:
    from kron_ppg.model import KronPPG, KronPPGConfig, patchify
    from kron_ppg.train import load_encoder

    p = argparse.ArgumentParser(prog="kron_embed")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("name")
    p.add_argument("variants", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-subject", type=int, default=24)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--random", action="store_true", help="untrained encoder with the checkpoint's configuration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    dev = torch.device(a.device)
    if a.random:
        torch.manual_seed(a.seed)
        config = KronPPGConfig(**torch.load(a.checkpoint, map_location="cpu", weights_only=True)["config"])
        encoder = KronPPG(config).to(dev)
    else:
        encoder = load_encoder(a.checkpoint, dev)
    encoder.eval()

    @torch.no_grad()
    def embed(x: np.ndarray, fs: float) -> np.ndarray:
        if abs(fs - encoder.config.sample_rate) > 1e-6:
            raise ValueError(f"windows at {fs} Hz, encoder reads {encoder.config.sample_rate} Hz")
        out = []
        for i in range(0, len(x), a.batch):
            t = torch.from_numpy(np.ascontiguousarray(x[i:i + a.batch])).to(dev)
            patches, ok = patchify(t, encoder.config.patch)
            out.append(encoder.embedding(encoder.encode(patches, ok), ok).float().cpu().numpy())
        return np.concatenate(out)

    embed_with(a.name, embed, a.variants, a.out, a.per_subject, a.device,
               {"checkpoint": str(a.checkpoint), "random": a.random, "seed": a.seed,
                "parameters": sum(q.numel() for q in encoder.parameters())})


if __name__ == "__main__":
    main()
