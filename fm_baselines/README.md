# fm_baselines

Open PPG foundation models as frozen encoders with linear probes. They run on the windows prepared by
kron-fm's `kron-ppg prepare`, with kron-ppg's own person-level split, so their numbers sit beside
kron-ppg's clocks. Why these models, their licences and the job schedule are in
[docs/galaxy-watch-ppg-foundation-models.md](../docs/galaxy-watch-ppg-foundation-models.md).

This is a separate uv project from the prototype at the repository root: it needs CUDA torch on the
cluster, while the root pins CPU torch for the laptop.

| Model | Upstream (pinned commit) | Fed |
|---|---|---|
| `pulseppg` | maxxu05/pulseppg, MIT code and weights (Zenodo 17345536) | 30 s at 50 Hz |
| `papagei_s` | Nokia-Bell-Labs/papagei-foundation-model, BSD-3-Clear code, Zenodo 13983110 weights with no licence | 3 × 10 s at 125 Hz |
| `anyppg` | PKUDigitalHealth/AnyPPG, no licence on code or weights | 3 × 10 s at 125 Hz |

No upstream code is copied here. `FM_HOME` (default `/data/uq2347/fm-baselines`) holds `upstream/<repo>`
clones and `weights/`; `models.py` checks each clone's commit and each checkpoint's SHA-256, imports
only the network file, and reads checkpoints with `weights_only=True`.

```bash
uv sync
uv run fm-baselines embed pulseppg <kron-fm>/data/interim/ppg/vitaldb/fs25-w30s ... --out runs/emb
uv run fm-baselines probe runs/emb --out runs/probe      # every model and rate, plus an HR-only baseline
uv run pytest -q
```

On the cluster, `slurm/run_after_run2.sh` waits for kron-ppg run 2's prepared windows and submits one GPU
job per model and rate, then a CPU probe job per rate. Cluster names go in the git-ignored
`slurm/env.local.sh` (`FM_GPU_PARTITION`, `FM_CPU_PARTITION`, `FM_ACCOUNT`, `FM_GPU_CONSTRAINT`,
`FM_CPU_EXCLUDE`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`).
