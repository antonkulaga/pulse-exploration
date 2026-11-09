# fm_baselines

KronLSTM PPG encoders (kron-fm's kron-ppg) and open PPG foundation models, compared on the same people, the
same targets and one evaluation. Inputs are the windows prepared by kron-fm's `kron-ppg prepare`, with
kron-ppg's own person-level split. The targets are age and aging-relevant health measures from VitalDB:
- numbers: BMI, eGFR, albumin, hemoglobin, glucose;
- flags: hypertension, diabetes, ASA ≥ 3, CKD, anemia, low albumin, abnormal lung function, and sex as a
  control.

Results and the job schedule are in
[docs/galaxy-watch-ppg-foundation-models.md](../docs/galaxy-watch-ppg-foundation-models.md).

This is a separate uv project from the prototype at the repository root: it needs CUDA torch on the
cluster, while the root pins CPU torch for the laptop.

| Encoder | What it is | Fed |
|---|---|---|
| `kronlstm_pretrain` | kron-ppg's KronLSTM, self-supervised (next patch + same-person contrast), frozen | 30 s at 25 Hz |
| `kronlstm_untrained` | the same architecture with random weights: what the architecture gives without training | 30 s at 25 Hz |
| `kronlstm_traits_finetune` | kron-ppg's encoder after fine-tuning on the six-trait head, frozen | 30 s at 25 Hz |
| `pulseppg` | maxxu05/pulseppg, MIT code and weights (Zenodo 17345536) | 30 s at 50 Hz |
| `papagei_s` | Nokia-Bell-Labs/papagei-foundation-model, BSD-3-Clear code, Zenodo 13983110 weights with no licence | 3 × 10 s at 125 Hz |
| `anyppg` | PKUDigitalHealth/AnyPPG, no licence on code or weights | 3 × 10 s at 125 Hz |
| `hr` | no model: mean and SD of each window's heart rate and periodicity | – |
| `kron-head-*` | kron-ppg's own trained heads (`traits`, `clock` runs), scored by the same evaluation | – |

**How models are compared** (`probe.py`):
1. Every encoder's mean embedding per person gets a linear probe per target, trained on train people.
2. Each probe's output, and each kron-ppg head's output, is one PPG score per target.
3. On validation people, two models are fitted per target: age+sex, and age+sex+score.
4. Both are scored on test people. The paired bootstrap of the difference (ΔAUC or ΔR²) says whether the
   PPG adds anything beyond age.
5. Age clocks are judged by their bias-corrected gap's AUC for each condition.

**Upstream code and weights.** No upstream code is copied here. `FM_HOME` (default `/data/uq2347/fm-baselines`)
holds the `upstream/<repo>` clones and `weights/`. `models.py`:
- checks each clone's commit and each checkpoint's SHA-256;
- imports only the network file;
- reads checkpoints with `weights_only=True`.

KronLSTM encoders are embedded in kron-fm's environment by `python -m fm_baselines.kron_embed`.

```bash
uv sync
uv run fm-baselines embed pulseppg <kron-fm>/data/interim/ppg/vitaldb/fs25-w30s ... --out runs/emb
uv run fm-baselines probe runs/emb --cases <kron-fm>/data/input/ppg/vitaldb/cases.csv.gz --out runs/probe
uv run fm-baselines kron-eval <kron-ppg run>... --subjects <variant>/subjects.parquet --cases ... --out runs/probe
uv run pytest -q
```

On the cluster:
- `slurm/run_all.sh` chains everything to kron-ppg run 2's drivers. Each GPU job is one model and rate;
  CPU jobs do the probes and the summary.
- `slurm/run_partial.sh` is the preliminary run on the windows that existed before run 2.
- Cluster names go in the git-ignored `slurm/env.local.sh`: `FM_GPU_PARTITION`, `FM_CPU_PARTITION`,
  `FM_ACCOUNT`, `FM_GPU_CONSTRAINT`, `FM_CPU_EXCLUDE`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`.
