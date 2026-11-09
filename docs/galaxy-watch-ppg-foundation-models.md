# Galaxy Watch PPG and the open PPG foundation models we can run on it

Written 2026-11-09. It brings together what this repository already found about the watch (Phase 0, the
model-feasibility check, the 2026-10-24 model survey) and adds three things:

- a fresh check of the open models' licences and weights;
- a choice of three models;
- linear-probe baselines for those three, scheduled on the university's GPU cluster beside kron-fm's
  kron-ppg run 2.

## 1. What the watch gives us

Measured on our Watch6 Classic (SM-R965F, One UI Watch 8, Android 16) with `watch/probe`. The details are in
[phase0-offline-findings.md](phase0-offline-findings.md) and [model-feasibility.md](model-feasibility.md).

| Source | Channels | Rate | Raw or processed | Delivery | Access | Status here |
|---|---|---|---|---|---|---|
| `com.samsung.sensor.hr_raw`, read through plain Android `SensorManager` | 16 integer channels. v5 = green PPG (r = 0.98 with the SDK's green); v1/v4 and v9/v11/v13 are other LEDs (IR/red, inferred from correlation); v0 is the on-wrist flag; the rest are ambient or unused | 24.96 Hz | raw LED counts | continuous stream while a foreground service runs | `READ_HEART_RATE` permission only: **tier `25hz`, no setup** | works. It hides about one lost sample every 10–12 s behind evenly spread timestamps; `foot_template` corrects for this |
| `TYPE_HEART_RATE` | HR | 1 Hz | processed | stream | standard | works |
| Accelerometer, gyroscope | 3-axis each | 100 Hz | raw | stream | standard | works |
| Samsung Health Sensor SDK, continuous PPG | green, IR, red | 25 Hz | raw | batches of data points to a listener | SDK + Health Platform developer mode on our own watch: **tier `all`**. Distributing an app that uses raw PPG needs Samsung partner approval | recorded 2026-10-22 together with `hr_raw` |
| SDK HR + IBI | HR, beat-interval list, status flags | about 1 Hz | processed | batched | tier `all` | works; cleaned IBIs reach 4.4 ms RMSSD error on GalaxyPPG |
| SDK `PPG_ON_DEMAND` | green, IR, red | 100 Hz | raw | **foreground only, at most 30 s, one on-demand tracker at a time** | tier `all` | recorded; used by AI-PPG Age |
| SDK `ECG_ON_DEMAND` | single-lead ECG in mV, plus a co-sampled green PPG | 500 Hz | raw | foreground, 30 s | tier `all` | recorded 2026-10-26; 37 of 38 `hr_raw` beats matched |
| SDK SpO2, skin temperature, BIA | – | spot checks | processed | on demand | tier `all` | not used yet |
| Samsung vendor sensors (100 Hz factory PPG, 32 g accelerometer) through `SensorManager` | – | – | – | – | refused: the `SSENSOR` permission is signature-only | no |
| Samsung Health history (phone) | HR minute bins, HRV, SpO2, sleep stages; probably overnight RR in `sleep_raw_data` | – | processed | "Download personal data" export | the user, on the phone | not yet exported; RR history is probably kept only about a week |

**Consequences for the models:**

- The signal every user has is **25 Hz green PPG**, from `hr_raw` or from the SDK.
- 100 Hz arrives only in 30 s foreground spot checks.
- Wrist green PPG holds almost nothing above 12.5 Hz, and average-beat shape survives 25 Hz
  (`benchmarks/morph25`). So a model fed 25 Hz upsampled to its own rate loses little in the 0.5–8 Hz band
  these models filter to.

## 2. Open PPG foundation models: licences and fit

Checked 2026-11-09 against GitHub's licence API, the Zenodo records and the Hugging Face API. The weights
were downloaded to the cluster. Older entries come from
[the 2026-10-24 survey](../research_notes/PPG%20health%20and%20aging%20measures/open_models_datasets.md) and
kron-fm's `reports/PPG foundation models and aging clocks 2026.md`.

| Model | Code licence | Weights: licence, where | Trained on | Fit for 25 Hz Galaxy Watch green PPG | Verdict |
|---|---|---|---|---|---|
| **Pulse-PPG** (IMWUT 2025) | MIT | **MIT**, Zenodo 17345536 (315 MB zip), ungated | wrist WearOS (Fossil) field PPG, 120 people, 50 Hz, 4 min windows | **best**: wrist, WearOS, field data; 25 → 50 Hz is a 2× upsample | **run** |
| **PaPaGei-S** (ICLR 2025) | BSD-3-Clause-Clear | **none stated**, Zenodo 13983110 (23 MB), ungated | finger PPG (VitalDB, MIMIC-III, MESA), 125 Hz, 10 s | fair: finger domain; 25 → 125 Hz | **run**, for research only |
| **AnyPPG** (KDD 2026) | **none** (no LICENSE file) | **none**, `load_anyppg/anyppg_ckpt.pth` in the repository (16 MB) | finger PPG + ECG alignment, 58,796 people, 125 Hz, 10 s | fair; it is the only one reported on 30 Hz phone PPG (BUT PPG) | **run**, for research only, never redistributed |
| GPT-PPG | – | not released (no HF or GitHub weights; HF search empty on 2026-11-09) | 200 M 30 s windows | – | cannot run |
| SiamQuality | none | no weights | ICU finger PPG, 30 s | – | cannot run |
| `lowdown-labs/fela-ppg` | CC BY-NC 4.0 **and** Hippocratic 3.0 (both apply) | gated: our token is not on the access list | MIMIC PERform AF (35 ICU people); 32 Hz, 25 s | input fits 25 Hz, but it is an **AF classifier** (0.77M parameters), not an encoder for age or traits | skip here; compare against our AF screen once access is granted |
| AI-PPG Age | MIT | MIT, HF `Ngks03/PPG-VascularAge` | UK Biobank finger beats | one averaged 100-sample beat, not a window | already in `ppg_age.py`; VitalDB zero-shot MAE 11.5 y |
| NormWear, NormWear-2 | Apache-2.0 | Apache-2.0, HF | generalist multi-sensor | generic; a specialist PPG model wins 27% more often | not chosen |
| Apple PPG FM and PpgAge; Samsung HiMAE, MMR, xMAE | – | closed | wrist green PPG at 64 or 100 Hz | the evidence that wrist PPG carries age, HTN and diabetes | cannot run |

**Licences in practice.**

- Pulse-PPG is the only one we may use and ship freely.
- PaPaGei's code is BSD, but its weights carry no licence, so they default to all rights reserved.
- AnyPPG has no licence at all.
- So PaPaGei and AnyPPG serve as research baselines here: no product use and no redistribution of their
  code or weights. The code in `fm_baselines/` imports them from upstream clones and does not copy them.

**Chosen:**
- **Pulse-PPG**: the domain match for the watch.
- **PaPaGei-S**: the most-cited finger baseline.
- **AnyPPG**: the best reported open age probe (MAE 6.08 y), and it has phone-camera evidence.

## 3. What runs on the cluster

Code: [`fm_baselines/`](../fm_baselines). It is a separate uv project, because it needs CUDA torch while the
prototype pins CPU torch.

**Inputs.** kron-ppg run 2's prepared windows: 30 s, band-passed to 0.5–8 Hz and robust-scaled.
- VitalDB (about 5,900 people, surgical finger PPG) at 25 and 125 Hz;
- BIDMC (53 ICU people) at 25 and 125 Hz;
- GalaxyPPG (24 people aged 20–29, Galaxy Watch5 green PPG) at 25 Hz.

The 25 Hz VitalDB set is finger PPG decimated to the watch's rate. It tests what the watch's rate costs,
not the wrist.

**Split.** kron-ppg's own person-level hash split (`kron_ppg.data.subject_split` over `<source>/<subject>`),
reproduced in `fm_baselines.data.kron_split` and pinned by a test. The test people are kron-ppg's test people.

**Embedding.** Up to 24 windows per person, spread over the recording, each through the frozen encoder:
- Pulse-PPG: 30 s at 50 Hz, max-pooled to 512-d;
- PaPaGei-S and AnyPPG: three 10 s segments at 125 Hz, z-scored, embeddings averaged.

A person's feature is the mean of their window embeddings.

**Probes**, fitted on VitalDB train people, with hyperparameters chosen on validation people and scored on
test people:
- **age**: ridge. Reports MAE and r against predicting the median, then on BIDMC and GalaxyPPG with the
  VitalDB fit unchanged.
- **age gap**: bias-corrected on validation people, as kron-ppg does. Reports its AUC for hypertension,
  diabetes and ASA ≥ 3, beside age's own AUC. This is the outcome test; MAE alone does not make a clock.
- **flags** `preop_htn`, `preop_dm`, `asa` ≥ 3, `sex`: logistic regression on age+sex alone, on the
  embedding, and on both. A paired bootstrap of (embedding+age+sex) − (age+sex) says whether the PPG adds
  anything beyond age.
- **`hr`**: a no-model baseline from the same windows (mean and SD of heart rate and periodicity).

## 4. Job schedule

State at 2026-11-09 22:05 UTC. Run 2 was still downloading VitalDB (4,208 of about 5,900 tracks), so
none of the inputs existed yet. The GPU and CPU jobs are therefore queued behind run 2 by a driver
rather than submitted.

- **Driver:** `fm_baselines/slurm/run_after_run2.sh`, running on the haumea login node as pid 904199
  (started 22:01 UTC). Its log is `/data/uq2347/fm-baselines/run-r1.log`; every job ID is logged as
  `job <name> <id> on <node>` when submitted. Outputs go to `/data/uq2347/fm-baselines/runs/r1/{emb,probe}`,
  and the final table to `probe/summary.md`.
- **Code on the cluster:** `/data/uq2347/fm-baselines/code`, a copy of `fm_baselines/` at this commit.
- **Weights:** `/data/uq2347/fm-baselines/weights`; upstream clones in `upstream/`.
- **Smoke test:** job 6423285 (gpu12, 1 min 13 s) embedded run 1's BIDMC and GalaxyPPG 25 Hz windows with
  all three models. Each checkpoint loaded strictly, and 422 windows took 0.6–3.4 s per model. A 10-fold
  age probe on BIDMC's 52 people gave r = 0.28 (Pulse-PPG), 0.16 (PaPaGei-S) and 0.60 (AnyPPG): noisy,
  but the embeddings are not degenerate. Job 6423284 found both gpu11 cards busy and exited 3, as designed.

| Step | Waits for | Jobs (name: kind) | Expected (UTC, 2026-11-10) |
|---|---|---|---|
| 25 Hz embeddings | run 2's prepare job (`prep=` in `ppg-run2b.log`), after the download | `fm-emb25-pulseppg`, `fm-emb25-papagei_s`, `fm-emb25-anyppg`: GPU, about 5–10 min each, mostly parquet reading | start about 00:30–02:00 |
| 25 Hz probes | the three jobs above | `fm-probe25`: CPU, 8 cores, about 15–30 min | about 01:00–02:30 |
| 125 Hz embeddings | the publish job's 125 Hz prepare (`hub=` in `ppg-publish.log`) | `fm-emb125-<model>` × 3: GPU | start about 02:30–04:00 |
| 125 Hz probes (final) | the three jobs above | `fm-probe125`: CPU, rewrites `summary.md` over both rates | about 03:00–05:00 |

The times are estimates. They hinge on the download rate (10–35 tracks a minute) and on the `itmz`
account's shared CPU cap, which held the smoke job for a while.

**GPU placement.** Run 2 pins gpu13 and its jobs fail when both cards there are taken, so the driver uses
gpu13 only after run 2 logs `run2 finished`. Until then it uses gpu11 and gpu12, which are often busy with
another user. A job that finds every card busy exits 3, and the driver resubmits it on the next node every
10 min, for up to 12 h.

**If run 2's download is incomplete** (fewer than 5,800 tracks), run 2 stops before preparing anything. The
driver then logs the failure and submits nothing.

**Comparing with kron-ppg.** The run 2 clocks land in kron-fm's
`data/output/runs/kron-ppg/r2-clock-{scratch,finetune,probe}/metrics.json`, on the same test people.

## 5. Caveats

- Pulse-PPG was pretrained on 4-minute windows and is fed 30 s here. Its max-pool allows any length, but
  this is off its training distribution.
- Everything is trained and tested on surgical finger PPG from one Korean hospital. No probe here says
  anything about a wrist until it is checked on wrist data from people of many ages, which no public set
  has; GalaxyPPG's 20–29-year-olds only show domain shift.
- A linear probe on frozen features is the standard FM baseline, not fine-tuning. If one encoder clearly
  beats kron-ppg's frozen probe, fine-tuning it is the next run.
- As everywhere in this project: a PPG age is a within-person trend, and a clock earns its place by
  outcome evidence (the gap AUCs and the deltas), not by MAE.
