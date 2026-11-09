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

## 3. KronLSTM against the foundation models: what runs

Code: [`fm_baselines/`](../fm_baselines). It is a separate uv project, because it needs CUDA torch while the
prototype pins CPU torch.

**Encoders.** Every encoder sees the same windows of the same people and passes through the same probes and
evaluation:

| name | what |
|---|---|
| `kronlstm_pretrain` | kron-ppg's KronLSTM (5.85M parameters, 25 Hz, causal), self-supervised (next patch + same-person contrast); frozen |
| `kronlstm_traits_finetune` | the same encoder after kron-ppg fine-tuned it with its six-trait head on train people; frozen |
| `kronlstm_untrained` | the same architecture with random weights: what the architecture gives without training |
| `pulseppg`, `papagei_s`, `anyppg` | the open foundation models, frozen (section 2) |
| `hr` | no model: mean and SD of each window's heart rate and periodicity |
| `kron-head:*` | kron-ppg's own trained heads (age clocks, six-trait head), scored on their predictions |

**Targets** ([`labels.py`](../fm_baselines/src/fm_baselines/labels.py)), from kron-ppg's subject table and
VitalDB's preoperative records:
- numbers: age, BMI, eGFR (CKD-EPI 2021), albumin, hemoglobin, glucose;
- flags: hypertension, diabetes, ASA ≥ 3, CKD (eGFR < 60), anemia (WHO), low albumin (< 3.5 g/dL),
  abnormal lung function (preoperative PFT), and sex as a control.

Atrial fibrillation (16 cases) and in-hospital death (57) are too rare to test.

**Split.** kron-ppg's own person-level hash split, reproduced in `fm_baselines.data.kron_split` and pinned
by a test.

**Evaluation** ([`probe.py`](../fm_baselines/src/fm_baselines/probe.py)):
1. Each person's mean embedding over up to 24 windows gets a linear probe per target (ridge or logistic),
   trained on train people, with its penalty chosen on validation people. The probe's output is that
   target's PPG score. kron-ppg's heads already give scores.
2. On validation people, age+sex and age+sex+score models are fitted.
3. On test people, the gain of the second model (ΔAUC or ΔR²) gets a paired-bootstrap 95% interval.

**This is the question that matters for aging research:** does the pulse wave say anything about a
person's health that their age does not already say?

Age clocks are also judged by the AUC of their bias-corrected gap for each condition.

**Inputs.**
- **Run 1 era (r0, done):** kron-ppg's 3,986-person `vitaldb-partial` windows, run 1's encoder, the r2a
  trait heads and the r1 clocks. External sets: BIDMC and GalaxyPPG.
- **Run 2 (r1, scheduled):** about 5,900 VitalDB people at 25 Hz and 125 Hz, run 2's encoder, clocks and
  trait heads.

The 25 Hz VitalDB windows are finger PPG decimated to the watch's rate. They test the watch's rate, not the
wrist.

## 4. Preliminary results (r0, 2026-11-09)

614 test people, 25 Hz. The test set has 183 people with hypertension, 63 with diabetes, 56 with ASA ≥ 3,
34 with CKD, 172 with anemia, 37 with low albumin and 93 with abnormal lung function. The r1 clocks were
fitted on run 1's 1,996 people, so they are scored on the 308 of their test people. Full table:
`/data/uq2347/fm-baselines/runs/r0-partial/probe/summary.md`.

**Age.**
- The frozen KronLSTM with a linear probe reads age best: 6.92 y MAE (r 0.79), or 6.78 y after trait
  fine-tuning.
- Next come kron-ppg's own MLP heads (7.3–7.8 y), then AnyPPG 8.10, Pulse-PPG 8.85, PaPaGei-S 9.44, and the
  untrained KronLSTM 8.91.
- Heart rate alone gets nothing (11.57 against 11.63 for predicting the median).
- Pretraining is worth 2 y over the same architecture with random weights.
- On BIDMC (another hospital), PaPaGei-S transfers best (12.7 y); every model is far from in-domain.
- On GalaxyPPG wrist data from people aged 20–29, the mean predicted age ranges from 15.5 to 58.6 y.
  **No VitalDB-trained age reading means anything on a wrist.**

**Age gap.**
- No model's bias-corrected gap carries outcome signal: AUC 0.47–0.60 for every condition. Age alone
  gives 0.55–0.76.
- This repeats run 1. On this data, a PPG "age" is an age regressor, not an aging clock.

**Health beyond age and sex: ΔAUC or ΔR², with the 95% interval.**

| target | KronLSTM pretrain | best foundation model | heart rate only | reading |
|---|---|---|---|---|
| CKD (eGFR < 60) | **+0.156** [+0.055, +0.267] | PaPaGei-S +0.128 [+0.045, +0.213] | +0.028 (n.s.) | pulse shape carries kidney function; only 34 cases |
| eGFR (R²) | **+0.039** [+0.015, +0.062] | Pulse-PPG +0.015 [+0.001, +0.027] | +0.001 (n.s.) | as above, smaller |
| ASA ≥ 3 | **+0.121** [+0.014, +0.232] | AnyPPG +0.093 (n.s.) | −0.034 (n.s.) | severe systemic disease; 56 cases |
| hypertension | **+0.021** [+0.002, +0.040] | none (≤ +0.000) | −0.003 | small but beyond age; only the KronLSTM gets it |
| BMI (R²) | **+0.190** [+0.135, +0.241] | AnyPPG +0.147 | +0.004 | strong |
| albumin (R²), low albumin | +0.188, +0.244 | AnyPPG +0.173, +0.253 | **+0.102, +0.224** | mostly heart rate: sicker patients run faster |
| hemoglobin (R²), anemia | +0.093, +0.090 | Pulse-PPG +0.100, +0.098 | **+0.062, +0.066** | partly heart rate |
| abnormal lung function | +0.006 [+0.001, +0.011] | none | +0.003 | negligible |
| diabetes, glucose | n.s. | n.s. | glucose +0.014 | nothing beyond age |
| sex (beyond age) | +0.161 | AnyPPG +0.119 | n.s. | the expected control |

**Reading these.**
- The KronLSTM's self-supervised encoder beats all three open foundation models on age and on the
  conditions where PPG adds information. The untrained KronLSTM falls to foundation-model level or below,
  so the advantage comes from pretraining, not only the architecture.
- The KronLSTM is pretrained on VitalDB itself, so this is an in-domain advantage. The foundation models
  never saw VitalDB's sensor or its patients. A fair claim is "in-domain pretraining beats off-the-shelf
  encoders here", not "KronLSTM is a better PPG foundation model".
- Kidney function, BMI and severe systemic disease are the targets where the pulse shape, not just the heart
  rate, adds to age. They are worth following up on the full run.
- Albumin and anemia ride mostly on heart rate. That is acute surgical illness, not aging biology.
- All of this is intraoperative finger PPG under anaesthesia, with preoperative labels.

#### Age

| model | rate | test MAE (median baseline) | test r | BIDMC MAE / r | GalaxyPPG mean predicted (true 20–29) |
|---|---|---|---|---|---|
| anyppg | fs25 | 8.10 (11.63) | 0.69 | 19.95 / 0.55 | 15.5 |
| hr | fs25 | 11.57 (11.63) | 0.17 | 15.99 / 0.34 | 58.6 |
| kron-head:r1-clock-finetune | fs25 | 7.45 (11.38) | 0.77 | nan / nan | nan |
| kron-head:r1-clock-probe | fs25 | 7.33 (11.38) | 0.78 | nan / nan | nan |
| kron-head:r1-clock-scratch | fs25 | 7.79 (11.38) | 0.75 | nan / nan | nan |
| kron-head:r2a-traits-finetune | fs25 | 7.54 (11.63) | 0.78 | nan / nan | nan |
| kron-head:r2a-traits-probe | fs25 | 7.53 (11.63) | 0.78 | nan / nan | nan |
| kronlstm_pretrain | fs25 | 6.92 (11.63) | 0.79 | 14.75 / 0.69 | 33.6 |
| kronlstm_traits_finetune | fs25 | 6.78 (11.63) | 0.80 | 15.01 / 0.74 | 43.5 |
| kronlstm_untrained | fs25 | 8.91 (11.63) | 0.64 | 15.43 / 0.38 | 54.5 |
| papagei_s | fs25 | 9.44 (11.63) | 0.58 | 12.69 / 0.67 | 45.7 |
| pulseppg | fs25 | 8.85 (11.63) | 0.64 | 14.36 / 0.60 | 53.8 |

#### Age gap: AUC of the bias-corrected gap on test people (age's own AUC in brackets)

| model | rate | preop_htn | preop_dm | asa_ge3 | ckd | anemia | low_albumin | pft_abnormal |
|---|---|---|---|---|---|---|---|---|
| anyppg | fs25 | 0.51 (0.74) | 0.52 (0.68) | 0.53 (0.65) | 0.53 (0.64) | 0.52 (0.59) | 0.54 (0.55) | 0.56 (0.67) |
| hr | fs25 | 0.49 (0.74) | 0.47 (0.68) | 0.51 (0.65) | 0.38 (0.64) | 0.44 (0.59) | 0.31 (0.55) | 0.52 (0.67) |
| kron-head:r1-clock-finetune | fs25 | 0.51 (0.75) | 0.54 (0.76) | 0.50 (0.68) | 0.55 (0.67) | 0.49 (0.59) | 0.39 (0.57) | 0.51 (0.67) |
| kron-head:r1-clock-probe | fs25 | 0.54 (0.75) | 0.54 (0.76) | 0.51 (0.68) | 0.60 (0.67) | 0.49 (0.59) | 0.39 (0.57) | 0.53 (0.67) |
| kron-head:r1-clock-scratch | fs25 | 0.52 (0.75) | 0.56 (0.76) | 0.51 (0.68) | 0.54 (0.67) | 0.47 (0.59) | 0.43 (0.57) | 0.55 (0.67) |
| kron-head:r2a-traits-finetune | fs25 | 0.55 (0.74) | 0.53 (0.68) | 0.52 (0.65) | 0.54 (0.64) | 0.51 (0.59) | 0.50 (0.55) | 0.56 (0.67) |
| kron-head:r2a-traits-probe | fs25 | 0.53 (0.74) | 0.52 (0.68) | 0.51 (0.65) | 0.52 (0.64) | 0.50 (0.59) | 0.49 (0.55) | 0.54 (0.67) |
| kronlstm_pretrain | fs25 | 0.55 (0.74) | 0.54 (0.68) | 0.54 (0.65) | 0.57 (0.64) | 0.52 (0.59) | 0.52 (0.55) | 0.55 (0.67) |
| kronlstm_traits_finetune | fs25 | 0.54 (0.74) | 0.53 (0.68) | 0.52 (0.65) | 0.54 (0.64) | 0.54 (0.59) | 0.53 (0.55) | 0.55 (0.67) |
| kronlstm_untrained | fs25 | 0.51 (0.74) | 0.53 (0.68) | 0.53 (0.65) | 0.57 (0.64) | 0.51 (0.59) | 0.49 (0.55) | 0.53 (0.67) |
| papagei_s | fs25 | 0.50 (0.74) | 0.53 (0.68) | 0.48 (0.65) | 0.49 (0.64) | 0.51 (0.59) | 0.53 (0.55) | 0.53 (0.67) |
| pulseppg | fs25 | 0.51 (0.74) | 0.53 (0.68) | 0.49 (0.65) | 0.53 (0.64) | 0.51 (0.59) | 0.51 (0.55) | 0.54 (0.67) |

#### Flags: PPG AUC / Δ AUC over age+sex [95% CI]; ** = interval above 0

| model | rate | preop_htn | preop_dm | asa_ge3 | ckd | anemia | low_albumin | pft_abnormal | sex_male |
|---|---|---|---|---|---|---|---|---|---|
| anyppg | fs25 | 0.64 / -0.000 [-0.014, +0.013] | 0.59 / -0.004 [-0.028, +0.020] | 0.73 / +0.093 [-0.005, +0.205] | 0.68 / +0.035 [-0.001, +0.074] | 0.66 / +0.077 [+0.026, +0.130] ** | 0.82 / +0.253 [+0.139, +0.376] ** | 0.64 / +0.008 [-0.009, +0.025] | 0.70 / +0.119 [+0.072, +0.170] ** |
| hr | fs25 | 0.55 / -0.003 [-0.006, -0.000] | 0.52 / -0.008 [-0.021, +0.004] | 0.54 / -0.034 [-0.124, +0.057] | 0.55 / +0.028 [-0.015, +0.069] | 0.61 / +0.066 [+0.015, +0.113] ** | 0.77 / +0.224 [+0.107, +0.354] ** | 0.54 / +0.003 [-0.009, +0.017] | 0.49 / -0.011 [-0.031, +0.010] |
| kron-head:r2a-traits-finetune | fs25 | 0.72 / +0.017 [-0.000, +0.035] | 0.67 / +0.016 [-0.005, +0.038] | 0.73 / +0.106 [-0.001, +0.222] | – | – | – | – | 0.75 / +0.162 [+0.112, +0.213] ** |
| kron-head:r2a-traits-probe | fs25 | 0.71 / +0.013 [-0.003, +0.030] | 0.65 / +0.009 [-0.022, +0.041] | 0.73 / +0.104 [-0.001, +0.222] | – | – | – | – | 0.76 / +0.173 [+0.121, +0.226] ** |
| kronlstm_pretrain | fs25 | 0.73 / +0.021 [+0.002, +0.040] ** | 0.65 / +0.009 [-0.037, +0.052] | 0.76 / +0.121 [+0.014, +0.232] ** | 0.76 / +0.156 [+0.055, +0.267] ** | 0.67 / +0.090 [+0.032, +0.146] ** | 0.83 / +0.244 [+0.138, +0.354] ** | 0.67 / +0.006 [+0.001, +0.011] ** | 0.75 / +0.161 [+0.110, +0.213] ** |
| kronlstm_traits_finetune | fs25 | 0.73 / +0.017 [-0.005, +0.038] | 0.62 / -0.013 [-0.052, +0.027] | 0.75 / +0.111 [+0.006, +0.224] ** | 0.77 / +0.153 [+0.049, +0.257] ** | 0.69 / +0.103 [+0.050, +0.158] ** | 0.85 / +0.262 [+0.152, +0.378] ** | 0.70 / +0.005 [+0.001, +0.009] ** | 0.76 / +0.169 [+0.117, +0.218] ** |
| kronlstm_untrained | fs25 | 0.63 / -0.001 [-0.013, +0.010] | 0.60 / -0.006 [-0.035, +0.020] | 0.66 / +0.049 [-0.034, +0.140] | 0.68 / +0.090 [+0.008, +0.176] ** | 0.64 / +0.057 [+0.012, +0.098] ** | 0.80 / +0.232 [+0.122, +0.350] ** | 0.59 / -0.002 [-0.017, +0.012] | 0.63 / +0.081 [+0.034, +0.126] ** |
| papagei_s | fs25 | 0.63 / +0.000 [-0.009, +0.010] | 0.59 / -0.003 [-0.026, +0.019] | 0.63 / +0.005 [-0.100, +0.109] | 0.74 / +0.128 [+0.045, +0.213] ** | 0.64 / +0.070 [+0.018, +0.120] ** | 0.78 / +0.221 [+0.097, +0.352] ** | 0.58 / +0.001 [-0.010, +0.011] | 0.62 / +0.070 [+0.020, +0.121] ** |
| pulseppg | fs25 | 0.64 / -0.001 [-0.008, +0.007] | 0.65 / +0.013 [-0.013, +0.039] | 0.69 / +0.063 [-0.029, +0.159] | 0.70 / +0.090 [+0.034, +0.153] ** | 0.67 / +0.098 [+0.045, +0.149] ** | 0.79 / +0.236 [+0.117, +0.361] ** | 0.61 / +0.002 [-0.010, +0.013] | 0.65 / +0.088 [+0.040, +0.138] ** |

#### Numbers: PPG r / Δ R² over age+sex [95% CI]; ** = interval above 0

| model | rate | bmi | egfr | albumin | hemoglobin | glucose |
|---|---|---|---|---|---|---|
| anyppg | fs25 | 0.36 / +0.147 [+0.100, +0.191] ** | 0.35 / +0.009 [+0.002, +0.017] ** | 0.44 / +0.173 [+0.091, +0.247] ** | 0.38 / +0.102 [+0.051, +0.147] ** | 0.21 / +0.007 [-0.025, +0.040] |
| hr | fs25 | 0.10 / +0.004 [+0.000, +0.008] ** | 0.12 / +0.001 [-0.011, +0.012] | 0.29 / +0.102 [+0.031, +0.173] ** | 0.25 / +0.062 [+0.026, +0.094] ** | 0.11 / +0.014 [+0.003, +0.027] ** |
| kron-head:r2a-traits-finetune | fs25 | 0.37 / +0.160 [+0.108, +0.204] ** | – | – | – | – |
| kron-head:r2a-traits-probe | fs25 | 0.38 / +0.176 [+0.120, +0.225] ** | – | – | – | – |
| kronlstm_pretrain | fs25 | 0.41 / +0.190 [+0.135, +0.241] ** | 0.49 / +0.039 [+0.015, +0.062] ** | 0.46 / +0.188 [+0.113, +0.257] ** | 0.39 / +0.093 [+0.035, +0.143] ** | 0.27 / +0.019 [-0.013, +0.053] |
| kronlstm_traits_finetune | fs25 | 0.42 / +0.193 [+0.142, +0.243] ** | 0.47 / +0.042 [+0.008, +0.073] ** | 0.45 / +0.184 [+0.113, +0.244] ** | 0.41 / +0.099 [+0.041, +0.149] ** | 0.29 / +0.026 [-0.008, +0.061] |
| kronlstm_untrained | fs25 | 0.27 / +0.083 [+0.048, +0.120] ** | 0.29 / +0.009 [-0.007, +0.025] | 0.42 / +0.157 [+0.082, +0.231] ** | 0.30 / +0.055 [+0.020, +0.089] ** | 0.16 / +0.001 [-0.032, +0.031] |
| papagei_s | fs25 | 0.25 / +0.071 [+0.037, +0.104] ** | 0.30 / +0.008 [-0.005, +0.019] | 0.36 / +0.121 [+0.052, +0.180] ** | 0.30 / +0.052 [+0.022, +0.078] ** | 0.16 / +0.004 [-0.017, +0.024] |
| pulseppg | fs25 | 0.27 / +0.086 [+0.048, +0.124] ** | 0.35 / +0.015 [+0.001, +0.027] ** | 0.39 / +0.144 [+0.062, +0.219] ** | 0.36 / +0.100 [+0.046, +0.145] ** | 0.18 / +0.003 [-0.021, +0.031] |


## 5. Job schedule

**r0 (done, 2026-11-09):**
- `slurm/run_partial.sh`;
- GPU 6423398 (foundation models) and 6423401 (KronLSTM encoders), both on gpu13 for 4 min while run 2
  was still downloading;
- CPU 6423418, the serial probe: cancelled because it was too slow and replaced by parallel probes;
- CPU 6423464, the probes and kron-ppg heads, 5 min.

**r1 (run 2, scheduled).** Driver `fm_baselines/slurm/run_all.sh`, pid 1007851 on the haumea login node
(started 22:28 UTC), log `/data/uq2347/fm-baselines/run-r1.log`. Each job ID is logged when submitted as
`job <name> <id> on <node>`. Outputs go to `/data/uq2347/fm-baselines/runs/r1/{emb,probe}`, and the tables
to `probe/summary.md`.

| step | waits for | jobs | expected (UTC, 2026-11-10) |
|---|---|---|---|
| A | run 2's 25 Hz prepare (`prep=` in `ppg-run2b.log`) | `fm-emb25-{pulseppg,papagei_s,anyppg}`: GPU, a few min each | about 00:00–00:30 |
| B | run 2's pretraining (`pretrain=`) | `fm-emb25-kronlstm_pretrain`, then `fm-emb25-kronlstm_untrained`: GPU | about 01:00–01:30 |
| interim | A and B | `fm-probe-interim`: CPU, about 5–10 min, first full-size table | about 01:15–01:45 |
| C | kron-ppg's trait heads (`traits finished` in `ppg-traits-b.log`) | `fm-emb25-kronlstm_traits_finetune`: GPU | about 02:30–03:30 |
| D | the 125 Hz prepare (`hub=` in `ppg-publish.log`) | `fm-emb125-{pulseppg,papagei_s,anyppg}`: GPU | about 02:00–04:00 |
| final | C and D | `fm-probe-final`: CPU; probes plus kron-ppg's r2 clocks and trait heads, scored the same way | about 03:00–04:30 |

The KronLSTM itself, meaning run 2's pretraining, three age clocks and two trait heads, is trained by
kron-ppg's own drivers (`ppg-run2b.sh`, `ppg-traits-b.sh`). These jobs read their checkpoints and touch
nothing of theirs.

**GPU placement.** kron-ppg pins gpu13, and its jobs fail when both cards there are taken. So this driver
uses only gpu11 and gpu12 until kron-ppg logs `traits finished`. A job that finds every card busy exits 3,
and the driver retries it on the next node every 10 min, for up to 12 h. If run 2 stops (fewer than 5,800
tracks downloaded, or no encoder), the steps that depend on it log the failure and submit nothing.

The times are estimates: run 2's pretraining is 40,000 steps (run 1's 20,000 took 28 min), and the `itmz`
account's shared CPU cap can hold CPU jobs.

## 6. Caveats

- Pulse-PPG was pretrained on 4-minute windows and is fed 30 s here. Its max-pool allows any length, but
  this is off its training distribution.
- Everything is trained and tested on surgical finger PPG from one Korean hospital. No probe here says
  anything about a wrist until it is checked on wrist data from people of many ages, which no public set
  has. GalaxyPPG's 20–29-year-olds show that the age readings do not transfer.
- The probes on frozen encoders are linear. kron-ppg's heads are MLPs trained on windows, and on age they
  do slightly worse than a linear probe on the mean embedding.
- As everywhere in this project: a PPG age is a within-person trend, and a clock earns its place by
  outcome evidence (the gap AUCs and the deltas), not by MAE.
