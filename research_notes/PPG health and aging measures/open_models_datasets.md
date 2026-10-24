# Open PPG models, signal-processing libraries and labelled datasets for health and aging measures from Galaxy Watch and phone-camera PPG

Scope: the concrete toolset (model weights, licences, input specs, reported downstream numbers), libraries and datasets as of 2026-10-24. This extends `docs/model-feasibility.md`, which already covers the Manhph2211 models (PulseLM, CP-PPG, D-BETA, Q-HEART, ECG-Scan), ECGFounder and ME-rPPG at a high level; those are not repeated here except where new evidence was found. Our signals: Watch6 Classic `hr_raw` at about 24.9 Hz (no SDK); with the Samsung SDK, 25 Hz continuous green, Samsung IBIs, ≤30 s blocks of 100 Hz green/IR/red PPG and 500 Hz single-lead ECG; phone fingertip PPG at about 30 fps; face video.

## Q1. PPG foundation models: weights, licences, input requirements, training data and downstream results; 25 Hz usability; transfer to wrist green PPG

### Takeaway
Three open, permissively licensed single-channel PPG encoders are usable today: Pulse-PPG (MIT, the only one pretrained on wrist field data), PaPaGei (BSD-3 code, Zenodo weights with no stated licence) and AnyPPG (weights on GitHub but no licence file). All are small (4–29 M parameters) and expect 10 s at 125 Hz (PaPaGei, AnyPPG) or 50 Hz (Pulse-PPG). The only models shown to read age and cardiometabolic status well from wrist green PPG are Apple's closed models, trained on 141k–214k people: age MAE 3.19 years, and 2.43 years for PpgAge. On wrist data, the open clinically pretrained encoders transfer poorly: HR MAE 9–12 bpm on PPG-DaLiA and near-chance affect on WESAD. Pulse-PPG is the best open choice for watch PPG. The one open age model, AI-PPG Age, is MIT but needs UK-Biobank-style averaged single beats and failed to generalise in an independent benchmark.

### Cited Findings

#### Open-weight PPG encoders (catalogue)

| Model | Weights / licence / gating | Input spec | Pretraining data | Params |
|---|---|---|---|---|
| **PaPaGei-P / PaPaGei-S** (Nokia Bell Labs, ICLR 2025, arXiv 2410.20542) | Zenodo record 13983110 (open access, ungated) with `papagei_p.pt` (20 MB), `papagei_s.pt` (23 MB), `papagei_s_svri.pt`; the Zenodo record lists **no licence** — [Zenodo API](https://zenodo.org/records/13983110). The GitHub code is **BSD-3-Clause** — [GitHub](https://github.com/Nokia-Bell-Labs/papagei-foundation-model). The arXiv paper itself is CC BY-NC-ND 4.0, and the paper gives no separate weights licence — [arXiv](https://arxiv.org/html/2410.20542) | Single channel, resampled to **125 Hz, 10 s (1,250 samples)**, band-pass filtered, flatline detection via BioBSS — [GitHub](https://github.com/Nokia-Bell-Labs/papagei-foundation-model); [arXiv](https://arxiv.org/html/2410.20542) | 57,641 h / 20.75 M segments / 13,517 people, all **finger** clinical PPG: VitalDB (500 Hz, surgery), MIMIC-III matched subset (125 Hz, ICU), MESA (256 Hz, PSG) — [arXiv](https://arxiv.org/html/2410.20542) | P: 5 M; S: 5.7 M in the text, 5 M in the tables; 512-d embedding — [arXiv](https://arxiv.org/html/2410.20542) |
| **Pulse-PPG** (Memphis/UIUC, UbiComp/IMWUT 2025, arXiv 2502.01108) | **MIT**, Zenodo 17345536 (`pulseppg_model_weights.zip`, 315 MB, open) — [Zenodo](https://zenodo.org/records/17345536); GitHub MIT — [GitHub](https://github.com/maxxu05/pulseppg) | **Wrist** PPG recorded at 100 Hz, pretrained on **50 Hz, 4-min windows** (12,000 samples), no filtering, per-person z-score — [arXiv v2](https://arxiv.org/html/2502.01108v2) | MOODS field study: Fossil Sport (Gen 4) **WearOS** watch, 120 participants, up to 100 days, about 55k h, 822,247 four-minute segments — [arXiv v2](https://arxiv.org/html/2502.01108v2) | 1D ResNet-26, about 28.5 M; 512-d; plus a 127k-parameter learned distance model — [arXiv v2](https://arxiv.org/html/2502.01108v2) |
| **AnyPPG** (Peking Univ., KDD 2026, arXiv 2511.01747 v3 Mar 2026) | Checkpoint in repo `load_anyppg/anyppg_ckpt.pth`; **no LICENSE file in the repo** (GitHub API reports no licence) — [GitHub](https://github.com/PKUDigitalHealth/AnyPPG). The README says the MC-MED disease checkpoints are "will be updated in the future" — [GitHub](https://github.com/PKUDigitalHealth/AnyPPG) | **125 Hz, 10 s, z-score, shape (B,1,L)**; band-pass 0.5–8 Hz — [GitHub README](https://github.com/PKUDigitalHealth/AnyPPG); [arXiv v3](https://arxiv.org/html/2511.01747v3) | 109,909 h, 58,796 subjects, about 39.6 M segments of synchronised PPG+ECG: MC-MED 78,945 h, PulseDB 12,768 h, MESA 7,947 h, HSP 9,260 h, CFS 989 h. CLIP-style alignment to a frozen ECGFounder ECG encoder — [arXiv v3](https://arxiv.org/html/2511.01747v3) | Net1D CNN, about 4.04 M, 512-d — [arXiv v3](https://arxiv.org/html/2511.01747v3) |
| **NormWear** (ACM Trans. Computing for Healthcare 2026, arXiv 2412.09758) | **Apache-2.0**, HF `mosaic-laboratory/normwear` ungated; checkpoint also in a GitHub release — [HF](https://huggingface.co/mosaic-laboratory/normwear) | **Multichannel**, any modality (PPG, ECG, EEG, GSR, IMU). `sampling_rate` is passed at inference; the README example uses 64 Hz, 2 s — [GitHub](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear) | "Various public datasets"; 11 downstream datasets / 18 applications; zero-shot via text alignment — [arXiv](https://arxiv.org/abs/2412.09758) | not stated |
| **NormWear-2** (arXiv 2605.15465) | **Apache-2.0**, HF `mosaic-laboratory/normwear2`, ungated, `trust_remote_code` — [HF](https://huggingface.co/mosaic-laboratory/normwear2) | Multivariate (B, L, nvar); a "world model" for forecasting from waveform to day level — [HF](https://huggingface.co/mosaic-laboratory/normwear2) | chaos-balanced corpus (details in paper) — [HF](https://huggingface.co/mosaic-laboratory/normwear2) | not stated |
| **AI-PPG Age / VascularAge** (Nie et al.; Communications Medicine Nov 2025; arXiv 2502.12990, 2406.14953) | **MIT**, HF `Ngks03/PPG-VascularAge` **ungated** (`model.pth`, `net1d.py`) — [HF](https://huggingface.co/Ngks03/PPG-VascularAge) | **One averaged PPG beat formatted like UK Biobank**, (B,1,100) in the example, z-scored — [HF card](https://huggingface.co/Ngks03/PPG-VascularAge) | UK Biobank, 212,231 participants (later version) — [search summary of arXiv 2502.12990](https://arxiv.org/html/2502.12990v4) | small Net1D |
| **PPGFlowECG** (PKU; PPG→ECG generation) | Weights on HF `XiaochengFang/PPGFlowECG` (no licence tag) and Baidu; repo has no licence — [GitHub](https://github.com/PKUDigitalHealth/PPGFlowECG); [HF](https://huggingface.co/XiaochengFang/PPGFlowECG) | PPG in → synthetic ECG (scored with ECGFounder features) | 4 datasets — [GitHub](https://github.com/PKUDigitalHealth/PPGFlowECG) | — |
| **SiamQuality** (Ding et al., arXiv 2404.17667) | Code-only repo with **no licence**; no weights confirmed — [GitHub](https://github.com/chengding0713/SiamQuality) | 30 s PPG pairs — [arXiv](https://arxiv.org/abs/2404.17667) | >36 M 30-s PPG pairs from ICU patients (finger); 6 external wearable heart-monitoring tasks — [arXiv](https://arxiv.org/abs/2404.17667) | CNN; not stated |
| **GPT-PPG** (Emory/GaTech/UCSF, arXiv 2503.08015) | Paper says code and weights "will later be made publicly available"; no HF or GitHub release found (HF search for "PPG-GPT" returned nothing, checked 2026-10-24) — [Emory PDF via search](https://cs.emory.edu/~jyang71/files/gpt-ppg.pdf) | 30 s samples — [arXiv](https://www.arxiv.org/abs/2503.08015) | >200 M 30-s PPG samples; AF detection, denoising without fine-tuning — [arXiv](https://www.arxiv.org/abs/2503.08015) | 19 M and 85 M — [search summary](https://cs.emory.edu/~jyang71/files/gpt-ppg.pdf) |
| **PPG-Distill** (Emory, NeurIPS 2025) | GitHub reportedly released per a survey list; it distils GPT-PPG and PaPaGei teachers, with "up to 21.8%" student gain, 7× faster and 19× less memory — [arXiv](https://arxiv.org/pdf/2509.19215); [awesome list](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) | — | — | — |

#### Downstream numbers per model (as reported by authors)

- **PaPaGei-S** (linear probe; AUROC, or MAE where noted). Finger/clinical, in-domain held-out sets: VitalDB ICU admission 0.79; MIMIC-III mortality 0.67; MESA smoker 0.61; MESA AHI MAE 12.97 (3%) / 10.56 (4%); nuMoM2b pregnancy stage 0.78; PPG-BP SBP / DBP / HR MAE 14.39 / 8.71 / 4.00; PPG-BP hypertension 0.77; VitalVideos SBP / DBP MAE 14.65 / 8.29; SDB 0.70. **Wrist, out-of-domain**: WESAD valence / arousal 0.56 / 0.55; ECSMP mood 0.56; PPG-DaLiA HR MAE 11.53, activity accuracy 0.37. Demographics: **age MAE 7.78 y**, age-class accuracy 0.85, sex accuracy 0.79. There is no AF, diabetes or BMI task — [arXiv](https://arxiv.org/html/2410.20542)
- **Pulse-PPG**: beats PaPaGei on 10 of 11 tasks (the exception is sleep disturbance). Averages: F1 0.587 / AUROC 0.748 / MAE 8.86 vs PaPaGei 0.499 / 0.684 / 10.75. Instantaneous HR MAE on PPG-DaLiA is 6.28 better than PaPaGei. Tasks: MOODS stress and activity, PPG-DaLiA HR and activity, WESAD stress, SDB, PPG-BP SBP/DBP/HR/hypertension. **No AF or age task.** The same architecture re-pretrained on MIMIC-III finger PPG was worse on 10 of 11 tasks, so field pretraining beats clinical pretraining for these tasks — [arXiv v2](https://arxiv.org/html/2502.01108v2)
- **AnyPPG**, best on 13 of 15 dataset-task pairs:
  - PPG-DaLiA HR MAE 9.27 (SimCLR 9.37); Gyro-Acc-PPG (wrist) HR MAE 10.82.
  - UCI-BP SBP / DBP MAE 15.62 / 7.14.
  - **BUT PPG (smartphone camera, 30 Hz)**: HR MAE 5.47, SBP 13.31, DBP 9.62 (PaPaGei-P is better at 9.30), quality AUC 0.79.
  - DeepBeat AF AUC 0.69; WESAD 4-class AUC 0.76 (SimCLR 0.77).
  - WenXinWuYang HR 6.83, RR 2.30, **age MAE 6.08**.
  - Phenome-wide on held-out MC-MED (15,759 patients): 307 of 1,468 phenotypes reach AUC ≥0.70, e.g. CHF 0.84, diabetic retinopathy 0.83, dementia 0.81, Parkinson's 0.77, CKD 0.75. The authors note this is a single-cohort, non-causal result.
  - Baselines: MOMENT, Chronos-2, Pulse-PPG, PaPaGei-S/P, SimCLR, BYOL.
  - [arXiv v3](https://arxiv.org/html/2511.01747v3)
- **Generalist vs specialist** (51 PPG tasks): a specialist PPG model gets a 27% higher "win score" than generalist TSFMs such as MOMENT under full fine-tuning — [arXiv 2510.14254](https://arxiv.org/abs/2510.14254)
- **ICML 2026 "A robust PPG foundation model using multimodal physiological supervision"** (Geenjaar et al.): pretrained on public ICU data with ECG and respiration guiding contrastive pairs, using 3× fewer subjects than prior SOTA. Improves 14 of 15 tasks, including field daily activity and HR. No code or weights link on the poster page — [ICML](https://icml.cc/virtual/2026/poster/64846)

#### Aging-specific models and evidence

- **Apple PpgAge** (Miller, Futoma, Abbaspourazad et al., *Nature Communications* 16:9264, Oct 2025):
  - Wrist PPG from 213,593 AHMS participants and >149 M participant-days.
  - **MAE 2.43 y (healthy cohort), 3.18 y (general cohort)**.
  - A 6-year age gap has HR 1.40 for adverse cardiac events after adjustment.
  - People with a gap >6 y have 1.5–5× the age/sex-adjusted rates of heart disease, heart failure and diabetes.
  - PpgAge tracks smoking, exercise and sleep, and rises sharply in pregnancy.
  - **No weights** (Apple data).
  - [Princeton listing](https://collaborate.princeton.edu/en/publications/a-wearable-based-aging-clock-associates-with-disease-and-behavior/); [FightAging summary](https://www.fightaging.org/archives/2025/10/wearable-device-measurement-of-blood-circulation-as-a-basis-for-an-aging-clock/)
- **Apple PPG/ECG foundation models** (Abbaspourazad et al., ICLR 2024, arXiv 2312.05409):
  - PPG input: Apple Watch **4-channel green PPG, 60 s, 64 Hz** (downsampled from 256 Hz), low-motion periods only. ECG input: lead I, 30 s, 512→128 Hz.
  - Encoders: 3.3 M (PPG) and 2.5 M (ECG) EfficientNet-style.
  - Data: 141,207 participants and 19.85 M PPG segments.
  - Participant-level linear-probe results, PPG vs ECG:

    | Target | PPG | ECG |
    |---|---|---|
    | Age MAE | 3.19 y | 6.33 y |
    | Age >50 AUC | 0.976 | — |
    | Sex AUC | 0.993 | — |
    | BMI MAE | 2.54 | — |
    | Hypertension AUC | 0.819 | — |
    | Diabetes AUC | 0.866 | — |
    | AF AUC | 0.818 | — |
    | Heart disease AUC | 0.874 | — |
    | Beta-blockers AUC | 0.801 | — |

  - **Weights and code are not released**; the sensor data "cannot be shared".
  - [arXiv HTML](https://arxiv.org/html/2312.05409v2)
- **AI-PPG Age** (UK Biobank, MIT weights above):
  - An age gap >9 y is associated with higher MACCE risk; external check on 2,343 MIMIC-III-derived patients — [search summary of arXiv 2502.12990](https://arxiv.org/html/2502.12990v4).
  - The independent benchmark (Brag, arXiv 2603.14030, Mar 2026) on 906 PulseDB surgical patients found:
    - AI-PPG Age MAE 9.72 y, with predictions collapsed into a 38–67 y range.
    - Pulse-PPG frozen embeddings + ridge: 9.28 y.
    - HR/HRV + demographics: 9.59 y.
    - Pulse-PPG + demographics: 8.22 y (r = 0.725).
    - The age gap correlates with DBP (r = −0.188).
    - The author attributes the gap to Apple's 2.43 y mainly to sample size; the learning curve has not plateaued.
    - Code is MIT.
  - [arXiv](https://arxiv.org/abs/2603.14030); [GitHub](https://github.com/Misterbra/ppg-age-benchmark)
- **Weizmann/Pheno.AI WatchPAT foundation model** (medRxiv 2026.07.01.26356992):
  - Input: **finger** peripheral arterial tonometry + SpO2 + pulse rate.
  - Model: transformer MAE on 2-min segments; about 310k h from about 11.3k participants (counts inconsistent within the paper).
  - Age r = 0.69 internal and **0.70 on external VitalDB**, vs pyPPG features 0.65.
  - Hypertension AUC 0.67→0.74.
  - 2-year incident hypertension AUC 0.64 with demographics, 0.66 with embeddings alone.
  - Weights and licence not confirmed.
  - [medRxiv](https://www.medrxiv.org/content/10.64898/2026.07.01.26356992v1.full)

#### Closed big-tech models (no weights)

- **Google SensorFM** (arXiv May 2026, blog July 2026):
  - Input is **34 one-minute aggregate features** from PPG, accelerometer, EDA, skin temperature and altimeter, not raw waveform. Pretrained on about 5 M Fitbit/Pixel Watch users.
  - Beats feature baselines on 34 of 35 tasks.
  - Secondary sources only: [The Decoder](https://the-decoder.com/sensorfm/); [LetsDataScience](https://letsdatascience.com/news/google-presents-sensorfm-for-wearable-health-data-f65b9fe5).
  - A survey list marks code and weights as not released — [awesome list](https://github.com/tujy859/Wearable-Physiological-Foundation-Models).
- **Google LSM/LSM-2 and Apple WBM**: not released; LSM-2 appears only as a reimplementation inside OpenMHC — [awesome list](https://github.com/tujy859/Wearable-Physiological-Foundation-Models).
- **OpenMHC** (Stanford/Imperial, arXiv 2607.16235):
  - My Heart Counts data: 11,894 participants, >60 M h across 19 channels such as steps, HR, sleep and workouts, plus 169 linked health variables.
  - Released to qualified researchers, with open code and weights for reimplemented wearable FMs.
  - It holds minute-level aggregates, not raw PPG.
  - [arXiv](https://arxiv.org/abs/2607.16235)

#### LLM-style PPG models (beyond PulseLM)

- **PPG-LM** (Emory/UCLA et al., arXiv 2609.33516): a "PPG-language model family" trained on about 73k h aligned to EHR-derived captions, evaluated on MC-MED, MIMIC-III and VitalDB. No code or weights linked — [arXiv](https://arxiv.org/abs/2609.33516)
- **OpenTSLM** (Stanford; **MIT**): Llama-3.2-1B/3B or Gemma-3 backbones, with pretrained checkpoints on HF. Tasks: HAR, EEG sleep staging, 12-lead ECG-QA, captioning. **No PPG task**. The Llama bases are gated by Meta — [GitHub](https://github.com/OpenTSLM/OpenTSLM)
- **HEARTS benchmark** (arXiv 2603.06638; 16 LLMs, 110 tasks, 20 modalities): LLMs "substantially underperform specialized models" on health time series — [arXiv](https://arxiv.org/abs/2603.06638)
- **CBPM-LLaMA** (cuffless BP, arXiv 2406.18069): listed with code and weights released — [awesome list](https://github.com/tujy859/Wearable-Physiological-Foundation-Models)

#### Sampling-rate and wrist-transfer evidence

- PaPaGei and AnyPPG band-limit PPG to 0.5–8 Hz; HiMAE also band-passes 0.5–8 Hz — [AnyPPG](https://arxiv.org/html/2511.01747v3); [HiMAE](https://arxiv.org/html/2510.25785)
- AnyPPG runs on **30 Hz smartphone PPG** (BUT PPG, upsampled to 125 Hz) with HR MAE 5.47 and SBP MAE 13.31 — [arXiv v3](https://arxiv.org/html/2511.01747v3)
- A wrist-PPG authentication study (NDSS 2026) downsampled from 500 Hz: performance is statistically indistinguishable down to **25 Hz** and drops about 5% at 20 Hz. The authors call 25 Hz the practical lower bound, though this is a task-specific model, not an FM — [arXiv 2508.13690](https://arxiv.org/html/2508.13690v1)
- Our own benchmark on GalaxyPPG (25 Hz Galaxy Watch5 green PPG): with sub-sample beat timing and artifact handling, 60 s RMSSD error is about 3 ms median at rest. It fails in locomotion. Naive peak picking overestimates RMSSD by about 180 ms — `/home/antonkulaga/sources/pulse-exploration/docs/hrv-25hz-benchmark.md`
- No paper found ablates a PPG FM's embeddings at 25 Hz (see Gaps).

### Inferences
- **25 Hz input.** All three open encoders can be fed 25 Hz watch PPG by upsampling: to 125 Hz for PaPaGei and AnyPPG, to 50 Hz for Pulse-PPG. The 0.5–8 Hz passband they use sits below the 12.5 Hz Nyquist limit of 25 Hz data. What is lost is fine morphology: dicrotic notch, second-derivative a–e waves and sub-40 ms timing. Interpolation does not recover it. Morphology-driven targets (age, arterial stiffness, BP) should therefore use the **100 Hz on-demand blocks**, or the 30 fps phone PPG where AnyPPG has evidence. HR, HRV, sleep and activity-type uses can run on the 25 Hz continuous stream.
- **Model choice by signal.**
  - Pulse-PPG is the closest domain match to our continuous watch stream: wrist, WearOS, field, MIT, 100 Hz raw data downsampled to 50 Hz. It also had the best open-model age MAE in the one independent age benchmark.
  - AnyPPG and PaPaGei are finger-domain models; they suit the phone fingertip PPG and the 100 Hz SDK blocks.
- **What transfers to wrist green PPG.** Apple's results (age MAE 3.19, diabetes 0.866, HTN 0.819, AF 0.818) show that age and cardiometabolic status are encoded in wrist green PPG at 64 Hz. Apple needed about 140k participants, participant-level averaging of many low-motion segments, and four channels. Open finger-trained encoders transfer weakly to wrist HR and affect tasks (WESAD about 0.55 AUROC; DaLiA HR MAE 9–12 bpm). BP from wrist PPG is poor for every model.
- **Realistic open baseline for an aging marker.** Expect frozen embeddings + ridge to give an age MAE of about 6–9 y on hundreds-to-thousands of subjects (PaPaGei 7.78, AnyPPG 6.08, Pulse-PPG 9.28 on a different cohort). Getting near Apple's 2.4–3.2 y needs a large labelled wrist cohort, which no public dataset provides (see Q4).
- **Age from face video.** The MCD-rPPG age MAE of 3.91 y (Q4) is likely driven by facial appearance, not pulse physiology. It should not be cited as evidence for camera-PPG aging markers.

### Gaps
- No paper found ablates any PPG FM at 25 Hz or 32 Hz input, so robustness to `hr_raw` is unverified. A direct test (GalaxyPPG 25 Hz vs E4 64 Hz embeddings) is cheap.
- GPT-PPG weights: promised but not found on HF or GitHub as of 2026-10-24.
- SiamQuality weights not confirmed.
- PaPaGei's Zenodo weights carry no explicit licence, and AnyPPG's repo has none. Legally that defaults to all rights reserved; contact the authors before any commercial use.
- PaPaGei per-dataset sex/age numbers beyond the summary line (Appendix Table 16) were not fully read.
- NormWear and NormWear-2 parameter counts and per-task numbers were not extracted.
- Peer-reviewed numbers for the ICML 2026 robust-PPG FM and PPG-LM were not available on the pages read.

## Q2. Which models have been trained or evaluated on Samsung Galaxy Watch data?

### Takeaway
Samsung Research has trained at least three PPG foundation models on large Galaxy Watch corpora. HiMAE used 100 Hz green PPG from 47,644 people, MMR used about 32k users, and xMAE used about 9,400 h of paired ECG+PPG. **None has released weights**, and HiMAE explicitly withholds code. No open model was pretrained on Samsung data. The only public Galaxy Watch PPG data is GalaxyPPG (Galaxy Watch5, 25 Hz green via the SDK, 24 young adults, CC BY 4.0). It is a fine validation set for HR/HRV and the 25 Hz pipeline, but useless for aging, since all participants are 20–29.

### Cited Findings
- **HiMAE** (Samsung Research, ICLR 2026, arXiv 2510.25785):
  - Pretraining data: **green PPG at 100 Hz, 10 s windows**, about 80,000 h from 7 free-living studies and 47,644 participants. Devices: mainly Galaxy Watch Active 2, Watch 3 and Watch Active, plus Gear S3. Collected in the USA, Brazil, Bangladesh and South Korea.
  - Preprocessing: 0.5–8 Hz band-pass, scaled to [−1, 1].
  - Model: about 1.2 M parameters (4.8 MB), 0.99 ms on a Galaxy Watch8 CPU (prototype).
  - Tasks: hypertension, PVC, DREAMT sleep staging and Tulane abnormal labs. BP regression was poor for all models.
  - Authors: "we are unable to release the full source code"; nothing is said about weights.
  - [arXiv HTML](https://arxiv.org/html/2510.25785); [Samsung Research](https://research.samsung.com/research-papers/HiMAE-Hierarchical-Masked-Autoencoders-Discover-Resolution-Specific-Structure-in-Wearable-Time-Series)
- **Masked Multiscale Reconstruction (MMR)** (Thukral, Tanade, Lee … Arcot Desai; arXiv 2601.12215): wavelet-based masked reconstruction on about 17 M unlabeled 10 s PPG segments from about 32,000 users of "various types of Samsung Galaxy smartwatches". Improves or matches SOTA on 17 of 19 tasks. No code or weights linked — [arXiv abs](https://arxiv.org/abs/2601.12215); [arXiv PDF](https://arxiv.org/pdf/2601.12215)
- **xMAE** ("Physiology-Aware Masked Cross-Modal Reconstruction", Samsung, ICML, listed 2026-05-01):
  - Method: reconstructs masked ECG from PPG; pretrained on about 9,400 h of paired ECG+PPG.
  - Results: beats unimodal and multimodal baselines on 15 of 19 tasks, including CV outcomes, abnormal labs, sleep staging and demographics.
  - No public weights found.
  - [Samsung Research](https://research.samsung.com/research-papers/Physiology-Aware-Masked-Cross-Modal-Reconstruction-for-Biosignal-Representation-Learning); [Samsung Newsroom](https://news.samsung.com/global/from-biosignals-to-health-insights-samsung-researchs-work-on-health-foundation-models)
- **GalaxyPPG** (Park, Zheng & Lee, KAIST, *Scientific Data* 2025):
  - Signals: Galaxy Watch5 **25 Hz green PPG** via the Samsung Health Sensor SDK (must be inverted); Galaxy HR with IBI and status flags; Empatica E4 BVP at 64 Hz; Polar H10 ECG at 130 Hz.
  - Participants: 24, aged 20–29, East Asian.
  - Activities: rest, TSST/SSST stress, meditation, typing, walking, jogging, running; 7-point stress ratings.
  - Hosting: Zenodo 14635823, **CC BY 4.0**, 481 MB.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12119839/); [Zenodo](https://zenodo.org/records/14635823)
- Pulse-PPG's wrist pretraining data is a **Fossil** WearOS watch, not Samsung — [arXiv v2](https://arxiv.org/html/2502.01108v2). PaPaGei and AnyPPG pretrain only on clinical finger PPG — [PaPaGei](https://arxiv.org/html/2410.20542); [AnyPPG](https://arxiv.org/html/2511.01747v3).
- The Samsung Health Sensor SDK exposes raw PPG green, red and IR on Galaxy Watch4 and later, and needs a matching Health Platform version — [Samsung SDK](https://developer.samsung.com/health/sensor/overview.html). Developers report that raw PPG distribution needs partnership approval — [Samsung forum](https://forum.developer.samsung.com/t/raw-ppg-access/42055).
- Other consumer-watch datasets (non-Samsung):
  - HEART-Watch (Google Pixel Watch) — [arXiv 2512.03988](https://arxiv.org/pdf/2512.03988)
  - Multi-site PPG (custom ring, earring, watch and necklace with Polar H10 reference; 20 participants; NeurIPS 2026) — [arXiv 2605.17859](https://arxiv.org/pdf/2605.17859)

### Inferences
- HiMAE and MMR show that Samsung has internally validated 100 Hz green PPG for hypertension, PVC, sleep staging and abnormal-lab detection. They establish feasibility on Galaxy hardware but give us nothing runnable. The HiMAE paper's architecture and hyperparameters are detailed enough to reimplement and pretrain on our own data; at 1.2 M parameters that is cheap.
- For our devices, the right open stack is: Pulse-PPG embeddings on the 25 Hz and 100 Hz watch PPG; AnyPPG/PaPaGei on the 100 Hz SDK blocks and fingertip camera; GalaxyPPG as the public Samsung-format check set.

### Gaps
- Exact downstream AUROCs for HiMAE, MMR and xMAE: they are in the figures, which were not read.
- Whether MMR or xMAE weights will be released: no statement found.
- No public dataset of Samsung watch PPG with older participants or disease labels was found.

## Q3. Signal-processing libraries: licences and features

### Takeaway
The permissive core is NeuroKit2 (MIT), HeartPy (MIT in the current repo), BioBSS (MIT), vital_sqi (MIT), pyHRV (BSD-3) and BioSPPy (BSD-3). pyPPG, which gives the richest morphology set (74 biomarkers), is **GPL-3 with an added "academic research purpose only" sentence**, so keep it server-side and out of anything distributed. rPPG-Toolbox uses a Responsible-AI licence with use restrictions; pyVHR is GPL-3; open-rppg is MIT.

### Cited Findings

| Library | Licence (repo) | What it gives us | Notes |
|---|---|---|---|
| **pyPPG** (Goda, Charlton, Behar; Physiol. Meas. 2024, doi 10.1088/1361-6579/ad33a2) | **GPL-3.0**, LICENSE adds "provided for academic research purpose only" — [GitHub](https://github.com/godamartonaron/GODA_pyPPG) | Per-beat fiducial points and **74 PPG digital biomarkers**; optional SQI after Li et al. 2015; resamples filtered signals to 75 Hz for systolic peak detection — [docs](https://pyppg.readthedocs.io/en/latest/) | Used as the hand-crafted baseline (age r 0.65) in the Weizmann PAT FM paper — [medRxiv](https://www.medrxiv.org/content/10.64898/2026.07.01.26356992v1.full) |
| **NeuroKit2** | **MIT**, about 2.4k stars, active (pushed 2026-10-01) — [GitHub](https://github.com/neuropsychology/NeuroKit) | PPG/ECG/EDA/RSP processing, peaks, quality, HRV | Our reference R-peak pipeline in the 25 Hz HRV benchmark — `docs/hrv-25hz-benchmark.md` |
| **HeartPy** | **MIT** (LICENSE © 2021 Paul van Gent) — [GitHub](https://github.com/paulvangentcom/heartrate_analysis_python) | PPG/ECG peak detection, HR/HRV, breathing rate | — |
| **BIOBSS** | **MIT** — [GitHub](https://github.com/obss/BioBSS) | ECG/PPG/EDA/ACC processing and features | PaPaGei uses it for flatline detection — [PaPaGei GitHub](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) |
| **vital_sqi** | **MIT**, last push 2024-11 — [GitHub](https://github.com/meta00/vital_sqi) | Physiological signal quality control (SQI suite) | — |
| **pyHRV** | **BSD-3-Clause** — [GitHub](https://github.com/PGomes92/pyhrv) | HRV time, frequency and nonlinear features | — |
| **hrv-analysis** (Aura) | **GPL-3.0** — [GitHub](https://github.com/Aura-healthcare/hrv-analysis) | HRV features and RR cleaning | — |
| **BioSPPy** | BSD-3-style (custom text), last push 2022 — [GitHub](https://github.com/PIA-Group/BioSPPy) | General biosignal processing | Stale |
| **FLIRT** (ETH) | licence not machine-readable (NOASSERTION) — [GitHub](https://github.com/im-ethz/flirt) | Wearable (E4) HRV/EDA/ACC feature windows | — |
| **PPG-beats** (Charlton, MATLAB) | **GPL-3.0 / GPL-2.0** — [GitHub](https://github.com/peterhcharlton/ppg-beats) | Benchmark of PPG beat detectors | — |
| **RRest** (Charlton, MATLAB) | licence not machine-readable — [GitHub](https://github.com/peterhcharlton/RRest) | Respiratory-rate algorithms from PPG/ECG | — |
| **open-rppg** | **MIT** (LICENSE has an extra "Licensing Notice") — [GitHub](https://github.com/KegangWangCCNU/open-rppg) | `pip install open-rppg`, JAX. Face-video HR plus SQI, IBI, SDNN, RMSSD, pNN50, LF/HF, breathing rate; default `FacePhys.rlap` model; real-time webcam — [GitHub](https://github.com/KegangWangCCNU/open-rppg) | Per-model weight terms not checked |
| **rPPG-Toolbox** (NeurIPS 2023) | **Responsible AI Source Code License v1.1** (use restrictions) — [GitHub](https://github.com/ubicomplab/rPPG-Toolbox) | Training and evaluation of many rPPG models | — |
| **pyVHR** | **GPL-3.0** — [GitHub](https://github.com/phuselab/pyVHR) | Classical and deep rPPG pipeline | — |

### Inferences
- For a pipeline that might ship to users, build on NeuroKit2, HeartPy, BIOBSS, vital_sqi and pyHRV (all permissive). Keep pyPPG for server-side research features, or reimplement the handful of morphology indices we need. Under GPL, distributing a linked binary forces GPL; the "academic only" clause conflicts with GPL freedoms and is legally ambiguous.
- At 25 Hz, pyPPG's internal 75 Hz resampling cannot restore notch and APG detail. Run pyPPG morphology only on the 100 Hz SDK blocks or 30 fps fingertip PPG, and treat the latter's second-derivative features as noisy.

### Gaps
- The exact NeuroKit2 PPG quality methods and HeartPy feature lists were not re-read in this pass.
- vital_sqi's maintenance status: no release since 2024-11.
- No PPG quality index has been validated specifically on Galaxy `hr_raw`.

## Q4. Public datasets with labels for health and aging

### Takeaway
For **age/aging** labels at scale, the realistic options are UK Biobank (averaged finger pulse waveform for about 236k participants plus linked outcomes, paid access), MC-MED (Stanford ED: ECG+Pleth+Resp with ICD, labs and age for 70k patients, credentialed), VitalDB/PulseDB (about 5–6k surgical patients, open) and MESA/CFS via NSRR (finger PPG during PSG, DUA). **No public wrist dataset combines a broad age range with outcome labels.** The wrist sets are small lab or sleep studies: DREAMT (100), WESAD (15), PPG-DaLiA (15), GalaxyPPG (24, ages 20–29) and the AF sets DeepBeat (about 170) and Vilnius (45). Aurora-BP (1,125, wrist PPG + BP, ages 21–85) is the best wrist set for BP and vascular age, under a DUA. For camera PPG: BUT PPG (phone fingertip, 30 Hz), MCD-rPPG (600 people, phone face video, 13 biomarkers incl. age, HbA1c, cholesterol), VitalVideos (about 900, free subset for non-profit), MMPD, UBFC-rPPG and PURE.

### Cited Findings

#### Large clinical / cohort sets with age and outcome labels

| Dataset | Signals | Labels | Size | Access |
|---|---|---|---|---|
| **UK Biobank** field 4205 "Pulse wave pressure versus time response curve" (arterial stiffness category) | Finger PPG pulse curve; AI-PPG Age consumes it as an averaged single beat (100 samples) — [HF card](https://huggingface.co/Ngks03/PPG-VascularAge) | Age, all UKB phenotypes and linked outcomes (MACCE used by AI-PPG Age) — [search summary](https://arxiv.org/html/2502.12990v4) | **236,313 participants**, 286,819 items, 4 instances — [UKB Showcase](https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=4205) | Approved application. Fees: Tier 1 £3,000 (questionnaire, physical measures, outcomes), Tier 2 £6,000, Tier 3 £9,000 for 3 years; £500 for students and LMICs; access only via UKB-RAP — [UKB fees](https://www.ukbiobank.ac.uk/use-our-data/fees/) |
| **MC-MED** v1.0.1 (Stanford ED 2020–22) | ECG lead II, **PPG (Pleth)**, Resp in WFDB | ICD-9/10 diagnoses, disposition (admit/ICU), LOS, age, sex, triage vitals, labs, radiology | 118,385 visits / 70,545 patients | **Credentialed** + CITI + DUA (PhysioNet Credentialed Health Data Licence 1.5.0) — [PhysioNet](https://physionet.org/content/mc-med/) |
| **VitalDB** (SNUH 2016–17) | Up to 12 waveform tracks incl. arterial pressure; PPG at 500 Hz per PaPaGei — [PaPaGei](https://arxiv.org/html/2410.20542) | 72–74 clinical parameters (demographic, surgical, outcomes); 34 lab time series | 6,388 non-cardiac surgery cases | **Open**, CC BY 4.0 on PhysioNet — [PhysioNet](https://physionet.org/content/vitaldb/). Conflict: PulseDB labels its VitalDB-derived files CC BY-NC-SA 4.0 — [PulseDB](https://github.com/pulselabteam/PulseDB) |
| **PulseDB** v2 | 10 s ECG, PPG, ABP (raw and filtered) from MIMIC-III matched subset + VitalDB | SBP/DBP, age, gender; height/weight only for VitalDB | 5,361 subject files — [PulseDB](https://github.com/pulselabteam/PulseDB); 4,964 subjects / 12,768 h per AnyPPG — [AnyPPG](https://arxiv.org/html/2511.01747v3) | MIMIC part ODbL 1.0; VitalDB part CC BY-NC-SA 4.0; Box / Google Drive / Kaggle — [PulseDB](https://github.com/pulselabteam/PulseDB) |
| **MIMIC-III matched subset** | ICU finger PPG at 125 Hz | In-hospital mortality (PaPaGei T3 AUROC 0.67), age | 5,596 participants used by PaPaGei — [PaPaGei](https://arxiv.org/html/2410.20542) | ODbL per PulseDB — [PulseDB](https://github.com/pulselabteam/PulseDB). Benchmark: MIMIC-III-Ext-PPG (arrhythmia, RR, HR, BP) — [arXiv 2603.21832](https://arxiv.org/html/2603.21832v1) |
| **MESA sleep** (NSRR) | Finger PPG during PSG (Nonin; 256 Hz per PaPaGei, 64 Hz in another source) | Smoking and AHI (PaPaGei tasks); MESA cohort phenotypes | 2,055 participants (PaPaGei); 2,010 (AnyPPG) | NSRR DUA. A forum user could not find raw PPG in one EDF, so check channel labels — [NSRR forum](https://sleepdata.org/forum/dataset-sleep-ppg-and-ecg); [SleepPPG-Net2](https://arxiv.org/pdf/2404.06869) |
| **CFS** (NSRR) | Finger PPG 128 Hz (Nonin 8000) | PSG and family study | 322 subjects in AnyPPG | Non-commercial only — [search summary](https://arxiv.org/pdf/2404.06869); [AnyPPG](https://arxiv.org/html/2511.01747v3) |
| **HSP** (Human Sleep Project) | PSG PPG | Sleep | 1,584 subjects in AnyPPG — [AnyPPG](https://arxiv.org/html/2511.01747v3) | via BDSP |
| **SHHS** (NSRR) | Raw PPG **not confirmed** in the EDFs | — | — | NSRR forum question unanswered — [NSRR forum](https://sleepdata.org/forum/dataset-sleep-ppg-and-ecg) |
| **nuMoM2b** | PSG PPG | Pregnancy stage and gestational age (PaPaGei T7–T8) | — | NSRR — [PaPaGei](https://arxiv.org/html/2410.20542) |
| **OpenMHC** (My Heart Counts) | Apple-Watch-derived minute-level aggregates (HR, steps, sleep, workouts), 19 channels | 169 linked health, lifestyle and mood variables | 11,894 participants, >60 M h | Qualified researchers — [arXiv](https://arxiv.org/abs/2607.16235) |

#### Wrist / wearable PPG sets

| Dataset | Device / rate | Labels | Size | Access |
|---|---|---|---|---|
| **Aurora-BP** (Microsoft) | Wrist optical PPG (MAX30101, left wrist) + tonometry, ECG, accelerometer, brachial BP | BP, age 21–85 | 1,125 subjects | Some papers cite a DUA, others call it open — [Microsoft Aurora](https://www.microsoft.com/en-us/research/project/aurora/); [medRxiv using Aurora](https://www.medrxiv.org/content/10.1101/2025.05.22.25328194.full.pdf) |
| **DREAMT** v2.2.0 (Duke) | Empatica E4 BVP 64 Hz (left wrist) + full PSG at 100 Hz | 30 s sleep stages, AHI/OAHI, OSA severity, age, BMI | 100 participants | **Restricted** (DUA; PhysioNet Restricted Health Data Licence 1.5.0) — [PhysioNet](https://physionet.org/content/dreamt/) |
| **GalaxyPPG** | Galaxy Watch5 25 Hz green + E4 64 Hz + Polar H10 ECG 130 Hz | Activities, stress ratings, ECG HR/IBI, age, sex | 24 (ages 20–29) | **CC BY 4.0**, Zenodo — [Zenodo](https://zenodo.org/records/14635823) |
| **PPG-DaLiA** | E4 BVP 64 Hz + RespiBAN ECG 700 Hz | ECG HR ground truth, activities | 15 subjects, 2.7 GB | **CC BY 4.0** — [UCI](https://archive.ics.uci.edu/dataset/495/ppg+dalia) |
| **WESAD** | E4 BVP 64 Hz, EDA, temperature, ACC 32 Hz + RespiBAN 700 Hz | Baseline / stress / amusement + questionnaires | 15 subjects | Licence not on UCI page — [UCI](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) |
| **DeepBeat** (Stanford; Torres-Soto & Ashley 2020) | Wrist PPG, 25 s segments, 32 Hz | AF vs non-AF; signal quality (partly model-generated) | 175 individuals (108 AF / 67 non-AF); about 3.3 M segments from 167 subjects in another account | Synapse syn21985690 — [arXiv 2308.02433](https://arxiv.org/pdf/2308.02433); [arXiv 2607.19999](https://arxiv.org/pdf/2607.19999); [DeepBeat](https://arxiv.org/pdf/2001.00155) |
| **Vilnius long-term ECG + wrist PPG AF** (Zenodo v2, 2024) | Custom wrist device, **100 Hz green PPG** + ACC 50 Hz; Bittium Faros ECG 500 Hz | Beat-to-beat AF annotations, RR, demographics and comorbidities | 45 patients × 5–8 days (306 days) | "Other (Non-Commercial)"; files restricted (login/request) — [Zenodo](https://zenodo.org/records/11242869) |
| **MOODS** (Pulse-PPG) | Fossil WearOS, 100 Hz | Stress, activity | 120 × 100 days | Not stated as public; only the model is released — [Pulse-PPG](https://arxiv.org/html/2502.01108v2) |
| **PPG-BP** (Liang et al. 2018) | Finger PPG, short records | SBP, DBP, HR, hypertension | 219 subjects | Open; used by PaPaGei and Pulse-PPG — [PaPaGei](https://arxiv.org/html/2410.20542); [Pulse-PPG](https://arxiv.org/html/2502.01108v2) |

#### Camera PPG (fingertip and face)

| Dataset | Device / rate | Labels | Size | Access |
|---|---|---|---|---|
| **BUT PPG** | Smartphone fingertip PPG at 30 Hz | HR, SBP, DBP, quality (as used by AnyPPG) | — | PhysioNet; licence not verified (page fetch failed) — [AnyPPG](https://arxiv.org/html/2511.01747v3) |
| **MCD-rPPG** ("Gaze into the Heart", 2025) | Face video from a **Samsung A3s phone**, camcorder and webcam (24–30 fps, 640×480); reference PPG 100 Hz + ECG | Weight, height, BMI, **age**, SBP/DBP, SpO2, temperature, Hb, **HbA1c, cholesterol**, RR, HR, **arterial stiffness**, stress (PSM-25), sex | 600 subjects, 3,600 videos (rest + post-exercise) | Hugging Face, marked "open"; dataset licence not stated. Baseline model: age MAE 3.91 vs 5.71 naive; SBP 12.82; 3.9 MB, runs on CPU — [arXiv](https://arxiv.org/html/2508.17924v1) |
| **VitalVideos-Europe** | Face video + PPG + BP | BP | 850–893 participants | Free subset for non-profit users who must publish code and models; commercial licence for a fee — [arXiv 2306.11891](https://arxiv.org/pdf/2306.11891). VitalVideos-Worldwide: about 7,000 participants; free academic 300- and 900-participant subsets (secondary source) — [LinkedIn](https://be.linkedin.com/in/pjtoye) |
| **MMPD** | Phone video (Galaxy S22 Ultra) | Fitzpatrick 3–6, 4 lightings, 4 activities | 33 subjects, 11 h | [arXiv 2302.03840](https://arxiv.org/pdf/2302.03840) |
| **UBFC-rPPG** | Logitech C920 webcam, 30 fps | — | 42–43 subjects | [arXiv 2604.01679](https://arxiv.org/pdf/2604.01679) |
| **PURE** | Camera, CMS50E oximeter reference | 6 head-motion scenarios | 10 subjects / 60 sequences | [arXiv 2604.01679](https://arxiv.org/pdf/2604.01679) |

The **Apple AHMS** sensor data "cannot be shared" — [Apple ICLR 2024](https://arxiv.org/html/2312.05409v2).

### Inferences
- **Training an aging marker.** UK Biobank (age + outcomes, about 236k) or MC-MED (age + ICD phenome, 70k) are the only public-access routes at a scale comparable to Apple's. Both are finger PPG: averaged beats in UKB, ED bedside monitor waveforms in MC-MED. A model trained there would apply to our phone fingertip PPG and, after beat-averaging, to the 100 Hz watch blocks. Transfer to 25 Hz wrist data needs its own validation.
- **Validating on wrist.** Aurora-BP (ages 21–85, wrist PPG + BP) and DREAMT (wrist E4 + age/AHI) are the only wrist sets with a usable age spread, and both are small. An honest wrist age validation therefore probably needs our own cohort.
- **Licences for a possibly commercial product.** VitalDB (via vitaldb.net terms), CFS, the Vilnius AF set and VitalVideos (free tier) are non-commercial. MC-MED and DREAMT carry credentialed/restricted PhysioNet licences that forbid redistribution.

### Gaps
- BUT PPG version, size and licence: the PhysioNet page fetch failed.
- WESAD licence (UCI defers to the original site).
- Aurora-BP current access route.
- Whether SHHS EDFs contain raw PPG.
- MESA PPG sampling rate: 256 vs 64 Hz conflict.
- MESA outcome/mortality linkage: NSRR hosts the sleep files; event data likely needs a separate MESA Coordinating Center application. Not verified.
- WildPPG, UBFC-Phys and HPP (Weizmann) access terms were not checked.
- The Vilnius AF Zenodo page states 36.0 TB, which is likely an error.

## Q5. Practicalities: what runs on an RTX 4090 16 GB, and which models are non-commercial or gated

### Takeaway
Every open PPG encoder here is tiny: AnyPPG 4 M, PaPaGei 5–5.7 M, Pulse-PPG 28.5 M, HiMAE-class 1.2 M and AI-PPG Age small. They run inference on CPU, and fine-tuning or even re-pretraining fits easily on a 16 GB laptop 4090. Only the LLM-style models need care: PulseLM 7B must be quantised, while OpenTSLM's 1B fits. Licensing is the real constraint.
- Clean permissive weights: Pulse-PPG (MIT), NormWear and NormWear-2 (Apache-2.0), AI-PPG Age (MIT), ECGFounder (MIT).
- PaPaGei: BSD-3 code, but the weights have no licence.
- AnyPPG, SiamQuality and PPGFlowECG: no licence.
- CP-PPG: CC BY-NC (from earlier notes).
- Gated: PulseLM, D-BETA, ECG-Scan (manual), and Llama bases.

### Cited Findings
- **Parameter counts:**
  - PaPaGei 5 M / 5.7 M; weight files 20–23 MB — [arXiv](https://arxiv.org/html/2410.20542); [Zenodo](https://zenodo.org/records/13983110)
  - AnyPPG about 4.04 M — [arXiv v3](https://arxiv.org/html/2511.01747v3)
  - Pulse-PPG about 28.5 M; pretraining took about 5 days on one L40S at batch 64 — [arXiv v2](https://arxiv.org/html/2502.01108v2)
  - HiMAE about 1.2 M — [arXiv](https://arxiv.org/html/2510.25785)
  - Apple PPG encoder 3.3 M — [arXiv](https://arxiv.org/html/2312.05409v2)
  - GPT-PPG 19 M / 85 M (unreleased) — [Emory PDF](https://cs.emory.edu/~jyang71/files/gpt-ppg.pdf)
- **Licence and gating status checked via GitHub, HF and Zenodo APIs on 2026-10-24:**
  - Pulse-PPG: MIT (GitHub and Zenodo) — [Zenodo](https://zenodo.org/records/17345536)
  - PaPaGei: BSD-3 repo; Zenodo weights with no licence field — [GitHub](https://github.com/Nokia-Bell-Labs/papagei-foundation-model); [Zenodo](https://zenodo.org/records/13983110)
  - AnyPPG: no licence — [GitHub](https://github.com/PKUDigitalHealth/AnyPPG)
  - ECGFounder: MIT — [GitHub](https://github.com/PKUDigitalHealth/ECGFounder)
  - NormWear and NormWear-2: Apache-2.0, ungated — [HF](https://huggingface.co/mosaic-laboratory/normwear); [HF](https://huggingface.co/mosaic-laboratory/normwear2)
  - AI-PPG Age: MIT, ungated. The card also advertises a remote inference API on a raw IP address; do not send personal data to it — [HF](https://huggingface.co/Ngks03/PPG-VascularAge)
  - CP-PPG: HF gating "auto" — [HF API listing](https://huggingface.co/Manhph2211/CP-PPG)
  - OpenTSLM: MIT, but its default Llama-3.2 bases need Meta access — [GitHub](https://github.com/OpenTSLM/OpenTSLM)
  - NormWear-2 and the Manhph2211 models need `trust_remote_code` — [HF](https://huggingface.co/mosaic-laboratory/normwear2)

### Inferences
- **Memory.** In fp32, a 28.5 M-parameter encoder takes about 115 MB of weights. Inference on hours of 25 Hz data in 10 s windows is batch-bound, not memory-bound. Fine-tuning Pulse-PPG on 16 GB is feasible at reduced batch size, and re-pretraining a HiMAE-sized model on our own continuous watch data is a one-GPU job.
- **Commercial path.** Only MIT/Apache/BSD components are safe: Pulse-PPG, NormWear(-2), AI-PPG Age, ECGFounder, and PaPaGei code. PaPaGei weights and AnyPPG need written permission from Nokia Bell Labs and PKU. The data licences of the corpora these models were trained on (VitalDB CC BY-NC-SA via PulseDB, MC-MED credentialed) may also matter to a cautious legal reading.
- **Run hygiene.** Small encoders can share one uv environment. Keep `trust_remote_code` models isolated, as `docs/model-feasibility.md` already recommends.

### Gaps
- NormWear and NormWear-2 parameter counts and VRAM were not measured. They are expected to fit easily, but this is not verified.
- No checkpoint was load-tested in this pass.
- Licence intent for the PaPaGei weights and AnyPPG was not confirmed with the authors.
