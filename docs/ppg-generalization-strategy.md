# Making our PPG models survive a change of device: a strategy that ends on the Galaxy Watch

Written 2026-11-10. It builds on:

- [galaxy-watch-ppg-foundation-models.md](galaxy-watch-ppg-foundation-models.md), section 4: the r0 results;
- [phase0-offline-findings.md](phase0-offline-findings.md) and [model-feasibility.md](model-feasibility.md): what the
  watch delivers;
- the model and dataset survey in
  [research_notes/…/open_models_datasets.md](../research_notes/PPG%20health%20and%20aging%20measures/open_models_datasets.md);
- kron-fm's `reports/PPG foundation models and aging clocks 2026.md` and the `kron_ppg` code.

It does not repeat those. It adds:

- a diagnosis of why transfer fails, from our own results and code;
- a literature check (2026-11-10) of how others make PPG models generalise;
- ranked recommendations, each with its evidence, expected gain, cost and a test;
- a critique of the draft five-phase plan;
- an ordered experiment plan that ends in a Galaxy Watch validation.

Every claim from the literature carries its source; "[snippet]" marks claims seen only in a search result.

## 1. Short answers

**Do we need input adapters?** Not for the sampling rate.
- Every PPG foundation model we checked resamples to one fixed rate (40, 50, 64, 100 or 125 Hz) and
  works.
- Samsung's MMR, the closest precedent, pretrained on watch PPG that was "mostly 25–100 Hz" and simply
  polyphase-resampled it to 100 Hz ([MMR](https://arxiv.org/html/2601.12215)).
- The deployed input is one fixed rate (25 Hz watch green), and the 0.5–8 Hz pulse band sits under its
  12.5 Hz Nyquist limit.
- Learned per-device front ends have no published evidence in PPG. They become worth trying only once we
  have paired recordings that show a device effect left over after the fixes below. Even then, the form we
  would try is a tiny per-device FIR equaliser (section 4, R10).

**Do we need smart normalisation?** We already have most of what helps.
- Per-window robust scaling removes gain and offset, which are site and device cues, not age cues.
- Add three things:
  - a deterministic polarity rule;
  - handling of gain steps inside a window;
  - per-person baselines on the output, for the within-person trend the app shows.
- Do **not** re-centre embeddings per device. Device and age are confounded in our data (VitalDB is old,
  GalaxyPPG is 20–29), and plain re-centring would erase the age signal (section 3.5).

**Something else?** Yes. The bigger levers are data and the yardstick, not adapters.
- **The yardstick.** Our "transfer fails" mixes four shifts: input format, sensor site, physiological
  state (anaesthesia against awake and moving) and age range. Most of them are not format problems. The
  heart-rate-only baseline, which sees no pulse shape at all, already predicts a mean of 58.6 y for
  GalaxyPPG's 20–29-year-olds.
- **Train/serve parity.** Training uses a zero-phase filter; the watch and phone will use a causal one.
- **Positives across sessions and devices.** Today's same-person positives come from one recording.
- **Unlabelled wrist and field PPG in pretraining.**
- **An ECG anchor.** It costs us little, because VitalDB, GalaxyPPG, two open multi-site sets and the
  watch all record ECG.

## 2. Why transfer fails: what our own results and code say

### 2.1 Four shifts, not one

| shift | VitalDB (training) | GalaxyPPG (wrist check) | BIDMC | fixed by an input adapter? |
|---|---|---|---|---|
| **format**: rate chain, device filter, filter phase, polarity, gain steps, hidden lost samples | 500 Hz clinical monitor, decimated cleanly to 25 Hz | Watch5 SDK green, 25 Hz, inverted, device filtering unknown | 125 Hz ICU monitor | yes, mostly by deterministic canonicalisation |
| **site and optics** | finger, transmissive red/IR | dorsal wrist, reflective green (shallow capillaries, smoother beat, weaker notch) | finger | no; needs wrist data or paired data |
| **physiological state** | general anaesthesia: vasodilation, low HR, ventilator-driven breathing, vasopressors | awake, and the reader keeps the **whole session** including walking and jogging (`read_galaxyppg` takes all of `PPG.csv`; only the periodicity gate filters) | ICU patients | no; needs awake data and state-matched evaluation |
| **population and labels** | ages 0.4–94, mean 57, SD 15 | ages 20–29 only | elderly, n = 53 | no; needs age-matched comparison |

Evidence that state and label shift matter, not only format:

- The **HR-only baseline** (mean and SD of heart rate, no shape) predicts a mean of 58.6 y on GalaxyPPG. A
  feature that is identical on every device still breaks the readout, so the domain gap is partly
  physiological.
- **Regression to the cohort mean.** In an independent benchmark on 906 VitalDB patients, Pulse-PPG plus
  demographics had MAE 15.69 y for ages 20–40 against 5.59 y for ages 50–60
  ([Brag](https://arxiv.org/html/2603.14030v1)). A cohort aged 20–29 sits exactly where every VitalDB-trained
  probe is weakest, even in-domain.
- VitalDB has 299 cases under 30 y (counted 2026-11-10 from `cases.csv.gz`). They are the right comparison
  group for GalaxyPPG, not the whole test set.
- Cross-dataset BP shows the same pattern: the error grows mainly with the difference in **label**
  distribution between datasets ([Moulaeifard, Charlton & Strodthoff 2025](https://arxiv.org/html/2502.19167)).

### 2.2 Three problems in our own pipeline

1. **Filter phase: train/serve mismatch.** `kron_ppg.signal.bandpass` is zero-phase (`sosfiltfilt`). Its
   docstring says a phone will use "a causal band-pass". A causal Butterworth delays each frequency
   differently, so it shifts crest time and smears the dicrotic notch: exactly the shape features that
   carried CKD, BMI and ASA signal in r0. Every model we would ship is fed something it never saw.
2. **Polarity is hard-coded per source.** `read_galaxyppg` returns `-x`. The no-setup `hr_raw` stream, the
   SDK stream, the phone camera and any new dataset each need someone to know the sign.
3. **The same-person positives are same-recording positives.**
   - Run 1 used one VitalDB case per person, so the InfoNCE term rewarded whatever made windows of one
     recording alike: the monitor, the probe placement, the surgery's drug state. These are session and
     device fingerprints, the opposite of what we want to transfer.
   - VitalDB's `subjectid` links repeat surgeries: 216 people have two or more cases with PLETH (486
     cases). These are real cross-session positives that we are not using as such.

### 2.3 What the deployed number is

The project's rule is that a PPG age is a within-person trend. That changes what has to transfer:

- A **constant** device or site offset cancels in a personal baseline.
- What must transfer is the **slope**: how the reading moves when the person's vascular state moves.
- The **within-person nuisance variance** must also stay small. That variance comes from posture,
  temperature, contact pressure, time of day and motion.
- So the watch-side test is **test-retest repeatability and response to known within-person changes**, not
  cross-sectional MAE on a wrist cohort. Cross-sectional outcome evidence has to come from cohorts that
  have outcomes (section 6), connected to the watch by paired recordings.

## 3. What the literature says (checked 2026-11-10)

### 3.1 Sampling rate and input format

| model | rate, window | filter | normalisation | polarity | rate or device experiments |
|---|---|---|---|---|---|
| Apple PPG FM ([2312.05409](https://arxiv.org/html/2312.05409v2)) | 64 Hz (from 256), 60 s, 4 green channels | band-pass, cut-offs unstated | per-segment, per-channel z-score | – | none |
| HiMAE ([2510.25785](https://arxiv.org/html/2510.25785)) | 100 Hz, 10 s, patch 5 | 0.5–8 Hz | scaled to [−1, 1], SQI gate | – | 7 devices, no per-device result; "resolution" means encoder depth, not rate |
| MMR ([2601.12215](https://arxiv.org/html/2601.12215)) | most data 25–100 Hz → polyphase to 100 Hz, 10 s | zero-phase 0.5–10 Hz Chebyshev, then drops wavelet bands above 10 Hz | z-score | – | none |
| xMAE ([2605.00973](https://arxiv.org/html/2605.00973)) | 100 Hz, 10 s, patch 40 | Butterworth 0.5–8 Hz | [−1, 1] | – | pretrained on MIMIC-III, evaluated on smartwatch PPG |
| Pulse-PPG ([2502.01108](https://arxiv.org/html/2502.01108v2)) | 50 Hz, 4 min | **none** | per-person z-score + InstanceNorm at the input | – | field vs clinical pretraining (below) |
| PaPaGei ([2410.20542](https://arxiv.org/html/2410.20542v2)) | 125 Hz, 10 s | 0.5–12 Hz Chebyshev | per-window z-score | negation only as augmentation (PaPaGei-P) | "sampling rate effects" listed as future work |
| AnyPPG ([2511.01747](https://arxiv.org/html/2511.01747v2)) | 125 Hz, 10 s | 0.5–8 Hz | z-score | auto-corrects ECG polarity, not PPG | none |
| GPT-PPG ([2503.08015](https://arxiv.org/html/2503.08015)) | 40 Hz, 30 s, 1 s patches | none | min-max to [0.1, 0.9] | – | reports "limited OOD generalization" |
| Geenjaar et al., ICML 2026 ([2606.07365](https://arxiv.org/html/2606.07365v1)) | 125 Hz, 10 s | raw in pretraining | InstanceNorm1d at the input | negation augmentation, p = 0.2 | WildPPG by body site: every model does best on head PPG |
| NormWear ([code](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear), Apache-2.0) | fixed 65 Hz | – | scaled by mean \|x\| | – | its `sampling_rate` argument only resamples above 256 Hz. Its wavelet tokenizer counts scales in samples, so a 25 Hz input lands on the wrong frequency scale |
| MOMENT, Chronos-2, TimesFM 2.x, Moirai | – | – | – | – | none models rate explicitly. TimesFM 2.x dropped 1.0's frequency token; Moirai picks a patch size per frequency ([Moirai](https://arxiv.org/html/2402.02592)) |

**Rate-agnostic architectures.**

- S4 keeps its accuracy when only its step Δ is rescaled for a new rate:
  - speech commands at half the training rate: 98.32 → 96.30 ([S4](https://arxiv.org/pdf/2111.00396));
  - on ECG, a 100/200/500 Hz train × test matrix gives macro-AUC 0.941–0.942 in every cell
    ([Mehari & Strodthoff](https://arxiv.org/abs/2308.15291), MIT).

  This is the strongest biosignal evidence for such an architecture. It also shows that rate stops
  mattering once the band fits under Nyquist, which correct resampling already gives us.
- Neural CDEs, ODE-RNNs and mTAN have no PPG waveform results and stream poorly.
- Sampling-frequency-independent convolutions ([Saito et al.](https://arxiv.org/abs/2105.04079)) are shown
  on audio only.
- No PPG or wearable paper reports a cross-device gain from device tokens, FiLM, LoRA or per-device stems.

**Verdict.** Resample to 25 Hz with a proper anti-alias filter, and randomise the resampling chain in
training. Rate-agnostic layers buy nothing at one fixed deployment rate.

### 3.2 Normalisation

- **Scope.** The best quantitative evidence comes from EEG under subject shift. Per-subject z-scoring
  raised leave-one-subject-out accuracy on SEED from 52.96 % to 74.71 %. Domain adaptation (DANN, ADDA) added
  only about 6 points on top ([2210.01081](https://arxiv.org/html/2210.01081v3)). Normalising at the right
  scope beats adversarial alignment.
- **RevIN.** No ECG or PPG use found. Our per-window robust scaling is already its forward half.
- **Test-time adaptation.** Re-estimating BN statistics or TENT gains 2–7 points in cross-person wearable
  activity recognition (74.71 → 76.53 BN, 76.87 TENT; [OFTTA](https://arxiv.org/html/2310.18562v2)) and in
  EEG (57.38 → 64.32; [2311.18520](https://arxiv.org/html/2311.18520v2)). Small buffers make the
  statistics unreliable. Nothing has been shown for PPG regression, and the KronLSTM has no BN.
- **Amplitude is a site cue.**
  - Finger PPG has a larger amplitude and a smaller crest time and reflection index than every other
    site.
  - The share of analysable beats falls from 95 % at the finger to 86 % at the volar wrist and 67 % at
    the dorsal wrist ([Hartmann et al. 2019](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2019.00198/full)).
  - Removing amplitude is right for transfer. Shape features move with site too, so they need wrist data.
- **Polarity.**
  - A morphology rule (steep systolic upstroke, gradual diastolic decay, measured by mean absolute first
    derivative in ~0.3 s windows, plus a peak and trough "sharpness" check) detects inverted PPG with
    98.3 % accuracy on uncorrupted signals.
  - In the same study, inversion was 12.4 % more frequent with red than with green LEDs, and 39.6 % more
    frequent at maximum than at minimum contact pressure
    ([Jerve et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11511109/)).
  - The skewness SQI ([Elgendi 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5597264)) is a weaker
    tie-breaker: noisy upright beats are also negatively skewed.

### 3.3 Augmentation and domain randomisation

- **No paper randomises filters or resampling chains for PPG self-supervised learning.** This is a gap,
  and cheap for us to fill.
- **What foundation models use:**
  - Apple: cut-out, magnitude warp, Gaussian noise, channel permutation, time warp.
  - PaPaGei-S: crop and noise only, avoiding transforms that change morphology. It beat PaPaGei-P (which
    adds negation, flips and scaling) on average.
  - Pulse-PPG: no standard augmentations, because they "can distort PPG semantics".
  - SiamQuality: real quality pairs from the same patient and time instead of synthetic augmentation.
- **Strength.** Moderate strength works best. For ECG contrastive learning, Gaussian noise at σ 0.1–0.2 was
  best; weaker and stronger both hurt ([Soltanieh et al.](https://arxiv.org/abs/2206.07656)).
- **Recurrent models.** On 128 UCR datasets, time warping, permutation and flips **hurt** an LSTM (time warp
  −2.37 points, flip −3.82); window warping helped ([Iwana & Uchida](https://ar5iv.labs.arxiv.org/html/2007.15951)).
  Our encoder is recurrent.
- **Resampling as heart-rate augmentation.** It helped PPG-DaLiA heart rate (MAE 3.68 → 3.54 on top of
  self-supervised learning; [EnhancePPG](https://arxiv.org/html/2412.17860v1)). Downsampling augmentation
  plus a sampling-rate-adversarial feature recovered activity-recognition accuracy under mixed rates
  ([Hasegawa](https://arxiv.org/abs/2101.00812)).
- **Contact pressure** reshapes the wrist pulse. CP-PPG corrects it generatively
  ([2504.02735](https://arxiv.org/abs/2504.02735); CC BY-NC), and WF-PPG records it under control (section 3.6).
- **Finger↔wrist translation.**
  - A cascaded CycleGAN maps wrist to finger PPG (TTR-GAN [snippet]).
  - Chest-to-finger style transfer raised correlation with finger PPG from 0.63–0.78 to 0.89–0.90
    ([Pagotto et al.](https://arxiv.org/html/2503.13496v1)).
  - Both target heart-rate quality. Neither shows that age-related shape survives translation.

### 3.4 Self-supervised positives

- **Participant positives.** These are Apple's: two segments of the same participant, typically from
  different days.
  - Age MAE 3.19 y, against 6.60 y with positives from the same segment only.
  - Sex AUC 0.993 against 0.870.
  - [Apple](https://arxiv.org/html/2312.05409v2). This is the largest documented effect of any choice
    in this survey. It is in-domain (one device family), and it depends on positives that span sessions.
- **ECG–PPG alignment.**
  - AnyPPG aligns PPG to a frozen ECG encoder (CLIP-style) and reports +12.8 % on regression and +9.1 % on
    classification over the next-best model.
  - It is the only open encoder with phone-camera results; there is no ablation without the ECG.
  - Geenjaar et al. pick PPG positives with ECG and respiration and improve 14 of 15 tasks with three
    times fewer subjects ([2606.07365](https://arxiv.org/abs/2606.07365)).
- **Field against clinical pretraining.** Pulse-PPG's architecture pretrained on field wrist data beat the
  same architecture pretrained on MIMIC-III finger PPG on 10 of 11 tasks. Against PaPaGei it had F1
  0.587 vs 0.499 and AUROC 0.748 vs 0.684 ([Pulse-PPG](https://arxiv.org/html/2502.01108v2)).
- **Cross-site positives** (finger and wrist recordings of the same heartbeats used as positives) are
  **unpublished**. Paired data for them exists (section 3.6).

### 3.5 Domain adaptation, and the label-shift danger

- **Theory.** When the label distributions of two domains differ, learning an invariant representation
  trades off against error on both domains ([Zhao et al. 2019](https://arxiv.org/abs/1901.09453v2)).
  Invariance is "often a far too strict requirement" ([Johansson et al. 2019](https://export.arxiv.org/abs/1903.03448v2)).
  Alignment under label shift has to be class-conditional and reweighted
  ([Tachet des Combes et al. 2020](https://arxiv.org/abs/2003.04475)).
- **A worked example of the risk.** ComBat on pathology-model embeddings removed the site signal (AUROC
  0.955 → 0.506) and kept the target (0.667 → 0.669). It also erased race, which was confounded with site
  (0.835 → 0.471) ([PMC11470259](https://pmc.ncbi.nlm.nih.gov/articles/PMC11470259/)).
- **Unbalanced designs.** Batch correction in an unbalanced group × batch design "may systematically
  induce incorrect group differences" ([Nygaard et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4679072/)).
- **For us.**
  - Re-centring per device, DANN, or statistics re-estimated over all of GalaxyPPG would remove the age
    difference between VitalDB and GalaxyPPG along with the device difference.
  - It could even look like a fix: the young cohort's predicted ages would move toward the source mean
    or the target-batch mean, for the wrong reason.
  - Safe variants:
    - estimate offsets only on age-overlapping subsets (VitalDB under 30 against GalaxyPPG);
    - ComBat with age as a protected covariate, only where ages overlap;
    - prefer invariance that augmentation imposes, because it is label-independent by construction.
- **PPG domain adaptation results.** No DANN or CORAL result exists for PPG age or for finger → wrist. The
  only one found is a heart-rate study ([Illinois, snippet](https://experts.illinois.edu/en/publications/unsupervised-adversarial-domain-adaptation-in-wearable-physiologi/)).

### 3.6 Datasets with paired, multi-site or wide-age wrist PPG

| dataset | sites, devices, wavelengths | rate | people, ages | labels | licence | access | use for us |
|---|---|---|---|---|---|---|---|
| **Aurora-BP** (Microsoft) | wrist tonometer + **green** MAX30101 on the palm side of the arm + ECG | 250 Hz → 500 Hz | about 1,220 folders; adults (21–85 per a secondary source) | cuff and ambulatory BP, hypertension, diabetes, CAD, heart failure, stroke, medications | Microsoft DUA: research only, no product use, no redistribution, **"no joining the data with other data sources"** | accept the DUA on [Zenodo 19099166](https://zenodo.org/doi/10.5281/zenodo.19099166); 6.7 GB zipped; sample and docs on [GitHub](https://github.com/microsoft/aurorabp-sample-data) | the only wide-age green wrist-type PPG with disease labels. The joining clause probably rules out pooled training; evaluating a frozen model needs a legal read |
| **Multi-site PPG** ([2605.17859](https://arxiv.org/html/2605.17859)) | earring, ring, watch, necklace; green + IR | 100 Hz | 20, aged 20–51 | Polar H10 ECG | CC BY-NC 4.0 | ungated, [HF snowballlab/Multisite-PPG](https://huggingface.co/datasets/snowballlab/Multisite-PPG), about 17–19 GB | 191 h with all four sites at once: cross-site positives and an ECG anchor |
| **WildPPG** (NeurIPS 2024) | forehead, sternum, wrist, ankle; red, green, IR | 128 Hz | 16 | lead-I ECG, temperature, altitude | not stated | open [polybox link](https://polybox.ethz.ch/index.php/s/NWTuyNojU7aya1y) via the [project page](https://siplab.org/projects/WildPPG) | about 216 h in the wild across sites and wavelengths |
| **WF-PPG** (Sci Data 2025) | **wrist + fingertip** at once, green/red/IR, measured contact pressure | 128 Hz | 27, aged 20–32 | ECG, SpO2, cuff BP | article CC BY-NC-ND; data licence to check | figshare 10.6084/m9.figshare.27011998 ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11790827/)) | the cleanest finger↔wrist pairing; contact pressure as a nuisance |
| **Mol et al. 2020**, PTT in younger and older adults | **wrist + finger PPG**, ECG, continuous BP | not stated | 86: 55 under 65, **31 over 70** | BP, cold pressor and active stand | to check | [Radboud Data Repository](https://data.ru.nl/collections/di/dcn/DSC_626810_0007_173), probably a DUA | the only paired wrist+finger set with older people |
| **HEART-Watch** | Google Pixel Watch (2 PPG channels) + chest ECG + cuff BP | not stated | 40, aged 19–75 | BP; sitting, standing, walking | data access agreement | study plan to the PI ([paper](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001647)) | wide-age consumer watch |
| **UTSA-PPG** ([2505.18165](https://arxiv.org/html/2505.18165)) | fingertip (IR, red) + wristband (green); Fitbit and O2Ring HR | 100 Hz | 12 | ECG Holter | repo MIT | Drive links from [GitHub](https://github.com/utsanpb2-234/UTSA-PPG) | finger↔wrist green pairs |
| **MAUS** | fingertip (256 Hz) + PixArt wrist watch (green, 100 Hz) + ECG | 256 / 100 Hz | 22, mean 23 | workload | IEEE DataPort terms | free IEEE login ([DataPort](https://ieee-dataport.org/open-access/maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor)) | finger↔wrist green pairs |
| **PhysioNet pulse-transit-time-ppg** | two finger sites, green 537 / red 660 / IR 880 nm | 500 Hz | 22, aged 20–53 | ECG, BP, SpO2 | ODbL | open, 2.9 GB ([PhysioNet](https://physionet.org/content/pulse-transit-time-ppg/)) | green against red/IR on the same beats: the wavelength part of finger↔wrist |
| **PWDB** (simulated) | PPG at radial (wrist) and finger, among others | 500 Hz | 4,374 virtual subjects, aged 25–75 | full haemodynamics, PWV | PDDL | [Zenodo 3275625](https://zenodo.org/record/3275625) | age-controlled site differences; volume-based, not optical (no green light, no capillary bed) |

Already in the earlier survey and still relevant:

- **MESA sleep** (NSRR DUA): finger PPG during polysomnography, about 2,000 older adults, with cohort
  outcomes.
- **PulseDB**: its MIMIC-III part (ODbL) carries ages. It is a much larger external hospital set than BIDMC.
- **GalaxyPPG** (CC BY 4.0): Watch5 green and E4 BVP on the same seconds, plus Polar H10 ECG.
- **The Oura vascular-age study**: its fingertip waveforms are public on
  [OSF](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001329); the ring
  waveforms are not.

**No open dataset has wrist PPG over ages 20–80 with an arterial-stiffness reference.**

### 3.7 What has been shown to transfer to consumer watch PPG at about 25 Hz

| transfers | evidence |
|---|---|
| HR at rest | GalaxyPPG's own validation: 3.80 % HR and 4.36 % beat-interval error at baseline, 92 % of ECG beats matched ([GalaxyPPG paper](https://ic.kaist.ac.kr/publications/papers/park2025galaxyppg.pdf)) |
| HRV at rest, with interpolation | our [25 Hz HRV benchmark](hrv-25hz-benchmark.md): about 3 ms RMSSD error |
| AF with a quality gate | DeepBeat at 32 Hz moved to another wrist device: episode sensitivity 0.97 on 15 free-living people ([DeepBeat](https://arxiv.org/pdf/2001.00155)) |
| beat-level identity, hence some shape | wrist authentication at 25 Hz: EER 2.76 %, dropping sharply at 20 Hz ([2508.13690](https://arxiv.org/abs/2508.13690)) |
| average-beat shape (crest time, width, b/a) | our `benchmarks/morph25`: rebuilt from 25 Hz within 0.01–0.04 of the spread between people |

| does not transfer, or is unproven | evidence |
|---|---|
| HR and HRV during motion | GalaxyPPG: watch 95 bpm against 154 bpm on ECG while running; SDNN 136 ms against 41 ms |
| BP | AnyPPG on phone PPG: SBP R² 0.06 ([AnyPPG](https://arxiv.org/html/2511.01747v2)); cross-dataset BP error grows with label shift ([Moulaeifard et al.](https://arxiv.org/html/2502.19167)) |
| age across cohorts, even finger to finger | AI-PPG Age: MIMIC-III MAE 11.81 y, r 0.41 ([2502.12990](https://arxiv.org/html/2502.12990v4)); collapses to 38–67 y on VitalDB ([Brag](https://arxiv.org/html/2603.14030v1)) |
| age works only in-domain | Apple Watch 3.19 y (own data); Oura ring 7.25 y against 6.28 y for fingertip on the same people; OPPO heart age 5.9 y |
| PPG-only arterial stiffness | the Huawei watch result (cfPWV CCC 0.90) needs ECG through a finger on the watch ([Frontiers 2022](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.893557/pdf)) |
| finger beat-timing features at the wrist | wrist pulse transit time was "poorly reliable" where finger PTT worked (Mol et al.) |

**No published study tests a finger- or clinic-trained age or stiffness model on 25 Hz Galaxy green PPG.**
Ours would be the first, so the validation has to be designed, not borrowed.

### 3.8 The WEAR / wearable foundation model page

The parent session read [wearable-foundation-model.web.app](https://wearable-foundation-model.web.app/)
(Santhakumar Ramamoorthy, 2026). I did not read it myself.

**What it describes:**
- Pulse, HRV, movement and skin signals on one shared timeline, as a grid. A quality gate masks
  unreliable cells with a reason: motion, poor contact, not worn, weak pulse.
- A 9.5 M-parameter transformer trained SensorLM/CoCa-style: signal–text matching plus captions computed
  from each window's own measurements. It also has a quantile forecast head and follows people against
  their own baseline.
- 3.0 M hours from 12 public datasets, split by person.

**Results:** on up to 57 unseen people,
- HR error 8.5 bpm, against 8.0 for hand-made features;
- HRV error 15.7 ms, against 20.7;
- sleep-stage error 31.9 %, against 45.6 %.

**Missing:** no age, cardiometabolic or cross-device results, no code and no weights.

**Take from it:**
- **Quality masks as an input, not as a reason to drop windows.** `kron_ppg` already keeps a per-patch
  valid mask. Giving the mask a reason code (gap, motion, clipping, low periodicity) and letting the
  encoder see it is cheap. It also matches how a watch stream really looks overnight.
- **Many datasets, split by person**: the same lesson as Pulse-PPG.
- **Personal baselines**: the project's rule already.

**Do not take:**
- **Aggregate channels** (minute HR, HRV) as the representation.
  - In r0 the HR-only baseline added nothing beyond age for CKD, BMI, ASA ≥ 3 or hypertension. The
    KronLSTM's waveform encoder did (CKD +0.156 ΔAUC, BMI +0.19 ΔR²).
  - A shared timeline is a fine **second** stream next to the waveform, never a replacement for it.
  - WFM's own heart-rate error is no better than hand-made features.

**Its raw-PPG sources worth checking for our pretraining pool:**
- PPG-DaLiA and WESAD (E4 BVP);
- the MIMIC-III matched subset and MIMIC-IV WDB (finger, ICU);
- DREAMT (E4, restricted);
- BIG IDEAs (E4 BVP with continuous glucose; to check);
- the Jarchi–Casson wrist exercise set (PhysioNet `wrist`, 8 people, open).

LifeSnaps, PMData, GLOBEM and Walch sleep hold aggregates or HR, not raw PPG. CAPTURE-24, NHANES and PAMAP2
are accelerometry.

## 4. Ranked recommendations

Ranked by expected gain for the watch × strength of evidence ÷ cost. "Cost" is on our GPU cluster:

- run 1's pretraining (20,000 steps) took 28 min on one A100, so one 40,000-step run is about 1 GPU-hour;
- a probe pass over all targets is minutes of CPU.

Expected gains are my judgement from the cited evidence, not measured. The tests decide.

The tests use one shared battery (B1–B7, section 6), so every change is scored the same way.

### R1. Fix the yardstick before the model

- **What.**
  - Score wrist data on rest segments only.
  - Compare GalaxyPPG with VitalDB people under 30, not with the whole test set.
  - Report rank metrics within a domain (Spearman of predicted against true age) separately from
    calibration (mean offset).
  - Add the format stress test (B1) and the paired-device check (B3).
  - Keep the r0 rule: a change is kept only if the beyond-age deltas on VitalDB (CKD, BMI, ASA ≥ 3,
    hypertension) do not fall outside their intervals.
- **Evidence.**
  - The HR-only baseline's 58.6 y on GalaxyPPG.
  - Regression to the mean at ages 20–40 even in-domain (Brag: 15.69 y MAE).
  - BP failures driven by label shift (Moulaeifard et al.).
- **Expected gain.** No model gain. It stops us from "fixing" a format problem that is really state or age
  shift, and the reverse.
- **Cost.** CPU minutes. The first part (E0a–E0c) has been handed to the fm_baselines session.
- **Test.** The output is the decomposition itself: how much of GalaxyPPG's offset is device (Watch5
  against E4 on the same seconds), state (rest against all activities) and site plus anaesthesia
  (GalaxyPPG at rest against VitalDB under 30).

### R2. A canonical, deployment-identical front end

- **What.** One function, used in training, on the server and in the ONNX path:
  - Put `hr_raw` samples on a uniform 25 Hz grid from their timestamps, with the `foot_template`
    lost-sample correction.
  - Polyphase anti-alias resampling from any source rate.
  - A **causal** 0.5–8 Hz band-pass in training as well as on the device.
  - **Polarity from the data.** Use the upstroke-shorter-than-decay rule on the average beat, with
    skewness as the tie-breaker (Jerve et al.; Elgendi). Remove the per-source `-x`.
  - **Gain-step handling.** Detect steps in the raw level, for example from automatic gain control or
    LED current changes. Then split the window or mark the step in the mask.
  - Robust scaling per window, as now.
- **Evidence.**
  - Our own train/serve mismatch.
  - Polarity detection at 98.3 %, with inversion depending on wavelength and contact pressure.
  - MMR and every foundation model resample to one rate with a proper filter.
- **Expected gain.**
  - Removes the format sensitivity that E0a measures.
  - Small effect on site shift.
  - The causal-filter fix may move shape features enough to change the in-domain deltas, so re-measure
    them.
- **Cost.**
  - One re-prepare of the 25 Hz pools (CPU, hours).
  - One pretraining run (about 1 GPU-h) to make the new baseline `canon`.
  - Phone and watch code: re-use the same function.
- **Test.**
  - B1 sensitivity for `causal`, `invert`, `drop` and `gain-step` falls below 1 y mean |Δ age| and a ratio
    of 0.1.
  - The VitalDB in-domain beyond-age deltas stay within their intervals.

### R3. Positives that span sessions and devices

- **What.** Replace same-recording positives with three kinds:
  - **different sessions of the same person**: the 216 VitalDB people with repeat surgeries, GalaxyPPG's
    separate protocol blocks, and our own watch on different nights;
  - **different devices on the same seconds**: GalaxyPPG Watch5 against E4, and later the paired
    finger↔wrist sets of R6;
  - **"virtual devices"**: the same window passed through two independently drawn format chains from R4.
- **Evidence.**
  - Apple: participant positives give age MAE 3.19 y against 6.60 y for segment positives, the largest
    effect in the survey.
  - SiamQuality builds positives from real quality pairs.
  - Cross-device positives are unpublished; this would be new.
- **Expected gain.**
  - Removes the session fingerprinting described in section 2.2.
  - Should raise the paired-device agreement (B3) and BIDMC and PulseDB rank correlation more than any
    single augmentation.
  - 216 real cross-session people is few, so the virtual-device positives carry most of the load at first.
- **Cost.** Sampler changes in `kron_ppg.train` (`SubjectPairs` keyed by session and device). One or two
  pretraining runs, about 2 GPU-h.
- **Test.**
  - B3 ICC between Watch5 and E4 rises.
  - B2 external rank correlation rises.
  - Ablation: virtual-device positives only, real cross-session only, both.

### R4. Format domain randomisation at moderate strength

- **What.** On-the-fly augmentation in the data loader, drawn per window, before robust scaling:
  - random causal IIR low-pass (3–12 Hz cut-off, order 1–4) and high-pass (0.05–0.7 Hz);
  - a random resampling chain through 20–128 Hz, with ±1–3 % rate jitter;
  - hidden sample drops at 1 per 8–15 s;
  - gain steps of 0.5–2×;
  - Gaussian noise at σ 0.1–0.2 of the IQR;
  - baseline wander;
  - an extra low-pass "site smoothing" that weakens the dicrotic notch, as at the wrist;
  - polarity flips at low probability, as a backstop to R2.
  - **No time warp, permutation or time flip**: they hurt recurrent models.
- **Evidence.**
  - Indirect but consistent: the moderate-strength results (Soltanieh), the recurrent-model results
    (Iwana & Uchida), downsampling augmentation (Hasegawa, EnhancePPG), and negation in PaPaGei-P and
    Geenjaar.
  - The filter and resampling-chain randomisation itself is untested in PPG.
- **Expected gain.**
  - Large on B1, moderate on B3, small on true site shift.
  - It is label-independent, so it is safe under the age confounding of section 3.5.
- **Cost.** CPU in the loader (scipy `sosfilt` and `resample_poly` per window). The same pretraining run as
  R3.
- **Test.**
  - B1 under perturbations held out from training (filter types and rates not drawn in training).
  - B3, and B2 on BIDMC and PulseDB.

### R5. Unlabelled wrist and field PPG in pretraining

- **What.** Add to the `next+subject` pool, all brought through R2's front end at 25 Hz:
  - GalaxyPPG (already in);
  - Multi-site PPG (191 h, four sites at once);
  - WildPPG;
  - PPG-DaLiA;
  - WESAD;
  - UTSA-PPG;
  - PhysioNet `wrist`;
  - our own watch nights.

  Then, if the DUA is granted, add MESA finger sleep PPG: awake and asleep, not anaesthetised, older people.
- **Evidence.**
  - Pulse-PPG: field pretraining beat clinical pretraining on 10 of 11 tasks.
  - WEAR's many-dataset, person-split pool.
  - Apple's and Samsung's in-domain wrist results show wrist PPG carries age and cardiometabolic signal
    once the encoder has seen wrist data.
- **Expected gain.**
  - The largest single step on true site shift (GalaxyPPG at rest, the paired sets).
  - Moderate on state shift once MESA is in.
- **Cost.**
  - Downloads of about 40–60 GB to the cluster's storage, never the laptop (DaLiA about 2.7 GB,
    Multi-site about 18 GB, WildPPG to check).
  - Readers for each set (a day of work).
  - One pretraining run with a longer schedule, about 2–3 GPU-h.
- **Test.**
  - B4 (GalaxyPPG at rest against VitalDB under 30) closes.
  - B3 rises.
  - The VitalDB deltas hold.
  - Watch the mixing weights: VitalDB must not be drowned. Cap windows per person, as now.

### R6. An ECG anchor, and cross-site pairs

- **What.** A second contrastive term that aligns the pooled PPG embedding with the ECG of the same
  seconds. ECG is the same whatever the PPG sensor, so it pulls the PPG embedding toward content that does
  not depend on the device.
  - ECG sources: VitalDB ECG II (already listed in `kron_ppg.sources`), GalaxyPPG Polar H10, Multi-site
    PPG and WF-PPG Polar H10, WildPPG lead I, our watch ECG spot checks.
  - Use a frozen ECG encoder with a permissive licence (ECGFounder, MIT) as the target, as AnyPPG did.
  - Add finger↔wrist positives from WF-PPG, UTSA-PPG and MAUS once their licences are confirmed.
- **Evidence.**
  - AnyPPG: +12.8 % regression and +9.1 % classification, and the only open encoder with phone-camera
    evidence.
  - Geenjaar et al.: 14 of 15 tasks with three times fewer subjects.
  - Neither ablates device transfer.
- **Expected gain.**
  - Moderate on B3 and B4.
  - **Risk:** it pulls the embedding toward electrical and timing content, while vascular aging lives in
    pulse shape.
  - Keep the next-patch loss dominant, and check that the shape-driven deltas (CKD, BMI) survive.
- **Cost.**
  - ECG windows prepared alongside PPG (CPU).
  - ECG embeddings computed once and cached (GPU, under 1 h).
  - One pretraining run, about 1–2 GPU-h.
- **Test.**
  - B3 and B4 improve.
  - The VitalDB CKD and BMI deltas hold.
  - Ablation with the ECG term's weight at 0, 0.05 and 0.2.

### R7. A beat-synchronous branch and baseline

- **What.** Averaged beats, time-normalised foot to foot, plus their first and second derivatives:
  - fed to a small head or to ridge on morphology indices (crest-time fraction, width, b/a, d/a, reflection
    index);
  - as a baseline beside the encoder, and later as a second input stream.
- **Evidence.**
  - A VitalDB-trained model on the ensemble-averaged beat plus its second derivative kept its bias and grew
    its error SD by only about 25 % on a Samsung phone camera (reflectance)
    ([Aguet et al., CinC 2022](https://cinc.org/archives/2022/pdf/CinC2022-262.pdf)).
  - Our morph25 shows these indices survive 25 Hz.
  - Against that: AI-PPG Age, a pure beat model, still collapses across cohorts. A beat representation
    removes rate, phase and lost-sample problems, not site physiology.
- **Expected gain.**
  - It shows how much of the encoder's in-domain advantage is beat shape that a cheap, interpretable model
    could carry across devices.
  - It is the honest comparison for every transfer number.
- **Cost.** CPU only (watch_session's beat code already exists). No GPU.
- **Test.** Every battery metric, side by side with the encoder.

### R8. Personal baselines and per-person scaling for the deployed reading

- **What.**
  - On the watch, report the PPG age, or any embedding-based score, as a deviation from the person's own
    rolling baseline over the first weeks.
  - Optionally add a causal per-person running scale (EMA of median and IQR over minutes) as a second
    normalisation channel next to the per-window one.
- **Evidence.**
  - EEG: per-subject normalisation 52.96 → 74.71 %.
  - Pulse-PPG uses per-person z-scoring.
  - The project's rule that a PPG age is a within-person trend.
  - Constant device and site offsets cancel within a person (section 2.3).
- **Expected gain.**
  - Removes constant device offsets from what the user sees.
  - Says nothing about between-person validity, which is why R9 and the outcome cohorts remain.
- **Cost.** App and server logic; no training for the baseline. One pretraining ablation for the
  per-person scale.
- **Test.**
  - B6: test-retest ICC across nights and spot checks on our own watches.
  - The VitalDB repeat-surgery people: same person, different day.
  - Response to known within-person changes (B6b).

### R9. Label-aware alignment only, mainly as a diagnostic

- **What.** If a device offset must be removed between people, estimate it on an age-overlapping slice:
  - VitalDB under 30 against GalaxyPPG;
  - later, age-matched pairs from our cohort.

  Alternatively, use ComBat with age as a protected covariate. Never use plain per-device re-centring,
  DANN or batch statistics over a whole cohort.
- **Evidence.** Section 3.5.
- **Expected gain.**
  - Small. Mainly it tells us how big the device offset is once age is held fixed.
  - It shows the danger concretely: re-centring GalaxyPPG on its own mean will "fix" its predicted age
    without any knowledge gain.
- **Cost.** CPU minutes on saved embeddings.
- **Test.** Report B4 with and without the aligned offset. The offset must be estimated on half of the
  overlap people and tested on the other half.

### R10. Later, and only if B3 shows a residual device effect: a tiny learned per-device equaliser

- **What.** A per-device K-tap FIR filter (K about 9–15 at 25 Hz), initialised at identity, in front of a
  frozen encoder.
  - Train it on paired recordings so that device B's embedding matches device A's on the same seconds.
  - Interpretable as a transfer function.
  - A few dozen parameters, so a few paired minutes per device fit it.
- **Evidence.** None published for PPG (section 3.1). The finger↔wrist translation GANs show such mappings
  can be learned, at far higher capacity.
- **Cost.** Minutes of GPU once paired data exists.
- **Test.** B3 on held-out paired people.

### Not recommended now

| idea | why not |
|---|---|
| Rate-agnostic architectures (S4 or S5 with Δ rescaling, neural CDEs, mTAN, sampling-frequency-independent convolutions) | one fixed deployment rate. S4's own ECG result shows correct resampling gives the same. Swapping out the KronLSTM is expensive |
| Device tokens or FiLM, per-device stems, LoRA per device | no published cross-device gain; they need labelled or paired data per device; they invite the model to memorise device = age |
| Plain per-device re-centring, DANN, CORAL, MMD | label shift between our domains (section 3.5) |
| AdaBN, TENT | the KronLSTM has no BN; TTA buffers on a single watch are small; no PPG regression evidence |
| Wrist → finger GAN translation, then the finger model | targets heart-rate quality; no evidence that age-related shape survives |
| RevIN beyond per-window scaling | no biosignal evidence; our scaling already is its forward half |
| Aggregate-feature representations as the main input (WEAR-style) | lose the pulse shape that carried CKD, BMI and ASA in r0 |

## 5. The draft five-phase plan, critiqued

The draft:
1. stress tests;
2. a canonical input;
3. domain randomisation with cross-device positives;
4. diverse unlabelled pretraining, including our own watch nights;
5. per-device re-centring and personal baselines, plus a labelled wrist cohort and MESA.

| phase | keep | change |
|---|---|---|
| 1. stress tests | yes, now running as E0a–E0c | add the **state** and **age-matching** axes (R1). Synthetic perturbations alone measure format invariance, not transfer. Add the paired Watch5 vs E4 check, the only real cross-device test we can run tonight |
| 2. canonical input | yes | make it **deployment-identical**: a causal filter in training, polarity from the data, gain steps. It is the cheapest fix with a known mismatch behind it |
| 3. randomisation + cross-device positives | yes, the core | the positives need **sessions**, not just devices: Apple's gain comes from participant positives across time, and ours are same-recording. Add virtual-device positives, because real paired data is small. Exclude time warp and permutation |
| 4. diverse unlabelled pretraining | yes | move it **earlier**, merged with phase 3 into one pretraining series, since field pretraining has the strongest transfer evidence (Pulse-PPG). Add the ECG anchor, which the same datasets support. Bring **MESA forward as pretraining and state-shift data**, not only as a phase-5 label set |
| 5. per-device re-centring | **no**, as stated | replace with label-aware alignment (R9) as a diagnostic. Keep personal baselines (R8). Split "labelled wrist cohort and MESA" into the outcome bridge and the watch validation (section 7, E7–E8) |
| missing | | **paired data collection** (watch + finger reference on the same seconds). It is cheap, it is the only way to tie the watch to finger cohorts that have outcomes, and it should start now, not in phase 5 |

## 6. The evaluation battery

Every run reports all of these. Tables go next to the r0 tables in fm_baselines, so earlier runs stay
comparable.

| id | what | data | metric | target |
|---|---|---|---|---|
| B1 | **format sensitivity** | VitalDB test people, perturbations of E0a, plus held-out ones | mean \|Δ predicted age\|; embedding shift ÷ median between-person distance | < 1 y and < 0.1 for deployment-real perturbations |
| B2 | **external hospital** | BIDMC (53), PulseDB MIMIC-III part (several thousand people with age) | Spearman of predicted against true age; MAE; mean offset reported separately | rank correlation up, beyond its bootstrap interval |
| B3 | **same site, two devices, same seconds** | GalaxyPPG Watch5 against E4 at rest; later Multi-site PPG (watch against ring against earring), WF-PPG, UTSA, our own `hr_raw` against SDK | ICC of per-person scores; cosine of per-person mean embeddings | ICC ≥ 0.8 |
| B4 | **wrist at rest against age-matched finger** | GalaxyPPG at rest against VitalDB people under 30 | difference in mean predicted age, with interval | the interval includes 0, or shrinks run on run |
| B5 | **beyond age, in-domain** | VitalDB test: CKD, eGFR, BMI, ASA ≥ 3, hypertension | ΔAUC / ΔR² over age+sex | no loss beyond the r0/r1 intervals: the guard rail |
| B6 | **within-person repeatability** | VitalDB repeat-surgery people; our own watch nights and spot checks | ICC; within-person SD against between-person SD | ICC ≥ 0.7 over nights |
| B6b | **known within-person change** | our own: after hard exercise, after alcohol, after a bad night; Mol et al. cold pressor and active stand if access is granted | direction and size of the change against the night-to-night SD | moves, and in the expected direction |
| B7 | **outcome evidence** | MESA (when granted), VitalDB | gap AUC against age's own AUC; ΔAUC beyond age+sex | the project's real bar |

## 7. Ordered experiment plan, ending in a Galaxy Watch validation

**Decision rule for every step:** keep a change only if B5 holds and at least one of B2, B3 or B4 improves
beyond its interval. Otherwise record it as tried and move on.

| step | what | needs | compute | decides |
|---|---|---|---|---|
| **E0** (tonight, handed over) | E0a format stress test, E0b fair GalaxyPPG check, E0c polarity detector, on r1 (or r0) frozen encoders and probes | nothing new | CPU, under 1 h | how much of the failure is format, device, state and site plus age (R1) |
| **E1** | canonical front end (R2), re-prepare, pretrain `canon` with today's objective | R2 code in kron_ppg's signal step | CPU hours + ~1 GPU-h | the new baseline; whether format alone moves B2–B4 |
| **E2** | `canon` + format randomisation (R4) + virtual-device positives (R3); ablation: augmentation only against augmentation + positives | loader changes | 2 runs, ~2 GPU-h | format invariance (B1, B3) |
| **E3** | + real cross-session positives (VitalDB repeat surgeries, GalaxyPPG blocks, Watch5↔E4 seconds) | sampler keyed by session and device | ~1 GPU-h | session fingerprinting (B3, B6) |
| **E4** | + unlabelled wrist and field pool (R5): Multi-site PPG, WildPPG, DaLiA, WESAD, UTSA, PhysioNet `wrist`, own nights | downloads and readers | ~2–3 GPU-h | site shift (B4, B3) |
| **E5** | + ECG anchor (R6), weights 0 / 0.05 / 0.2 | ECG windows, cached ECGFounder embeddings | ~3 GPU-h | whether ECG helps without costing shape (B5) |
| **E6** | beat-synchronous baseline (R7) on every battery set | CPU | minutes | how much a cheap beat model already transfers |
| **E7** | **outcome and external bridge**: PulseDB-MIMIC ages (B2); MESA if the DUA lands (B7, and a pretraining source); Aurora-BP evaluation of a frozen model **only if** the DUA allows it (no pooling) | access requests now | GPU-h per set | between-person validity where outcomes exist |
| **E8** | **paired recordings and the Galaxy Watch validation** (below) | ethics, participants, a raw finger reference | small | whether the watch reading is the finger reading, and whether it tracks the person |

### E8: Galaxy Watch validation

**Paired recordings, starting now with the team's own watches.**
- Watch on the left wrist (`hr_raw` and the SDK at 25 Hz, plus 100 Hz spot checks).
- A finger reference on the other hand, recorded at the same time, either:
  - the phone fingertip camera at 30 fps; or
  - a finger pulse oximeter that streams its raw pleth waveform (the CMS50E-type device used as the
    reference in the PURE camera-PPG dataset; check that the model exports the waveform before buying).
- A 30 s watch ECG at the start, for beat alignment.
- These feed B3 and R10 with the deployment device itself.

**A small age-spread cohort.**
- **Size.** About 40–60 people from 20 to 80, evenly by decade. n = 29 detects r = 0.5 between PPG age
  and age (80 % power, α = 0.05), and n = 47 detects r = 0.4.
- **Each person.**
  - Two weeks of nights.
  - Ten still, seated 30 s spot checks (the OPPO protocol).
  - One paired finger session.
  - Cuff BP.
  - If a clinic partner can provide it, a carotid–femoral PWV.
- **Ethics.** This needs approval and consent, and is not started by any session without the user.

**Reading.**
1. Within-person repeatability over nights (B6).
2. Watch↔finger agreement per person (B3).
3. Rank correlation of the watch reading with age, and beyond age with BP and PWV.
4. The same model's finger reading on the same people. Where watch and finger agree within a person,
   the finger cohorts' outcome evidence (MESA, UK Biobank, VitalDB) carries over in part. Where they
   disagree, the watch reading is a trend marker only.

**Outcome evidence.** It cannot come from a cohort of 60. The bar for calling the watch reading an aging
clock stays B7 on an outcome cohort, joined to the watch by E8's paired agreement.

## 8. Decisions taken without asking (the user was asleep), and open items

- **No cluster jobs from this session.** The brief said research and writing only. The cheap E0
  experiments were written up in the shared notes and sent to the fm_baselines session to run.
- **E0 uses the existing probes unchanged.** The question is how the current readout moves, not what a
  refitted probe could do.
- **The draft's phase 5 re-centring is replaced** by label-aware alignment, because of the age confounding.
  This is the one place where the strategy reverses the draft rather than refining it.
- **MESA moves earlier**, as pretraining and state-shift data. The NSRR DUA request should go in now; only
  the user can sign it.
- **Aurora-BP:** its DUA forbids "joining the data with other data sources". Pooled training is treated as
  ruled out. Evaluating a frozen model needs the user's or a lawyer's reading.
- **Licences to confirm before use:** WF-PPG data, WildPPG, Mol et al. (Radboud), MAUS (IEEE DataPort terms),
  HEART-Watch (data access agreement). Multi-site PPG is CC BY-NC 4.0: fine for research, not for a
  product.
- **Unverified here:** PpgAge's input details; the TTR-GAN numbers; whether BIG IDEAs has raw BVP; WildPPG's
  licence; WF-PPG's data licence; and the WEAR page, which was read by the parent session, not this one.

## Sources

Model papers and code:
- [Apple PPG/ECG FM](https://arxiv.org/html/2312.05409v2)
- [HiMAE](https://arxiv.org/html/2510.25785)
- [MMR](https://arxiv.org/html/2601.12215)
- [xMAE](https://arxiv.org/html/2605.00973)
- [Pulse-PPG](https://arxiv.org/html/2502.01108v2)
- [PaPaGei](https://arxiv.org/html/2410.20542v2)
- [AnyPPG](https://arxiv.org/html/2511.01747v2)
- [GPT-PPG](https://arxiv.org/html/2503.08015)
- [Geenjaar et al. 2026](https://arxiv.org/html/2606.07365v1)
- [SiamQuality](https://arxiv.org/abs/2404.17667)
- [NormWear code](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)
- [Moirai](https://arxiv.org/html/2402.02592)
- [Generalist vs specialist PPG benchmark](https://arxiv.org/html/2510.14254)
- [AI-PPG Age](https://arxiv.org/html/2502.12990v4)
- [Brag age benchmark](https://arxiv.org/html/2603.14030v1)

Architectures:
- [S4](https://arxiv.org/pdf/2111.00396)
- [S4 on ECG across rates](https://arxiv.org/abs/2308.15291)
- [Sampling-frequency-independent convolution](https://arxiv.org/abs/2105.04079)
- [Rate-unidentifiable features in activity recognition](https://arxiv.org/abs/2101.00812)

Normalisation, augmentation, adaptation:
- [EEG normalisation scope](https://arxiv.org/html/2210.01081v3)
- [OFTTA](https://arxiv.org/html/2310.18562v2)
- [EEG test-time adaptation](https://arxiv.org/html/2311.18520v2)
- [Hartmann et al. 2019](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2019.00198/full)
- [Jerve et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11511109/)
- [Elgendi 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5597264)
- [Soltanieh et al.](https://arxiv.org/abs/2206.07656)
- [Iwana & Uchida](https://ar5iv.labs.arxiv.org/html/2007.15951)
- [EnhancePPG](https://arxiv.org/html/2412.17860v1)
- [CP-PPG](https://arxiv.org/abs/2504.02735)
- [Pagotto et al.](https://arxiv.org/html/2503.13496v1)
- [Aguet et al. 2022](https://cinc.org/archives/2022/pdf/CinC2022-262.pdf)
- [Moulaeifard, Charlton & Strodthoff 2025](https://arxiv.org/html/2502.19167)
- [ComBat on embeddings](https://pmc.ncbi.nlm.nih.gov/articles/PMC11470259/)
- [Zhao et al. 2019](https://arxiv.org/abs/1901.09453v2)
- [Johansson et al. 2019](https://export.arxiv.org/abs/1903.03448v2)
- [Tachet des Combes et al. 2020](https://arxiv.org/abs/2003.04475)
- [Nygaard et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4679072/)

Datasets and transfer evidence:
- Aurora-BP: [Zenodo](https://zenodo.org/doi/10.5281/zenodo.19099166), [docs](https://github.com/microsoft/aurorabp-sample-data)
- [Multi-site PPG](https://arxiv.org/html/2605.17859)
- [WildPPG](https://siplab.org/projects/WildPPG)
- [WF-PPG](https://pmc.ncbi.nlm.nih.gov/articles/PMC11790827/)
- [Mol et al.](https://data.ru.nl/collections/di/dcn/DSC_626810_0007_173)
- [HEART-Watch](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001647)
- [UTSA-PPG](https://arxiv.org/html/2505.18165)
- [MAUS](https://ar5iv.labs.arxiv.org/html/2111.02561)
- [PhysioNet PTT-PPG](https://physionet.org/content/pulse-transit-time-ppg/)
- [PhysioNet wrist](https://physionet.org/content/wrist/)
- [PWDB](https://peterhcharlton.github.io/pwdb/pwdb.html)
- [Charlton's dataset list](https://peterhcharlton.github.io/post/ppg_datasets)
- [Oura vascular age](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0001329)
- [GalaxyPPG paper](https://ic.kaist.ac.kr/publications/papers/park2025galaxyppg.pdf)
- [DeepBeat](https://arxiv.org/pdf/2001.00155)
- [25 Hz wrist authentication](https://arxiv.org/abs/2508.13690)
- [Huawei watch PWV](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.893557/pdf)
- [WEAR / wearable foundation model page](https://wearable-foundation-model.web.app/)
