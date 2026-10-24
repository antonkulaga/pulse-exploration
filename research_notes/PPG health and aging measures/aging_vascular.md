# PPG-derived biological and vascular aging measures: evidence and feasibility on Galaxy Watch6 Classic + phone camera

Notes compiled 2026-10-24. Scope: aging and vascular-aging markers derived from PPG (plus ECG+PPG timing), and wearable heart-rate measures linked to mortality. VO2max is excluded (another researcher covers it). Resting HRV from 25 Hz wrist PPG is already shown in our own benchmark (~3 ms RMSSD error), so it is not re-researched. Extends `docs/model-feasibility.md`, which covered Pulse-PPG, PaPaGei, AnyPPG, CP-PPG and the other models.

Evidence-strength shorthand used below:
- **A**: large cohort with hard outcomes (incident disease or mortality).
- **B**: validation against a reference standard such as cfPWV in tens to hundreds of people.
- **C**: vendor claim, internal white paper or marketing only.
- **ext** means external validation; **prosp** means prospective validation.

---

## 1. AI-derived "PPG age" and the age gap: who trained what, how accurate, and does the gap predict outcomes?

### Takeaway
Large deep models predict age from PPG. Three are most relevant.
- **Apple PpgAge** (wrist, n=213,593): MAE 2.4 y in healthy people.
- **AI-PPG Age** (UK Biobank finger, n=212,231): MAE ~7.6 y.
- **OPPO "heart age"** (smartwatch 30 s spot checks, n=581,804): MAE ~5 y.

In each, the age gap predicts incident cardiovascular disease, hypertension or diabetes independently of standard risk factors. The hazard ratios are about 1.4–1.6 per 6-year gap (PpgAge) and about 2.4 for a gap above 9 years (AI-PPG Age). None of these models has been prospectively validated, or externally validated on a different consumer wearable. The only openly downloadable age model (AI-PPG Age) collapses when moved to another population. The usable signal for us is the method, not a drop-in model.

### Cited Findings

**Apple PpgAge (Miller, Futoma, Abbaspourazad, …, Sapiro; Nature Communications 16:9264, 20 Oct 2025; DOI 10.1038/s41467-025-64275-4).** Evidence A (internal), no ext, no prosp.
- **Cohort:** Apple Heart & Movement Study, n=213,593 participants and >149 million participant-days; an observational study — [PubMed record / DOI](https://doi.org/10.1038/s41467-025-64275-4)
- **Input:** green (525 nm) 60-s wrist PPG segments, recorded passively "during low-motion periods throughout the day". Sampled at 64 Hz or 256 Hz with four optical channels. Segments must pass a quality check: beat segmentation covering ≥90% of the segment — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
- **Model:**
  - A self-supervised encoder (the Abbaspourazad et al. foundation model), pretrained on 19,993,427 segments from 172,318 participants, maps each segment to a 256-d embedding.
  - PpgAge itself is a ridge-penalised *linear* model on the mean embedding over a 30-day window, which needs ≥30 segments per window.
  - [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
- **Accuracy:**
  - Healthy test cohort (n=1,373): MAE 2.42 y (men) and 2.45 y (women).
  - General held-out cohort (n=120,235): MAE 3.13 y (men) and 3.26 y (women).
  - [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
  - The abstract headline is MAE 2.43 y (95% CI 2.33–2.53) healthy and 3.18 y general — [Princeton record of abstract](https://collaborate.princeton.edu/en/publications/a-wearable-based-aging-clock-associates-with-disease-and-behavior/)
- **Baseline:** an HR/HRV-only model reached MAE 6.10 y (linear) and 6.14 y (random forest) in the healthy cohort, and its disease associations were weaker (diabetes relative rate 1.25 vs 2.1) — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
- **Age gap:** predicted minus chronological age, made age-independent by regressing on age with a spline and keeping residuals — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
- **Prevalent disease with a gap >6 y** — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/):
  - Diabetes in women aged 35–45: 14.9% vs 6.3% (relative rate 2.38).
  - Heart disease in men aged 35–45: relative rate 3.46.
  - Heart failure in men aged 45–55: 2.97.
- **Incident disease, Cox model per 6-year gap** — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/):
  - ASCVD: HR 1.464 (1.36–1.57), 1,328 events among 89,553 people, C-index 0.758.
  - Hypertension: HR 1.620 (1.51–1.74), 1,449 events among 70,636 people.
  - Adjusted for age, sex, BMI, VO2max, smoking, hypertension, cholesterol and diabetes.
  - Secondary summaries quote "HR 1.40" for adverse cardiac events — [Princeton record](https://collaborate.princeton.edu/en/publications/a-wearable-based-aging-clock-associates-with-disease-and-behavior/). The primary text gives 1.464 for ASCVD, and the difference is unresolved (possibly a different outcome definition).
- **Behaviour and events** — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/):
  - Smoking: dose-response (men 55–65: daily smokers' gap 5.27 y vs never-smokers 1.74 y).
  - Exercise: lowest vs highest quintile 3.85 vs 1.34 y (men 65–75).
  - Sleep: −0.99 y gap per extra hour of deep sleep.
  - Pregnancy: median +3.56 y (n=165).
  - Cardiac events: +1.7 to 2.5 y around CABG, MI and valve replacement.
- **Limitations stated by the authors:**
  - Observational, with self-reported diagnoses.
  - At most about 4 years of follow-up and no mortality data.
  - No external or prospective validation.
  - All authors are Apple employees.
  - Code is released at github.com/apple/ml-ppg-age-analysis with synthetic data only, and no trained weights were stated to be released.
  - [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
- **Precursor, superseded for age:** the same group's ICLR 2024 PPG/ECG foundation model (~141K AHMS participants) got participant-level age regression MAE of 3.38 y from PPG embeddings vs 6.33 y from ECG — [arXiv 2312.05409](https://arxiv.org/abs/2312.05409)

**AI-PPG Age (Nie, Zhao, Tang, Li, Hong; Peking University; Communications Medicine 5:481, 19 Nov 2025; DOI 10.1038/s43856-025-01188-9).** Evidence A (UK Biobank). Its external validation is in an ICU cohort, not a wearable.
- **Input and model:**
  - UK Biobank finger PPG (PulseTrace PCA2, field 4205): the UK Biobank-preprocessed *single averaged pulse waveform of 100 samples*, z-scored.
  - Net1D CNN with squeeze-and-excitation (~390k parameters) and a distribution-aware "Dist Loss".
  - [arXiv v4](https://arxiv.org/html/2502.12990v4)
  - UK Biobank's arterial-stiffness device uses infrared finger sensors and acquires data "within seconds" — [Fung 2019, Sci Rep](https://doi.org/10.1038/s41598-019-45703-0)
- **Accuracy is weak:**
  - UK Biobank hold-out: r=0.49, MAE 7.57 y.
  - UK Biobank clinical-evaluation set: r=0.38, MAE 8.18 y.
  - VitalDB: MAE ~9.6 y.
  - MIMIC: MAE 11.81 y.
  - No bias correction of the gap.
  - [arXiv v4](https://arxiv.org/html/2502.12990v4)
- **Outcomes (published version)** — [Commun Med abstract via PubMed](https://doi.org/10.1038/s43856-025-01188-9):
  - N=212,231.
  - A gap >9 y gives MACCE HR 2.37 (p=8.46×10⁻⁸⁰ per the preprint) plus seven secondary outcomes, all p<0.005.
  - Serial PPG adds predictive value.
  - External MIMIC-III cohort (n=2,343): in-hospital mortality OR 1.02 per year of gap (p=0.01).
- **Effect sizes (preprint v4; per-group HRs adjusted for age, sex, ethnicity and BMI):**
  - Overestimated (>9 y) vs correct, all-cause mortality: HR 2.51 (2.22–2.84).
  - Hypertension: HR 2.88 (2.66–3.11).
  - Diabetes: HR 2.69 (2.42–2.99).
  - Stroke: HR 2.55 (2.11–3.08).
  - MACCE stays at HR 2.24 after adding smoking, hypertension, diabetes, dyslipidaemia and CKD.
  - Participants overestimated at both of two visits (n=27,108 with repeat PPG) had HR 5.53 (2.73–11.23).
  - [arXiv v4](https://arxiv.org/html/2502.12990v4)
- **No wearable, wrist or smartphone validation** — [arXiv v4](https://arxiv.org/html/2502.12990v4)
- **Superseded:** arXiv v1–v4 (titled "AI-derived PPG age") are superseded by the Commun Med version, and numbers may differ slightly between them.
- **Weights:** public at huggingface.co/Ngks03/PPG-VascularAge — [Brag 2026 benchmark, data-availability section](https://arxiv.org/abs/2603.14030)

**ArtAgeGap (Xu, …, Dichgans, Georgakis; LMU Munich; medRxiv 17 Nov 2025; DOI 10.1101/2025.11.15.25340310; not peer reviewed).** Evidence A (UK Biobank), with genetics.
- Multimodal deep model: raw UK Biobank PPG waveforms plus "hemodynamic features" — [medRxiv metadata](https://doi.org/10.1101/2025.11.15.25340310)
- ArtAgeGap correlates with pulse pressure (r=0.60) — [medRxiv metadata](https://doi.org/10.1101/2025.11.15.25340310)
- Associations with comorbidities: diabetes +0.93 y (0.64–1.21) and hyperlipidaemia +0.49 y — [medRxiv metadata](https://doi.org/10.1101/2025.11.15.25340310)
- **Outcomes:** in 96,615 participants with median 13-y follow-up, the gap predicts incident hypertension, MACE, CV mortality and all-cause mortality, with HR 1.01–1.06 per +1 y on top of age — [medRxiv metadata](https://doi.org/10.1101/2025.11.15.25340310)
- **Genetics:**
  - GWAS in 114,098 people found 60 loci (NPR3, CLCN6, SLC24A3, HDAC9 …).
  - Mendelian randomisation puts ArtAgeGap between risk factors (adiposity, BP, T2D) and CHD/stroke.
  - [medRxiv metadata](https://doi.org/10.1101/2025.11.15.25340310)
- I could not open the full text (HTTP 403), so the sampling details and the exact "hemodynamic features" are unknown.

**OPPO smartwatch "heart age" (Xie, …, Nie, …, Hong; arXiv 2608.01620, 3 Aug 2026; preprint).** This is the closest analogue to our on-demand watch setup. Evidence B+ (cfPWV, cross-sectional); no outcomes.
- **Signals:**
  - OPPO smartwatch PPG at 250 Hz, 4 channels (green and red), in user-initiated 30-s recordings.
  - Single-lead smartwatch ECG at 250 Hz, used only for ECG-guided cross-modal (CLIP-style) pretraining; inference is PPG-only.
  - [arXiv HTML](https://arxiv.org/html/2608.01620v1)
- **Scale:** 581,804 participants and 7,452,131 recordings from three OPPO cohorts — [arXiv HTML](https://arxiv.org/html/2608.01620v1)
- **Accuracy (subject-level, PPG-only)** — [arXiv HTML](https://arxiv.org/html/2608.01620v1):
  - Internal: MAE 4.95 y, r=0.734. The PPG-only model beats ECG-only (MAE 5.57 y).
  - External cfPWV cohort (n=560): MAE 5.895 y, r=0.819.
  - External home-BP cohort (n=1,095): MAE 4.344 y, r=0.800.
- **Aggregation:** averaging 1→15 recordings within 3 months lowers MAE from 5.395 to 4.949 y, plateauing after about 10 recordings — [arXiv HTML](https://arxiv.org/html/2608.01620v1)
- **Vascular validity** — [arXiv HTML](https://arxiv.org/html/2608.01620v1):
  - The age-adjusted gap correlates with cfPWV (partial r=0.2627), +0.062 m/s per year of gap (0.040–0.084).
  - Accelerated vs decelerated agers differ by 0.91 m/s cfPWV.
  - Prevalent hypertension OR 1.72 (1.49–1.99) per SD of gap; highest vs lowest quartile OR 4.25 (2.80–6.46).
- **Limitations:**
  - Cross-sectional, with no incident outcomes.
  - Development cohort mostly young, Asian and male.
  - Code and data are proprietary (OPPO approval needed).
  - [arXiv HTML](https://arxiv.org/html/2608.01620v1)

**Open-model benchmark (Brag, arXiv 2603.14030, 14 Mar 2026; single-author preprint).** Evidence: internal CV only.
- **Setup:** 906 VitalDB surgical patients from PulseDB (finger PPG, 125 Hz, 10-s segments); frozen embeddings, ridge regression, 5-fold CV — [arXiv](https://arxiv.org/abs/2603.14030)
- **Results (MAE):**
  - Pulse-PPG: 9.28 y (r=0.645).
  - PaPaGei-S: 9.80 y (r=0.583).
  - HR/HRV features alone: 11.49 y (predicting the mean gives 11.91 y).
  - Pulse-PPG + demographics: 8.22 y (r=0.725).
  - [arXiv](https://arxiv.org/abs/2603.14030)
- **AI-PPG Age zero-shot:** MAE 11.50 y, with "predictions collapse to a narrow 38–67 year range regardless of true age"; a linear probe on it gives 9.72 y — [arXiv](https://arxiv.org/abs/2603.14030)
- MAE falls monotonically with training-set size, and the author attributes the gap to PpgAge to data scale (906 vs 213,593 subjects) — [arXiv](https://arxiv.org/abs/2603.14030)
- **Input specs:**
  - Pulse-PPG expects 50 Hz input and was pretrained on wearable PPG from a 100-day field study (120 participants).
  - PaPaGei-S runs at 125 Hz.
  - AI-PPG Age takes one averaged beat resampled to 100 samples.
  - [arXiv](https://arxiv.org/abs/2603.14030)

**Smartphone camera (older; Dall'Olio et al., Sci Rep 2020; DOI 10.1038/s41598-020-76816-6).** Evidence B− (no reference standard; chronological age as a proxy); no ext.
- **Data:** 3,612 crowd-sourced smartphone fingertip recordings from the Happitech app, about 30 samples/s, about 90 s each, red channel only — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7661535/)
- **Task:** classifying young (18–38) vs old (60–79). This is classification, not age regression — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7661535/)
- **Results (AUC):**
  - Covariates only: 0.742.
  - Covariates + SDPPG "a" wave + RR turning-point ratio: 0.947.
  - 12-layer ResNet: 0.953.
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7661535/)
- One author is a Happitech shareholder — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7661535/)

### Inferences
- **Waveform shape carries most of the age information, not beat timing.**
  - PpgAge's full-waveform model roughly halves the error of an HR/HRV-only model (2.4 vs 6.1 y).
  - In the VitalDB benchmark, HR/HRV alone barely beats predicting the mean.
  - So an aging clock that works only from our beat intervals (Samsung IBI list, 1 Hz HR) will be much weaker than one using the raw 25 Hz or 100 Hz waveform.
- **A gap model is mostly a data-scale problem.**
  - PpgAge's head is a *linear* ridge model on frozen self-supervised embeddings; the power comes from 172K-person pretraining and 30-day averaging.
  - With a handful of users we cannot train an age regressor. Realistic options:
    - (i) Run the public AI-PPG Age weights on beat-averaged waveforms, knowing about the domain shift (finger IR UK Biobank → wrist green).
    - (ii) Use Pulse-PPG / PaPaGei embeddings as a *within-person* trend signal rather than an absolute age.
    - (iii) Calibrate a linear age probe on a public labelled set (PulseDB/VitalDB, MIMIC, UK Biobank if accessible) and accept MAE around 8–10 y.
- **OPPO's design matches our "100 Hz on-demand ≤30 s PPG + single-lead ECG" mode almost exactly**, except that they sample at 250 Hz. It shows that repeated 30-s spot checks (about 10 or more per person over 3 months) reach MAE about 5 y and track cfPWV. A morphology-only open model is not available for this, though.
- **Age gaps are noisy per person.** Even the best model has a 2.4–3.2 y MAE, and gap thresholds of 6–9 y are used for "elevated". Personal use should therefore rely on long windows (PpgAge: 30 days; OPPO: ≥10 recordings) and on change within a person, not on single readings.

### Gaps
- PpgAge trained weights are not public, so the exact per-segment encoder input cannot be reproduced (64 Hz vs 256 Hz is device-dependent).
- ArtAgeGap full text not accessed. Unknown whether its "hemodynamic features" include BP, which would inflate the PP correlation.
- No Samsung, Google/Fitbit or Huawei peer-reviewed "PPG age" model found. OPPO is the only Android-vendor result, and it is a preprint.
- No prospective study shows that lowering a PPG age gap (by an intervention) lowers risk.

---

## 2. Classical vascular-aging indices from the pulse wave (SI, RI, AIx, SDPPG/APG b/a, d/a, aging index, PTT/PAT/PWV)

### Takeaway
- **Best evidence:** classical indices are validated mainly on **fingertip (often transmissive IR) PPG**.
  - Takazawa's aging index (AGI) correlates with age at r=0.42–0.80.
  - Stiffness index (SI) correlates with cfPWV at r≈0.65.
  - SDPPG d/a independently predicts CV mortality (HR ~2.3–2.6 for the upper quartiles).
  - UK Biobank's finger arterial stiffness index (ASI) predicts CVD and mortality, though less than pulse pressure does.
- **Wrist green-light PPG** has very little validation for these indices; one small study found the wrist worse than the finger.
- **ECG + wrist-PPG timing:** a proprietary Huawei smartwatch estimate agreed well with Complior cfPWV (r=0.92, n=347).
- **Sampling:** derivative-based indices need waveform detail up to about 10 Hz. That argues for the 100 Hz on-demand mode or heavy beat-ensemble averaging, not the raw 25 Hz stream.

### Cited Findings
- **Review anchor:** Charlton et al., "Assessing hemodynamics from the photoplethysmogram to gain insights into vascular age: a review from VascAgeNet", Am J Physiol Heart Circ Physiol 2022, DOI 10.1152/ajpheart.00392.2021 — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/)
  - Methods fall into three classes: single-PPG pulse wave analysis, multi-PPG pulse transit time, and PPG combined with another signal (pulse arrival time).
- **Stiffness index (SI = height / Δt between systolic and diastolic peaks)** — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/):
  - Correlates with cfPWV at r=0.65 (n=87 healthy; Millasseau).
  - Within-subject CV 9.6% over 1 week.
  - Associated with HeartScore CV risk.
  - Genetically causal for BP but not for CAD.
- **Aging index (AGI = (b−c−d−e)/a) and SDPPG ratios** — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/):
  - AGI vs age: r=0.80 (n=600; Takazawa 1998, older) and r=0.42 (n=848; Hashimoto).
  - b/a vs age: r=−0.35; d/a vs age: r=0.37 (n=848).
  - HR-adjusted AGI vs cfPWV: r=0.64 (n=112).
  - AGI ICC ≥0.80. The a–c and a–d intervals have poor repeatability (ICC <0.5).
- **AIx surrogate** correlates with radial AIx (r=0.77) and central AIx (r=0.78–0.86) — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/)
- **SDPPG d/a and CV mortality (Inoue et al., Hypertens Res 2017)** — [DOI 10.1038/hr.2016.123](https://doi.org/10.1038/hr.2016.123):
  - 4,373 Japanese women aged 50–79, median follow-up 9.0 y, finger PPG.
  - Adjusted CV-mortality HR 2.30 (1.06–4.99) for the third d/a quartile and 2.60 (1.21–5.60) for the fourth, vs the first.
  - Adjusted for age, height, BMI, BP, HR and atherosclerosis risk factors.
  - Evidence A, but a single cohort of women only.
- **UK Biobank finger ASI (Said et al., JAHA 2018)** — [DOI 10.1161/JAHA.117.007621](https://doi.org/10.1161/JAHA.117.007621):
  - n=169,613; mean ASI 9.30±3.1 m/s.
  - Higher ASI predicted CVD (unadjusted HR 1.27, 1.25–1.28), MI (1.38) and heart failure (1.31), as well as mortality.
  - ASI improved Framingham reclassification by 2.3%, vs 5.4% for pulse pressure; the authors conclude that PP "appears to have a larger clinical value than ASI".
- **Multi-site PTT (finger-toe)** — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/):
  - Finger-toe PWV vs cfPWV: r=0.87 (n=101), 0.66 (n=86) and 0.67 (n=100).
  - Finger-toe PWV ICC 0.959, short-term CV about 5%.
  - Needs two simultaneous PPG sites; we do not have this.
- **Site matters** — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/):
  - Pulse shape "differs with measurement site", and different models may be needed for finger and wrist.
  - Crest-time changes are larger and earlier in finger than in radial PPG.
  - The review lists wristbands and smartwatches as possible platforms for single-PPG pulse wave analysis but reports no wrist-specific validation of SI, AGI or AIx against cfPWV.
- **Wrist vs finger, green vs red (2026 conference paper, n=22)** — [Sciforum, 12th Int. Electronic Conf. on Sensors and Applications; search snippet only, page returned 403](https://sciforum.net/paper/26556):
  - A wrist-worn prototype recorded wrist and finger PPG at red and green.
  - Green correlated more strongly with age than red at both sites, and the finger was the most accurate site.
  - "At the wrist, only the green wavelength showed trends consistent with physiology."
- **Fingertip APG in a larger sample:** B-ratio vs age r=−0.40 (n=168) — [Healthc Inform Res 2015 (search snippet)](https://e-hir.org/journal/view.php?number=809)
- **Sampling and bandwidth** — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/):
  - Most pulse-wave information lies in roughly the first 10 harmonics (below about 10 Hz).
  - Noise must be filtered before differentiation, because differentiation amplifies high frequencies.
  - Averaging matters: baPWV CV is 2.52% for single pulses vs 0.27% when averaged over several.
- **HRV-specific sampling work does not settle morphology needs:**
  - Béres et al. (Meas Sci Rev 2019) is about pulse-rate and HRV accuracy vs sampling interval, not morphology. It finds that cubic-spline interpolation lowers the needed rate a lot — [Béres et al. 2019](https://www.measurement.sk/2019/msr-2019-0030.pdf)
  - For beat intervals, SNR matters more than sampling rate — [arXiv 2202.13651](https://ar5iv.labs.arxiv.org/html/2202.13651)
- **Wrist ECG + PPG cfPWV estimate (Huawei WATCH GT 2 Pro ECG; Sun et al., Front Cardiovasc Med 2022)** — [PMC9353553](https://pmc.ncbi.nlm.nih.gov/articles/PMC9353553/):
  - Setup: n=347, aged 18–80, 30-s watch measurements, reference Complior.
  - Agreement: r=0.918, mean difference 0.046 m/s, LoA −1.77 to 1.86 m/s, MAE 0.657 m/s.
  - Detecting cfPWV >10 m/s: AUC 0.968, sensitivity 0.899, specificity 0.927.
  - Errors widen above age 60 and at SBP ≥140, and the authors suggest the device suits people under 60 with SBP <140.
  - The algorithm is a proprietary built-in ML model. The study was a hospital-internal project with no Huawei funds reported.
  - Evidence B.
- **Caveats on the watch PWV:**
  - A 2026 nationwide Chinese study stresses that smartwatch PWV "is based on wrist-derived single-site signals and algorithmic estimation rather than direct carotid–femoral measurements" — [Front Cardiovasc Med 2026 (snippet)](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2026.1865667/full)
  - Huawei Watch D3 arterial stiffness reportedly lacks medical-device certification — [Notebookcheck review (secondary)](https://www.notebookcheck.net/Huawei-Watch-D3-review-How-the-smartwatch-measures-arterial-stiffness.1396000.0.html)
- **Face video:**
  - Facial RGB-camera PTT is threatened by skin variability — [Moco et al., CVPRW 2018](https://mlanthology.org/cvprw/2018/moco2018cvprw-impairing)
  - Searches found no study estimating vascular age or cfPWV from face video alone (see Gaps).

### Inferences
- **Where the classical indices transfer best:**
  - Classical evidence comes from **finger, mostly IR transmissive, clinical sensors**. The fingertip camera (finger site, reflectance, about 30 fps) is the nearest of our devices in site.
  - The 100 Hz on-demand watch PPG is nearest in sampling rate.
  - The 25 Hz continuous wrist stream is nearest to neither.
- **25 Hz vs the harmonics we need:**
  - At 25 Hz the Nyquist limit is 12.5 Hz. The 10th harmonic exceeds that once HR is above about 75 bpm, and the sensor's own anti-alias filter (unknown) probably cuts lower still.
  - Interpolation recovers *timing*, but it cannot restore lost harmonics, which matter for the SDPPG c, d and e waves.
  - Because the heartbeat is not phase-locked to the 25 Hz clock, aligning and averaging many beats (equivalent-time sampling) may rebuild a usable average beat. This is untested on our data, and it depends on the unknown hardware filter.
- **Most robust choices for us:**
  - SI and crest time / systolic upstroke time: low-order timing features of the average beat.
  - b/a: the a and b waves are the large early features.
  - Fragile: d/a, e/a and AGI, because the c, d and e waves are small and the a–c and a–d intervals repeat poorly even in clinical data.
- **PAT on the watch:**
  - From the Samsung SDK ECG (500 Hz) plus the co-sampled green PPG, PAT runs from the R peak to the wrist PPG foot. It includes the pre-ejection period, and the heart-to-wrist path mostly misses the aorta.
  - So PAT is a weak cfPWV proxy unless combined with waveform features and calibrated; Huawei's ML model presumably does this.
  - Within one person, PAT trends at a fixed posture and arm position may still track BP and stiffness changes.
- **Confounders:**
  - All morphology indices depend on HR, BP, vasomotor tone (temperature) and contact pressure.
  - Per our earlier notes, CP-PPG normalises waveforms toward a young-adult template, so it must **not** be applied before computing vascular-aging features.

### Gaps
- No verified primary source gives a minimum sampling rate for SDPPG and other morphology indices. Common practice and unverified search snippets suggest ≥100 Hz for <10 ms fiducial error. A dedicated down-sampling study (for example, Fujita and Suzuki, IEEE Access 2019) should be checked.
- No large validation of SI, AGI or b/a from commercial wrist green PPG against cfPWV was found.
- A claim seen in search summaries that wrist PPG pulse shape showed "no significant associations with age" could not be traced to a primary source.
- The Takazawa 1998 and Millasseau 2002 originals were not opened; their numbers come via the Charlton review.

---

## 3. Commercial aging and vascular features (Samsung, Apple, Oura, Whoop, Withings, Huawei, Garmin): methods and validation

### Takeaway
- **Samsung Vascular Load** (2025) is a sleep-time optical pulse-wave-analysis stiffness proxy. It is limited to the Watch8 and Ultra generation and later, so not our Watch6 Classic, and has no published validation.
- **Samsung Antioxidant Index** is a 5-s fingertip multi-wavelength skin-carotenoid reflectance measurement. It is not PPG-pulse-based, needs Watch8 hardware, and its validation is a Samsung-cited SNU diet study.
- **Oura Cardiovascular Age:**
  - The only vendor PWV-from-PPG product with any numbers.
  - Internal white paper: n=99, r=0.83 vs cfPWV; not peer reviewed.
- **Huawei** has the best-validated smartwatch cfPWV estimate (ECG+PPG, n=347).
- **Withings** PWV is not PPG: a BCG + foot impedance scale, validated against SphygmoCor.
- **Whoop Healthspan** is a behavioural/physiological composite with no published validation.
- **Apple** has research-only PpgAge; its shipping vascular feature is Hypertension Notifications (FDA-cleared 2025, sensitivity 41%).

### Cited Findings
**Samsung Vascular Load (Labs feature, One UI 8 Watch, 2025)**
- Measures "the amount of stress placed on your vascular system" during sleep "using pulse wave analysis via its optical sensor"; high values "suggest arterial stiffness" — [Samsung UK support](https://www.samsung.com/uk/support/mobile-devices/health-features-on-the-galaxy-watch8-and-watch8-classic/)
- Needs the watch worn while sleeping on ≥3 of the last 14 days, with initial calibration of 3–4 h per night — [Samsung UK support](https://www.samsung.com/uk/support/mobile-devices/health-features-on-the-galaxy-watch8-and-watch8-classic/)
- Supported on Galaxy Watch Ultra, Watch8 and later, with Android 10+ and Samsung Health 6.30.2+ — [Samsung UK support](https://www.samsung.com/uk/support/mobile-devices/health-features-on-the-galaxy-watch8-and-watch8-classic/); [Samsung IE support](https://www.samsung.com/ie/support/mobile-devices/health-features-on-the-galaxy-watch8-and-watch8-classic/)
- Labs features are wellness / personal reference only and can be switched off — [Samsung Newsroom (search summary)](https://news.samsung.com/global/new-features-on-one-ui-8-watch-help-users-build-healthier-habits)
- No peer-reviewed validation or Samsung white paper found. Independent research supports the concept of overnight pulse-wave stiffness (pulse propagation time shorter in hypertensive patients; PPG stiffness lowest in N3 sleep) but did not evaluate Samsung's algorithm — [PubMed 27607460](https://pubmed.ncbi.nlm.nih.gov/27607460/); [Sleep 2023](https://academic.oup.com/sleep/article/46/10/zsad172/7209634)

**Samsung Antioxidant Index (Watch8, 2025)**
- Estimates skin carotenoids in about 5 s with the thumb or fingertip on the rear sensor, using "multi-wavelength LEDs and a custom photodetector array" to measure light absorption; the fingertip is used because of lower melanin — [Samsung Mobile Press feature story](https://www.samsungmobilepress.com/feature-stories/galaxy-watch8-nutrition-tracking-sensor-antioxidant-index/)
- 0–100 score, banded against the WHO 400 g/day fruit-and-vegetable recommendation — [Business Standard (secondary)](https://www.business-standard.com/technology/tech-news/samsung-galaxy-watch-8-can-now-measure-antioxidant-levels-how-it-works-125102400568_1.html)
- Samsung calls the Watch8 series "the first Antioxidant Index feature in a smartwatch" — [Samsung UK support](https://www.samsung.com/uk/support/mobile-devices/health-features-on-the-galaxy-watch8-and-watch8-classic/)
- Validation is Samsung-cited only: a Seoul National University dietary study (80 participants, 6 weeks) found skin carotenoids rose within 1–2 weeks of carotenoid-rich meals — [Samsung newsroom via search summary](https://news.samsung.com/uk/from-lab-to-wrist-samsungs-industry-first-nutrition-tracking-tech-for-galaxy-watch)

**Oura Cardiovascular Age (CVA, 2024)** — evidence C (internal, not peer reviewed)
- Uses "Pulse Wave Analysis (PWA) on the high-resolution PPG signal" from the finger ring, recorded during rest and sleep, to estimate PWV and CVA. Needs 14 days and nights of data within 30 days — [Oura CVA white paper PDF](https://ouraring.com/blog/wp-content/uploads/2024/05/FINAL_Oura_Cardiovascular_Age_White_Paper.pdf)
- Internal validation: estimated PWV vs cfPWV r=0.83 in N=99 healthy people (age 46.6±16.6 y, cfPWV 7.77±1.49 m/s); said to be "acceptable accuracy … based on the recent guidelines" — [Oura CVA white paper PDF](https://ouraring.com/blog/wp-content/uploads/2024/05/FINAL_Oura_Cardiovascular_Age_White_Paper.pdf)
- The age-related waveform-morphology figure is built from age groups of 10,000 individuals each — [Oura CVA white paper PDF](https://ouraring.com/blog/wp-content/uploads/2024/05/FINAL_Oura_Cardiovascular_Age_White_Paper.pdf)
- Output categories: "Above" (CVA ≥6 y over chronological age), "Aligned" (within ±5 y), "Below" (≥6 y younger) — [Oura CVA white paper PDF](https://ouraring.com/blog/wp-content/uploads/2024/05/FINAL_Oura_Cardiovascular_Age_White_Paper.pdf)

**Huawei (smartwatch cfPWV / arterial stiffness)**
- WATCH GT 2 Pro ECG, ECG+PPG ML: r=0.918, MAE 0.657 m/s, AUC 0.968 for cfPWV >10 m/s (n=347 vs Complior) — [PMC9353553](https://pmc.ncbi.nlm.nih.gov/articles/PMC9353553/)
- The Watch D3's arterial-stiffness feature reportedly lacks medical certification — [Notebookcheck (secondary)](https://www.notebookcheck.net/Huawei-Watch-D3-review-How-the-smartwatch-measures-arterial-stiffness.1396000.0.html)

**Withings (scale PWV, not PPG)**
- Body Cardio estimates aortic PWV from aorta-to-leg pulse transit time, using ballistocardiography plus impedance plethysmography on one foot while standing — [Campo et al., Am J Hypertens 2017](https://doi.org/10.1093/ajh/hpx059)
- Withings/Boutouyrie validation — [Campo et al. 2017](https://doi.org/10.1093/ajh/hpx059):
  - Calibrated on 106 people, validated on 86.
  - Bias 0.25 m/s, SE 1.39 m/s ("acceptable" per ARTERY).
  - R=0.76 with cfPWV.
  - The authors say repeatability still needs improvement.
- Independent check: in n=20 young adults vs SphygmoCor, the scale overestimated PWV by about 0.68 m/s (MAPE 9.7%) — [Collier et al. 2020, PMC7151625](https://pmc.ncbi.nlm.nih.gov/articles/PMC7151625/)

**WHOOP Healthspan / WHOOP Age / Pace of Aging (2025)** — evidence C
- WHOOP Age turns about 6 months of sleep, activity and fitness data into a "physiological age". Pace of Aging updates weekly — [WHOOP: Healthspan data meets longevity](https://www.whoop.com/ca/en/thelocker/Healthspan-Data-Meets-Longevity/)
- The "nine contributors" include sleep duration and consistency, steps, HR-zone time, strength activity, VO2max, resting HR and lean body mass. VO2max reportedly carries heavy weight — [WHOOP 2025 launches](https://www.whoop.com/thelocker/everything-whoop-launched-in-2025); [WHOOP podcast 353](https://www.whoop.com/ch/en/thelocker/podcast-353-whoop-2025-year-in-review/)
- It is a composite of summary metrics, not waveform-derived. No published weighting or validation found.

**Apple**
- PpgAge is research only (section 1).
- The shipping vascular feature is Hypertension Notifications, FDA-cleared in Sept 2025 (K250507) — [FDA 510(k) mirror](https://fda.innolitics.com/device/K250507); [AAFP blog](https://www.aafp.org/pubs/afp/afp-community-blog/entry/smartwatch-screening-for-hypertension.html):
  - ML on 30 days of background PPG.
  - Pivotal study: 2,229 enrolled, 1,863 analysable.
  - Sensitivity 41.2% (37.2–45.3), specificity 92.3% (90.6–93.7).

### Inferences
- Samsung Vascular Load and Oura CVA show a common vendor design:
  - **Low-motion sleep or rest PPG, aggregated over multiple nights** (Samsung: 3 of 14 nights; Oura: 14 of 30 days).
  - Reported as a **relative category** against age norms.
- We can copy this design on the Watch6 Classic: the 25 Hz continuous green PPG during sleep, motion-gated with the 100 Hz accelerometer and averaged per night. Without a cfPWV reference it can only show within-person trends.
- Samsung's support page excludes pre-Watch8/Ultra models. We should not expect Vascular Load on the Watch6 Classic, or expect to read it through the Samsung Health export (not verified).
- The Antioxidant Index is a spectroscopic carotenoid measurement, not a pulse-wave measure. It is likely not reproducible with the Watch6 SDK's green/IR/red PPG channels; this is unverified.

### Gaps
- No Samsung white paper or peer-reviewed validation of Vascular Load was found, and its algorithm, output units and reference are unknown.
- Whether the Samsung Health data export contains Vascular Load or Antioxidant Index values is unknown.
- I found no Garmin PPG-derived vascular or aging metric. Garmin "Fitness Age" is VO2max-based and out of scope.
- I found no published validation of the Withings "vascular age" output (as distinct from PWV), and no Withings wrist-PPG stiffness feature.
- The Oura AHA 2025 abstracts on cardiovascular metrics were not retrieved.

---

## 4. Other aging-relevant wearable signals: resting HR, HRV decline, HR recovery and mortality

### Takeaway
- **Resting HR:** a robust mortality marker. All-cause mortality RR is 1.09 per +10 bpm in the general population (46 cohorts), and HR is about 1.2 per 10 bpm in UK Biobank.
- **HRV:** low SDNN gives about 40% higher first-CVD risk (8 cohorts, n≈22k). HRV falls steeply with age in 8.2M Fitbit users.
- **HR recovery:** attenuated recovery gives all-cause mortality HR 1.68 (9 cohorts).
- All three need only beat timing or 1 Hz HR, so they are the most feasible aging-related outputs on every one of our wrist modes.
- The outcome evidence comes from clinical and ECG measurements, not from consumer wearables.

### Cited Findings
- **Resting HR, general population (Zhang et al., CMAJ 2016)** — [PubMed 26598376](https://pubmed.ncbi.nlm.nih.gov/26598376/); [CMAJ PDF](https://www.cmaj.ca/content/cmaj/188/3/e53.full.pdf):
  - Meta-analysis of 46 prospective cohorts.
  - Per +10 bpm: all-cause mortality RR 1.09 (1.07–1.12), CV mortality RR 1.08 (1.06–1.10).
  - RHR >80 bpm vs the lowest category: all-cause mortality RR 1.45 (1.34–1.57).
  - Roughly linear above 45 bpm, and unchanged after adjusting for traditional risk factors.
- **Resting HR in UK Biobank (n=502,534, 7–12 y follow-up)** — [PLOS ONE 2020, PMC7259773](https://pmc.ncbi.nlm.nih.gov/articles/PMC7259773/):
  - All-cause mortality HR per +10 bpm: 1.22 (1.20–1.24) in men and 1.19 (1.16–1.22) in women.
  - Associations are stronger at younger ages.
- **HRV and first CV event (Hillebrand et al., Europace 2013; 8 studies, 21,988 participants without known CVD)** — [Leiden repository](https://scholarlypublications.universiteitleiden.nl/handle/1887/102345); [Europace record](https://dbpia.nl.go.kr/europace/article/15/5/742/673395):
  - Low SDNN gives about 40% higher risk.
  - Each +1% SDNN gives about 1% lower risk.
  - The association depended on recording duration.
- **HRV in patients with existing CVD (Fang et al. 2019 meta-analysis):** pooled HR 2.12 (1.64–2.75) for all-cause death and 1.46 (1.19–1.77) for CV events — [Biol Res Nurs 2019](https://journals.sagepub.com/doi/full/10.1177/1099800419877442)
- **HRV and age in wearables (Natarajan et al., Lancet Digital Health 2020)** — [DOI 10.1016/S2589-7500(20)30246-6](https://doi.org/10.1016/S2589-7500(20)30246-6):
  - 8,203,261 Fitbit users; wrist PPG beat intervals, 5-min windows.
  - HRV declines with age, and parasympathetic indices decline faster.
  - From age 20 to 60 (06:00–07:00 h), HF power falls 82.0% in men and 80.9% in women; LF power falls 66.5% and 69.3%.
  - Strong diurnal variation; more daily activity is associated with higher HRV in a dose-dependent way.
  - Gives age- and sex-specific RMSSD/SDRR benchmark tables.
- **HR recovery (Qiu et al., JAHA 2017; 9 prospective cohorts)** — [PMC5524096](https://pmc.ncbi.nlm.nih.gov/articles/PMC5524096):
  - Attenuated vs fast HRR: all-cause mortality HR 1.68 (1.51–1.88).
  - Per 10 bpm lower HRR: HR 1.09 (1.01–1.19).
  - Applies to a standardised post-exercise recovery protocol.
- **HRV and age in PpgAge:**
  - In PpgAge, HR/HRV-only age models reached MAE 6.1 y vs 2.4 y for the waveform model — [PMC12537950](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
  - On clinical finger PPG, HR/HRV features barely predicted age at all (MAE 11.49 vs 11.91 baseline) — [arXiv 2603.14030](https://arxiv.org/abs/2603.14030)

### Inferences
- **Resting HR:** night-time or lowest-sustained RHR from 1 Hz HR or 25 Hz PPG is trivially feasible and has the strongest mortality evidence per unit of engineering effort. Use a consistent definition, such as the nightly minimum 5-min mean HR, to cut day-to-day noise.
- **HRV:** nightly RMSSD/SDNN from 25 Hz PPG (already benchmarked by us) can be read against the Natarajan age/sex benchmark tables as an "autonomic age" percentile. The outcome evidence uses 24-h or short ECG SDNN, so the transfer is approximate.
- **HRR:** compute it from watch HR (1 Hz, or about 1 s during workouts in the Samsung export) as the HR drop 60 s after a hard bout ends. Free-living HRR is unstandardised (exercise intensity, posture and cooldown all vary), so treat it as a within-person trend.
- **Combination:** RHR, HRV and HRR are complementary to waveform-based vascular age (timing vs morphology). A personal "aging dashboard" could combine both families.

### Gaps
- I found no cohort linking *consumer-wearable-measured* RHR, HRV or HRR (Fitbit, Apple, Samsung) to incident mortality. The searches returned only clinical-measurement cohorts.
- Free-living HRR from wearables has no mortality validation that I found.

---

## 5. Feasibility on our devices: (a) 25 Hz continuous wrist PPG, (b) 100 Hz on-demand wrist PPG ≤30 s (+500 Hz ECG), (c) fingertip camera ~30 fps, (d) face video

### Takeaway
- **(a) 25 Hz continuous wrist PPG:** the right source for RHR, HRV and nightly trend features. Usable for a PpgAge-*style* within-person embedding trend after upsampling. Marginal for derivative-based indices.
- **(b) 100 Hz on-demand wrist PPG + 500 Hz ECG:** the best watch mode for pulse-wave-analysis indices and for ECG→PPG pulse arrival time. It mirrors the OPPO heart-age protocol (30-s spot checks, repeated about 10 or more times).
- **(c) Fingertip camera:** at the measurement site where most classical evidence lies. Demonstrated for coarse young/old classification at 30 Hz. The closest domain for the open AI-PPG Age (finger) model.
- **(d) Face video:** no evidence for any aging or vascular-age marker. HR only.

### Cited Findings
- **Input requirements of the reference models:**
  - PpgAge: 60-s, 4-channel, 64/256 Hz green wrist PPG during low motion, averaged over 30-day windows of ≥30 segments — [PMC12537950](https://pmc.ncbi.nlm.nih.gov/articles/PMC12537950/)
  - OPPO heart age: 250 Hz, 4-channel (green/red), 30-s spot checks, optionally with 250 Hz single-lead ECG for pretraining — [arXiv 2608.01620](https://arxiv.org/html/2608.01620v1)
  - AI-PPG Age: a single averaged finger beat resampled to 100 samples — [arXiv 2603.14030](https://arxiv.org/abs/2603.14030)
  - Pulse-PPG: 50 Hz wearable PPG — [arXiv 2603.14030](https://arxiv.org/abs/2603.14030)
- **Sleep and rest aggregation in vendor features:** Samsung Vascular Load needs ≥3 of 14 nights; Oura CVA needs 14 days and nights of rest/sleep data out of 30 — [Samsung UK support](https://www.samsung.com/uk/support/mobile-devices/health-features-on-the-galaxy-watch8-and-watch8-classic/); [Oura white paper](https://ouraring.com/blog/wp-content/uploads/2024/05/FINAL_Oura_Cardiovascular_Age_White_Paper.pdf)
- **Smartwatch spot-check precedent:** Huawei's smartwatch cfPWV estimate used 30-s ECG+PPG measurements — [PMC9353553](https://pmc.ncbi.nlm.nih.gov/articles/PMC9353553/)
- **Smartphone precedent:** smartphone fingertip PPG at about 30 samples/s, about 90 s per recording (red channel), separated young from old with AUC 0.95 — [PMC7661535](https://pmc.ncbi.nlm.nih.gov/articles/PMC7661535/)
- **Spectral content:** most pulse-wave information lies below about 10 Hz (first ~10 harmonics) — [PMC8917928](https://pmc.ncbi.nlm.nih.gov/articles/PMC8917928/)
- **Wrist wavelength:** at the wrist, only green-light APG indices behaved physiologically, and the finger beat the wrist (n=22) — [Sciforum 2026 (snippet)](https://sciforum.net/paper/26556)
- **Face video:** facial RGB PTT is limited by skin variability — [CVPRW 2018](https://mlanthology.org/cvprw/2018/moco2018cvprw-impairing). Searches found rPPG work on HR, BP and multi-biomarker datasets but no face-video age or vascular-age estimator — [MCD-rPPG dataset, arXiv 2508.17924](https://arxiv.org/pdf/2508.17924)

### Inferences
Feasibility matrix (inference, built from the cited input requirements; ✓ = feasible, ~ = partial or experimental, ✗ = not feasible):

| Measure | (a) 25 Hz continuous wrist green | (b) 100 Hz on-demand wrist (G/IR/R) ≤30 s + 500 Hz ECG | (c) Fingertip camera ~30 fps | (d) Face video |
|---|---|---|---|---|
| Resting HR (nightly) | ✓ best source (also from 1 Hz HR) | ✓ but spot only | ✓ spot | ✓ spot, still/lit only |
| HRV (RMSSD/SDNN) vs age norms | ✓ (our benchmark) | ✓ (30 s is short; use ECG RR) | ~ (HRV poor at 30 fps per earlier notes) | ✗ |
| HR recovery | ✓ from 1 Hz HR after workouts | ✗ (on-demand only at rest) | ✗ | ✗ |
| PpgAge-style embedding age / trend | ~ upsample 25→50/64 Hz, low-motion/sleep windows, 30-day means. No public PpgAge weights. Use Pulse-PPG (50 Hz wearable) embeddings for within-person trend only | ~ closer to the 64 Hz spec; but at most 30 s per block and foreground only | ~ AI-PPG Age (finger, averaged beat) is the closest domain, but expect collapse/bias (zero-shot MAE 11.5 y on VitalDB) | ✗ |
| OPPO-style heart age (30-s spot checks ×10+) | ✗ | ~ protocol matches, 100 vs 250 Hz; no open model; would need own labelled data | ✗ | ✗ |
| SI, crest time, upstroke time (average beat) | ~ via beat-ensemble average; unverified | ✓ best watch option; site evidence weak at wrist | ✓ finger site matches literature; 30 fps marginal | ✗ |
| SDPPG b/a | ~ / ✗ (second derivative at 25 Hz; harmonics lost) | ~ (green; filter before differentiating) | ~ (Dall'Olio used SDPPG a-wave at 30 Hz) | ✗ |
| SDPPG d/a, e/a, AGI | ✗ | ~ fragile (small c/d/e waves; poor repeatability even clinically) | ~ fragile | ✗ |
| PAT (ECG R peak → wrist PPG foot) as PWV proxy | ✗ (no ECG) | ✓ measurable; includes PEP and misses the aorta; within-person trend; Huawei-style ML calibration would need cfPWV labels | ~ phone camera + watch ECG need cross-device time sync (ms level), so unlikely | ✗ |
| Night-time "Vascular Load"-like index | ✓ concept: sleep PWA features averaged over ≥3 nights, as relative trend | ✗ (on-demand, foreground only) | ✗ | ✗ |
| Antioxidant / carotenoid index | ✗ (needs Watch8 spectroscopic sensor) | ✗ (unverified whether the IR/red channels suffice; likely not) | ✗ | ✗ |

- **Recommended order for the prototype:**
  1. RHR, HRV and HRR trends against published age norms (the strongest outcome evidence; already feasible).
  2. A nightly 25 Hz average-beat morphology trend: SI, crest time, b/a where stable; motion-gated; per night.
  3. Daily 100 Hz + ECG 30-s spot checks at a fixed posture for PAT and a higher-fidelity average beat. Repeat ≥10 times to reach the OPPO-like stability regime.
  4. A fingertip camera session (torch, locked exposure, 60–90 s) as a finger-site cross-check, and as input to AI-PPG Age with a domain-shift caveat.
- **Server-side compute:** all of the above is cheap. The RTX 4090 is needed only for foundation-model embeddings (Pulse-PPG / PaPaGei), which are small 1D ResNets.
- **Honest output framing:** absolute "vascular age" numbers from our devices would have no reference validation. Report within-person change and percentile-vs-norms (RHR, HRV), and keep any model "age" output as an exploratory number with its expected MAE (≥5–10 y outside its training domain).

### Gaps
- Unknown: the anti-alias filtering and exact channel semantics of `hr_raw` at 25 Hz, which decide whether beat-ensemble averaging can recover morphology. Needs a side-by-side recording of the 25 Hz stream vs the 100 Hz on-demand stream on the same wrist.
- Unknown: whether the Samsung SDK ECG tracker's co-sampled PPG runs at the full 500 Hz and is time-aligned well enough for PAT. This needs checking on device.
- No study validates any aging index from 25 Hz wrist PPG, and none from face video.
- No open model is trained on wrist green PPG with age labels at a scale comparable to PpgAge or OPPO. Our Pulse-PPG-based trend would be unvalidated for aging.
