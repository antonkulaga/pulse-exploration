# Cardiometabolic disease detection and risk estimation from wearable/smartphone PPG and single-lead watch ECG

Scope: which cardiometabolic conditions can be detected or risk-scored from PPG (and the Galaxy Watch's
30 s single-lead ECG), with evidence strength, the signal each method needs, and feasibility on our devices
(Galaxy Watch6 Classic SM-R965F; phone fingertip camera ~30 fps; face video). Status as of October 2026.
Out of scope here (other researchers): vascular ageing / arterial stiffness, camera-measurement technique,
HRV accuracy (already benchmarked: resting RMSSD error ~3 ms from 25 Hz wrist PPG).
Earlier context: `docs/model-feasibility.md` (ECGFounder, PulseLM, Pulse-PPG, PaPaGei, AnyPPG).

Evidence-strength shorthand used below: **A** = regulator-cleared + large prospective study; **B** = large
retrospective/prospective peer-reviewed study, not cleared or cleared only in one country; **C** = small or
single-centre studies, preprints, conference abstracts; **D** = marketing / no peer-reviewed validation.

---

## 1. Atrial fibrillation and irregular rhythm (wrist PPG, smartphone camera PPG, watch ECG)

### Takeaway
AF is the only cardiometabolic condition where PPG detection is mature (evidence A): FDA-cleared wrist-PPG
notifications from Apple, Fitbit and Samsung run on beat-interval "tachograms", not on waveform morphology,
and have very high PPV (84–98%) but modest sensitivity (Samsung 68%), while smartphone fingertip apps
(FibriCheck, Preventicus) reach ~95–99% sensitivity/specificity on spot checks. Published wrist-PPG AF models
work at 20–32 Hz, so our 25 Hz continuous `hr_raw` stream is adequate for an AF screen; diagnosis requires a
≥30 s single-lead ECG reviewed by a physician, which the Watch6's ECG provides.

### Cited Findings

**Apple (wrist PPG, tachogram-based)**
- Apple Heart Study (Perez et al., NEJM 2019, DOI 10.1056/NEJMoa1901183): 419,297 participants; over a median 117 days 2,161 (0.52%) received an irregular-pulse notification; notification rates ranged from 3.2% in those >65 y to 0.16% in those <40 y; AF was present in 34% of the 450 analysable ECG patches (applied ~13 days after notification); PPV 0.84 (95% CI 0.76–0.92) for AF on the patch simultaneous with a subsequent notification and 0.71 for an irregular tachogram — [eScholarship copy of NEJM paper](https://escholarship.org/uc/item/5n08v244); [Healio](https://www.healio.com/news/cardiology/20190404/apple-heart-study-forms-solid-foundation-for-af-detection-via-smartwatch)
- Apple's algorithm: the optical sensor reads pulse waves during minimal arm movement; times between PPG peaks form beat-to-beat intervals ("tachogram"); each tachogram is classified regular/irregular by a proprietary classifier; an irregular tachogram triggers more frequent sampling (≥15 min apart) and a notification fires when 5 of 6 sequential tachograms (including the first) are irregular within 48 h — [Apple Watch Arrhythmia Detection (clinician document)](https://www.apple.com/uk/healthcare/docs/site/Apple_Watch_Arrhythmia_Detection.pdf); [clinicaltrials.gov Apple Heart Study protocol](https://cdn.clinicaltrials.gov/large-docs/00/NCT03335800/Prot_SAP_000.pdf)
- Regulatory: Irregular Rhythm Notification Feature De Novo DEN180042 granted 11 Sep 2018 (product code QDB, 21 CFR 870.2790, "photoplethysmograph analysis software for OTC use"); the 2023 update K231173 describes a tachogram classifier plus a confirmation-cycle algorithm and includes a predetermined change control plan — [regdatalab DEN180042](https://www.regdatalab.com/510k/DEN180042); [Innolitics K231173](https://fda.innolitics.com/device/K231173). Apple's clinician page cites sensitivity 88.6% / specificity 99.3% vs patch ECG for the current feature — [apple.co link](https://apple.co/34flP6v) (not independently verified). The 2019 Apple Heart Study figures describe the first-generation algorithm and are superseded by the IRNF 2.0 clearance.

**Fitbit / Google (wrist PPG, tachogram-based)**
- Fitbit Heart Study (Lubitz et al., Circulation 2022): 455,699 participants enrolled May–Oct 2020 (median age 47, 71% female); the algorithm requires 11 of 11 consecutive tachograms to be irregular, corresponding to ≥30 min of irregular rhythm; PPV of a subsequent irregular-heart-rhythm detection for concurrent AF on ECG patch was 98.2% (95% CI 95.5–99.5%), 97.0% in those ≥65 y — [PMC9640290](https://pmc.ncbi.nlm.nih.gov/articles/9640290); [MGH press release](https://www.massgeneral.org/news/press-release/fitness-trackers-may-diagnose-potentially-irregular-heartbeats). FDA clearance of Fitbit Irregular Rhythm Notifications dated 2022-04-08 — [510k database K212372](https://510k.innolitics.com/device/K212372)

**Samsung (our watch's own features)**
- Samsung Irregular Heart Rhythm Notification (IHRN), FDA 510(k) K230292 (announced 8 May 2023): background PPG analysis; after a number of consecutive irregular measurements it prompts an on-watch ECG; subject-level sensitivity 68.0% (95% CI 60.5–75.5) and specificity 98.8% (98.0–99.6); per the label it notified 102/150 participants with a continuous AF episode ≥1 h and stayed silent for 652/660 without concordant AF; notification followed by ECG gave overall sensitivity 66.7% and specificity 100%; not for users <22 y or with other known arrhythmias — [FDA K230292 summary](https://www.accessdata.fda.gov/cdrh_docs/pdf23/K230292.pdf); [Samsung IHRN IFU](https://image-us.samsung.com/SamsungUS/home/apps/samsung-health-monitor/IFU_IHRN_US_202306.pdf); [Samsung press](https://samsungmobilepress.com/articles/samsung-announces-fda-cleared-irregular-heart-rhythm-notification-for-galaxy-watch)
- Samsung ECG Monitor app: K201168 (Aug 2020) reported 98.1% AF sensitivity (95% CI 96.3–99.9) and 100% sinus-rhythm specificity in 544 subjects vs cardiologist-read 12-lead ECG, with 2.9% inconclusive (Apple ECG predicate 2.2%); superseded by K240909 (Aug 2024, ECG app v1.3): 96.0% AF sensitivity / 98.7% sinus specificity in 1,013 subjects — [Innolitics K201168](https://fda.innolitics.com/device/K201168); [Innolitics K240909](https://fda.innolitics.com/device/K240909). A secondary source cites a systematic review finding only 86% sensitivity / 76% specificity when unclassified ECGs are counted as errors — [Qaly blog](https://www.qaly.co/post/samsung-ecg-accuracy) (unverified secondary).
- Stanford + Samsung (Voisin, Shen, Aliamiri, Avati, Hannun, Ng; arXiv 1811.07774): deep model on raw 30 s PPG from a **Samsung wrist wearable sampled at 20 Hz**; 510,566 records from 81 patients (>4,000 h); test AUC 0.95; no code/data release — [Stanford ML Group project page](https://stanfordmlgroup.github.io/projects/ppg/); [arXiv](https://arxiv.org/pdf/1811.07774)

**Smartphone fingertip camera PPG**
- FibriCheck: first FDA clearance K173872 (2018): sensitivity 95.60%, specificity 96.55% in 223 subjects (100 with AF) vs 12-lead ECG — [Innolitics K173872](https://fda.innolitics.com/device/K173872). 2025 multicentre study across ten smartphone models, 236 subjects: accuracy 98.5%, sensitivity 96.3%, specificity 99.3% vs 12-lead ECG, stable across phones and comorbidities — [PMC12635274](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12635274/). A 2024 FDA clearance (K232804) followed; press figures of 98.3%/99.9% come from a 50-patient trial (NCT06282380) and company statements — [DAIC](https://www.dicardiology.com/content/fibricheck-enters-us-market-fda-clearance); [Innolitics K232804](https://510k.innolitics.com/device/K232804)
- Preventicus (eBRAVE-AF RCT, Rizas et al., Nature Medicine 2022): site-less RCT of 5,551 adults aged 50–90; smartphone PPG pre-screen + 14-day ECG patch for abnormal results; detection of treatment-relevant AF (new OAC prescription) within 6 months doubled vs usual care, OR 2.12 (95% CI 1.19–3.76); after crossover OR 2.75 (1.42–5.34); no outcome (stroke) data — [eBRAVE-AF commentary, PMC9798038](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9798038/); [TCTMD](https://www.tctmd.com/news/smartphone-app-can-detect-af-influence-care-ebrave-af)

**Screening RCTs and guidelines**
- A randomized trial in 437 adults ≥65 y (JACC, Jan 2026) found 6 months of Apple Watch rhythm monitoring identified new AF in 9.6% vs 2.3% with standard care — [Epocrates summary](https://epocrates.com/online/article/smartwatch-boosts-atrial-fibrillation-detection-in-randomized-trial). Heartline (J&J + Apple, ≥65 y Medicare, stroke-reduction question, launched 2020) — no published primary results found in my search — [J&J launch release](https://www.jnj.com/johnson-johnson-launches-heartline-the-first-of-its-kind-virtual-study-designed-to-explore-if-a-new-iphone-app-and-apple-watch-can-help-reduce-the-risk-of-stroke)
- ESC: a single-lead ECG of ≥30 s showing irregular RR intervals without discernible repeating P waves is diagnostic of clinical AF, but only after physician review (2020 ESC text; reported as reinforced in the 2024 ESC AF guideline, Van Gelder et al., Eur Heart J 2024, DOI 10.1093/eurheartj/ehae176) — [ESC 2020 guideline slides](https://omronhc.lampyon.com/img/afib/ESC_guidelines_2020.pdf); [RCM review 2026](https://www.imrpress.com/journal/RCM/27/10/10.31083/RCM51122). The 2023 ACC/AHA AF guideline is reported to state that PPG alone is not reliable enough to establish an AF diagnosis; ESC regards PPG's role in screening/stroke reduction as unclear — [Health Technology Wales EAR067 (2025)](https://healthtechnology.wales/wp-content/uploads/EAR067-PPAF-0.27-CB.pdf)

**Sampling rate and beat-timing precision**
- No study I found directly ablates 25 Hz vs lower rates for AF. But wrist-PPG AF models are routinely built at low rates: 20 Hz (Samsung device, AUC 0.95, above); DeepBeat at 32 Hz (25 s windows); and a JACC: Clinical EP 2024 study (Antiperovitch et al., 204 free-living patients, ~4 weeks of simultaneous PPG + ECG) resampled 5-min PPG segments to 25 Hz: deep network AUROC 0.973 (sens 92.2%, spec 95.5%) analysing ~95% of data vs heuristic RR-interval processing AUROC 0.972 (sens 99.6%, spec 94.4%) analysing only ~62% (motion rejection); external validation AUROC 0.994 (DNN) vs 0.989 (heuristic) — [JACC EP 2024 (Europe PMC record, PMID 38340117)](https://www.jacc.org/doi/10.1016/j.jacep.2024.01.008); [DeepBeat dataset description](https://arxiv.org/pdf/2308.02433)
- A 20-beat PPG window can be enough to detect AF correctly — [PubMed 33018131](https://pubmed.ncbi.nlm.nih.gov/33018131/). For pulse-rate *variability* (not AF), studies disagree on minimum rate: ~100 Hz without interpolation for time-domain PRV in real PPG, much lower with spline interpolation for frequency-domain — [Biomed Signal Process Control 2021](https://www.sciencedirect.com/science/article/pii/S1746809421001865); [Comput Methods Programs Biomed 2022](https://www.sciencedirect.com/science/article/pii/S0169260722001109)

**Open algorithms, models and data**
- DeepBeat (Torres-Soto & Ashley, npj Digit Med 2020): multitask CNN (signal quality + rhythm); denoising-autoencoder pretraining raised AF F1 from 0.54 to 0.96; dataset (described in later papers as >3.3 M 25-s segments at 32 Hz from 167 subjects, recorded at 128 Hz and downsampled) is public on Synapse (syn21985690), but only part of the labels are human-assigned (rest model-generated) — [PMC7481177](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7481177/); [arXiv 2001.00155](https://arxiv.org/pdf/2001.00155); [label-noise note, arXiv 2308.02433](https://arxiv.org/pdf/2308.02433)
- FELA-PPG (Hugging Face, research preview): single-channel PPG → AF probability per 25-s window plus signal-quality score; trained on MIMIC PERform AF (ODbL share-alike) — [HF model card](https://huggingface.co/lowdown-labs/fela-ppg). BayesBeat (2020) with public weights/code — [arXiv 2011.00753](https://arxiv.org/pdf/2011.00753). CardioPPG checkpoint on Gitee (filename claims AUC 0.966 on MIMIC AF from PulseDB) — [Gitee](https://gitee.com/sea1234567/CardioPPG). No standard benchmark comparing these was found.

### Inferences
- **Signal need: beat timing only.** All three cleared wrist algorithms classify tachograms (inter-beat-interval series over short windows) and require persistence (Apple 5/6 tachograms in 48 h; Fitbit 11/11 consecutive tachograms, ≥30 min). At 25 Hz the sample spacing is 40 ms, but AF irregularity (successive RR differences of 100+ ms) is far above that, and our HRV benchmark already shows peak timing to ~ms precision after interpolation. The Samsung 20 Hz and resampled-25 Hz studies are the closest direct evidence that 25 Hz suffices. → **Feasible without the SDK** on `hr_raw` (continuous, opportunistic at rest), and with the SDK we also get Samsung's own beat-interval list.
- **Confirmation path is on-device:** SDK ECG_ON_DEMAND (500 Hz, 30 s) meets the ESC "≥30 s single-lead" bar; run NeuroKit2 R-peaks + ECGFounder (1-lead) or a PhysioNet/CinC-2017-style classifier, then physician review. This mirrors Samsung's own IHRN→ECG flow.
- **Phone camera AF is the strongest phone-only use** (FibriCheck/Preventicus at ~30 fps spot checks); our 30 fps fingertip capture is in the same regime as those apps.
- Main engineering risks: motion artefact rejection (heuristic methods discard ~40% of data), ectopic beats (main source of false positives), and the lack of a clean open model trained on Samsung green-channel PPG at 25 Hz — a small labelled set from our own watch (paired with SDK ECG) would be needed to calibrate thresholds.

### Gaps
- No study found that ablates PPG sampling rate (e.g., 25 vs 50 vs 100 Hz) specifically for AF detection performance.
- Apple's and Samsung's actual PPG sampling rates for IRNF/IHRN are not disclosed in the documents found.
- Heartline primary results were not found (may be unpublished as of Oct 2026).
- Preventicus standalone diagnostic-accuracy numbers and its CE/FDA status were not retrieved.
- Apple ECG app sensitivity/specificity (DEN180044) not retrieved in this pass.

---

## 2. Cuffless blood pressure and hypertension

### Takeaway
Expert bodies (ESH 2022/2023, AHA/ACC 2025 guideline, AHA statement Dec 2025, ESC/EAPC-type statement 2026)
agree cuffless BP is not accurate enough to diagnose or manage hypertension; ESH 2023 defines six validation
tests and ISO 81060-3:2022 distinguishes "trending" from "absolute" devices. Cleared products are either
calibrated pulse-wave-analysis trackers (Samsung, Hilo/Aktiia — 510(k) OTC July 2025) or a non-numeric
hypertension *screen* (Apple HTNF, K250507, Sept 2025: 30-day PPG, sensitivity 41.2%, specificity 92.3%).
For us, realistic targets are (a) an Apple-style multi-week hypertension-risk flag from PPG embeddings with
home-cuff labels and (b) within-person BP *trend* tracking after cuff calibration, possibly with ECG+PPG
pulse arrival time in 30 s SDK blocks — not calibration-free absolute mmHg.

### Cited Findings

**Apple Hypertension Notification Feature (HTNF) — primary FDA document**
- 510(k) K250507, cleared 11 Sep 2025; regulation 21 CFR 870.2380 "Cardiovascular Machine Learning-Based Notification Software"; product code later re-assigned to SFR; predicate Viz HCM (DEN230003), a 12-lead-ECG hypertrophic cardiomyopathy detector — [FDA K250507](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K250507.pdf)
- Method: the watch's ML model "gives each qualified PPG signal a score associated with risk of hypertension"; the iPhone aggregates qualified scores over 30-day periods and notifies if patterns suggest hypertension; no BP number is shown; not for people already diagnosed, pregnancy, treatment monitoring or BP surveillance; absence of a notification does not exclude hypertension — [FDA K250507](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K250507.pdf)
- Algorithm development: a self-supervised deep-learning PPG encoder trained on unlabeled Apple Watch data from >86,000 participants, plus a **linear model** on top trained with Apple Watch data + home BP references from >9,800 participants; Training/Train-Dev/Test-Dev/Test splits without subject overlap — [FDA K250507](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K250507.pdf). Apple's own materials describe >100,000 participants across studies and say the training data for the representation did not include BP — [Apple validation paper (Sept 2025)](https://www.apple.com/health/pdf/Hypertension_Notifications_Validation_Paper_September_2025.pdf); [Apple newsroom](https://www.apple.com/newsroom/2025/09/apple-debuts-apple-watch-series-11-featuring-groundbreaking-health-insights/)
- Pivotal validation: hypertension defined as average SBP ≥130 or DBP ≥80 mmHg (AHA 2017); 2,229 enrolled without prior diagnosis; 30 days of watch wear with an FDA-cleared home cuff as reference; 1,863 had ≥15 days usable data; sensitivity 41.2% (95% CI 37.2–45.3), specificity 92.3% (90.6–93.7); stage 2 sensitivity 53.7% (47.7–59.7); specificity in normotensives 95.3% (93.7–96.5); higher sensitivity/lower specificity at age ≥60 and BMI >30; no clinically meaningful differences by sex, race, or Fitzpatrick skin tone after covariate adjustment — [FDA K250507](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K250507.pdf)
- Longitudinal (2 years, six 30-day windows): specificity in non-hypertensives 86.4% (N=187), 92.5% in those with normal BP (N=121); most false-positive notifications occurred in the first month — [FDA K250507](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K250507.pdf)
- Apple chose an operating point prioritising specificity — [Apple validation paper](https://www.apple.com/health/pdf/Hypertension_Notifications_Validation_Paper_September_2025.pdf). Critique (Cohen, Brady, Juraschek, Picone, Yang, Schutte, Hypertension 2026;83:e26031): 41% sensitivity means ~59% of undiagnosed hypertensives get no alert, "not suitable for largescale, reliable hypertension screening"; compares with office BP over two visits (sensitivity 71%, specificity 59%); notes the predicate neither uses PPG nor screens hypertension; PPG affected by skin tone, contact pressure, habitus, hydrostatic pressure, vasoconstriction, activity — [PMC12826266](https://pmc.ncbi.nlm.nih.gov/articles/PMC12826266/)

**Samsung (calibrated pulse-wave analysis; available on our watch via Samsung Health Monitor)**
- Mechanism: pulse wave analysis from the HR (PPG) sensor; the app relates later pulse-wave changes to the calibration value; calibration = 3 cuff readings within 30 min (cuff on one arm, watch on the other), repeated every 28 days; launched in South Korea in 2020 after MFDS clearance — [Samsung support page](https://www.samsung.com/au/support/mobile-devices/measure-bp-on-samsung-watch); [Samsung press 2020](https://samsungmobilepress.com/articles/samsung-launches-the-samsung-health-monitor-application-with-blood-pressure-measurement)
- Independent evaluations: Galaxy Watch Active2 vs ambulatory BP (40 patients) showed bias toward the calibration point — low SBP overestimated, high SBP underestimated, DBP overestimated; "not yet ready for clinical usage" — [Front Cardiovasc Med 2022](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2022.958212/text). Real-world Galaxy Watch data: calibration instability decreased at night and when averaging several readings — [IEEE OJEMB 2024](https://doaj.org/article/5fbc188dd46e4d06ac01acb6637bb8e1). Galaxy Watch5 vs cuff over 28 days (37 participants): acceptable accuracy near the calibration point, declining as BP moves away from it — [Walzel et al., Digital Health 2026, DOI 10.1177/20552076261415923](https://doi.org/10.1177/20552076261415923). Korean feasibility study with 760 users uploading readings for 4 weeks after calibration — [Ewha](https://pure.ewha.ac.kr/en/publications/feasibility-and-measurement-stability-of-smartwatch-based-cuffles/)

**Hilo (formerly Aktiia)**
- FDA 510(k) clearance (July 2025) for OTC sale of the G0 Blood Pressure Monitoring System (Hilo band): wrist optical sensor, cloud pulse-wave analysis, readings only when still; described as the first FDA-cleared OTC cuffless BP monitor; already CE Class IIa; US launch planned 2026; >120,000 units sold — [Fierce Biotech](https://www.fiercebiotech.com/medtech/fda-clears-over-counter-cuffless-blood-pressure-monitor-hilo-bracelet); [MobiHealthNews](https://www.mobihealthnews.com/news/aktiia-gets-fda-clearance-otc-cuffless-blood-pressure-monitor). (Calibration requirement and 510(k) number not retrieved.)
- An independent EHJ Digital Health 2025 study "Validation of a popular consumer-grade cuffless blood pressure device for continuous 24 h monitoring" received an Expression of Concern (20 Nov 2025) after the manufacturer complained the authors misread the manual; PMC now lists it as RETRACTED; the notice does not name the device — [PMC12853114](https://pmc.ncbi.nlm.nih.gov/articles/PMC12853114/)

**Expert positions and standards**
- ESH Working Group statement (Stergiou, Mukkamala et al., J Hypertens 2022, DOI 10.1097/HJH.0000000000003224) and ESH validation recommendations (Stergiou et al., J Hypertens 2023;41:2074–2087, DOI 10.1097/HJH.0000000000003483): six tests for intermittent cuffless devices — static, device position, treatment, awake/asleep, exercise, recalibration — required according to each device's calibration and measurement features — [Europe PMC/PubMed 37303198](https://pubmed.ncbi.nlm.nih.gov/37303198/); [Ovid abstract](https://www.ovid.com/jnls/jhypertension/abstract/10.1097/hjh.0000000000003483~european-society-of-hypertension-recommendations-for-the)
- 2025 AHA/ACC high BP guideline recommends against cuffless devices for diagnosis or management until validation is adequate — [Hypertension 2025 guideline](https://www.ahajournals.org/doi/10.1161/HYP.0000000000000249). AHA scientific statement "Cuffless Devices for the Measurement of Blood Pressure" (11 Dec 2025): not proven accurate enough to diagnose or guide treatment; should not be used to start/titrate drugs; priorities are standardized validation, calibration stability, real-world utility, outcome links; FDA clearance does not require validation under an international protocol — [AHA professional summary](https://professional.heart.org/en/science-news/cuffless-devices-for-the-measurement-of-blood-pressure); [AHA newsroom](https://newsroom.heart.org/news/cuffless-blood-pressure-technologies-in-wearable-devices-show-promise-to-transform-care); [Medscape](https://www.medscape.com/viewarticle/aha-cuffless-devices-promising-not-ready-hypertension-2025a1000zxo)
- Scientific statement (Parati et al., Eur J Prev Cardiol 2026, DOI 10.1093/eurjpc/zwag058): categorises cuffless technologies and potential uses (out-of-office, nocturnal, hospital) but does not recommend them for clinical decisions — [PubMed search listing](https://pubmed.ncbi.nlm.nih.gov/?term=European+Society+of+Hypertension+recommendations+validation+cuffless+blood+pressure+measuring+devices). Korean Society of Hypertension classed cuffless monitors as Class IIb for out-of-office use, unlike AHA/ACC/ESC/ESH (Lee et al., Clin Hypertens 2026) — [DOI 10.5646/ch.2026.32.e40](https://doi.org/10.5646/ch.2026.32.e40)
- ISO 81060-3:2022 covers clinical investigation of continuous non-invasive automated sphygmomanometers, both "trending" and "absolute accuracy" devices; devices with output periods much longer than 30 s are generally not "continuous"; it provides no method for motion-artefact assessment; listed as "to be revised" (2025) — [ISO 71161](https://www.iso.org/standard/71161.html); [IEC webstore](https://webstore.iec.ch/publication/33607). The Hypertension 2026 commentary notes ISO protocols are not well suited to intermittent cuffless devices — [PMC12826266](https://pmc.ncbi.nlm.nih.gov/articles/PMC12826266/)
- A finger-PPG cuffless monitor met all ESH-recommendation criteria in 1,173 participants and its readings associated with hypertension-mediated organ damage (Qiu et al., JAHA 2026) — [DOI 10.1161/JAHA.126.050501](https://doi.org/10.1161/JAHA.126.050501)

**Pulse arrival time (ECG + PPG) and ML-from-PPG evidence**
- Finnegan et al. (Sci Rep 2021, 30 healthy volunteers, phenylephrine): compared population models and per-subject (a posteriori) models of BP from PTT/PAT; PEP rose only 5.5 ± 4.5 ms while PTT changes were much larger — [PMC8611024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8611024/). In ICU patients the overall PAT–BP correlation was only r ≈ −0.5 (SBP) / −0.42 (DBP) — [PMC6308183](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6308183/). Mukkamala et al. review of accuracy-evaluation challenges — [Hypertension 2021, DOI 10.1161/HYPERTENSIONAHA.121.17747](https://doi.org/10.1161/HYPERTENSIONAHA.121.17747)
- Open benchmark (González et al., Sci Data 2023): 11 ML methods on four public datasets with a scaled error (MASE) to compare against naive references; code at GitHub — [PMC10030661](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10030661/); [bp-benchmark repo](https://github.com/inventec-ai-center/bp-benchmark). On calibration-free VitalDB splits, the best deep models only reduced MAE ~17% vs baseline (MASE 0.83) — [arXiv 2502.19949](https://arxiv.org/pdf/2502.19949). A 2026 standardized benchmark (NBPDB, 101,453 segments, 1,103 adults from MIMIC-III/VitalDB) found no state-of-the-art PPG-BP model met AAMI/ISO 81060-2; adding age/sex/BMI cut error 23% (MInception MAE 4.75/2.90 mmHg) — [arXiv 2602.04725](https://arxiv.org/abs/2602.04725)

### Inferences
- **Hypertension screen (Apple-style) is the most realistic BP-adjacent target.** Apple's architecture — self-supervised PPG encoder + linear head + 30-day aggregation — maps directly onto open encoders (Pulse-PPG trained on WearOS watch PPG, PaPaGei, AnyPPG) with a linear probe. Labels would come from a home cuff over ≥15 days. Apple needed ~9,800 labelled people for a 41%/92% operating point, so a single-user or small-cohort version can at best track *within-person* change, not replicate a screening classifier.
- **Signal need:** morphology. Apple scores "qualified" (still, good-contact) PPG segments; BP information is largely in pulse shape, so our best inputs are the SDK's 100 Hz green/IR/red 30 s blocks; 25 Hz `hr_raw` may retain low-frequency shape (Nyquist 12.5 Hz) but loses dicrotic-notch timing. Long-term continuous data (weeks) are essential for the aggregation step.
- **Trend vs absolute:** every cleared numeric product (Samsung, Hilo) needs cuff calibration, and Samsung's readings regress toward the calibration point. Calibration-free absolute mmHg from PPG fails AAMI/ISO in public benchmarks. Target "change from personal baseline" (e.g., nocturnal vs daytime, week-over-week) and report ISO 81060-3 "trending" style metrics.
- **PAT with ECG + PPG is uniquely possible on our watch** via SDK ECG_ON_DEMAND (500 Hz ECG with co-sampled green PPG) in 30 s blocks; PAT = PEP + PTT, and PEP varies with sympathetic state/exercise, so it needs per-person calibration and posture control. Samsung's own calibrated BP reading (Samsung Health Monitor) can serve as a comparator, but itself is cuff-anchored.
- For evidence-strength: Apple HTNF = A (cleared, large pivotal) for a *screen with low sensitivity*; Samsung/Hilo = A/B for regulated calibrated devices but with independent studies showing poor tracking away from calibration; calibration-free ML BP = C.

### Gaps
- Apple's PPG sampling rate, segment length and what PPG "qualification" means were not disclosed in K250507.
- The Apple validation paper PDF (>10 MB) could not be parsed; per-subgroup risk ratios (Figure 1) not extracted.
- Hilo's 510(k) number, calibration scheme (initialization cuff?) and pivotal accuracy figures not retrieved.
- Whether Samsung Health Monitor BP is CE-marked and enabled on SM-R965F in Germany/EU was not verified.
- Full ESH 2023 test thresholds (e.g., required mean error/SD per test) not retrieved (paywalled).
- No peer-reviewed validation of Samsung wrist-ECG+PPG PAT for BP was found.

---

## 3. Diabetes, HbA1c and glucose (incl. CGM-paired research opportunity)

### Takeaway
Diabetes *risk* from PPG is real but modest (evidence B): Avram et al. 2020 (Nature Medicine) detected prevalent
diabetes from smartphone fingertip PPG with user-level AUC 0.766 (0.740 temporal validation; 0.830 with
age/sex/race/BMI), NPV 97% at 6.6% prevalence — and a model fed only beat intervals still got AUC 0.721.
There is no independent replication in peer-reviewed literature that I found; Huawei has deployed a PPG
"Diabetes Risk Study" (validation pending). Non-invasive glucose *values* from watches are not credible
(FDA safety communication, Feb 2024). The best evidence for CGM-paired work is small: Bent et al. 2021
(Empatica E4 + Dexcom G6, n=16; food logs and time-of-day dominate) and Lehmann et al. 2023 (hypoglycaemia
AUROC 0.76 from smartwatch data, n=22). Open CGM + wearable datasets exist (BIG IDEAs, PhysioCGM, CGMacros).

### Cited Findings

**Avram et al. 2020 (primary source, read from full text)**
- Avram R, Olgin JE, … Tison GH. "A digital biomarker of diabetes from smartphone-based vascular signals." Nat Med 2020;26:1576–1582, DOI 10.1038/s41591-020-1010-5, PMID 32807931 — [Nature Medicine](https://doi.org/10.1038/s41591-020-1010-5)
- Data: 53,870 Health eHeart participants contributing 2,589,448 fingertip PPG recordings (Azumio Instant Heart Rate iOS app), self-reported diabetes (6.6% prevalence); split 70/10/20; recordings limited to **100 or 120 Hz** (100 Hz upsampled to 120 Hz), detrended (0.4 Hz) and **low-pass filtered at ~10 Hz**; 39-layer CNN on 2,560 samples (~21 s) — [full text (gwern mirror)](https://www.gwern.net/doc/ai/nn/cnn/2020-avram.pdf)
- Results: test-set user-level AUC 0.766 (95% CI 0.750–0.782; recording-level 0.680); at threshold 0.427 sensitivity 75%, specificity 65%, PPV 13%, NPV 97%; contemporary cohort (7,806 newly enrolled, May–Dec 2018) AUC 0.740 (0.723–0.758); prospective clinic cohort of 181 patients (21% diabetic) recording-level AUC 0.682; among 21 newly HbA1c-diagnosed clinic patients AUC 0.644, 16/21 flagged; adding age, sex, race, BMI → AUC 0.830; DNN score independent of hypertension, hypercholesterolaemia, CAD; HRV lost significance once the DNN score was added — [full text](https://www.gwern.net/doc/ai/nn/cnn/2020-avram.pdf)
- Mechanism ablations: a DNN on a single cardiac-cycle waveform (morphology only) reached user-level AUC 0.691; a DNN on peak-to-peak intervals only (no morphology) reached 0.721; DNN score correlated with HbA1c (β = 2.28) and performance did not differ between HbA1c 7–8% and >8% — [full text](https://www.gwern.net/doc/ai/nn/cnn/2020-avram.pdf)
- No independent replication of the Avram approach was found by my searches; a 2021 conference paper ("Deep Learning based non-invasive diabetes predictor using PPG signals") exists but content was not reviewed — [ResearchGate listing via search](https://www.researchgate.net/publication/331612255_PREDICTING_DIABETES_FROM_PHOTOPLETHYSMOGRAPHY_USING_DEEP_LEARNING)

**Later large-scale PPG work (mostly conference/preprint, evidence C)**
- CinC 2025 abstract: PaPaGei foundation-model embeddings vs morphological PPG features for hypertension and type 2 diabetes in ~215,000 UK Biobank subjects (results not retrieved) — [CinC 2025 #362](https://cinc.org/2025/Program/accepted/362.html). Another CinC 2025 abstract on longitudinal PPG reports AUROC 0.833–0.888 for type 2 diabetes and 0.746–0.879 for hypertension depending on temporal-change features (dataset not verified) — [CinC 2025 #374](https://cinc.org/2025/Program/accepted/374.html)
- Wearable Behavior Foundation Model (Erturk, Kamran, Abbaspourazad, … Futoma; ICML 2025): 2.5 billion hours from 162,000 individuals, 57 health tasks; a search snippet of its table shows heart-failure AUROC 0.865 for PPG embeddings and 0.889 for behaviour+PPG (diabetes/hypertension rows not retrieved; affiliation not shown on the arXiv page) — [arXiv 2507.00191](https://arxiv.org/pdf/2507.00191)

**Consumer deployments and claims**
- Huawei "Diabetes Risk Study" (Watch GT 6 Pro, later GT 7/7 Pro): PPG-based, needs 3–14 days of wear, outputs low/medium/high risk, does not measure glucose; excludes type 1 and gestational diabetes; validation with Dubai Health in 150 participants (healthy/prediabetic/diabetic) against fasting glucose, HbA1c, OGTT was in progress, no published accuracy found — [Wareable](https://www.wareable.com/smartwatches/huawei-diabetes-risk-study-watch-gt-6-pro); [Huawei support](https://consumer.huawei.com/ph/support/content/en-us15953826/); [Wareable GT 7 update](https://www.wareable.com/smartwatches/huawei-watch-gt-7-series-diabetes-risk-study-update)
- FDA safety communication (21 Feb 2024): no smartwatch or smart ring that claims to measure/estimate blood glucose on its own without piercing the skin is authorized; wrong readings could cause dosing errors; does not apply to apps displaying data from authorized CGMs — [FDA](https://www.fda.gov/medical-devices/safety-communications/do-not-use-smartwatches-or-smart-rings-measure-blood-glucose-levels-fda-safety-communication)
- Samsung "AGEs Index" (Galaxy Watch7 and later): optical "indicator of metabolic health and biological ageing" from the BioActive sensor, measured during sleep; a wellness feature not to be used for diabetes treatment decisions; no independent validation found; reviewers reported near-identical values across users at launch — [Wareable explainer](https://www.wareable.com/features/samsung-galaxy-watch-ages-index-explained); [Samsung newsroom (AT)](https://news.samsung.com/at/praventive-gesundheitsfunktionen-mit-der-neuen-galaxy-watch-dank-bioactive-sensor). (Evidence D. Our Watch6 Classic is not listed as supported.)
- 2026 preprint: incremental-learning framework on a self-collected longitudinal benchmark of PPG + capillary glucose + cuff BP (183 participants, 285 days), subject-independent 5-fold MAE 0.64 mmol/L, 97.7% Clarke A+B; no time-of-day/subject-mean baseline reported — [arXiv 2607.04414](https://arxiv.org/abs/2607.04414)

**Work pairing CGM labels with wearable PPG/HRV features**
- Bent et al. (npj Digit Med 2021, DOI 10.1038/s41746-021-00465-w): 16 adults 35–65 y with high-normal HbA1c or prediabetes; Empatica E4 (PPG 64 Hz, EDA/temp 4 Hz, accel 32 Hz) + **Dexcom G6** for 8–10 days with food logs; 69 features incl. 8 IBI-derived HRV metrics; XGBoost glucose prediction RMSE ~21 mg/dL, MAPE 13–14% (population leave-one-person-out and personalized); personalized excursion classification accuracy 84%; feature importance: food 37%, activity 17%, circadian 10.6%, stress 8.2%; top single features were minutes-from-midnight, previous-24 h sugar intake, HbA1c; no Clarke grid — [PMC8172541](https://pmc.ncbi.nlm.nih.gov/articles/PMC8172541/). Data released on PhysioNet as "BIG IDEAs Lab Glycemic Variability and Wearable Device Data" (BVP, ACC, Dexcom CSVs per participant) — [PhysioNet](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/)
- Same group (BMJ Open Diabetes Res Care 2021): wrist wearables predicted HbA1c and glucose-variability metrics in the same 16 people; heart rate ranked last of four sensors (after skin temperature, EDA, accelerometry) for HbA1c — [DOAJ](https://doaj.org/article/ddf3ede38fa540469fa43f66d7630855); [Duke](https://comphealth.duke.edu/publications/non-invasive-wearables-for-remote-monitoring-of-hba1c-and-glucose-variability-proof-of-concept)
- Lehmann et al. (Diabetes Care 2023;46:993–997): Garmin vivoactive 4S + Empatica E4 with CGM in insulin-treated adults; ML hypoglycaemia (<3.9 mmol/L) detection from wearable data only, AUROC 0.76 ± 0.07 in 22 people (leave-person-out); hypoglycaemia associated with higher HR, lower HRV, higher tonic EDA — [Bern repository](https://boris.unibe.ch/179063/); [St. Gallen](https://www.alexandria.unisg.ch/259515/); [PracticeUpdate](https://www.practiceupdate.com/content/noninvasive-hypoglycemia-detection-in-people-with-diabetes-using-smartwatch-data/148903/45)
- PhysioCGM dataset: raw ECG, PPG (Empatica E4), EDA, skin temperature, accelerometry with Dexcom G6 ground truth in 10 people with type 1 diabetes — [PMC12630648](https://pmc.ncbi.nlm.nih.gov/articles/PMC12630648/)
- CGMacros (PhysioNet, Jan 2025): 45 participants (15 healthy, 16 prediabetes, 14 T2D), 10 free-living days, Dexcom G6 Pro + Libre Pro, Fitbit Sense minute HR/METs, meal photos and macronutrients — [PhysioNet](https://physionet.org/content/cgmacros/); [PMC12462512](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12462512/)
- HRV and acute glucose (25 healthy volunteers, fingerstick + 10-min ECG): fasting LF power and total power correlated negatively with the fasting-to-postprandial glucose rise — [UTS repository](https://opus.cloud1.lib.uts.edu.au/handle/10453/154328)

### Inferences
- **Diabetes risk score: feasible on phone and probably on watch.** Avram's input band (<10 Hz, ~21 s) is representable at 25–30 Hz sampling in principle, but they excluded <100 Hz recordings, so our 30 fps phone PPG is out of their distribution (would need retraining); the intervals-only AUC of 0.721 suggests much of the signal is resting HR/HRV/autonomic, which our 25 Hz watch stream measures well. Expect AUC ~0.70–0.77 at best and low PPV; useful only as a "get an HbA1c test" nudge. No public Avram weights were identified.
- **Glucose values: do not attempt as a product.** Treat any PPG→mg/dL claim as hype unless it beats time-of-day + last-meal + persistence baselines in subject-independent validation (Bent's top features were clock time and sugar intake; the 2026 arXiv paper reports no such baseline).
- **CGM research opportunity (user's Dexcom):** an N-of-1 / small-cohort study pairing Dexcom 5-min glucose with (i) continuous 25 Hz `hr_raw` beat intervals → HR, RMSSD, LF/HF-like indices, (ii) skin temperature and accelerometer (SDK), and (iii) scheduled SDK 30 s 100 Hz PPG morphology blocks pre-/post-meal and overnight. Most tractable targets, in order: postprandial excursion detection (relative change), nocturnal hypoglycaemia-like autonomic signatures (Lehmann-style), and day-level glycaemic variability. Must include baselines (time-of-day, meal log, glucose persistence) and leave-day-out validation. Public datasets (BIG IDEAs, CGMacros, PhysioCGM) give pretraining/benchmark data, though none uses Samsung PPG.
- Evidence strength: diabetes risk B (one large study + validation cohorts, no replication); hypoglycaemia detection C; PPG→glucose value C/D; AGEs/“metabolic” indices D.

### Gaps
- No peer-reviewed independent replication of Avram 2020 found; no published accuracy for Huawei's Diabetes Risk Study.
- AUROC values of the UK Biobank PaPaGei diabetes/hypertension abstract and the WBM diabetes/hypertension rows were not retrieved (CinC pages returned 403).
- D1NAMO / OhioT1DM dataset contents were not verified in this pass.
- How Dexcom data can be exported/streamed for research (Dexcom developer API, Clarity export) was not researched.

---

## 4. Heart failure / low ejection fraction, valvular and coronary disease, cardiovascular event prediction

### Takeaway
Low-EF (LVEF ≤40%) detection from a single-lead *watch ECG* has prospective evidence (Mayo, Apple Watch,
Nat Med 2022, AUC ~0.88, but only 16 cases) and one regulatory approval (Samsung + Medical AI "AiTiA
LVSD-1L", Korea MFDS 2025, Galaxy Watch8/Ultra). PPG-based LV-dysfunction detection is still small-sample
(camera rPPG AUC 0.9; PPG features AUC 0.79; wearable PPG for obstructive HCM in 19 patients).
Coronary disease: multi-position watch ECG can show ST changes (feasibility only). CV event risk from
fingertip PPG + age/sex/smoking matched an office-based score in UK Biobank (C 0.711). No wearable
valvular-disease evidence was found.

### Cited Findings
- Mayo Clinic (Attia et al., "Prospective evaluation of smartwatch-enabled detection of left ventricular dysfunction", Nat Med 2022;28:2497–2503, DOI 10.1038/s41591-022-02053-1): 2,454 patients from 46 US states and 11 countries sent 125,610 Apple Watch ECGs (Aug 2021–Feb 2022); 421 had a sinus-rhythm watch ECG within 30 days of an echocardiogram, 16 (3.8%) with EF ≤40%; AUC 0.881–0.885 (single nearest ECG) and ~0.885–0.89 (mean of ECGs); at threshold 0.67 mean-prediction sensitivity 68.8% / specificity 83.7%, at 0.6 sensitivity 87.5% / specificity 80.7%; the model was **retrained on lead I only** from the 12-lead derivation set, bandpass-filtered 0.2–25 Hz to mimic watch morphology, using median beats; limitations: HFpEF not targeted, few AF cases, Mayo-app patients only, proprietary code, licensing conflicts (Anumana) — [PMC9805528](https://pmc.ncbi.nlm.nih.gov/articles/PMC9805528/); [Mayo News Network](https://newsnetwork.mayoclinic.org/discussion/ai-transforms-smartwatch-ecg-signals-into-a-diagnostic-tool-for-heart-failure/); [EurekAlert](https://www.eurekalert.org/news-releases/951446)
- Samsung + Medical AI (Korea): LVSD detection from Galaxy Watch ECG via the AiTiA LVSD-1L app approved by Korea's MFDS; validated against 12-lead analysis; offered on Galaxy Watch8 and Watch Ultra in Korea; no FDA clearance found (Oct 2025 reports) — [SamMobile](https://www.sammobile.com/news/samsung-bringing-potentially-life-saving-feature-galaxy-watches/); [Notebookcheck](https://www.notebookcheck.net/Samsung-s-smartwatch-gains-regulatory-approval-for-LVSD-detection-while-Brainwave-EEG-stays-in-prototype-phase.1133136.0.html); [Android Authority](https://www.androidauthority.com/samsung-galaxy-watch-lvsd-ecg-3602963)
- Smartwatch-ECG AI pilots for LVSD in real-world settings and for post-discharge HF rehospitalisation exist but are small; validation is limited by scarce self-recorded watch ECGs — [Circulation 2025 abstract 4368006](https://www.ahajournals.org/doi/10.1161/circ.152.suppl_3.4368006); [JACC: Basic Transl Sci 2025 feasibility study](https://www.sciencedirect.com/science/article/pii/S2452302X25000142)
- Single-lead AI-ECG predicted LVEF ≤40% in 33.7% of a cohort and those had higher MACE rates, suggesting prognostic value beyond EF — [PMC12282343](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12282343/)
- PPG for LV dysfunction: contactless camera rPPG AI identified HF with LVSD with AUC 0.90 (accuracy 89.7%) — [J Cardiol 2025 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0914508725002643); PPG-derived features (LASSO) AUC 0.79 ± 0.06 in a small observational study — [EHJ Digital Health 2025 suppl.](https://academic.oup.com/ehjdh/article/7/Supplement_1/ztaf143.113/8422872); wearable-PPG ML detected obstructive HCM with C-statistic 0.99 but in only 19 patients vs 64 controls — [npj Digit Med 2019](https://www.nature.com/articles/s41746-019-0130-0)
- Behaviour + PPG foundation-model embeddings: heart-failure AUROC ~0.865 (PPG) and ~0.889 (behaviour+PPG) vs ~0.824 baseline (search-snippet values; unverified table) — [arXiv 2507.00191](https://arxiv.org/pdf/2507.00191)
- Coronary: a smartwatch moved to 9 body positions (I, II, III, V1–V6) recorded multichannel ECGs agreeing with standard 12-lead in 100 participants (54 STEMI, 27 NSTEMI, 19 controls); reported sensitivity 93%/specificity 95% for ST-elevation — [Spaccarotella et al., JAMA Cardiol 2020, PMC7466842](https://pmc.ncbi.nlm.nih.gov/articles/PMC7466842); [ACC journal scan](https://acc.org/latest-in-cardiology/journal-scans/2020/09/02/13/05/multichannel-ecgs-obtained-by-a-smartwatch-esc-2020)
- CV event risk: Google's PPG deep-learning score (UK Biobank fingertip PPG + age, sex, smoking) predicted 10-year MACE with C-statistic 71.1% vs 70.9% for a refit office-based WHO score (which uses BP, height, weight); adding DL features to the office score raised C by 1.0%; calibration error ~1.8% — [Weng, Baur et al., PLOS Glob Public Health 2024, PMC11149850](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11149850/); [Google Research blog](https://research.google/blog/a-step-towards-making-heart-health-screening-accessible-for-billions-with-ppg-signals/)
- Apple Heart & Movement Study "PpgAge" (~213,593 participants): a 6-year PPG-age gap carried HR 1.40 for adverse cardiac events (event listing; vascular-ageing details belong to the other researcher) — [CityU event page](https://www.cityu.edu.hk/ma/news-events/event/2025/wearable-based-aging-clock-associates-disease-and-behavior)

### Inferences
- **Low EF: needs ECG, not PPG.** Our SDK ECG (500 Hz, 30 s, lead-I-like) is the right input; Mayo's recipe (lead I, 0.2–25 Hz bandpass, median beat, retrain from a 12-lead corpus) can be reproduced with open 12-lead datasets that have echo labels, or by fine-tuning ECGFounder's 1-lead checkpoint — but no open, validated single-lead LVSD model was found, so this is a training project, not a download. Watch6 Classic is not in Samsung's LVSD rollout.
- PPG-based HF detection is evidence C; foundation-model embeddings on long-term PPG may carry HF signal (WBM numbers), which supports logging embeddings for drift against a personal baseline rather than diagnosing.
- CV event risk scoring from short fingertip PPG is reproducible in principle (Google model used UK Biobank finger PPG at rest), but labels require outcome cohorts — not feasible to validate ourselves; only the published UK Biobank approach could be re-implemented with UK Biobank access.
- Coronary/ischaemia: single-lead watch ECG cannot replace 12-lead; multi-position recording is a manual research curiosity.

### Gaps
- No wearable PPG or single-lead ECG evidence for valvular disease (e.g., aortic stenosis) was found.
- Yale/Khera-lab single-lead "noise-adapted" LVSD models and any open-weight single-lead LVEF model were not verified.
- Medical AI's AiTiA LVSD-1L accuracy figures on Galaxy Watch were not found.
- Whether ECGFounder's label set includes any LV-function label was not checked.

---

## 5. Other blood or cardiometabolic markers claimed from PPG/camera/ECG (haemoglobin, lipids, kidney, potassium)

### Takeaway
Credible: hyperkalaemia from ECG (12-lead/2-lead AI, AUC 0.85–0.90; single-lead/smartwatch AUC 0.81–0.88 in
preprints) and CKD from ECG (12-lead AUC 0.77; 1-lead 0.82 in <60 y). Plausible but weak: haemoglobin from
phone camera (HemaApp n=31; fingernail-photo app degrades in field use to ±2–4 g/dL). Not credible: lipids/
cholesterol and kidney function from PPG — no supporting studies found.

### Cited Findings
- Hyperkalaemia, 12-lead-derived: Galloway et al. (JAMA Cardiol 2019;4:428–436) trained on 1,576,581 ECGs from 449,380 Mayo patients, validated in 61,965 CKD stage ≥3 patients; using only leads I and II AUC 0.853–0.883 (4 leads up to 0.901), sensitivity 88.9–91.3%, specificity 55–63.2% — [PMC6537816](https://pmc.ncbi.nlm.nih.gov/articles/PMC6537816)
- Hyperkalaemia, single-lead/smartwatch: Kardio-Net (Cedars-Sinai/Stanford, medRxiv 2024) AUC 0.876 on single-lead ECGs and 0.831 on prospective smartwatch ECGs from 40 ESRD patients (Taiwan), MAE ~0.58 mEq/L — [medRxiv 2024.05.08.24307064](https://www.medrxiv.org/content/10.1101/2024.05.08.24307064v1.full.pdf). Pocket-K (arXiv 2026): lead-I model on an ECG foundation model, 34,439 patients / 62,290 ECG–K⁺ pairs, AUROC 0.936 internal, 0.858 temporal, 0.808 external; K⁺ >5.5 mmol/L — [arXiv 2603.14177](https://arxiv.org/abs/2603.14177)
- CKD from ECG: Holmstrom et al. (Communications Medicine 2023): 12-lead AUC 0.767 held-out, 0.709 external; in patients <60 y AUC 0.843 (12-lead) and 0.824 (1-lead) — [Commun Med](https://link.springer.com/10.1038/s43856-023-00278-w)
- Haemoglobin: HemaApp (UW) shines light through the finger with the phone camera/flash; in 31 patients rank-order correlation 0.82 with lab Hb, 85.7% sensitivity and 76.5% precision for anaemia — [UbiComp Lab](https://ubicomplab.cs.washington.edu/publications/hemaapp). Fingernail-photo app (Nat Commun 2018): ±2.4 g/dL in 100-person validation, ±0.92 g/dL with personal calibration in 16 people — [HealthDay](https://www.healthday.com/healthpro-news/blood-disorders/smartphone-app-uses-fingernail-bed-images-to-detect-anemia-740281.html); later real-world evaluation preprint — [medRxiv 2022](https://www.medrxiv.org/content/10.1101/2022.12.01.22282972.full.pdf); independent field test in rural Bihar vs HemoCue: ±4.43 g/dL (adults) and ±3.54 g/dL (children), ±2.25 g/dL after retraining — screening-only accuracy — [Göttingen repository](https://publications.goettingen-research-online.de/ID-124159)
- Lipids/cholesterol: my searches found no study showing PPG or smartphone camera predicts cholesterol/lipid levels; the Google PPG CVD-risk model did not use cholesterol, and its blog only claims similar risk discrimination to screenings that use cholesterol — [PMC11149850](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11149850/); [Google blog](https://research.google/blog/a-step-towards-making-heart-health-screening-accessible-for-billions-with-ppg-signals/)
- Avram's diabetes DNN score stayed predictive after adjusting for hypercholesterolaemia (i.e., PPG captures something beyond lipid status, not lipid levels) — [Avram full text](https://www.gwern.net/doc/ai/nn/cnn/2020-avram.pdf)

### Inferences
- **Potassium/CKD need ECG:** feasible research target on our SDK 30 s ECG (lead-I-like) by fine-tuning an ECG foundation model (Pocket-K builds on one); validation needs paired lab K⁺, so for a single healthy user it is only a sanity/plumbing exercise. Expect lower accuracy than 12-lead.
- **Haemoglobin:** the watch's SDK 100 Hz green/IR/red PPG enables multi-wavelength ratio features (as in SpO2), conceptually close to HemaApp; the phone's white-flash RGB camera is weaker. Needs lab Hb labels; evidence C.
- **Lipids, kidney function from PPG:** treat as hype; no credible evidence.

### Gaps
- KardiaMobile/AliveCor K⁺ product status and any FDA clearance for ECG-based potassium were not retrieved.
- Masimo SpHb and other multi-wavelength Hb accuracy literature not reviewed.
- No search for PPG-based CKD/eGFR studies returned results (absence noted, not exhaustively proven).

---

## 6. Signal requirements per method and feasibility on our devices

### Takeaway
Rhythm-type targets (AF, diabetes-risk-via-autonomic, hypoglycaemia autonomic signatures) need only beat
timing and run on the 25 Hz continuous stream; BP/hypertension and morphology-based risk scores need
≥100 Hz on-demand blocks plus weeks of data; LV dysfunction, potassium and CKD need the 500 Hz watch ECG;
Hb needs multi-wavelength PPG; lipid and glucose-value claims should be dropped.

### Cited Findings
- Cleared AF notifications use tachograms of inter-beat intervals (Apple: 5/6 irregular in 48 h; Fitbit: 11/11 consecutive irregular tachograms, ≥30 min) — [Apple arrhythmia document](https://www.apple.com/uk/healthcare/docs/site/Apple_Watch_Arrhythmia_Detection.pdf); [Fitbit Heart Study, PMC9640290](https://pmc.ncbi.nlm.nih.gov/articles/9640290)
- AF models on 20 Hz (Samsung wrist) and 25 Hz (resampled) wrist PPG reach AUC 0.95–0.97 — [Stanford ML Group](https://stanfordmlgroup.github.io/projects/ppg/); [JACC EP 2024](https://www.jacc.org/doi/10.1016/j.jacep.2024.01.008)
- Apple HTNF scores qualified PPG signals and aggregates over 30 days (≥15 days of data in validation) — [FDA K250507](https://www.accessdata.fda.gov/cdrh_docs/pdf25/K250507.pdf)
- Avram's diabetes model used 100–120 Hz fingertip PPG low-passed at ~10 Hz, ~21 s; intervals-only variant AUC 0.721 vs full 0.766 — [Avram full text](https://www.gwern.net/doc/ai/nn/cnn/2020-avram.pdf)
- Mayo low-EF model: lead I, 0.2–25 Hz bandpass, median beat — [PMC9805528](https://pmc.ncbi.nlm.nih.gov/articles/PMC9805528/)
- Hyperkalaemia/CKD models: lead I or leads I+II ECG — [Galloway, PMC6537816](https://pmc.ncbi.nlm.nih.gov/articles/PMC6537816); [Pocket-K](https://arxiv.org/abs/2603.14177); [Holmstrom](https://link.springer.com/10.1038/s43856-023-00278-w)

### Inferences
Feasibility matrix (our devices; "no-SDK" = `hr_raw` ~24.9 Hz continuous + 1 Hz HR + 100 Hz accel; "SDK" =
25 Hz green + Samsung IBIs + skin temp + on-demand ≤30 s blocks of 100 Hz PPG G/IR/R, 500 Hz ECG + co-sampled
green PPG, SpO2, BIA):

| Target | Evidence | Signal needed | No-SDK watch | SDK watch | Phone camera | Notes |
|---|---|---|---|---|---|---|
| AF screen | A | beat timing, minutes, persistence over hours | yes | yes (+ Samsung IBIs) | yes, 30 fps spot checks (FibriCheck-like) | needs motion/quality gating; ectopy → false positives |
| AF confirmation | A | ≥30 s single-lead ECG | no | yes (ECG 500 Hz) | no | physician review per ESC |
| Hypertension risk flag (Apple-style) | A (Apple), C for us | qualified resting PPG morphology, weeks | weak (25 Hz shape only) | yes (100 Hz blocks, many per day) | possible but no evidence | needs home-cuff labels; linear probe on PPG embeddings |
| BP trend (calibrated) | B/C | PWA morphology or PAT (ECG+PPG), cuff calibration | weak | yes (PAT in ECG blocks) | no | report as trend vs baseline; ISO 81060-3 "trending" framing |
| Absolute cuffless BP (no calibration) | C (fails AAMI/ISO) | — | no | no | no | do not pursue as output |
| Diabetes risk score | B | ~20 s PPG at ≥100 Hz (Avram) or beat intervals (AUC 0.72) | intervals only | yes | needs retraining (30 fps < Avram's 100 Hz) | low PPV; nudge to HbA1c |
| Glucose value | C/D | — | no | no | no | FDA warning; drop |
| Glycaemic excursion / hypoglycaemia (with Dexcom labels) | C | HR/HRV + temp + motion, continuous | yes | yes (+temp, morphology blocks) | no | research only; beat time-of-day/meal baselines |
| Low EF (LVSD) | B (ECG); C (PPG) | 30 s lead-I ECG | no | yes (needs model training) | rPPG AUC 0.9 small study | no open validated model found |
| Hyperkalaemia / CKD | B (12-lead), C (single-lead) | lead-I ECG | no | yes (plumbing only without labs) | no | requires lab labels |
| Haemoglobin / anaemia | C | multi-wavelength PPG | no | yes (G/IR/R 100 Hz) | weak (RGB + flash) | requires lab Hb labels |
| CV event risk (UK Biobank-style) | B | resting fingertip PPG + age/sex/smoking | — | possible | possible | cannot validate without outcome cohort |
| Lipids, kidney from PPG | none | — | — | — | — | hype |

- Long-term continuous data are the differentiator for our setup: Apple HTNF (30-day aggregation), Fitbit/Apple AF (persistence rules) and PPG foundation models (WBM, PpgAge) all gain from many recordings per person; Avram also found best performance with >6 recordings per user — [Avram full text](https://www.gwern.net/doc/ai/nn/cnn/2020-avram.pdf).

### Gaps
- Exact anti-aliasing/bandwidth of Samsung's `hr_raw` 25 Hz channels is unknown; this determines how much waveform shape survives for morphology-based models.
- The co-sampled PPG rate inside ECG_ON_DEMAND (per ECG sample at 500 Hz vs lower) should be confirmed from the SDK before relying on it for PAT.
</content>
</invoke>
