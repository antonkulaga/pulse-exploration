# Wrist PPG + accelerometer + SpO2 + skin temperature: sleep, breathing, oxygenation, stress, illness, recovery and fitness

Researched 2026-10-24 for the Galaxy Watch6 Classic (SM-R965F) prototype. Data tiers used below:

- **Tier A (no SDK):** `hr_raw` green PPG at about 25 Hz, continuous; 1 Hz HR; 100 Hz accelerometer and gyroscope.
- **Tier B (Samsung Health Sensor SDK, developer mode):**
  - Continuous: 25 Hz green PPG, Samsung's beat-interval (IBI) list, skin temperature.
  - On demand, at most 30 s per block: 100 Hz PPG (green/IR/red), 500 Hz ECG, SpO2, BIA.
- **Tier C (Samsung Health export):** HR, HRV, SpO2, skin temperature, stress score, sleep stages, and probably overnight RRI.
- **Tier D (closed Samsung features):** for example the Samsung Health Monitor sleep apnea feature. The result is usable, but the method cannot be reproduced.

Scope excludes HRV accuracy (already benchmarked: about 3 ms RMSSD error at rest, nothing usable during walking or running). It also excludes the aging/vascular, cardiometabolic, camera-PPG and foundation-model topics, which other researchers cover.

## Respiratory rate from PPG: accuracy at rest and in sleep, and is 25 Hz wrist PPG enough?

### Takeaway
Overnight respiratory rate (RR) from wrist PPG is solved well enough for trend tracking.

- **Samsung's own algorithm** uses 25 Hz green PPG plus 25 Hz accelerometer, which is exactly our Tier A. Against a PSG thermocouple it reaches RMSE 1.13 brpm for the overnight average and 1.62 brpm for 1-min values. Accuracy degrades in severe apnea.
- **Fitbit's IBI/RSA method** restricted to deep sleep reaches MAE 0.46 brpm.
- **Awake, at rest, spot RR from PPG** is much worse. The best of 314 published algorithms has 95% limits of agreement of about ±6 brpm even under ideal conditions.

25 Hz is sufficient for overnight RR. Breathing sits at 0.1–0.5 Hz, and Samsung's validated product runs at 25 Hz.

### Cited Findings
- **Samsung Galaxy Watch4/5 nocturnal RR (Jung, Kim, Choi, Joo, *Sensors* 2023, DOI 10.3390/s23187976)** — [PMC10536355](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536355/)
  - Signals: green PPG (525 nm) at **25 Hz** plus 3-axis accelerometer at **25 Hz**, worn on the non-dominant wrist.
  - Method (classical, not deep learning): three PPG respiratory modulations (RIFV frequency, RIAV amplitude, RIIV intensity/baseline) plus three accelerometer axes.
    - PPG band-pass 0.1–1.5 Hz with slope-based beat detection; accelerometer band-pass 0.1–1 Hz.
    - Lomb–Scargle spectra are fused, the dominant peak is tracked, and the output updates every minute.
  - Reference: PSG nasal thermocouple; 195 sleep-clinic patients (31 normal, 46 mild, 45 moderate, 73 severe OSA).
  - Overnight average: RMSE 1.13 brpm, bias 0.39, LoA −1.68 to 2.46, 91.8% within ±2 brpm.
  - Continuous 1-min values: RMSE 1.62, bias 0.37, LoA −2.73 to 3.47, 86.7% within ±2 brpm.
  - With AHI < 30: overnight RMSE 0.46, 1-min RMSE 1.22.
  - Severe OSA: overnight RMSE 1.74, 79.5% within ±2 brpm; continuous RMSE 2.17, 75.8% within ±2 brpm. Results by sleep stage were not reported.
- **Fitbit RR (Natarajan, Su, Heneghan et al., *npj Digital Medicine* 2021;4:136, DOI 10.1038/s41746-021-00493-6)** — [PMC8443549](https://pmc.ncbi.nlm.nih.gov/articles/PMC8443549/)
  - Method: RR inferred from respiratory sinus arrhythmia in PPG inter-beat intervals. Power spectra of 5-min segments in **deep sleep** (light sleep if deep-sleep SNR < 2.5, REM ignored) are averaged over the night; range 10–26 brpm.
  - Validation: 28 participants (52 nights) against PSG or home-test airflow, with AHI ≥ 30 excluded. r = 0.95, bias −0.24 brpm, RMSE 0.65, MAE 0.46, MAPE 3.0%.
  - Population norm (n = 10,000): mean nocturnal RR 15.4 brpm (SD 2.35); 90% of values fall in 11.8–19.2.
  - All authors were Fitbit employees and the data are not open.
- **Awake / ideal-condition benchmark (Charlton et al., *Physiol Meas* 2016, 314 algorithms, VORTAL dataset)** — [PMC5390977](https://pmc.ncbi.nlm.nih.gov/articles/PMC5390977/)
  - Population: 39 healthy adults aged 18–40, 500 Hz signals, at rest.
  - Best PPG algorithm: bias 1.0 brpm, 95% LoA −5.1 to 7.2 (2SD 6.2), and it produced an estimate for only 54% of windows.
  - Best ECG algorithm: LoA ±4.7. Impedance pneumography: LoA −5.6 to 5.2.
  - 63.6% of algorithms were more precise with ECG than with PPG.
  - All top algorithms used time-domain breath detection, and most used fusion of BW/AM/FM modulations. Frequency-domain method EF4 performed poorly.
  - The authors warn the results may not extrapolate to elderly or unwell people.
- **Open code:** Charlton's RRest toolbox (MATLAB/GitHub) implements the extraction, estimation and fusion stages — [MathWorks File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/55289-respiratory-rate-estimation)
- **Sampling-rate evidence**
  - A 2022 review summarising Choi et al. found no differences in PRV indices above 25 Hz, but that work did not test interpolation or protocol effects. Cubic-spline interpolation restores HRV accuracy at 20–50 Hz — [Sensors 22(4):1428 review](https://www.mdpi.com/1424-8220/22/4/1428/htm)
  - A CinC 2021 RR-from-PPG method downsampled to 25 Hz to save computation — [CinC 2021 preprint](https://www.cinc.org/2021/Program/accepted/15_Preprint.pdf)
  - No study directly sets a minimum sampling rate for PPG-derived RR (same review).
- **Apple's approach:** Apple's breathing analysis during sleep uses the wrist **accelerometer**, which captures breathing micro-motion, not PPG — [Apple white paper, Sept 2024](https://apple.com/health/pdf/sleep-apnea/Sleep_Apnea_Notifications_on_Apple_Watch_September_2024.pdf)

### Inferences
- **Tier A is sufficient for overnight RR.** Samsung's validated pipeline uses the same 25 Hz green PPG and an accelerometer that we have at 100 Hz. Our 25 Hz `hr_raw` should support a re-implementation:
  - band-pass 0.1–1.5 Hz;
  - RIIV/RIAV/RIFV plus accelerometer;
  - Lomb–Scargle fusion, or Charlton-style time-domain breath detection.
- The 16 `hr_raw` channels still need decoding to find a clean green channel.
- **Tier B (SDK IBI list) supports the Fitbit-style RSA method** in deep sleep. Tier C (export RRI) may allow it retrospectively.
- **Nightly median RR is the robust product.** Expect about 1 brpm error for the overnight value but ±3 brpm per minute. Severe OSA inflates error, so RR and apnea estimation interact.
- **Daytime or awake spot RR is not worth promising.** Even research-grade PPG has ±6 brpm LoA at rest, and our benchmark shows nothing works during movement.

### Gaps
- No wrist-PPG study reports RR error by sleep stage, or for awake, seated spot checks on a watch. Charlton's numbers are for finger/ear PPG at 500 Hz.
- Whether the 100 Hz on-demand SDK PPG improves RR over 25 Hz is untested. In principle it should not matter for 0.1–0.5 Hz content.
- The Samsung export's sleep RR field format and granularity were not verified.
- No independent replication of Samsung's or Fitbit's RR algorithms was found.

## Sleep staging from PPG/HRV + accelerometer vs polysomnography

### Takeaway
Two tiers of evidence differ sharply:

- **Commercial watches**, including Samsung, reach only fair-to-moderate four-stage agreement with PSG: κ ≈ 0.30–0.53, accuracy ≈ 56–65%. Galaxy Watch5: κ 0.42.
- **Open deep models on raw PSG-finger PPG** reach κ ≈ 0.74–0.79 on NSRR cohorts: SleepPPG-Net, SleepPPG-Net2 and wav2sleep, the last with open weights.
- **Externally**, on a wrist oximeter, κ is about 0.66–0.67.

Models that use only beat intervals lose roughly 0.05–0.08 κ compared with raw-waveform models. 25 Hz PPG resampled to the 34 Hz these models use is plausible, but wrist-green-PPG domain shift is the real risk.

### Cited Findings
- **SleepPPG-Net (Kotzen, Charlton, Salabi, Amar, Landesberg, Behar; *IEEE JBHI* 2023, 27:924–932)** — [arXiv 2202.05735](https://arxiv.org/pdf/2202.05735)
  - Residual CNN plus temporal CNN on raw PPG, 4 classes (wake/light/deep/REM).
  - MESA held-out test (n = 204): κ 0.75.
  - Benchmarks on the same split: κ 0.66 for feature engineering (BM-FE) and 0.69 for a derived-time-series beat-interval model (BM-DTS).
  - External database after transfer learning: κ 0.74. Pre-training used SHHS ECG.
- **SleepPPG-Net2 (Attia, …, Behar, arXiv 2404.06869, 2024)** — [arXiv](https://arxiv.org/html/2404.06869v1)
  - Adds a domain-shift-uncertainty layer; PPG downsampled to **34.13 Hz**; leave-one-dataset-out evaluation.
  - External κ: CFS 0.73, ABC 0.68, HomePAP 0.68, CAP 0.54, SLEEPAI 0.67. MESA source test: κ 0.74, accuracy 84%.
  - **SLEEPAI used a wrist-worn oximeter: κ 0.66–0.67, accuracy 79%.**
  - Lower κ with older age, male sex and higher AHI. REM is misclassified as wake in REM sleep behaviour disorder.
  - The paper does not state whether code or weights are released.
- **wav2sleep (Carter & Tarassenko, PMLR 259 / ML4H 2025)** — [arXiv 2411.04644](https://arxiv.org/html/2411.04644v1)
  - One model trained on more than 10,000 NSRR nights. Each 30-s epoch is resampled to 1024 points (about 34 Hz).
  - PPG-only κ: MESA 0.742, CHAT 0.793, CFS 0.763, CCSHS 0.832. SleepPPG-Net on the same splits: 0.713, 0.757, 0.731, 0.803.
  - ECG-only κ on SHHS: 0.74 (accuracy 82.3%).
  - **Beat-interval-only baseline:** Sridhar et al. 2020 (instantaneous heart rate) reaches κ 0.66 (accuracy 77%) on SHHS.
  - Code, weights and NSRR pipelines are at github.com/joncarter1/wav2sleep; no licence is stated in the paper.
  - The NSRR blog also covers it — [sleepdata.org](https://sleepdata.org/blog/2025/06/wav2sleep-a-unified-multi-modal-approach-to-sleep-stage-classification-from-physiological-signals)
- **Lee et al. 2023, 11 consumer trackers vs PSG (JMIR mHealth uHealth 11:e50983, DOI 10.2196/50983)** — [PMC10654909](https://pmc.ncbi.nlm.nih.gov/articles/PMC10654909/)
  - 75 Korean adults with sleep complaints, four-stage results:

    | Device | n | Accuracy | κ | Macro F1 |
    |---|---|---|---|---|
    | Galaxy Watch5 | 22 | 64.9% | 0.418 | 0.576 |
    | Fitbit Sense 2 | — | 64.6% | 0.419 | 0.581 |
    | Pixel Watch | — | 63.6% | 0.404 | — |
    | Oura Ring 3 | — | 54.3% | 0.349 | — |
    | Apple Watch 8 | — | 56.4% | 0.298 | — |

  - Galaxy Watch5 per stage: deep F1 0.50 (sensitivity 0.48), REM F1 0.60, wake sensitivity 0.48. It had the smallest sleep-efficiency bias (−0.4%).
- **Schyvens et al. 2025, *SLEEP Advances* 6(2):zpaf021** — [OUP](https://academic.oup.com/sleepadvances/article/6/2/zpaf021/8090472)
  - Six wrist devices, 62 adults, one PSG night. Four-stage κ: Apple Watch Series 8 0.53, Fitbit Sense 0.42, Fitbit Charge 5 0.41.
  - Garmin Vivosmart 4 κ 0.21 and wake detection of 29–52% come from a secondary source and are unverified.
  - No Samsung device was tested.
- **Galaxy Watch3 vs PSG (Kim, Joo, Choi, *J Sleep Med* 2023, DOI 10.13078/jsm.230004)** — [e-jsm.org](https://www.e-jsm.org/journal/view.php?number=358)
  - 32 adults. Four-stage accuracy 0.651; sleep sensitivity 0.954; wake specificity 0.524.
  - Per-subject κ: wake 0.47, light 0.34, deep 0.34, REM 0.39.
  - REM overestimated by 26 min (ICC 0.15).
- **Huawei Watch GT2 in a sleep clinic (Guo et al., *PLoS One* 2025, DOI 10.1371/journal.pone.0330774)** — [PMC12483222](https://pmc.ncbi.nlm.nih.gov/articles/PMC12483222/)
  - Per-stage κ: wake 0.43, light 0.12, deep 0.32, REM 0.12.
  - TST overestimated by 28 min, WASO underestimated by 37 min.
  - Worse in OSA and insomnia: an example of clinical-population degradation.
- **DREAMT (Duke)** — [PhysioNet](https://physionet.org/content/dreamt/)
  - 100 sleep-clinic participants wearing an Empatica E4 (BVP 64 Hz, ACC 32 Hz, EDA, TEMP) with simultaneous PSG stages and apnea events. Credentialed access; v2.2.0.
  - HiMAE uses it for 4-class staging with subject-independent 5-fold CV. Deep sleep is only about 3% of epochs — [arXiv 2510.25785](https://arxiv.org/pdf/2510.25785)
- **Other open tools and data**
  - SleepECG: Python, heartbeat-time-based staging with classifiers pre-trained on public NSRR records — [JOSS](https://joss.theoj.org/papers/10.21105/joss.05411)
  - OxWearables `asleep`: wrist accelerometer only, trained on more than 1000 PSG nights — [GitHub](https://github.com/OxWearables/asleep)
  - Walch et al. 2019 *SLEEP* (DOI 10.1093/sleep/zsz180): Apple Watch accelerometer plus PPG heart rate, with open code `ojwalch/sleep_classifiers` and data on PhysioNet `sleep-accel` — [SleepECG paper lists it](https://joss.theoj.org/papers/10.21105/joss.05411)
  - A PhysioNet dataset pairs Apple Watch instantaneous HR plus accelerometer with EEG stage labels (2025 IEEE TBME) — [PhysioNet](https://www.physionet.org/content/bidsleep-dataset/1.0.0/Bidslab11/)

### Inferences
- **Fastest path:** Samsung's own staging (Tier C) is already at Fitbit/Pixel level (κ about 0.42). Re-implementing it from IBIs alone is unlikely to beat it by much.
- **Real upside:** run wav2sleep or SleepPPG-Net style raw-waveform models on Tier A/B 25 Hz green PPG resampled to 34 Hz, with accelerometer added.
  - PSG-finger results suggest κ of about 0.74, and wrist-oximeter external results κ of about 0.66.
  - Wrist green reflectance PPG differs from finger transmissive IR, so expect a drop unless fine-tuned on DREAMT (wrist BVP + PSG).
- **Beat intervals at 25 Hz are adequate for IBI-based staging**, given our 3 ms RMSSD error. IBI-only models top out around κ 0.66–0.69 (Sridhar on SHHS, BM-DTS on MESA), compared with about 0.74–0.75 from raw waveforms.
- **The accelerometer mainly helps sleep/wake.** Wake detection is the weakest point of every commercial watch (wake sensitivity about 0.27–0.52).
- **Use multi-night trends.** Per-night stage minutes, especially REM and deep, carry wide individual error.

### Gaps
- No peer-reviewed study validates Galaxy Watch4–8 staging against PSG with an overall κ beyond Lee 2023 (Watch5). Two Samsung-related trials are registered: [NCT06363656](https://clinicaltrials.gov/study/NCT06363656) and [NCT06696196](https://clinicaltrials.gov/study/NCT06696196).
- No published result runs SleepPPG-Net or wav2sleep on 25 Hz wrist green PPG. The wav2sleep repository licence is unverified.
- No DREAMT benchmark κ table from the original Wang et al. 2024 paper was retrieved.

## Sleep apnea: Samsung and Apple features, PPG/SpO2-based AHI/ODI, wrist SpO2 accuracy and skin-tone bias

### Takeaway
The two cleared features use different signals and therefore suit different tiers.

- **Samsung (FDA De Novo, Feb 2024):** PPG SpO2 desaturations plus actigraphy, over 2 nights within 10 days, about 4 h of sleep each. Sensitivity 82.7%, specificity 87.7% against 2-night PSG.
- **Apple (510(k) K240929, Sept 2024):** accelerometer-only "breathing disturbances" over a 30-day window. Sensitivity 66.3% (43% moderate, 89% severe), specificity 98.5%.
- **Green-only wrist PPG plus accelerometer (no SpO2)** estimates AHI with ρ ≈ 0.61–0.67: sensitivity 62% and specificity 91% at AHI ≥ 15.
- **Wrist SpO2** is within about 2–3% RMSE of reference in controlled settings. Skin-tone data for wrist devices are thin. Our SDK gives red/IR only in 30-s blocks, so overnight ODI must come from Samsung's own outputs (Tier C/D).

### Cited Findings
- **Samsung sleep apnea feature** (Samsung Health Monitor, Galaxy Watch4 or later on Wear OS 5+ with Android 12+ phone) — [Samsung IFU PDF, Dec 2024](https://image-us.samsung.com/SamsungUS/home/apps/samsung-health-monitor/IFU_SLEEP-APNEA_US_v1.0_2024-12_v1.2.pdf)
  - Method: the watch uses its "PPG-based SpO2 sensor and actigraphy" and counts relative decreases in blood oxygen during sleep.
  - Requirements: 2 nights within 10 days, about 4 h of sleep each. It is a one-off screen ("not a tracking or monitoring tool"), with a 30-day wait before reuse.
  - Validation: multi-centre, 470 subjects aged 22+, 2-night PSG.
    - Sensitivity 82.7% (167/202 moderate-to-severe on both nights); specificity 87.7% (235/268).
    - Agreement on 84.2% of analysable nights.
    - Correctly called: 89/89 normal, 98/108 mild, 60/91 moderate, 107/111 severe.
  - Subgroups (sensitivity / specificity):

    | Subgroup | Sensitivity | Specificity |
    |---|---|---|
    | Age < 40 | 73.7% | 92.6% |
    | Age > 55 | 82.7% | 79% |
    | Female | 76.7% | 90.9% |
    | Male | 86.1% | 83.5% |
    | BMI < 25 | 70% | 91% |
    | Fitzpatrick 1–2 | 74.1% | 94.8% |
    | Fitzpatrick 3–4 | 85.7% | 82.7% |
    | Fitzpatrick 5–6 | 87.5% | 90.9% |

  - The IFU says skin pigmentation or low perfusion can prevent data collection, and that the watch is "not approved … as a pulse oximeter".
- **De Novo granted 6 Feb 2024** for detecting signs of moderate-to-severe OSA in adults 22+, OTC — [MedTech Dive](https://www.medtechdive.com/news/samsung-fda-watch-sleep-apnea/707335/)
  - DEN230041 is cited only by a secondary blog and is unverified.
  - An independent study (reported via [HME News](https://www.hmenews.com/article/researchers-found-samsung-galaxy-watch-accurately-identifies-sleep-apnea)) found AUROC 0.94. At the default threshold of 15: sensitivity 94.1%, specificity 66.7%. Primary paper not retrieved.
- **Apple Sleep Apnea Notification Feature** — [FDA 510(k) K240929 summary](https://fda.innolitics.com/device/K240929); [Apple white paper](https://apple.com/health/pdf/sleep-apnea/Sleep_Apnea_Notifications_on_Apple_Watch_September_2024.pdf)
  - Signal: triaxial accelerometer only, used to classify nightly "Breathing Disturbances" (events/h, explicitly not equal to AHI). Series 9+ / Ultra 2.
  - Notification rule: assessed every non-rolling 30 days. It requires at least 10 nights with values and notifies if at least 50% are elevated.
  - Development: 3,936 nights / 2,160 participants for design; 7,220 nights / 2,542 participants held out (sensitivity 66.6%, specificity 95.9%). AASM 4% hypopnea rule.
  - Clinical validation: 1,499 enrolled, 1,448 completed, up to 30 nights each, at least 2 nights of type-3 HSAT.
    - Sensitivity: moderate 43.4% (89/205), severe 89.1% (164/184), weighted overall 66.3% (95% CI 62.2–70.3).
    - Specificity: normal 100% (543/543), mild 91.0%, weighted 98.5%.
  - 91.4% of nightly values fell in a pre-specified error funnel against HSAT AHI. No subgroup differences were reported by sex, BMI or race.
- **Green-only wrist PPG for AHI (Papini et al., *Sci Rep* 2020, DOI 10.1038/s41598-020-69935-7)** — [PMC7421543](https://pmc.ncbi.nlm.nih.gov/articles/PMC7421543/)
  - Signals: Philips wrist reflective PPG, **green, 32 Hz**, no SpO2; accelerometer 128 Hz.
  - Model: deep learning on 212 features (IBI HRV, pulse-amplitude respiratory surrogate, activity, sleep-stage probabilities) flags 30-s epochs; AHI comes from the epoch proportion.
  - Data: 502 participants, hold-out test 252 (188 after quality exclusion). AHI Spearman 0.67 (0.61 on all); severity weighted κ 0.51.

    | AHI threshold | Sensitivity | Specificity | AUC |
    |---|---|---|---|
    | ≥ 5 | 77% | 72% | 0.84 |
    | ≥ 15 | 62% | 91% | 0.86 |
    | ≥ 30 | 46% | 98% | 0.85 |

  - Tends to underestimate AHI; epoch-level event sensitivity is only about 38%. Data are available only on request.
- **Galaxy Watch4 SpO2 vs arterial co-oximetry (Browne, Bernstein, Bickler, *Sensors* 2025, DOI 10.3390/s25051286)** — [PMC11902706](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902706/)
  - UCSF hypoxia lab, SaO2 100→70%; 12 watch participants, 247 pairs.
  - Reflectance PPG at **25 Hz per wavelength**, SpO2 every 1 s.
  - **RMSD 1.8%**, which met the then-current FDA/ISO limit of < 3%; 10.2% of readings rejected.
  - Fitzpatrick V–VI was about 25% of the cohort; no per-tone bias was reported; the authors call for more dark-skin testing.
- **Consumer watches vs a clinical finger oximeter (Jiang, …, Dunn, *PLOS Digital Health* 2023, DOI 10.1371/journal.pdig.0000296)** — [PMC10337940](https://pmc.ncbi.nlm.nih.gov/articles/PMC10337940/)
  - 49 patients (34.7% Black, mostly pulmonary disease).
  - Apple Watch 7: MAE 2.2%, RMSE 2.9%, 11% missing. Garmin Venu 2s: MAE 5.8%, RMSE 6.7%. Withings ScanWatch and Garmin Fenix: 28–31% missing.
  - No significant skin-tone differences after correction, but only 3 patients had SpO2 < 90% and the reference was not arterial blood.
- **Withings ScanWatch**: a company-run hypoxia study of 14 healthy adults — [PMC8078374](https://pmc.ncbi.nlm.nih.gov/articles/PMC8078374/). No published wrist-watch apnea sensitivity was found; the trial [NCT04496843](https://trial.medpath.com/zh/clinical-trial/9afd3852f57dde30/nct04496843-validation-sleep-apnea-syndrome-diagnostic-performance-withings-hwa09) is registered.
- **FDA draft pulse-oximeter guidance (Jan 2025)** — [TCTMD](https://www.tctmd.com/news/fda-issues-draft-guidance-improving-skin-tone-disparities-pulse-oximeters)
  - Calls for at least 150 diverse participants and Monk Skin Tone plus objective Individual Typology Angle (ITA) measurement, with labelling warnings.
  - Applies to medical oximeters, not wellness wearables. Final status unverified.

### Inferences
- **Apnea screening by tier:**
  - **Tier D:** our Watch6 Classic is eligible for Samsung's cleared feature, which is the best-validated option (Watch4+ on Wear OS 5+; regional availability unverified).
  - **Tier A:** reproducing Apple's accelerometer approach with our 100 Hz accelerometer is feasible in principle but needs labelled training data. DREAMT has wrist accelerometer plus apnea events, though only at 32 Hz.
  - **Tier A green PPG plus accelerometer** (Papini-style) is realistic for moderate-to-severe screening, at about 60% sensitivity and 90% specificity at AHI ≥ 15. It combines naturally with our RR and IBI pipelines.
- **Overnight ODI from our own red/IR is not possible.** The SDK limits red/IR to 30-s foreground blocks. We have to rely on Samsung's overnight SpO2 in the export (Tier C), if its granularity allows desaturation counting.
- **Treat wrist SpO2 as a trend.** Single readings carry about 2–3% RMSE in good conditions and lose 10–30% of readings in the field.

### Gaps
- The FDA De Novo decision summary for Samsung (performance requirements, special controls) was not retrieved.
- The Samsung feature's ODI threshold and algorithm details are not public.
- The sampling rate and granularity of Samsung Health's overnight SpO2 export are unknown, as is whether per-minute values exist.
- No wrist SpO2 study stratified by ITA or Monk Skin Tone with arterial reference in large samples was found.
- Apple's accelerometer sampling rate for Breathing Disturbances is not disclosed.
- The independent Samsung OSA study behind the HME News report (authors, n) was not retrieved.

## Stress and autonomic state: HRV-based detection, WESAD-type benchmarks, cognitive load, EDA, consumer stress scores

### Takeaway
Detecting acute lab stress such as the Trier Social Stress Test (TSST) from wrist BVP works reasonably under leave-one-subject-out (LOSO) evaluation.

- **WESAD wrist BVP alone:** 84% accuracy (stress vs non-stress). All wrist physiology together: 88%.
- **Chest physiology:** 93%.
- **Three-class (baseline/stress/amusement):** wrist physiology about 76%.

Free-living stress scores barely track self-reported stress. In 781 students over 3 months, Garmin stress and self-reported stress "did not overlap for most individuals".

EDA_CONTINUOUS in the Samsung SDK is Galaxy Watch8+ only, so it is not on our Watch6. Composite readiness and stress scores across 10 vendors are undisclosed and mostly unvalidated.

### Cited Findings
- **WESAD (Schmidt et al., ICMI 2018)** — [paper PDF](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf); [dataset page](https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/)
  - Setup: 15 subjects; wrist Empatica E4 (BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz) and chest RespiBAN at 700 Hz. TSST stress plus amusement, 60-s windows, LOSO.
  - Binary stress vs non-stress:

    | Input | Classifier | Accuracy | F1 |
    |---|---|---|---|
    | Wrist BVP alone | Random forest | 84.18% | 81.35 |
    | Wrist BVP alone | LDA | 85.83% | — |
    | Wrist EDA alone | — | 76–80% | — |
    | Wrist TEMP alone | — | about 68% | — |
    | Wrist physiology (BVP+EDA+TEMP) | Random forest | 88.33% | 86.10 |
    | Chest physiology | LDA | 93.12% | 91.47 |
    | Chest RESP alone | LDA | 88.09% | — |

  - Three-class: wrist physiology 76.17% (F1 66.33); chest physiology 80.34% (F1 72.51).
  - Chest EDA beat wrist EDA, and wrist BVP slightly beat chest ECG. RESP may partly reflect speaking during the TSST.
- **Later WESAD claims:** binary LOSO accuracies of 97–99% (NeurIPS 2025 with minority-class augmentation; other conference papers) — [NeurIPS 2025](https://nips.cc/virtual/2025/132980)
  - Earlier work reports wrist 3-class macro F1 of about 55–63 — [arXiv 2303.08215](https://arxiv.org/pdf/2303.08215)
  - Accuracy is inflated by class imbalance on WESAD — [arXiv 2203.09669](https://arxiv.org/pdf/2203.09669)
- **Free-living validity of a consumer stress score (Siepe, Tutunji, Rieble, Proppert, Fried; *J Psychopathol Clin Sci* 2025, DOI 10.1037/abn0001013)** — [Marburg summary](https://www.uni-marburg.de/en/fb04/team-heck/news/news/new_publication_about_ema); [press coverage](https://athletechnews.com/smartwatch-stress-scores-rarely-match-reality-researchers-say/)
  - 781 students (WARN-D) wearing Garmin Vivosmart 4 for 3 months, with 4 EMA surveys a day.
  - Sleep agreed robustly with self-report, tiredness weakly, and stress measures "did not overlap for most individuals".
  - Caveat: one older device ([Notebookcheck](https://www.notebookcheck.net/Smartwatch-tech-fails-to-measure-stress-in-new-study.1088069.0.html)).
  - A Tel Aviv preprint found HRV explained about 1–2% of variance in daily stress (secondary report, same coverage).
- **Composite health scores (Doherty, Baldwin, Lambe, Burke, Altini; *Translational Exercise Biomedicine* 2025, DOI 10.1515/teb-2025-0001)** — [De Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/teb-2025-0001/html)
  - 14 readiness, recovery, stress and strain scores from 10 manufacturers, including Samsung, Garmin, Oura, WHOOP, Fitbit and Polar.
  - HRV and RHR are the dominant inputs.
  - No manufacturer disclosed its formula, and few provided empirical validation.
- **Samsung SDK EDA:** `EDA_CONTINUOUS` delivers raw 1 Hz EDA and is available **only on Galaxy Watch8 series or later** — [Samsung data specifications](https://developer.samsung.com/health/sensor/guide/data-specifications.html); [codelab](https://developer.samsung.com/codelab/health/electrodermal-activity.html)
  - Samsung positions it for stress and sleep insights. Hot-flash monitoring is a stated plan ([Samsung dev blog](https://developer.samsung.com/health/blog/en/how-the-galaxy-watchs-eda-sensor-enhances-your-health-monitoring)).
- **Fitbit/Google cEDA (Sense 2, Pixel Watch 2/3)** — [Google Research blog](https://research.google/blog/what-does-electrodermal-sensing-reveal-insights-from-the-pixel-watch-fitbit-sense-2/); [Body Response design](https://blog.google/products/fitbit/how-we-trained-fitbits-body-response-feature-to-detect-stress/)
  - About 10 million hours from about 16,000 people. Skin conductance level is lowest near 22:00 and highest near 06:00.
  - Higher SCL, skin temperature and HR with elevated depression or anxiety symptoms (both preprints, 2024).
  - Body Response combines cEDA, HR, HRV and skin temperature against a personal baseline learned over about 1 month.
  - No accuracy metrics have been published.

### Inferences
- **What our tiers can support:**
  - Tier A: acute-arousal and stress-episode detection at rest (25 Hz PPG-derived HR/IBI plus accelerometer to reject motion).
  - Tier B: adds skin temperature, which is weak alone. EDA is unavailable on the Watch6, so we lose the modality that best separates arousal from physical load.
- **Validate and report honestly:**
  - Validate personally against EMA or known stressors, not against Samsung's stress score.
  - Frame outputs as "autonomic arousal relative to personal baseline", not "stress".
  - Expect lab-level accuracy (about 85% binary) only in controlled, seated conditions.
- **Gate on motion.** HR/HRV confounds (posture, caffeine, positive excitement, light activity) dominate in free living, and our benchmark shows PPG HRV fails during movement.

### Gaps
- No peer-reviewed validation of Samsung's stress score was found.
- The 2024 cognitive-load literature (n-back, HRV) was not reviewed in this pass.
- The WESAD-type LOSO results with 97–99% accuracy were not checked for leakage.
- No independent validation of Samsung Watch8 EDA exists yet.

## Illness, infection and recovery: early infection detection, fever, readiness, menstrual cycle

### Takeaway
Overnight resting HR is the workhorse signal; respiratory rate and skin temperature add to it.

- Personal-baseline deviations flag about 63–80% of COVID-19 infections, often 1–3 days before symptoms.
- Discrimination reaches AUC about 0.80–0.82 when combined with symptoms or temperature.
- Specificity is the weak point: alcohol, stress and travel also trigger alerts.

Wrist temperature supports retrospective ovulation estimation: Apple MAE 1.2 days (89% within ±2 days); Galaxy Watch5 64% within 2 days. Readiness and recovery scores are unvalidated composites.

### Cited Findings
- **Mishra et al. 2020, *Nat Biomed Eng* (Stanford, DOI 10.1038/s41551-020-00640-6)** — [summary of abstract](https://www.s4me.info/threads/pre-symptomatic-detection-of-covid-19-from-smartwatch-data-2020-mishra-et-al.29270)
  - 32 infected of about 5,300 participants; 26 (81%) showed HR, step or sleep changes, 22 of 25 at or before symptom onset.
  - A resting-HR-based retrospective alarm would have caught 63% before symptoms.
- **Alavi et al., *Nat Med* 28:175–184 (online Nov 2021), DOI 10.1038/s41591-021-01593-2** — [PMC8799466](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8799466/)
  - 3,318 participants, 84 infected. Real-time NightSignal algorithm on **overnight resting HR**, with red/green alerts.
  - Alerts for 67/84 (80%) of pre-symptomatic or asymptomatic infections; the preprint gives a median of 3 days before symptoms.
  - Non-infection events (stress, alcohol, travel) also triggered alerts: 1.15 alert-days per person against 3.42 for COVID cases.
- **Quer et al., DETECT (Scripps), *Nat Med* (online 29 Oct 2020)** — [Scripps release](https://www.scripps.edu/news-and-events/press-room/2020/20201029-topol-radin-quer-detect.html)
  - 30,529 participants, 78% using Fitbit.
  - Symptoms plus sensor deviations (RHR, sleep, steps) separated COVID-positive from negative symptomatic people with AUC 0.80 (IQR 0.73–0.86).
- **Natarajan et al. 2020, *npj Digit Med* (Fitbit, 2,745 PCR-confirmed cases)** — [DOAJ record](https://doaj.org/article/02b99d5ad127459fb9f207548d75cd8c)
  - RR and HR rise and HRV falls with illness.
  - Follow-up 2021: 36.4% of symptomatic and 23.7% of asymptomatic cases had nightly RR at least 3 brpm above baseline in days −1 to +5, against 4.3% in a control window — [PMC8443549](https://pmc.ncbi.nlm.nih.gov/articles/PMC8443549/)
- **TemPredict (Oura) first study, *Sci Rep* 2022, DOI 10.1038/s41598-022-07314-0** — [PubMed 35236896](https://pubmed.ncbi.nlm.nih.gov/35236896/)
  - Multimodal model (HR, HRV, RR, continuous skin temperature): AUC 0.819, sensitivity 82%, specificity 63%, an average of 2.75 days before testing.
  - Adding continuous temperature raised AUC by 4.9%. Accuracy varied by age and sex.
- **Readiness/recovery:** composite scores are undisclosed and largely unvalidated (Doherty et al. 2025, above) — [De Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/teb-2025-0001/html)
  - Recovery scores have no reference standard to validate against — [SciTePress 2025](https://www.scitepress.org/publishedPapers/2025/137331/pdf/index.html)
- **Menstrual cycle**
  - **Apple (Human Reproduction 2025, DOI 10.1093/humrep/deaf005)** — [record](https://www.nature.com/nature-index/article/10.1093/humrep%2Fdeaf005); [Apple support](https://support.apple.com/120357)
    - Wrist-temperature retrospective ovulation estimate against urinary LH: MAE about 1.2 days, 89.0% within ±2 days for completed cycles (80% within ±2 days for ongoing cycles).
    - Available in about 80% of cycles. Next-period prediction: 89.4% within ±3 days.
    - Requires at least 4 h of sleep tracking for 5 nights.
  - **Ava wrist skin temperature vs oral basal body temperature (Zhu et al., *JMIR* 2021)** — [PMC8238491](https://pmc.ncbi.nlm.nih.gov/articles/PMC8238491/)
    - 57 women, 193 cycles. Sensitivity 0.62 vs 0.23; specificity 0.26 vs 0.70 (more false positives).
  - **Galaxy Watch5 wrist skin temperature vs serum-progesterone-confirmed ovulation (*Biosensors and Bioelectronics* 2026, SKKU)** — [record](https://scholarx.skku.edu/item/c076f260-522c-43cb-a0fd-3d330decb736)
    - Ovulation date within 2 days in 64% of cycles and within 3 days in 78%. Details are from a search snippet only.
  - **Samsung Cycle Tracking** (Galaxy Watch5+) uses Natural Cycles' temperature algorithm; it is described as FDA-*registered*, EU class I — [GSMArena](https://m.gsmarena.com/samsung_galaxy_watch5_watch5_pro_temperature_based_cycle_tracking-news-57538.php)
  - **Natural Cycles 510(k) clearances** cover Oura (2021) and Apple Watch wrist temperature (Sept 2023) — [BioSpace](https://www.biospace.com/natural-cycles-receives-fda-clearance-to-integrate-its-birth-control-app-with-data-measured-by-apple-watch)

### Inferences
- **Tier A** (25 Hz PPG → overnight minimum or median HR, plus nightly RR) suffices for a NightSignal-style infection or strain alert. NightSignal needs only overnight resting HR.
- **Tier B or C** add skin temperature (about +5% AUC in TemPredict) and IBI-based HRV.
- **All of these need continuous overnight data and a personal baseline of several weeks.** None needs 100 Hz, red/IR or ECG.
- **Expect frequent false alarms.** Alcohol, stress, travel and the luteal phase all raise RHR and temperature. Cycle-phase modelling (temperature plus RHR) should run before illness alerts in menstruating users.
- **Readiness:** build transparent sub-signals (RHR deviation, RMSSD deviation, RR deviation, temperature deviation, sleep duration) rather than one opaque score.

### Gaps
- The influenza and cold pre-symptomatic study (Grzesiak et al., *JAMA Netw Open* 2021) and the Oura fever-feasibility study (Smarr et al., *Sci Rep* 2020) were not verified in this pass, so no numbers are given.
- No validation exists for Samsung's own skin-temperature accuracy (its absolute vs relative stability) or for its "Energy Score".
- Post-COVID-era validation on other infections (influenza, RSV) at scale was not reviewed.

## Cardiorespiratory fitness: VO2max from wrist HR/PPG, heart-rate recovery, VO2max and longevity

### Takeaway
Wrist VO2max estimates are good at the population level but poor for individuals.

- **INTERLIVE meta-analysis:** exercise-based estimates have near-zero bias but LoA about ±10 mL/kg/min; resting-based estimates are biased upward by about 2 with LoA about −13 to +17.
- **Apple Watch:** underestimates by about 6 mL/kg/min (MAPE 13–16%).
- **Samsung:** only a company-reported r = 0.82 exists; no independent peer-reviewed VO2max validation was found.

Fitness is among the strongest mortality predictors (low vs elite HR 5.0), so the trend is worth tracking even when the absolute value is uncertain.

### Cited Findings
- **INTERLIVE systematic review and meta-analysis (Molina-Garcia et al., *Sports Med* 2022;52:1577–1597, DOI 10.1007/s40279-021-01639-y)** — [PMC9213394](https://pmc.ncbi.nlm.nih.gov/articles/PMC9213394)
  - 14 studies in the meta-analysis.
  - Resting-based estimates: bias +2.17 mL/kg/min, LoA about −13 to +17.
  - Exercise-based estimates: bias −0.09, LoA about −9.9 to +9.7.
  - Acceptable for groups, not individuals. Partly funded by Huawei ([UGR record](https://digibug.ugr.es/handle/10481/72723)).
- **Apple Watch (Lambe et al., *PLOS One* 2025)** — [PMC12080799](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12080799/)
  - 30 participants against indirect calorimetry: underestimated by 6.07 mL/kg/min (95% CI 3.77–8.38), MAPE 13.31%, MAE 6.92.
- **Apple Watch Series 7 (Caserman et al., *JMIR Biomed Eng* 2024)** — [JMIR](https://biomedeng.jmir.org/2024/1/e59459)
  - MAPE 15.79%, ICC 0.47.
  - Overestimates in less-fit and underestimates in highly fit people.
- **Samsung**
  - University of Michigan studies report an "82% correlation" between watch VO2max and CPET; the result is company-reported and unpublished — [Samsung Newsroom](https://news.samsung.com/us/galaxy-watch-shows-strong-correlation-with-clinical-and-sports-science-equipment-in-university-of-michigan-studies)
  - Galaxy Watch6 HR during maximal treadmill CPET (55 adults) underestimated a chest strap by 2.67 bpm on average, typically within 1–3 bpm — [JMIR Cardio 2026](https://cardio.jmir.org/2026/1/e81917)
- **Free-living VO2max from wearables (Spathis, Perez-Pozuelo, …, Mascolo; *npj Digit Med* 2022, DOI 10.1038/s41746-022-00719-1)** — [PMC9718831](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718831/)
  - Cohorts: Fenland (n = 11,059), its longitudinal cohort (n = 2,675), and UK Biobank validation (n = 181), using wrist ECG-derived HR and accelerometer plus anthropometrics.
  - Holdout r = 0.82 against maximal or submaximal tests; tracked change over 7 years.
- **Longevity**
  - Mandsager et al., *JAMA Netw Open* 2018 (122,007 treadmill-tested patients, median 8.4-y follow-up): low vs elite fitness HR 5.04 (95% CI 4.10–6.30), with no upper limit of benefit — [PMC6324439](https://pmc.ncbi.nlm.nih.gov/articles/PMC6324439); [TCTMD](https://www.tctmd.com/news/big-fitness-equals-big-survival-benefit-new-study-shows)
  - Kodama et al., *JAMA* 2009 meta-analysis: each 1-MET higher fitness is associated with about 13% lower all-cause mortality — [DOI 10.1001/jama.2009.681](https://doi.org/10.1001/jama.2009.681)
- **Heart-rate recovery:** a fall of ≤ 12 bpm in the first minute after a symptom-limited exercise test strongly predicted death (Cole et al., *NEJM* 1999) — [DOI 10.1056/NEJM199910283411804](https://doi.org/10.1056/NEJM199910283411804)

### Inferences
- **VO2max estimation needs exercise-phase data:**
  - outdoor runs or walks with GPS speed and steady HR (Samsung's own estimate requires a running workout);
  - or a free-living model in the Spathis style (HR-vs-activity slope from 1 Hz HR plus 100 Hz accelerometer, Tier A).
- **Our benchmark shows PPG HR during running may be unreliable from 25 Hz raw.** Use Samsung's 1 Hz HR (Tier A/C), which is validated to within about 3 bpm even at maximal effort on the Watch6.
- **Heart-rate recovery** is measurable from 1 Hz HR at the end of workouts (Tier A/C) and needs no SDK. The 60-s post-exercise window is exactly where wrist PPG recovers quickly.
- **Report VO2max as a personal trend** with about ±10% uncertainty, not as an absolute value.

### Gaps
- No independent peer-reviewed validation of Galaxy Watch VO2max against CPET was found.
- Samsung's algorithm inputs and resting-vs-exercise basis are undisclosed.
- Heart-rate recovery measured by a wrist watch has not been validated against ECG in a published study found in this pass.

## For each measure: which data tier does it need (continuous overnight, on-demand 100 Hz, red/IR, or 25 Hz green + accelerometer only)?

### Takeaway
Almost everything in scope runs on **Tier A**: continuous 25 Hz green PPG plus accelerometer, overnight. Overnight is essential because sleep is the only reliably motion-free window. Tier B mainly adds Samsung's IBI list and continuous skin temperature.

Red/IR (SpO2) is needed only for desaturation-based apnea and SpO2 itself. Because the SDK limits red/IR to 30-s foreground blocks, overnight oxygenation must come from Samsung's own outputs (Tier C/D). EDA is not available on the Watch6.

### Cited Findings
- **Respiratory rate:** Samsung's validated nocturnal RR uses 25 Hz green PPG plus 25 Hz accelerometer — [PMC10536355](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536355/). Fitbit uses IBIs in deep sleep — [PMC8443549](https://pmc.ncbi.nlm.nih.gov/articles/PMC8443549/).
- **Sleep staging:** open models resample PPG to about 34 Hz — [arXiv 2404.06869](https://arxiv.org/html/2404.06869v1); [arXiv 2411.04644](https://arxiv.org/html/2411.04644v1). IBI-only models reach κ about 0.66 against about 0.74 for raw waveforms (same sources).
- **Apnea:**
  - Samsung: PPG SpO2 (red/IR) plus actigraphy over 2 nights — [Samsung IFU](https://image-us.samsung.com/SamsungUS/home/apps/samsung-health-monitor/IFU_SLEEP-APNEA_US_v1.0_2024-12_v1.2.pdf)
  - Apple: accelerometer only, at least 10 nights in 30 days — [Apple](https://apple.com/health/pdf/sleep-apnea/Sleep_Apnea_Notifications_on_Apple_Watch_September_2024.pdf)
  - Green-only 32 Hz PPG plus accelerometer — [PMC7421543](https://pmc.ncbi.nlm.nih.gov/articles/PMC7421543/)
- **SpO2:** the Galaxy Watch4 samples red/IR at 25 Hz per wavelength — [PMC11902706](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902706/)
- **EDA:** Watch8+ only — [Samsung SDK spec](https://developer.samsung.com/health/sensor/guide/data-specifications.html)
- **Infection alerts:** overnight resting HR only (NightSignal) — [PMC8799466](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8799466/). Temperature adds about 5% AUC — [PubMed 35236896](https://pubmed.ncbi.nlm.nih.gov/35236896/).
- **Cycle phase:** nightly wrist temperature with at least 4 h of sleep tracking on 5 nights (Apple) — [Apple support](https://support.apple.com/120357)

### Inferences
Tier mapping per measure (A = no SDK 25 Hz green + 100 Hz accelerometer + 1 Hz HR; B = SDK; C = Samsung export; D = closed Samsung feature):

| Measure | Minimum tier | Continuous overnight? | 100 Hz on-demand? | Red/IR? | Notes |
|---|---|---|---|---|---|
| Overnight RR | **A** | yes | no | no | Mirrors Samsung's validated 25 Hz pipeline; B's IBIs enable the Fitbit RSA method |
| Daytime spot RR | A or B | no | optional | no | Poor accuracy even in research (±6 brpm) |
| Sleep/wake + 4-stage | **A** (raw PPG + accelerometer) | yes | no | no | C already gives Samsung stages (κ about 0.42); open raw-PPG models offer a possible gain |
| Apnea, accelerometer/PPG-based | **A** | yes, multi-night | no | no | Needs training labels (DREAMT); expect sensitivity about 60%, specificity about 90% at AHI ≥ 15 |
| Apnea, desaturation-based / ODI | **D** (or C if SpO2 granularity allows) | yes | no | **yes** | SDK red/IR is limited to 30-s blocks, so it cannot run overnight |
| SpO2 spot check | **B** on-demand | no | yes (SDK SpO2) | yes | About 2–3% RMSE at best; many rejected readings |
| Acute stress / arousal | A (B for IBIs) | daytime, resting only | no | no | Gate by accelerometer; no EDA on Watch6 |
| Infection / strain alert | **A** (C for history) | yes, weeks of baseline | no | no | RHR alone works; add RR (A), temperature (B/C) |
| Fever / cycle phase | **B or C** (skin temperature) | yes | no | no | Not possible on Tier A, which has no temperature |
| Readiness / recovery | A + B/C | yes | no | no | Build transparent components |
| VO2max | A/C (1 Hz HR + accelerometer/GPS during exercise) | no (workouts) | no | no | Use Samsung's HR in motion, not raw 25 Hz |
| Heart-rate recovery | A/C | no (end of workout) | no | no | 1 Hz HR suffices |

- The 500 Hz ECG and bioimpedance blocks contribute nothing essential to these measures. ECG could serve as a periodic reference to calibrate PPG-derived IBIs.

### Gaps
- Whether Tier A `hr_raw` can be logged reliably all night has not been tested (battery, Wear OS background limits).
- Decoding which of the 16 `hr_raw` channels is the usable green signal is still open.
- The time resolution of the Samsung export (SpO2, skin temperature, sleep RRI) needs confirming on a real export before Tier C-dependent plans (ODI, temperature baselines) are committed.
