# Which models can we use with a Galaxy Watch and a phone?

Checked 2026-10-21/22: two research workflows (32 agents with adversarial review) plus a probe of the
real watch. Scope: a prototype on consumer devices (Galaxy S Ultra or Honor flagship phone, Galaxy
Watch), with inference on a server and no regulatory work. "Zero-shot" means no labelled training on
our side.

## Short answer

- **Nothing from the Manhph2211 profile is a drop-in zero-shot model for our devices.** None was trained
  on Samsung watch PPG, phone-camera PPG or face video. PulseLM is the only one with readable zero-shot
  outputs, and those are categories ("normal", "tachycardia"). Five of its 12 tasks are thresholds on
  HR, HRV and signal quality that classical processing computes directly.
- **Classical signal processing is the backbone** (NeuroKit2, pyPPG, HeartPy). It is the bar every
  model has to beat, and on HR and HRV no model is expected to beat it.
- **Open models better suited than the profile's:**
  - ECGFounder: single-lead ECG, 150 labels, MIT, ungated.
  - Pulse-PPG: pretrained on WearOS watch PPG, MIT.
  - AnyPPG: best of these encoders on phone-camera PPG.
  - ME-rPPG: face-video HR, 2.5 MB ONNX.

## The Manhph2211 models

| Model | What it is | Our signals | Zero-shot? | Access |
|---|---|---|---|---|
| PulseLM (HF) | Qwen2.5-7B-Instruct + Pulse-PPG encoder (inferred from exact parameter count), 15.3 GB merged bf16 | 10 s single-channel PPG: watch 100 Hz on-demand or phone fingertip | Yes: 12 closed-ended categories. Only the HR category is credible on our signals; HRV categories need ECG/IBI-grade timing | gated-manual; input API in `modeling_pulselm.py` unreadable until approved |
| CP-PPG | 1.1 M-param U-Net that removes contact-pressure distortion from wrist green PPG, 128 Hz × 8 s | watch 100 Hz on-demand (upsampled) | Yes (waveform in → waveform out), but it pushes every waveform to a young-adult template; unsafe for vascular-ageing features | gated-auto; CC BY-NC (HF card wrongly says Apache) |
| D-BETA | 12-lead 500 Hz ECG-text model (ICML 2025) | watch single-lead ECG zero-padded to 12 | Only via text-prompt notebook; single lead is far out of distribution | gated-manual; CC BY-NC |
| Q-HEART | 12-lead ECG QA, Llama-3.2-1B | same as D-BETA | Plumbing checks only | open, but needs Meta's Llama-3.2-1B-Instruct gate |
| ECG-Scan | CLIP ViT image encoder for photos of 12-lead printouts | phone photo of a paper ECG | No: embeddings only, trained on synthetic renders | gated-manual; CC BY-NC |
| Cheap2Chic | earphone audio enhancement | none | Unrelated, and the HF repo is empty | — |

Paper reality check (PulseLM v2, Table 11): the best model averages 70.6% exact match, against 50.2%
for a model that never sees the PPG. On AF (85.4 vs 85.3) and respiratory rate (32.7 vs 33.1) it gains
nothing over the question-only baseline. Several cells look like majority-class predictions. There is
no label leak: the per-sample `text` field is never fed to the model (verified in code).

## What our watch actually gives (probed on SM-R965F, One UI Watch 8, Android 16)

| Signal | Route | Rate | Status |
|---|---|---|---|
| Raw PPG, 16 integer channels | plain Android `SensorManager`, `com.samsung.sensor.hr_raw`, permission `READ_HEART_RATE` | 24.96 Hz measured | **Works without the Samsung SDK** (our probe app). Channel meaning not yet decoded |
| HR | `TYPE_HEART_RATE` | 1 Hz | works |
| Accelerometer, gyroscope | standard sensors | 100 Hz | works |
| 100 Hz factory PPG, ECG, skin temp, 32 g accel | Samsung vendor sensors | — | registration refused (`SSENSOR` is signature-only); needs the Samsung Health Sensor SDK in Health Platform developer mode |
| Samsung Health history | adb backup | — | impossible: the app does not allow backup |

Samsung Health Sensor SDK (developer mode, own watches only) adds:
- PPG_ON_DEMAND at 100 Hz (green/IR/red).
- ECG_ON_DEMAND at 500 Hz in mV, with a co-sampled PPG green value.
- HR + IBI, SpO2 and BIA.

On-demand trackers are foreground only, at most 30 s each, and one at a time. Continuous trackers may
return invalid values while one runs, so plan sequential 30 s blocks.

History lives on the phone, never on the watch. Samsung Health → Settings → "Download personal data"
gives:
- HR as hourly rows plus 1-min bins in `jsons/…binning_data.json`, and about 1 s HR during workouts.
- HRV (SDNN/RMSSD) bins, SpO2, skin temperature, stress and sleep stages.
- ECG records as a PDF only.
- Probably overnight beat-to-beat RR in `sleep_raw_data` `rri_info`. This is inferred and should be
  confirmed on our own export; it is probably kept only for about the last week.

Parse the export with `Livia-Zaharia/samsung_health_extractor`, keeping the files its default ignore
list skips.

## Phone

- **Fingertip PPG:**
  - About 30 fps for third-party apps on Samsung. 60 fps is reported on S24/S25 Ultra but unconfirmed.
  - The 120/240 fps mode forces auto exposure, so it is useless for PPG.
  - Use torch on, locked exposure, linear tone.
  - Published contact-camera HR error is 2–4.5 bpm. HRV is poor at 30 fps.
  - A Chrome `getUserMedia` page with the torch and manual exposure is closer to the validated setups
    than stock-camera MP4s.
- **Face video:** HR only. About 1.5–3 bpm error when still and well lit, 8–15+ bpm otherwise. HRV is
  unusable.
- **Honor:** the phone only. Honor watches expose no raw data. Manual Camera2 control on global Honor
  ROMs is unverified, so probe it on the handset.

## Zero-shot plan, in order

1. Classical HR/HRV/quality on the watch's 25 Hz raw PPG and the 1 Hz HR. Compare against Samsung
   Health's own numbers at the same timestamps; this needs no labels.
2. Watch ECG through the Samsung SDK, run through NeuroKit2 R-peaks and then ECGFounder (1-lead
   checkpoint). Check at load time that the 150-class head is present.
3. Phone fingertip and face capture, against the watch ECG and HR as reference.
4. Embeddings (Pulse-PPG, PaPaGei-S, AnyPPG), used only for drift and anomaly tracking against a
   personal baseline.
5. PulseLM, once approved, reported next to the classical numbers it bins.

## Access to-do (user actions)

- [ ] HF: request access to PulseLM, D-BETA and ECG-Scan (manual); accept the CP-PPG and PulseLM-dataset
  terms (auto).
- [ ] HF: accept Meta's Llama-3.2-1B-Instruct licence (for Q-HEART).
- [ ] Samsung: download Health Sensor SDK 1.4.1 `.aar` → `watch/probe/libs/`.
- [ ] Watch: Health Sensor Service developer mode (Settings → Apps → Health Sensor Service, tap the title
  about 10 times).
- [ ] Phone: Samsung Health "Download personal data", soon, because sleep RR history is short.

## Hygiene

D-BETA and Q-HEART load multi-GB pickles. PulseLM, CP-PPG, D-BETA and ECG-Scan all need
`trust_remote_code`. Read that code before running it, and run each in its own uv environment, because
the pins conflict (PulseLM pins torch 2.1.2 / transformers 4.43.3). The laptop RTX 4090 (16 GB) fits
PulseLM only quantised to 8-bit or 4-bit.
