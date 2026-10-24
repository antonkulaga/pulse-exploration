# Phase 0 without the watch: what our recordings and GalaxyPPG already answer

Checked 2026-10-24 from the 30 s two-tier recording (`data/runs/20261022-015638-record`, `hr_raw`
and SDK green/IR/red recorded together on the SM-R965F), the 30 s 100 Hz SDK recording
(`20261022-015808-record`), and GalaxyPPG. Phase 0 of the [research report](../reports/PPG%20health%20and%20aging%20measures.md)
still needs one watch-ECG + `hr_raw` recording and one still 25 vs 100 Hz recording; this note
narrows what those have to settle.

## `hr_raw` channels

Each of the 16 channels was correlated with SDK green, IR and red on a common clock (band 0.5–5 Hz,
lag search ±400 ms).

| Channel | Behaviour | Best match | Reading |
|---|---|---|---|
| v0 | constant 2 on the wrist | — | wrist status (1 off, 2 on) |
| v1, v4 | ~2.7 × 10⁶, pulsatile | IR r = 0.97, green 0.95 | raw light level, closer to IR/red |
| v5 = v7 | 2^21 offset, largest swing | **green r = 0.98** | AC pulse channel, closest to green |
| v9, v11, v13 | pulsatile | IR/red r = 0.96–0.97 | further light channels, probably IR/red |
| v6, v8, v10, v12, v14 | ~2^21, no pulse (r < 0.2) | — | ambient or idle |
| v2, v3, v15 | always 0 | — | unused |

The server already uses v5 (`watch_session._ppg_input`). v9/v11/v13 are candidates for a
multi-wavelength no-SDK signal, but their LED colours are inferred from correlation only.

## `hr_raw` hides its lost samples

- `hr_raw` timestamps are spread almost perfectly evenly: 40.16 ms median, 39.92–40.34 ms range.
  The SDK stream sits on a 40.00 ms grid and shows its losses as visible gaps (two of ~75 ms in
  30 s).
- Both streams deliver about two fewer samples per 30 s than 25.0 Hz would. `hr_raw` spreads the
  loss over its stamps, which is exactly the pattern GalaxyPPG showed and `foot_template`'s
  lost-sample fit corrects. The SDK stream on this watch does not need that correction.
- Beat by beat, `hr_raw` and SDK green differ by a constant ~60 ms plus a sawtooth: a slow ramp of
  ~4 ms per beat (the 0.4 % stretch of the stamps) and jumps of about ±40 ms (one sample). The
  interval difference between the two streams has an SD of 24 ms, so the two disagree more than
  either does with ECG on GalaxyPPG. RMSSD came out 63 vs 59 ms with the correction, 70 vs 64
  without. Only the watch ECG can say which stream is right.
- Spectrally, v5 matches SDK green: relative power at 5–8 Hz 1.9 × 10⁻³ vs 1.4 × 10⁻³ and at
  8–10 Hz 1.9 × 10⁻⁴ vs 0.9 × 10⁻⁴. `hr_raw` is not more heavily filtered than the SDK stream.

## Pulse shape survives 25 Hz when beats are averaged

- **Wrist green PPG has little above 12.5 Hz.** On our watch's 100 Hz SDK recording (30 s, with
  motion), 12.5–20 Hz holds 2.5 × 10⁻⁴ and 20–50 Hz 6.4 × 10⁻⁵ of the 0.8–3 Hz pulse-band
  power, about 1.7 % of the amplitude. IR and red carry more high-frequency power, most likely
  noise.
- **Reconstruction from 25 Hz keeps average-beat shape.** `benchmarks/morph25/run.py` takes
  GalaxyPPG's 64 Hz E4 wrist BVP (23 people, 430 seated-rest 60 s windows), decimates it to 25 Hz
  with a random sample phase, rebuilds it at 1 kHz and compares average-beat indices with the
  64 Hz reference:

| Index | Reference mean | Error SD, spline | Error SD, band-limited | Spread between people | Error / spread |
|---|---|---|---|---|---|
| Crest time | 234 ms | 1.2 ms | 0.4 ms | 44 ms | 0.03 / 0.01 |
| Width at half height | 496 ms | 1.3 ms | 0.9 ms | 36 ms | 0.04 / 0.03 |
| b/a (second derivative) | −0.575 | 0.003 | 0.001 | 0.068 | 0.04 / 0.02 |
| d/a (second derivative) | −0.163 | 0.013 | 0.005 | 0.042 | 0.30 / 0.12 |

Beats were found on the rebuilt signal itself; aligning them at the reference beat times changed
nothing, so beat timing at 25 Hz is not the limit either. Band-limited (polyphase) upsampling beats
cubic splines, most visibly for d/a.

**Caveat.** E4 BVP is already filtered by Empatica (almost no power above 10 Hz), so this tests
reconstruction of content below 12.5 Hz, not content the E4 never had. Together with the watch's
100 Hz spectrum it says the 25 Hz limit is not what stops pulse-shape work on wrist green PPG.
What can still stop it: how `hr_raw` is filtered and timed on the device (the hidden lost samples
above shift beats by 40 ms unless corrected), and noise. A still same-wrist 25 vs 100 Hz recording
settles it. Note also that within-person variation between 60 s windows is as large as the spread
between people for crest time (65 vs 44 ms), so single windows will not separate people; nightly
averages may.

## Reproduce

```bash
uv run python benchmarks/morph25/run.py
```

The channel and timing analysis used one-off scripts on the two recordings; their numbers are
above.
