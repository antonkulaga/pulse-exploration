/* Pure signal helpers for the capture page: ROI colour means, timestamp choice, trace assembly.
 * No DOM access, so Node can test them. In a browser this defines window.PulseSignal.
 */
(function (root) {
  "use strict";

  /** Mean R, G, B over RGBA pixel data, plus the fraction of pixels with any channel >= 250. */
  function roiStats(data) {
    let r = 0;
    let g = 0;
    let b = 0;
    let sat = 0;
    const n = Math.floor(data.length / 4);
    if (n === 0) return null;
    for (let i = 0; i < n * 4; i += 4) {
      const R = data[i];
      const G = data[i + 1];
      const B = data[i + 2];
      r += R; g += G; b += B;
      if (R >= 250 || G >= 250 || B >= 250) sat++;
    }
    return { r: r / n, g: g / n, b: b / n, sat: sat / n, n: n };
  }

  // Timestamp sources per frame, in order of preference. mediaTime is the frame's presentation
  // timestamp (seconds, converted to ms when recorded); captureTime is when the camera captured it;
  // expectedDisplayTime is quantised to the display refresh; "now" is the callback time.
  const TS_SOURCES = ["mediaTime", "captureTime", "expectedDisplayTime", "now"];

  function strictlyIncreasing(values) {
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (typeof v !== "number" || !Number.isFinite(v)) return false;
      if (i > 0 && !(v > values[i - 1])) return false;
    }
    return values.length > 0;
  }

  /** Pick the first source present on every frame and strictly increasing; t_ms is relative to frame 0. */
  function chooseTimestamps(frames) {
    for (const source of TS_SOURCES) {
      const vals = frames.map((f) => f[source]);
      if (strictlyIncreasing(vals)) {
        const t0 = vals[0];
        return { source: source, t_ms: vals.map((v) => round(v - t0, 3)) };
      }
    }
    // Should not happen (callback time always advances), but never send unordered times silently.
    const vals = frames.map((f, i) => (Number.isFinite(f.now) ? f.now : i));
    const t0 = vals[0] || 0;
    return { source: "now-unordered", t_ms: vals.map((v) => round(v - t0, 3)) };
  }

  function round(x, digits) {
    const k = Math.pow(10, digits);
    return Math.round(x * k) / k;
  }

  function median(values) {
    if (!values.length) return null;
    const s = values.slice().sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  /** Frame-rate statistics from relative timestamps in ms. */
  function fpsStats(tMs) {
    const n = tMs.length;
    if (n < 2) return { n_frames: n, duration_s: 0, mean_fps: null, median_fps: null, jitter_ms: null };
    const dts = [];
    for (let i = 1; i < n; i++) dts.push(tMs[i] - tMs[i - 1]);
    const dur = (tMs[n - 1] - tMs[0]) / 1000;
    const med = median(dts);
    const mean = dts.reduce((a, b) => a + b, 0) / dts.length;
    const sd = Math.sqrt(dts.reduce((a, b) => a + (b - mean) * (b - mean), 0) / dts.length);
    return {
      n_frames: n,
      duration_s: round(dur, 3),
      mean_fps: dur > 0 ? round((n - 1) / dur, 2) : null,
      median_fps: med > 0 ? round(1000 / med, 2) : null,
      jitter_ms: round(sd, 2),
      max_gap_ms: round(Math.max.apply(null, dts), 2),
    };
  }

  /** Frames the page never saw, from rVFC presentedFrames (null when unknown). */
  function framesMissed(frames) {
    const pf = frames.map((f) => f.presentedFrames);
    if (!pf.length || pf.some((v) => typeof v !== "number")) return null;
    let missed = 0;
    for (let i = 1; i < pf.length; i++) missed += Math.max(0, pf[i] - pf[i - 1] - 1);
    return missed;
  }

  /**
   * Build the POST /camera/trace body from recorded frames:
   * {mode, t_ms, r, g, b, meta}. Extra per-frame details stay inside meta.
   */
  function buildTrace(frames, mode, meta) {
    const ts = chooseTimestamps(frames);
    const alt = {};
    for (const source of ["captureTime", "expectedDisplayTime"]) {
      if (source === ts.source) continue;
      const vals = frames.map((f) => f[source]);
      if (vals.length && vals.every((v) => typeof v === "number" && Number.isFinite(v))) {
        alt[source] = vals.map((v) => round(v - vals[0], 3));
      }
    }
    const sats = frames.map((f) => (typeof f.sat === "number" ? f.sat : 0));
    const fullMeta = Object.assign({}, meta || {}, {
      timestamp_source: ts.source,
      measured_fps: fpsStats(ts.t_ms),
      frames_missed: framesMissed(frames),
      saturation: {
        mean_fraction: sats.length ? round(sats.reduce((a, b) => a + b, 0) / sats.length, 4) : null,
        max_fraction: sats.length ? round(Math.max.apply(null, sats), 4) : null,
      },
    });
    if (Object.keys(alt).length) fullMeta.timestamps_alt_ms = alt;
    return {
      mode: mode,
      t_ms: ts.t_ms,
      r: frames.map((f) => round(f.r, 3)),
      g: frames.map((f) => round(f.g, 3)),
      b: frames.map((f) => round(f.b, 3)),
      meta: fullMeta,
    };
  }

  /** Detrend for display: subtract a centred moving mean (window in samples). */
  function detrend(values, win) {
    const n = values.length;
    const out = new Array(n);
    const half = Math.max(1, win >> 1);
    let sum = 0;
    let lo = 0;
    let hi = -1;
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - half);
      const b = Math.min(n - 1, i + half);
      while (hi < b) { hi++; sum += values[hi]; }
      while (lo < a) { sum -= values[lo]; lo++; }
      out[i] = values[i] - sum / (hi - lo + 1);
    }
    return out;
  }

  const api = { roiStats, chooseTimestamps, fpsStats, framesMissed, buildTrace, detrend, median, TS_SOURCES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.PulseSignal = api;
})(typeof window !== "undefined" ? window : null);
