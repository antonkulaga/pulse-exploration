/* Shared helpers for the phone pages: result rendering, waveform drawing, small utilities.
 * Plain script (no modules, no libraries). In a browser it defines window.PulseUI; under Node it
 * exports the same object so the pure parts (resultHTML, parseWaveform) can be tested.
 *
 * The result JSON comes from POST /camera/video or /camera/trace. Only some keys are guaranteed
 * (tier, mode, hr_bpm, hr_peaks_bpm, quality metrics, warnings); everything else is optional, and
 * any key the page does not know is still shown under "Details" so nothing is hidden.
 */
(function (root) {
  "use strict";

  const HANDLED = new Set([
    "tier", "mode", "hr_bpm", "hr_peaks_bpm", "quality", "warnings", "hrv", "hrv_reliable", "hrv_note",
    "resp_rate", "respiration", "deep", "methods", "primary_method", "primary_reason",
    "waveform", "waveform_fs", "waveform_peaks",
  ]);
  const HR_KEYS = ["hr_bpm", "hr", "bpm", "hr_fft_bpm"];

  function escapeHTML(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /** A finite number from a number or numeric string, else null. */
  function num(value) {
    if (typeof value === "number") return Number.isFinite(value) ? value : null;
    if (typeof value === "string" && value.trim() !== "") {
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function fmtNum(n) {
    if (Number.isInteger(n)) return String(n);
    const a = Math.abs(n);
    if (a >= 100) return n.toFixed(0);
    if (a >= 10) return n.toFixed(1);
    if (a >= 1) return n.toFixed(2);
    return n.toPrecision(3);
  }

  function prettyKey(key) {
    return String(key).replace(/_/g, " ");
  }

  /** HTML for any JSON value: scalars inline, short numeric arrays joined, objects as nested tables. */
  function valueHTML(value, depth) {
    depth = depth || 0;
    if (value === null || value === undefined) return '<span class="muted">—</span>';
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "number") return Number.isFinite(value) ? fmtNum(value) : '<span class="muted">—</span>';
    if (typeof value === "string") return escapeHTML(value);
    if (Array.isArray(value)) {
      if (value.length === 0) return '<span class="muted">none</span>';
      const scalars = value.every((v) => v === null || ["number", "string", "boolean"].includes(typeof v));
      if (scalars && value.length <= 8) return value.map((v) => valueHTML(v, depth + 1)).join(", ");
      return '<span class="muted">[' + value.length + " values]</span>";
    }
    if (isObject(value)) {
      if (depth >= 2) return "<code>" + escapeHTML(JSON.stringify(value)) + "</code>";
      return kvTableHTML(value, depth + 1);
    }
    return escapeHTML(String(value));
  }

  function kvTableHTML(obj, depth, skip) {
    const rows = Object.keys(obj)
      .filter((k) => !(skip && skip.has(k)))
      .map((k) => "<tr><th>" + escapeHTML(prettyKey(k)) + "</th><td>" + valueHTML(obj[k], depth) + "</td></tr>");
    if (rows.length === 0) return '<span class="muted">—</span>';
    return '<table class="kv"><tbody>' + rows.join("") + "</tbody></table>";
  }

  /**
   * Normalise the waveform field into {y, t, fs, peaks} or null.
   * Accepts: [numbers]; [[t, y], ...]; {y|values|signal|data, t|t_s|time, fs|sample_rate_hz|..., peaks}.
   */
  function parseWaveform(result) {
    if (!isObject(result)) return null;
    const w = result.waveform;
    let y = null;
    let t = null;
    let fs = num(result.waveform_fs);
    let peaks = Array.isArray(result.waveform_peaks) ? result.waveform_peaks : null;
    let peakTimes = null;
    let label = null;
    if (Array.isArray(w)) {
      if (w.length && Array.isArray(w[0])) {
        t = w.map((p) => num(p[0]));
        y = w.map((p) => num(p[1]));
      } else {
        y = w.map(num);
      }
    } else if (isObject(w)) {
      const ys = w.y || w.values || w.signal || w.data || w.ppg;
      if (Array.isArray(ys)) y = ys.map(num);
      const ts = w.t || w.t_s || w.time || w.time_s;
      if (Array.isArray(ts)) t = ts.map(num);
      fs = num(w.fs) || num(w.sample_rate) || num(w.sample_rate_hz) || num(w.fs_hz) || fs;
      if (Array.isArray(w.peaks)) peaks = w.peaks;
      peakTimes = Array.isArray(w.peaks_t_s) ? w.peaks_t_s : Array.isArray(w.peaks_t) ? w.peaks_t : null;
      label = typeof w.signal === "string" ? w.signal : null;
    }
    if (!y) return null;
    if (y.filter((v) => v !== null).length < 2) return null;
    if (t && t.length !== y.length) t = null;
    if (!peaks && peakTimes) peaks = timesToIndices(peakTimes, t, fs);
    if (peaks) peaks = peaks.map(num).filter((p) => p !== null && p >= 0 && p < y.length);
    return { y: y, t: t, fs: fs, peaks: peaks && peaks.length ? peaks : null, label: label };
  }

  /** Peak times (s) to sample indices, using the time axis if there is one, else the sample rate.
   *  Peaks outside the plotted time range are dropped. */
  function timesToIndices(times, t, fs) {
    const ts = times.map(num).filter((v) => v !== null).sort((a, b) => a - b);
    const axis = t ? t.map((v, i) => [v, i]).filter((p) => p[0] !== null) : [];
    if (axis.length >= 2) {
      const first = axis[0][0];
      const last = axis[axis.length - 1][0];
      const half = (last - first) / (axis.length - 1) / 2;
      const out = [];
      let j = 0;
      for (const pt of ts) {
        if (pt < first - half || pt > last + half) continue;
        while (j < axis.length - 1 && Math.abs(axis[j + 1][0] - pt) <= Math.abs(axis[j][0] - pt)) j++;
        out.push(axis[j][1]);
      }
      return out;
    }
    if (fs) return ts.map((pt) => Math.round(pt * fs));
    return null;
  }

  function waveDuration(wave) {
    if (!wave) return null;
    if (wave.t) {
      const ts = wave.t.filter((v) => v !== null);
      if (ts.length >= 2) return ts[ts.length - 1] - ts[0];
    }
    if (wave.fs) return (wave.y.length - 1) / wave.fs;
    return null;
  }

  function hrHTML(result) {
    const hr = num(result.hr_bpm);
    const peaks = num(result.hr_peaks_bpm);
    const main = hr !== null ? hr : peaks;
    let html = '<div class="hr">';
    if (main === null) {
      html += '<span class="hr-none">No heart rate</span>';
    } else {
      html += '<span class="hr-value">' + escapeHTML(Math.round(main)) + '</span><span class="hr-unit">bpm' +
        (hr !== null ? " (spectral)" : " (peaks)") + "</span>";
    }
    html += "</div>";
    const parts = [];
    if (hr !== null) parts.push("Spectral (FFT): <b>" + fmtNum(hr) + "</b> bpm");
    if (peaks !== null) parts.push("Peak count: <b>" + fmtNum(peaks) + "</b> bpm");
    if (hr !== null && peaks !== null) {
      const gap = Math.abs(hr - peaks);
      parts.push(gap > 5
        ? '<span class="badge warn">methods differ by ' + fmtNum(gap) + " bpm</span>"
        : '<span class="badge ok">methods agree within ' + fmtNum(gap) + " bpm</span>");
    }
    if (typeof result.primary_method === "string") {
      parts.push("signal: " + escapeHTML(result.primary_method) +
        (typeof result.primary_reason === "string" ? " (" + escapeHTML(result.primary_reason) + ")" : ""));
    }
    if (parts.length) html += '<p class="muted">' + parts.join(" · ") + "</p>";
    return html;
  }

  function qualityLabel(result) {
    const q = result.quality;
    return isObject(q) && typeof q.label === "string" ? q.label : null;
  }

  function labelBadge(label) {
    const cls = { good: "ok", fair: "warn", poor: "bad" }[String(label).toLowerCase()] || "plain";
    return '<span class="badge ' + cls + '">quality: ' + escapeHTML(label) + "</span>";
  }

  function badgesHTML(result) {
    const out = [];
    if (typeof result.tier === "string" && result.tier) {
      out.push('<span class="badge">tier: ' + escapeHTML(result.tier) + "</span>");
    } else {
      out.push('<span class="badge bad">tier missing</span>');
    }
    if (result.mode !== undefined && result.mode !== null) {
      out.push('<span class="badge plain">mode: ' + escapeHTML(result.mode) + "</span>");
    }
    const label = qualityLabel(result);
    if (label) out.push(labelBadge(label));
    return '<div class="result-badges">' + out.join("") + "</div>";
  }

  function warningsHTML(result) {
    const w = result.warnings;
    if (w === undefined || w === null) return "";
    const list = Array.isArray(w) ? w : [w];
    if (list.length === 0) return '<h3>Warnings</h3><p class="muted">None.</p>';
    return '<h3>Warnings</h3><ul class="warnings">' + list.map(warningItem).join("") + "</ul>";
  }

  function warningItem(x) {
    if (typeof x === "string") return "<li>" + escapeHTML(x) + "</li>";
    if (isObject(x) && (typeof x.message === "string" || typeof x.msg === "string")) {
      const sev = typeof x.severity === "string" ? x.severity : null;
      const cls = sev === "error" || sev === "critical" ? "bad" : sev === "info" ? "plain" : "warn";
      return '<li class="sev-' + escapeHTML(sev || "warning") + '">' +
        (sev && sev !== "warning" ? '<span class="badge ' + cls + '">' + escapeHTML(sev) + "</span> " : "") +
        escapeHTML(x.message || x.msg) +
        (typeof x.code === "string" ? ' <small class="muted">' + escapeHTML(x.code) + "</small>" : "") + "</li>";
    }
    return "<li>" + valueHTML(x, 2) + "</li>";
  }

  function qualityHTML(result) {
    const q = result.quality;
    if (q === undefined || q === null) return "";
    if (!isObject(q)) return "<h3>Signal quality</h3><p>" + valueHTML(q, 1) + "</p>";
    const label = qualityLabel(result);
    return "<h3>Signal quality " + (label ? labelBadge(label) : "") + "</h3>" + kvTableHTML(q, 1, new Set(["label"]));
  }

  function hrvHTML(result) {
    const h = result.hrv;
    const note = typeof result.hrv_note === "string" ? '<p class="muted">' + escapeHTML(result.hrv_note) + "</p>" : "";
    if (h === undefined || h === null) return note ? "<h3>HRV</h3>" + note : "";
    if (!isObject(h)) return "<h3>HRV</h3><p>" + valueHTML(h, 1) + "</p>" + note;
    const flag = h.hrv_reliable !== undefined ? h.hrv_reliable : result.hrv_reliable;
    let badge = "";
    if (flag === true) badge = '<span class="badge ok">reliable</span>';
    else if (flag === false) badge = '<span class="badge warn">not reliable: for reference only</span>';
    else badge = '<span class="badge plain">reliability unknown</span>';
    const skip = new Set(["hrv_reliable"]);
    return "<h3>HRV " + badge + "</h3>" + note + '<div class="' + (flag === false ? "muted" : "") + '">' +
      kvTableHTML(h, 1, skip) + "</div>";
  }

  function respHTML(result) {
    const r = result.resp_rate !== undefined && result.resp_rate !== null ? result.resp_rate : result.respiration;
    if (r === undefined || r === null) return "";
    const n = num(r);
    if (n !== null) return "<h3>Breathing</h3><p>about <b>" + fmtNum(n) + "</b> breaths/min</p>";
    if (!isObject(r)) return "<h3>Breathing</h3><p>" + valueHTML(r, 1) + "</p>";
    let rate = null;
    for (const k of ["rate_brpm", "brpm", "rate", "bpm"]) { if (rate === null) rate = num(r[k]); }
    const badge = r.reliable === true ? ' <span class="badge ok">reliable</span>'
      : r.reliable === false ? ' <span class="badge warn">not reliable</span>' : "";
    const rest = {};
    Object.keys(r).forEach((k) => { if (!["rate_brpm", "brpm", "rate", "bpm", "reliable"].includes(k)) rest[k] = r[k]; });
    return "<h3>Breathing" + badge + "</h3>" +
      (rate !== null ? "<p>about <b>" + fmtNum(rate) + "</b> breaths/min</p>" : '<p class="muted">No breathing rate.</p>') +
      (Object.keys(rest).length ? "<details><summary>Breathing details</summary>" + kvTableHTML(rest, 1) + "</details>" : "");
  }

  /** One row per model or method: HR first, then SQI/SNR, an error if any, everything else folded. */
  function modelTableHTML(entries) {
    const rows = entries.map(([name, v]) => {
      let hr = num(v);
      let line = "";
      let rest = "";
      if (isObject(v)) {
        hr = null;
        for (const k of HR_KEYS) { if (hr === null) hr = num(v[k]); }
        const bits = [];
        if (num(v.hr_peaks_bpm) !== null && num(v.hr_bpm) === null) bits.push("peaks " + fmtNum(num(v.hr_peaks_bpm)) + " bpm");
        if (num(v.sqi) !== null) bits.push("SQI " + fmtNum(num(v.sqi)));
        if (num(v.snr_db) !== null) bits.push("SNR " + fmtNum(num(v.snr_db)) + " dB");
        if (typeof v.error === "string") bits.push('<span class="status error">' + escapeHTML(v.error) + "</span>");
        line = bits.join(" · ");
        const others = {};
        Object.keys(v).forEach((k) => {
          if (!HR_KEYS.concat(["hr_peaks_bpm", "sqi", "snr_db", "error", "model", "name"]).includes(k)) others[k] = v[k];
        });
        if (Object.keys(others).length) rest = "<details><summary>more</summary>" + kvTableHTML(others, 2) + "</details>";
      } else if (hr === null) {
        line = valueHTML(v, 2);
      }
      return "<tr><th>" + escapeHTML(name) + "</th><td>" + (hr !== null ? "<b>" + fmtNum(hr) + "</b> bpm " : "") +
        (line ? '<span class="muted">' + line + "</span>" : "") + rest + "</td></tr>";
    });
    return '<table class="kv"><tbody>' + rows.join("") + "</tbody></table>";
  }

  function entriesOf(d) {
    if (Array.isArray(d)) {
      return d.map((item, i) => [isObject(item) ? (item.model || item.name || "model " + (i + 1)) : "model " + (i + 1), item]);
    }
    return Object.keys(d).map((k) => [k, d[k]]);
  }

  function deepHTML(result) {
    const d = result.deep;
    if (d === undefined || d === null) return "";
    const title = "<h3>Pretrained models</h3>";
    if (!isObject(d) && !Array.isArray(d)) return title + "<p>" + valueHTML(d, 1) + "</p>";
    // Either {model: result, ...} or a wrapper {available, reason, models: {...}, ...}.
    const wrapped = isObject(d) && (isObject(d.models) || Array.isArray(d.models) || "available" in d || "skipped" in d);
    const models = wrapped ? d.models : d;
    let html = title;
    if (wrapped) {
      if (d.available === false) html += '<p class="muted">Not available' + (d.reason ? ": " + escapeHTML(d.reason) : "") + ".</p>";
      else if (d.skipped) html += '<p class="muted">Skipped for this recording.</p>';
    }
    const entries = models && (isObject(models) || Array.isArray(models)) ? entriesOf(models) : [];
    if (entries.length) html += modelTableHTML(entries);
    else if (!wrapped) html += '<p class="muted">None ran.</p>';
    if (wrapped) {
      const info = {};
      Object.keys(d).forEach((k) => { if (!["models", "available", "reason", "skipped"].includes(k)) info[k] = d[k]; });
      if (Object.keys(info).length) html += "<details><summary>Model run details</summary>" + kvTableHTML(info, 1) + "</details>";
    }
    return html;
  }

  function methodsHTML(result) {
    const m = result.methods;
    if (!isObject(m) && !Array.isArray(m)) return "";
    const entries = entriesOf(m);
    if (!entries.length) return "";
    return "<details><summary>Per-method estimates</summary>" + modelTableHTML(entries) + "</details>";
  }

  function otherHTML(result) {
    const keys = Object.keys(result).filter((k) => !HANDLED.has(k));
    if (keys.length === 0) return "";
    const rest = {};
    keys.forEach((k) => { rest[k] = result[k]; });
    return "<details><summary>Details</summary>" + kvTableHTML(rest, 1) + "</details>";
  }

  /** The whole result block as an HTML string. Never throws on odd or missing fields. */
  function resultHTML(result) {
    if (!isObject(result)) {
      return '<h2>Result</h2><p class="status error">The server returned something unexpected.</p>' +
        '<div class="raw"><pre>' + escapeHTML(JSON.stringify(result, null, 2)) + "</pre></div>";
    }
    const wave = parseWaveform(result);
    let waveBlock = "";
    if (wave) {
      const dur = waveDuration(wave);
      waveBlock = '<h3>Waveform</h3><canvas class="waveform" data-wave="1" aria-label="pulse waveform"></canvas>' +
        '<div class="wave-caption">' + (wave.label ? escapeHTML(wave.label) + " · " : "") + wave.y.length + " samples" +
        (dur ? " · " + fmtNum(dur) + " s" : "") +
        (wave.peaks ? " · " + wave.peaks.length + " detected beats" : "") + "</div>";
    }
    let raw = "";
    try { raw = JSON.stringify(result, null, 2); } catch (e) { raw = String(result); }
    return "<h2>Result</h2>" + badgesHTML(result) + hrHTML(result) + warningsHTML(result) + qualityHTML(result) +
      waveBlock + methodsHTML(result) + hrvHTML(result) + respHTML(result) + deepHTML(result) + otherHTML(result) +
      '<details class="raw"><summary>Raw JSON</summary><pre>' + escapeHTML(raw) + "</pre></details>";
  }

  function cssVar(name, fallback) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    } catch (e) {
      return fallback;
    }
  }

  /** Draw {y, t?, peaks?} on a canvas, auto-scaled, sharp on high-DPI screens. Nulls make gaps. */
  function drawWaveform(canvas, wave, opts) {
    opts = opts || {};
    canvas.__wave = wave;
    canvas.__waveOpts = opts;
    const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
    const cssW = canvas.clientWidth || 300;
    const cssH = canvas.clientHeight || 140;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const ys = (wave && wave.y) || [];
    const vals = ys.filter((v) => v !== null && Number.isFinite(v));
    ctx.strokeStyle = cssVar("--wave-grid", "#ddd");
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const gy = Math.round((cssH * i) / 4) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cssW, gy); ctx.stroke();
    }
    if (vals.length < 2) return;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (hi - lo < 1e-12) { hi += 0.5; lo -= 0.5; }
    const pad = 8;
    const n = ys.length;
    const xOf = (i) => pad + ((cssW - 2 * pad) * i) / Math.max(1, n - 1);
    const yOf = (v) => {
      const k = (v - lo) / (hi - lo);
      return cssH - pad - k * (cssH - 2 * pad);
    };
    ctx.strokeStyle = opts.color || cssVar("--wave", "#c8102e");
    ctx.lineWidth = opts.lineWidth || 1.6;
    ctx.lineJoin = "round";
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const v = ys[i];
      if (v === null || !Number.isFinite(v)) { pen = false; continue; }
      if (pen) ctx.lineTo(xOf(i), yOf(v)); else { ctx.moveTo(xOf(i), yOf(v)); pen = true; }
    }
    ctx.stroke();
    if (wave.peaks) {
      ctx.fillStyle = cssVar("--text", "#000");
      for (const p of wave.peaks) {
        const i = Math.round(p);
        const v = ys[i];
        if (v === null || !Number.isFinite(v)) continue;
        ctx.beginPath(); ctx.arc(xOf(i), yOf(v), 2.5, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  function redrawAll() {
    document.querySelectorAll("canvas").forEach((c) => {
      if (c.__wave) drawWaveform(c, c.__wave, c.__waveOpts);
    });
  }

  /** Render a result into a container element (and draw its waveform). */
  function renderResult(container, result) {
    container.hidden = false;
    container.innerHTML = resultHTML(result);
    const canvas = container.querySelector("canvas[data-wave]");
    const wave = parseWaveform(result);
    if (canvas && wave) drawWaveform(canvas, wave);
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /** A readable message for a failed API call (FastAPI errors carry {detail}). */
  function errorMessage(status, bodyText, endpoint) {
    let detail = "";
    try {
      const j = JSON.parse(bodyText);
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail !== undefined ? j.detail : j);
    } catch (e) {
      detail = (bodyText || "").slice(0, 300);
    }
    if (status === 404 || status === 405) {
      return "The laptop server does not offer " + endpoint + " (HTTP " + status + "). Is the camera API included in the running server?";
    }
    if (status === 413) return "The file is too large for the server (HTTP 413).";
    return "Server error HTTP " + status + (detail ? ": " + detail : "");
  }

  /**
   * Point elements with data-https-path at the HTTPS server when this page came over plain HTTP.
   * The HTTPS port is read from <body data-https-port>, default 8766.
   */
  function linkHttps() {
    const port = (document.body && document.body.dataset.httpsPort) || "8766";
    document.querySelectorAll("[data-https-path]").forEach((a) => {
      const path = a.getAttribute("data-https-path") || "/";
      const url = location.protocol === "https:" ? path : "https://" + location.hostname + ":" + port + path;
      a.setAttribute("href", url);
      if (a.hasAttribute("data-show-url")) a.textContent = location.protocol === "https:" ? location.origin + path : url;
    });
    document.querySelectorAll("[data-origin]").forEach((el) => { el.textContent = location.origin; });
  }

  function loadPref(key, fallback) {
    try { return localStorage.getItem("pulse." + key) || fallback; } catch (e) { return fallback; }
  }

  function savePref(key, value) {
    try { localStorage.setItem("pulse." + key, value); } catch (e) { /* storage blocked: fine */ }
  }

  function fmtBytes(n) {
    if (!Number.isFinite(n)) return "?";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(0) + " kB";
    return n + " B";
  }

  const api = {
    escapeHTML, num, fmtNum, fmtBytes, parseWaveform, resultHTML, drawWaveform, renderResult, errorMessage,
    linkHttps, loadPref, savePref,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.PulseUI = api;
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    let pending = 0;
    window.addEventListener("resize", () => {
      cancelAnimationFrame(pending);
      pending = requestAnimationFrame(redrawAll);
    });
    try {
      matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll); // colours come from CSS
    } catch (e) { /* old browsers: a resize still redraws */ }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", linkHttps);
    else linkHttps();
  }
})(typeof window !== "undefined" ? window : null);
