// app.js - wires the screening lab page: loads the model, runs the seven phases on the
// chosen leaf, and draws every intermediate result. All processing stays in the browser.
import * as ip from "./imgproc.js";
import { SmallCNN, probDiseased } from "./cnn.js";
import { bayesThreshold, decideFirst, decideSecond, expectedCosts, searchThresholds } from "./agent.js";

const $ = (id) => document.getElementById(id);
const S = 128;                                   // model input size
const state = { model: null, meta: null, clean: null, source: null, seed: 42, thr: null, costs: null,
                first: null, second: null, busy: false, pending: false };

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function putRGB(canvas, im) {
  canvas.width = im.w; canvas.height = im.h;
  const ctx = canvas.getContext("2d"), data = ctx.createImageData(im.w, im.h);
  for (let i = 0; i < im.w * im.h; i++) {
    if (im.c === 3) { data.data[4 * i] = im.d[3 * i]; data.data[4 * i + 1] = im.d[3 * i + 1]; data.data[4 * i + 2] = im.d[3 * i + 2]; }
    else { data.data[4 * i] = data.data[4 * i + 1] = data.data[4 * i + 2] = im.d[i]; }
    data.data[4 * i + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
}
const drawRGB = (id, im) => putRGB(typeof id === "string" ? $(id) : id, im);

// Inferno-like ramp for error / activation maps
const RAMP = [[0, 0, 4], [66, 10, 104], [147, 38, 103], [221, 81, 58], [252, 165, 10], [252, 255, 164]];
function ramp(t) {
  t = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i, a = RAMP[i], b = RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
// Scalar map (any numeric array) -> canvas. mode "gray" or "ramp"; lo/hi fix the scale.
function drawMap(canvas, arr, w, h, { mode = "gray", lo = null, hi = null } = {}) {
  if (typeof canvas === "string") canvas = $(canvas);
  canvas.width = w; canvas.height = h;
  let mn = lo, mx = hi;
  if (mn === null || mx === null) {
    let a = Infinity, b = -Infinity;
    for (let i = 0; i < w * h; i++) { if (arr[i] < a) a = arr[i]; if (arr[i] > b) b = arr[i]; }
    if (mn === null) mn = a; if (mx === null) mx = b;
  }
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h), span = mx - mn || 1;
  for (let i = 0; i < w * h; i++) {
    const t = (arr[i] - mn) / span;
    const [r, g, b] = mode === "ramp" ? ramp(t) : [255 * t, 255 * t, 255 * t];
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
const drawMask = (id, m) => drawMap(id, m, S, S, { lo: 0, hi: 1 });

function chips(id, items) {
  $(id).innerHTML = items.map(({ k, v, cls = "" }) =>
    `<div class="chip ${cls}"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
}
const gateChip = (k, ok, detail) => ({ k, v: `${ok ? "pass" : "fail"} · ${detail}`, cls: ok ? "pass" : "fail" });
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "∞");
const fmtP = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(p < 0.1 ? 4 : 3));

// Histogram chart (one or two series) with an optional vertical marker
function drawHist(id, series, marker = null) {
  const cv = $(id), dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 720, H = cv.clientHeight || 120;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const pad = 8, bw = (W - 2 * pad) / 256;
  const max = Math.max(...series.flatMap((s) => Array.from(s.h)), 1);
  for (const s of series) {
    ctx.fillStyle = s.color; ctx.globalAlpha = s.alpha ?? 0.85;
    for (let i = 0; i < 256; i++) {
      const bh = (s.h[i] / max) * (H - 2 * pad);
      ctx.fillRect(pad + i * bw, H - pad - bh, Math.max(bw - 0.3, 0.6), bh);
    }
  }
  ctx.globalAlpha = 1;
  if (marker !== null) {
    ctx.strokeStyle = css("--lesion"); ctx.lineWidth = 2;
    const x = pad + (marker + 0.5) * bw;
    ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke();
    ctx.fillStyle = css("--lesion"); ctx.font = "11px JetBrains Mono, monospace";
    ctx.fillText(`t = ${marker}`, Math.min(x + 4, W - 50), pad + 10);
  }
}

// ---------------------------------------------------------------------------
// Image input
// ---------------------------------------------------------------------------
async function bitmapToImage(bmp) {
  // Large photos are first scaled by the browser to <= 1024 px, then area-resized exactly.
  const scale = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale)), h = Math.max(1, Math.round(bmp.height * scale));
  const cv = document.createElement("canvas");
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data, im = ip.image(w, h, 3);
  for (let i = 0; i < w * h; i++) { im.d[3 * i] = rgba[4 * i]; im.d[3 * i + 1] = rgba[4 * i + 1]; im.d[3 * i + 2] = rgba[4 * i + 2]; }
  return ip.resize(ip.centerSquare(im), S, S);
}

async function decodeBlob(blob) {
  try { return await createImageBitmap(blob, { imageOrientation: "from-image" }); }
  catch { return await createImageBitmap(blob); }
}

async function useBlob(blob, source) {
  setStatus("Decoding the photo…");
  try {
    state.clean = await bitmapToImage(await decodeBlob(blob));
  } catch {
    setStatus("That file could not be read as an image. Try a JPEG or PNG.");
    return;
  }
  state.source = source;
  state.seed = 42;
  state.second = null;
  schedule();
}

// JPEG (or PNG) round trip through the browser's own encoder: returns the decoded image and size
async function transmit(im, quality) {
  const cv = document.createElement("canvas");
  putRGB(cv, im);
  const type = quality === null ? "image/png" : "image/jpeg";
  const blob = await new Promise((res) => cv.toBlob(res, type, quality === null ? undefined : quality / 100));
  if (quality === null) return { img: im, bytes: blob.size };
  const bmp = await createImageBitmap(blob), c2 = document.createElement("canvas");
  c2.width = im.w; c2.height = im.h;
  const ctx = c2.getContext("2d");
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, im.w, im.h).data, out = ip.image(im.w, im.h, 3);
  for (let i = 0; i < im.w * im.h; i++) { out.d[3 * i] = rgba[4 * i]; out.d[3 * i + 1] = rgba[4 * i + 1]; out.d[3 * i + 2] = rgba[4 * i + 2]; }
  return { img: out, bytes: blob.size };
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
function settings() {
  return {
    sim: $("c-sim").checked, sigma: +$("c-sigma").value, imin: +$("c-imin").value, imax: +$("c-imax").value,
    restore: $("c-restore").value, enhance: $("c-enhance").value,
    q: $("c-lossless").checked ? null : +$("c-q").value, seg: $("c-seg").checked,
  };
}
function readCosts() {
  const num = (id, d) => { const v = parseFloat($(id).value); return Number.isFinite(v) && v >= 0 ? v : d; };
  return { fn: Math.max(num("c-fn", 50), 0.01), fp: Math.max(num("c-fp", 1), 0.01), reimage: num("c-ri", 2) };
}
function updateOutputs() {
  $("o-sigma").textContent = $("c-sigma").value;
  $("o-imin").textContent = (+$("c-imin").value).toFixed(2);
  $("o-imax").textContent = (+$("c-imax").value).toFixed(2);
  $("o-q").textContent = $("c-lossless").checked ? "lossless" : $("c-q").value;
  $("c-q").disabled = $("c-lossless").checked;
  const sim = $("c-sim").checked;
  for (const id of ["c-sigma", "c-imin", "c-imax", "c-reroll"]) $(id).disabled = !sim;
}

function recomputeThresholds() {
  state.costs = readCosts();
  state.thr = searchThresholds(state.meta.val, state.costs);
  const tb = bayesThreshold(state.costs);
  $("o-thr").textContent = `Bayes threshold ${fmtP(tb)}. Re-image band [${fmtP(state.thr.tLow)}, ${fmtP(state.thr.tHigh)}], ` +
    `chosen on ${state.meta.val.y.length} validation leaves (validation cost ${state.thr.cost.toFixed(3)} per leaf).`;
}

// ---------------------------------------------------------------------------
// The pipeline for one capture (attempt 0 = first photo, 1 = re-image)
// ---------------------------------------------------------------------------
async function screen(clean, st, seed) {
  const g = state.meta.gates, t = {};
  const yieldUI = () => new Promise((r) => setTimeout(r, 0));
  const cap = st.sim ? ip.degrade(clean, { noise_sigma: st.sigma, illum_min: st.imin, illum_max: Math.max(st.imax, st.imin) }, seed)
                     : { degraded: clean, reference: clean, field: null };
  const meanG = ip.meanGray(cap.degraded), gate1 = meanG >= g.min_mean && meanG <= g.max_mean;

  let t0 = performance.now();
  const restored = ip.restore(cap.degraded, st.restore);
  t.restore = performance.now() - t0; await yieldUI();
  t0 = performance.now();
  const enhanced = ip.enhance(restored, st.enhance);
  t.enhance = performance.now() - t0; await yieldUI();
  t0 = performance.now();
  const tx = await transmit(enhanced, st.q);
  t.compress = performance.now() - t0;
  t0 = performance.now();
  let seg = null, crop = tx.img, gate2 = true;
  if (st.seg) {
    seg = ip.segment(tx.img);
    gate2 = seg.area >= g.mask_area_min && seg.area <= g.mask_area_max;
    crop = ip.applyMask(tx.img, seg.mask, S);
  }
  t.segment = performance.now() - t0; await yieldUI();
  t0 = performance.now();
  const out = state.model.forward(crop);
  t.cnn = performance.now() - t0;
  const p = probDiseased(out.logits, state.meta.temperature);
  return { clean, cap, meanG, gate1, restored, enhanced, tx, seg, crop, gate2, out, p, t, st };
}

// ---------------------------------------------------------------------------
// Rendering each phase
// ---------------------------------------------------------------------------
function render(r) {
  const { st } = r;
  // Phase 1
  drawRGB("p1-clean", r.clean);
  if (r.cap.field) drawMap("p1-illum", r.cap.field, S, S, { lo: 0, hi: 1.6 });
  else drawMap("p1-illum", new Float32Array(S * S).fill(1), S, S, { lo: 0, hi: 1.6 });
  drawRGB("p1-ref", r.cap.reference);
  drawRGB("p1-cap", r.cap.degraded);
  chips("p1-chips", [
    { k: "noise σ", v: st.sim ? st.sigma : "off" },
    { k: "light gain", v: st.sim ? `${st.imin.toFixed(2)}–${Math.max(st.imax, st.imin).toFixed(2)}` : "off" },
    gateChip("gate 1 exposure", r.gate1, `mean ${r.meanG.toFixed(0)}`),
  ]);

  // Phase 2
  drawRGB("p2-in", r.cap.degraded);
  drawRGB("p2-out", r.restored);
  drawRGB("p2-ref", r.cap.reference);
  const removed = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++)
    removed[i] = (Math.abs(r.cap.degraded.d[3 * i] - r.restored.d[3 * i]) + Math.abs(r.cap.degraded.d[3 * i + 1] - r.restored.d[3 * i + 1]) + Math.abs(r.cap.degraded.d[3 * i + 2] - r.restored.d[3 * i + 2])) / 3;
  drawMap("p2-removed", removed, S, S, { mode: "ramp", lo: 0, hi: 64 });
  $("p2-method").textContent = `(${$("c-restore").selectedOptions[0].text})`;
  if (st.sim) {
    chips("p2-chips", [
      { k: "PSNR before", v: `${fmt(ip.psnr(r.cap.reference, r.cap.degraded))} dB` },
      { k: "PSNR after", v: `${fmt(ip.psnr(r.cap.reference, r.restored))} dB` },
      { k: "SSIM before", v: fmt(ip.ssim(r.cap.reference, r.cap.degraded), 3) },
      { k: "SSIM after", v: fmt(ip.ssim(r.cap.reference, r.restored), 3) },
      { k: "time", v: `${r.t.restore.toFixed(0)} ms` },
    ]);
  } else {
    chips("p2-chips", [{ k: "scores", v: "need a reference: turn on simulation" }, { k: "time", v: `${r.t.restore.toFixed(0)} ms` }]);
  }

  // Phase 3
  drawRGB("p3-in", r.restored);
  drawRGB("p3-out", r.enhanced);
  $("p3-method").textContent = `(${$("c-enhance").selectedOptions[0].text})`;
  drawHist("p3-hist", [
    { h: ip.histogram(ip.gray(r.restored).d), color: css("--muted"), alpha: 0.55 },
    { h: ip.histogram(ip.gray(r.enhanced).d), color: css("--leaf"), alpha: 0.75 },
  ]);
  chips("p3-chips", [
    { k: "RMS contrast", v: `${fmt(ip.rmsContrast(r.restored), 3)} → ${fmt(ip.rmsContrast(r.enhanced), 3)}` },
    { k: "entropy (bits)", v: `${fmt(ip.entropy(r.restored))} → ${fmt(ip.entropy(r.enhanced))}` },
    { k: "time", v: `${r.t.enhance.toFixed(0)} ms` },
  ]);

  // Phase 4
  drawRGB("p4-in", r.enhanced);
  drawRGB("p4-out", r.tx.img);
  const err = new Float32Array(S * S);
  for (let i = 0; i < S * S; i++)
    err[i] = (Math.abs(r.enhanced.d[3 * i] - r.tx.img.d[3 * i]) + Math.abs(r.enhanced.d[3 * i + 1] - r.tx.img.d[3 * i + 1]) + Math.abs(r.enhanced.d[3 * i + 2] - r.tx.img.d[3 * i + 2])) / 3;
  drawMap("p4-err", err, S, S, { mode: "ramp", lo: 0, hi: 40 });
  drawRGB("p4-zoom", ip.crop(r.tx.img, 48, 48, 32, 32));
  const raw = S * S * 3, ratio = raw / r.tx.bytes;
  chips("p4-chips", [
    { k: "format", v: st.q === null ? "PNG (lossless)" : `JPEG q${st.q}` },
    { k: "bytes sent", v: `${r.tx.bytes.toLocaleString()} of ${raw.toLocaleString()}` },
    { k: "ratio", v: `${ratio.toFixed(1)} : 1` },
    { k: "bits / pixel", v: ((8 * r.tx.bytes) / (S * S)).toFixed(2) },
    { k: "PSNR", v: st.q === null ? "∞ (identical)" : `${fmt(ip.psnr(r.enhanced, r.tx.img))} dB` },
  ]);

  // Phase 5
  const segIds = ["p5-sat", "p5-bin", "p5-open", "p5-close", "p5-cc", "p5-fill", "p5-overlay", "p5-crop"];
  if (r.seg) {
    const s = r.seg;
    drawRGB("p5-sat", s.sat);
    drawMask("p5-bin", s.binary); drawMask("p5-open", s.opened); drawMask("p5-close", s.closed);
    drawMask("p5-cc", s.largest); drawMask("p5-fill", s.mask);
    $("p5-t").textContent = `(t = ${s.threshold})`;
    $("p5-ncc").textContent = `(of ${s.components})`;
    const ov = ip.clone(r.tx.img), m = s.mask;
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const i = y * S + x;
      const edge = m[i] && (x === 0 || y === 0 || x === S - 1 || y === S - 1 || !m[i - 1] || !m[i + 1] || !m[i - S] || !m[i + S]);
      if (edge) { ov.d[3 * i] = 255; ov.d[3 * i + 1] = 200; ov.d[3 * i + 2] = 0; }
      else if (!m[i]) for (let k = 0; k < 3; k++) ov.d[3 * i + k] = ov.d[3 * i + k] * 0.35;
    }
    drawRGB("p5-overlay", ov);
    drawRGB("p5-crop", r.crop);
    drawHist("p5-hist", [{ h: ip.histogram(s.blur.d), color: css("--muted"), alpha: 0.7 }], s.threshold);
    chips("p5-chips", [
      { k: "Otsu threshold", v: s.threshold },
      { k: "components", v: s.components },
      { k: "leaf area", v: `${(100 * s.area).toFixed(1)}%` },
      gateChip("gate 2 mask area", r.gate2, "5–95% allowed"),
      { k: "time", v: `${r.t.segment.toFixed(0)} ms` },
    ]);
  } else {
    for (const id of segIds) { const c = $(id), ctx = c.getContext("2d"); ctx.clearRect(0, 0, c.width, c.height); }
    drawRGB("p5-crop", r.crop);
    $("p5-t").textContent = ""; $("p5-ncc").textContent = "";
    drawHist("p5-hist", [{ h: new Float64Array(256), color: css("--muted") }]);
    chips("p5-chips", [{ k: "segmentation", v: "off: the whole frame goes to the CNN" }, gateChip("gate 2", true, "skipped")]);
  }

  // Phase 6
  const blocks = [["input", "3×128×128"], ["conv1 32", "→ 32×64×64"], ["conv2 64", "→ 64×32×32"], ["conv3 128", "→ 128×16×16"], ["conv4 128", "→ 128×8×8"], ["GAP", "→ 128"], ["linear", "→ 2 logits"], ["softmax / T", "→ P"]];
  $("p6-arch").innerHTML = blocks.map(([a, b]) => `<span>${a} <b>${b}</b></span>`).join("<span style='border:0;background:none;padding:0'>→</span>");
  const maps = $("p6-maps");
  maps.innerHTML = "";
  const inRow = document.createElement("div");
  inRow.className = "fmrow";
  inRow.innerHTML = `<div class="lab">input crop<br>128×128 RGB</div><div class="maps"></div>`;
  const ic = document.createElement("canvas");
  ic.style.gridColumn = "span 2";
  drawRGB(ic, r.crop);
  inRow.querySelector(".maps").appendChild(ic);
  maps.appendChild(inRow);
  r.out.acts.forEach((a, k) => {
    const row = document.createElement("div");
    row.className = "fmrow";
    row.innerHTML = `<div class="lab">block ${k + 1}: ${a.C}×${a.H}×${a.W}<br>8 strongest maps</div><div class="maps"></div>`;
    const means = Array.from({ length: a.C }, (_, c) => { let s = 0; for (let i = 0; i < a.H * a.W; i++) s += a.data[c * a.H * a.W + i]; return [s, c]; });
    means.sort((x, y) => y[0] - x[0]);
    for (const [, c] of means.slice(0, 8)) {
      const cv = document.createElement("canvas");
      drawMap(cv, a.data.subarray(c * a.H * a.W, (c + 1) * a.H * a.W), a.W, a.H, { mode: "ramp" });
      cv.title = `channel ${c}`;
      row.querySelector(".maps").appendChild(cv);
    }
    maps.appendChild(row);
  });
  const T = state.meta.temperature;
  chips("p6-chips", [
    { k: "logits z₀ / z₁", v: `${r.out.logits[0].toFixed(2)} / ${r.out.logits[1].toFixed(2)}` },
    { k: "temperature T", v: T.toFixed(3) },
    { k: "P(diseased)", v: fmtP(r.p) },
    { k: "parameters", v: state.model.nParams.toLocaleString() },
    { k: "time", v: `${r.t.cnn.toFixed(0)} ms` },
  ]);
}

// Log-odds axis with the CLEAR / RE-IMAGE / FLAG regions, the Bayes threshold and p
function drawAxis(p, p2) {
  const svg = $("p7-axis"), W = 640, x0 = 20, x1 = 620, lo = 1e-4, hi = 1 - 1e-4;
  const lg = (v) => Math.log(v / (1 - v));
  const X = (v) => x0 + ((lg(Math.min(hi, Math.max(lo, v))) - lg(lo)) / (lg(hi) - lg(lo))) * (x1 - x0);
  const { tLow, tHigh } = state.thr, tb = bayesThreshold(state.costs);
  const rect = (a, b, color) => `<rect x="${X(a)}" y="30" width="${Math.max(0, X(b) - X(a))}" height="18" fill="${color}"/>`;
  let s = rect(lo, tLow, css("--leaf-soft")) + rect(tLow, tHigh, css("--amber-soft")) + rect(tHigh, hi, css("--lesion-soft"));
  s += `<line x1="${x0}" y1="48" x2="${x1}" y2="48" stroke="${css("--line")}"/>`;
  for (const v of [0.001, 0.01, 0.1, 0.5, 0.9, 0.99]) s += `<line x1="${X(v)}" y1="48" x2="${X(v)}" y2="53" stroke="${css("--muted")}"/><text x="${X(v)}" y="66" text-anchor="middle">${v}</text>`;
  s += `<line x1="${X(tb)}" y1="26" x2="${X(tb)}" y2="52" stroke="${css("--fg")}" stroke-dasharray="3 2"/><text x="${X(tb)}" y="86" text-anchor="middle">Bayes ${fmtP(tb)}</text>`;
  const mid = (a, b) => (X(a) + X(b)) / 2;
  if (X(tLow) - x0 > 40) s += `<text x="${mid(lo, tLow)}" y="43" text-anchor="middle" class="lbl" style="fill:${css("--leaf")}">CLEAR</text>`;
  if (X(tHigh) - X(tLow) > 60) s += `<text x="${mid(tLow, tHigh)}" y="43" text-anchor="middle" class="lbl" style="fill:${css("--amber")}">RE-IMAGE</text>`;
  if (x1 - X(tHigh) > 40) s += `<text x="${mid(tHigh, hi)}" y="43" text-anchor="middle" class="lbl" style="fill:${css("--lesion")}">FLAG</text>`;
  const mark = (v, label) => `<path d="M${X(v)} 28 l-6 -10 h12 z" fill="${css("--fg")}"/><text x="${Math.min(W - 60, Math.max(30, X(v)))}" y="14" text-anchor="middle" class="lbl">${label} ${fmtP(v)}</text>`;
  s += mark(p, p2 === null ? "p =" : "p₁ =");
  if (p2 !== null) s += `<path d="M${X(p2)} 50 l-6 10 h12 z" fill="${css("--amber")}"/>`;
  svg.innerHTML = s;
}

function renderDecision() {
  const r = state.first;
  if (!r) return;
  const c = state.costs, { tLow, tHigh } = state.thr, gateOk = r.gate1 && r.gate2;
  const d1 = decideFirst(r.p, gateOk, tLow, tHigh);
  const e = expectedCosts(r.p, c);
  const best = e.clear <= e.flag ? "CLEAR" : "FLAG";
  $("p7-table").innerHTML = `<thead><tr><th>Action</th><th>Expected cost</th><th>How it is computed</th></tr></thead><tbody>
    <tr class="${best === "CLEAR" ? "best" : ""}"><td>CLEAR</td><td>${e.clear.toFixed(3)}</td><td>p · C_FN = ${fmtP(r.p)} · ${c.fn}</td></tr>
    <tr class="${best === "FLAG" ? "best" : ""}"><td>FLAG</td><td>${e.flag.toFixed(3)}</td><td>(1 − p) · C_FP = ${(1 - r.p).toFixed(4)} · ${c.fp}</td></tr>
    <tr><td>RE-IMAGE</td><td>${c.reimage} + 2nd decision</td><td>worth it when p is uncertain or a gate failed</td></tr></tbody>`;
  let d2 = null;
  if (state.second) d2 = decideSecond(state.second.p, state.second.gate1 && state.second.gate2, c);
  drawAxis(r.p, state.second ? state.second.p : null);
  chips("p7-chips", [
    { k: "P(diseased)", v: fmtP(r.p) },
    gateChip("safety gates", gateOk, gateOk ? "both passed" : (!r.gate1 ? "exposure" : "mask area")),
    { k: "re-image band", v: `[${fmtP(tLow)}, ${fmtP(tHigh)}]` },
    { k: "first decision", v: d1.action, cls: d1.action === "FLAG" ? "fail" : d1.action === "CLEAR" ? "pass" : "" },
  ]);
  const note = document.getElementById("p7-note") || Object.assign(document.createElement("p"), { id: "p7-note", className: "card-note" });
  $("p7-chips").after(note);
  note.textContent = state.thr.rate < 0.02
    ? `With these costs the validation search chose an empty uncertainty band (${(100 * state.thr.rate).toFixed(1)}% of validation leaves re-imaged, all from failed gates). For this model a retake of the same leaf gives almost the same P(diseased), so a second look is not worth paying for when the photo is good. Retakes still happen whenever a safety gate fails: drag both light-gain sliders down to make the capture too dark and watch the agent ask for one.`
    : `The validation search re-images ${(100 * state.thr.rate).toFixed(1)}% of leaves at these costs: inside the band, a second look is cheaper on average than deciding now.`;

  // Verdict panel
  const final = d2 || d1;
  const badge = $("v-badge");
  badge.className = `badge ${final.action}`;
  badge.textContent = final.action;
  let why = d1.reason;
  if (!d2 && d1.action === "FLAG")
    why = `P(diseased) = ${fmtP(r.p)} is above the upper threshold ${fmtP(tHigh)}. Clearing would cost ${e.clear.toFixed(2)} on average and flagging ${e.flag.toFixed(2)}: a miss costs ${c.fn}, an unnecessary inspection only ${c.fp}, so the agent flags even at moderate risk.`;
  else if (!d2 && d1.action === "CLEAR")
    why = `P(diseased) = ${fmtP(r.p)} is below the lower threshold ${fmtP(tLow)}. The expected cost of clearing (${e.clear.toFixed(3)}) is lower than flagging (${e.flag.toFixed(3)}).`;
  $("v-reason").textContent = d2 ? `After re-imaging: ${d2.reason}` : why;
  drawRGB("v-thumb", (state.second || r).cap.degraded);
  const truth = state.source && state.source.label !== undefined
    ? `${state.source.name} (${state.source.label ? "diseased" : "healthy"})` : "unknown (your photo)";
  const rows = [
    ["P(diseased)", d2 ? `${fmtP(r.p)} → ${fmtP(state.second.p)}` : fmtP(r.p)],
    ["gates", `exposure ${r.gate1 ? "pass" : "FAIL"} · mask ${r.gate2 ? "pass" : "FAIL"}`],
    ["sent", `${r.tx.bytes.toLocaleString()} bytes (${(S * S * 3 / r.tx.bytes).toFixed(1)}:1)`],
    ["PlantVillage label", truth],
  ];
  $("v-kv").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  const box = $("v-second");
  if (d1.action === "RE-IMAGE" && !state.second) {
    box.innerHTML = `<div class="second"><strong>The agent asks for a second photo.</strong>
      <span class="muted" style="font-size:0.85rem">${r.st.sim ? "Simulate a retake of the same leaf with new noise and lighting, or upload a second photo." : "Upload a second photo of the same leaf."}</span>
      <div style="display:flex;gap:8px;flex-wrap:wrap">${r.st.sim ? '<button class="btn primary" id="v-retake" type="button">Re-image this leaf</button>' : ""}
      <label class="btn" for="file2" style="cursor:pointer">Upload second photo</label><input type="file" id="file2" accept="image/*" style="position:absolute;width:1px;height:1px;opacity:0"></div></div>`;
    $("v-retake")?.addEventListener("click", () => retake(null));
    $("file2").addEventListener("change", (ev) => { const f = ev.target.files[0]; if (f) retake(f); });
  } else box.innerHTML = "";
}

async function retake(file) {
  setStatus("Re-imaging…");
  let clean = state.clean;
  if (file) {
    try { clean = await bitmapToImage(await decodeBlob(file)); }
    catch { setStatus("That file could not be read as an image."); return; }
  }
  state.second = await screen(clean, settings(), state.seed + 7919);
  renderDecision();
  setStatus(`Second capture screened (${state.second.t.restore + state.second.t.cnn | 0} ms for restoration + CNN).`);
}

// ---------------------------------------------------------------------------
// Run loop
// ---------------------------------------------------------------------------
function setStatus(s) { $("v-status").textContent = s; }

let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(run, 120); }

async function run() {
  if (!state.model || !state.clean) return;
  if (state.busy) { state.pending = true; return; }
  state.busy = true;
  setStatus("Screening…");
  try {
    const t0 = performance.now();
    state.second = null;
    state.first = await screen(state.clean, settings(), state.seed);
    render(state.first);
    renderDecision();
    setStatus(`All 7 phases ran in ${(performance.now() - t0).toFixed(0)} ms on this device.`);
  } catch (e) {
    console.error(e);
    setStatus(`Something went wrong: ${e.message}`);
  } finally {
    state.busy = false;
    if (state.pending) { state.pending = false; schedule(); }
  }
}

// ---------------------------------------------------------------------------
// Model card
// ---------------------------------------------------------------------------
function renderModelCard() {
  const m = state.meta, tm = m.test_metrics, pct = (v) => `${(100 * v).toFixed(1)}%`;
  $("modelcard").innerHTML = `
    <div class="panel"><h3>Training</h3><p class="muted">SmallCNN (${state.model.nParams.toLocaleString()} parameters) trained on PlantVillage tomato leaves, healthy vs. any of 9 diseases, after the full pipeline: field degradation, NLM, CLAHE, JPEG q${m.pipeline.jpeg_q}, segmentation. Validation picked the early-stopping epoch, the temperature (T = ${m.temperature.toFixed(3)}) and the decision thresholds.</p></div>
    <div class="panel"><h3>Held-out test set (${m.n_test} leaves)</h3><dl class="kv">
      <dt>CNN accuracy</dt><dd>${pct(tm.cnn_acc)}</dd><dt>ROC-AUC</dt><dd>${tm.auc.toFixed(4)}</dd>
      <dt>decision sensitivity</dt><dd>${pct(tm.sensitivity)}</dd><dt>decision specificity</dt><dd>${pct(tm.specificity)}</dd>
      <dt>re-image rate</dt><dd>${pct(tm.reimage_rate)}</dd><dt>missed diseased leaves</dt><dd>${tm.FN}</dd>
      <dt>cost per leaf</dt><dd>${tm.cost_per_sample.toFixed(3)}</dd></dl></div>
    <div class="panel"><h3>Limits</h3><p class="muted">PlantVillage photos are single leaves on a plain background, taken in controlled light. Real field photos with soil, other leaves or hands in frame are a different distribution, and the model can be confidently wrong on them. The browser ports of NLM, CLAHE and JPEG closely match the Python/OpenCV code but are not bit-identical. This is a teaching demo, not a diagnostic tool.</p></div>`;
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------
async function init() {
  updateOutputs();
  for (const id of ["c-sim", "c-sigma", "c-imin", "c-imax", "c-restore", "c-enhance", "c-q", "c-lossless", "c-seg"])
    $(id).addEventListener("input", () => { updateOutputs(); schedule(); });
  for (const id of ["c-fn", "c-fp", "c-ri"])
    $(id).addEventListener("input", () => { if (!state.meta) return; recomputeThresholds(); renderDecision(); });
  $("c-reroll").addEventListener("click", () => { state.seed = (Math.random() * 1e9) | 0; schedule(); });
  $("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) { markSample(null); useBlob(f, { name: f.name }); } });
  const drop = $("drop");
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f) { markSample(null); useBlob(f, { name: f.name }); }
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (state.first) { render(state.first); renderDecision(); } });

  try {
    const [meta, buf, samples] = await Promise.all([
      fetch("model/model.json").then((r) => r.json()),
      fetch("model/weights.bin").then((r) => r.arrayBuffer()),
      fetch("samples/samples.json").then((r) => r.json()),
    ]);
    state.meta = meta;
    state.model = new SmallCNN(meta, buf);
    recomputeThresholds();
    renderModelCard();
    buildSamples(samples);
    setStatus("Model loaded.");
    const first = samples.find((s) => /early/i.test(s.cls)) || samples.find((s) => s.label === 1) || samples[0];
    pickSample(first);
  } catch (e) {
    console.error(e);
    setStatus("The model files could not be loaded. Serve the docs/ folder over HTTP (not file://).");
  }
}

function markSample(cls) {
  for (const b of document.querySelectorAll(".sample")) b.setAttribute("aria-pressed", String(b.dataset.cls === cls));
}
async function pickSample(s) {
  markSample(s.cls);
  const blob = await fetch(s.file).then((r) => r.blob());
  useBlob(blob, s);
}
function buildSamples(samples) {
  $("samples").innerHTML = "";
  for (const s of samples) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "sample"; b.dataset.cls = s.cls;
    b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<img src="${s.file}" alt="" loading="lazy"><span><i class="dot ${s.label ? "" : "h"}"></i>${s.name}</span>`;
    b.addEventListener("click", () => pickSample(s));
    $("samples").appendChild(b);
  }
}

init();
