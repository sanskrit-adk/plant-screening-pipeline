// imgproc.js - the image-processing half of the pipeline, ported from OpenCV / scikit-image.
//
// Pure functions, no DOM: the same file runs in the browser and in Node (tools/web_parity.mjs).
// An image is { w, h, c, d } where d is a Uint8Array of h*w*c values, row-major, channels last
// (RGB for colour images, c = 1 for grey/masks). Each port follows the OpenCV algorithm the
// Python pipeline calls (same kernels, border modes and rounding where it matters), so the
// browser reproduces src/stages.py closely; tools/web_parity.mjs measures how closely.

export const image = (w, h, c, d) => ({ w, h, c, d: d || new Uint8Array(w * h * c) });
export const clone = (im) => image(im.w, im.h, im.c, new Uint8Array(im.d));
const clamp8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
const round8 = (v) => clamp8(Math.round(v));

// Border rules used by OpenCV
export function reflect101(i, n) {           // BORDER_DEFAULT: gfedcb|abcdefgh|gfedcba
  if (n === 1) return 0;
  while (i < 0 || i >= n) { if (i < 0) i = -i; if (i >= n) i = 2 * n - 2 - i; }
  return i;
}
const replicate = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);   // BORDER_REPLICATE
function reflectHalf(i, n) {                 // scipy 'reflect': dcba|abcd|dcba
  while (i < 0 || i >= n) { if (i < 0) i = -i - 1; if (i >= n) i = 2 * n - 1 - i; }
  return i;
}

// ---------------------------------------------------------------------------
// Geometry: centre crop and resize (cv2.INTER_AREA semantics)
// ---------------------------------------------------------------------------
export function centerSquare(im) {
  const s = Math.min(im.w, im.h), x0 = (im.w - s) >> 1, y0 = (im.h - s) >> 1;
  return crop(im, x0, y0, s, s);
}

export function crop(im, x0, y0, w, h) {
  const out = image(w, h, im.c);
  for (let y = 0; y < h; y++) {
    const s = ((y0 + y) * im.w + x0) * im.c;
    out.d.set(im.d.subarray(s, s + w * im.c), y * w * im.c);
  }
  return out;
}

// Area-coverage weights for one axis (downscaling): each output cell averages the
// source cells it covers, weighted by overlap - cv2.INTER_AREA's "pixel area relation".
function areaTab(src, dst) {
  const scale = src / dst, tab = [];
  for (let o = 0; o < dst; o++) {
    const a = o * scale, b = (o + 1) * scale, ws = [];
    for (let s = Math.floor(a); s < Math.min(src, Math.ceil(b)); s++) {
      const ov = Math.min(b, s + 1) - Math.max(a, s);
      if (ov > 1e-9) ws.push([s, ov / scale]);
    }
    tab.push(ws);
  }
  return tab;
}

// Linear coefficients that cv2.INTER_AREA uses when ENLARGING (area-mode linear).
function areaUpTab(src, dst) {
  const scale = src / dst, inv = dst / src, tab = [];
  for (let o = 0; o < dst; o++) {
    let sx = Math.floor(o * scale);
    let fx = (o + 1) - (sx + 1) * inv;
    fx = fx <= 0 ? 0 : fx - Math.floor(fx);
    if (sx < 0) { fx = 0; sx = 0; }
    if (sx >= src - 1) { fx = 0; sx = src - 1; }
    tab.push(fx === 0 ? [[sx, 1]] : [[sx, 1 - fx], [sx + 1, fx]]);
  }
  return tab;
}

export function resize(im, W, H) {
  if (im.w === W && im.h === H) return clone(im);
  const down = im.w >= W && im.h >= H;
  const tx = down ? areaTab(im.w, W) : areaUpTab(im.w, W);
  const ty = down ? areaTab(im.h, H) : areaUpTab(im.h, H);
  const c = im.c, tmp = new Float64Array(im.h * W * c), out = image(W, H, c);
  for (let y = 0; y < im.h; y++)
    for (let x = 0; x < W; x++)
      for (let k = 0; k < c; k++) {
        let s = 0;
        for (const [sx, wgt] of tx[x]) s += wgt * im.d[(y * im.w + sx) * c + k];
        tmp[(y * W + x) * c + k] = s;
      }
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
      for (let k = 0; k < c; k++) {
        let s = 0;
        for (const [sy, wgt] of ty[y]) s += wgt * tmp[(sy * W + x) * c + k];
        out.d[(y * W + x) * c + k] = round8(s);
      }
  return out;
}

// ---------------------------------------------------------------------------
// Colour conversions (OpenCV 8-bit formulas)
// ---------------------------------------------------------------------------
export function gray(im) {                   // cv2.COLOR_RGB2GRAY, fixed point
  if (im.c === 1) return im;
  const out = image(im.w, im.h, 1), n = im.w * im.h;
  for (let i = 0; i < n; i++) {
    const r = im.d[3 * i], g = im.d[3 * i + 1], b = im.d[3 * i + 2];
    out.d[i] = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
  }
  return out;
}

const srgbToLin = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
const linToSrgb = (v) => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
const fLab = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);

// RGB -> 8-bit Lab (L*255/100, a+128, b+128). gamma=false gives OpenCV's "linear" LRGB2Lab.
function rgbToLab8(r, g, b, gamma) {
  r /= 255; g /= 255; b /= 255;
  if (gamma) { r = srgbToLin(r); g = srgbToLin(g); b = srgbToLin(b); }
  const X = (0.412453 * r + 0.35758 * g + 0.180423 * b) / 0.950456;
  const Y = 0.212671 * r + 0.71516 * g + 0.072169 * b;
  const Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
  const fy = fLab(Y);
  const L = Y > 0.008856 ? 116 * fy - 16 : 903.3 * Y;
  return [L * 2.55, 500 * (fLab(X) - fy) + 128, 200 * (fy - fLab(Z)) + 128];
}

function lab8ToRgb(L8, a8, b8, gamma) {
  const L = L8 / 2.55, a = a8 - 128, bb = b8 - 128;
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - bb / 200;
  const inv = (f) => (f * f * f > 0.008856 ? f * f * f : (f - 16 / 116) / 7.787);
  const X = inv(fx) * 0.950456, Y = L > 7.9996 ? fy * fy * fy : L / 903.3, Z = inv(fz) * 1.088754;
  let r = 3.240479 * X - 1.53715 * Y - 0.498535 * Z;
  let g = -0.969256 * X + 1.875991 * Y + 0.041556 * Z;
  let b = 0.055648 * X - 0.204043 * Y + 1.057311 * Z;
  const enc = (v) => { v = Math.min(1, Math.max(0, v)); return gamma ? linToSrgb(v) : v; };
  return [round8(enc(r) * 255), round8(enc(g) * 255), round8(enc(b) * 255)];
}

// ---------------------------------------------------------------------------
// 1. Field degradation (same model as src/degrade.py; its own seeded RNG)
// ---------------------------------------------------------------------------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussRng(u) {                        // Box-Muller standard normal
  let s = null;
  return () => {
    if (s !== null) { const v = s; s = null; return v; }
    const r = Math.sqrt(-2 * Math.log(1 - u())), th = 2 * Math.PI * u();
    s = r * Math.sin(th);
    return r * Math.cos(th);
  };
}

// Random-direction linear gradient blended with an off-centre vignette, spanning [gmin, gmax].
export function illuminationField(w, h, rnd, gmin, gmax) {
  const theta = rnd() * 2 * Math.PI, cx = 0.2 + 0.6 * rnd(), cy = 0.2 + 0.6 * rnd();
  const a = 0.3 + 0.4 * rnd();
  const lin = new Float64Array(w * h), rad = new Float64Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const xx = x / Math.max(w - 1, 1), yy = y / Math.max(h - 1, 1), i = y * w + x;
      lin[i] = Math.cos(theta) * xx + Math.sin(theta) * yy;
      rad[i] = Math.hypot(xx - cx, yy - cy);
    }
  const norm = (arr, flip) => {
    let mn = Infinity, mx = -Infinity;
    for (const v of arr) { if (v < mn) mn = v; if (v > mx) mx = v; }
    for (let i = 0; i < arr.length; i++) {
      const t = (arr[i] - mn) / (mx - mn + 1e-8);
      arr[i] = flip ? 1 - t : t;
    }
  };
  norm(lin, false); norm(rad, true);
  const f = new Float64Array(w * h);
  for (let i = 0; i < f.length; i++) f[i] = a * lin[i] + (1 - a) * rad[i];
  norm(f, false);
  for (let i = 0; i < f.length; i++) f[i] = gmin + f[i] * (gmax - gmin);
  return f;
}

// Returns { degraded, reference, field }: reference is the noise-free, unevenly lit image.
export function degrade(clean, opt, seed) {
  const rnd = mulberry32(seed), nrm = gaussRng(rnd);
  const { w, h } = clean;
  const G = illuminationField(w, h, rnd, opt.illum_min, opt.illum_max);
  const ref = image(w, h, 3), out = image(w, h, 3);
  const sigma = opt.noise_sigma || 0;
  for (let i = 0; i < w * h; i++)
    for (let k = 0; k < 3; k++) {
      const lit = clean.d[3 * i + k] * G[i];
      ref.d[3 * i + k] = clamp8(Math.trunc(lit));          // numpy astype(uint8) truncates
      out.d[3 * i + k] = clamp8(Math.trunc(sigma > 0 ? lit + sigma * nrm() : lit));
    }
  return { degraded: out, reference: ref, field: G };
}

// ---------------------------------------------------------------------------
// 2. Restoration
// ---------------------------------------------------------------------------
export function gaussian5(im) {               // cv2.GaussianBlur(5x5, sigma 0): [1 4 6 4 1]/16
  const K = [1, 4, 6, 4, 1], { w, h, c } = im, tmp = new Int32Array(w * h * c), out = image(w, h, c);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let k = 0; k < c; k++) {
        let s = 0;
        for (let j = -2; j <= 2; j++) s += K[j + 2] * im.d[(y * w + reflect101(x + j, w)) * c + k];
        tmp[(y * w + x) * c + k] = s;
      }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let k = 0; k < c; k++) {
        let s = 0;
        for (let j = -2; j <= 2; j++) s += K[j + 2] * tmp[(reflect101(y + j, h) * w + x) * c + k];
        out.d[(y * w + x) * c + k] = (s + 128) >> 8;
      }
  return out;
}

export function median5(im) {                 // cv2.medianBlur(5), replicated border
  const { w, h, c } = im, out = image(w, h, c), buf = new Uint8Array(25);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let k = 0; k < c; k++) {
        let n = 0;
        for (let dy = -2; dy <= 2; dy++)
          for (let dx = -2; dx <= 2; dx++)
            buf[n++] = im.d[(replicate(y + dy, h) * w + replicate(x + dx, w)) * c + k];
        buf.sort();
        out.d[(y * w + x) * c + k] = buf[12];
      }
  return out;
}

export function bilateral(im, d = 7, sColor = 50, sSpace = 50) {   // cv2.bilateralFilter
  const { w, h } = im, out = image(w, h, 3), r = d >> 1;
  const cw = new Float64Array(256 * 3);
  for (let i = 0; i < cw.length; i++) cw[i] = Math.exp((-0.5 * i * i) / (sColor * sColor));
  const offs = [];
  for (let i = -r; i <= r; i++)
    for (let j = -r; j <= r; j++) {
      const rr = Math.hypot(i, j);
      if (rr <= r) offs.push([i, j, Math.exp((-0.5 * rr * rr) / (sSpace * sSpace))]);
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const p = (y * w + x) * 3, r0 = im.d[p], g0 = im.d[p + 1], b0 = im.d[p + 2];
      let sr = 0, sg = 0, sb = 0, ws = 0;
      for (const [i, j, sw] of offs) {
        const q = (reflect101(y + i, h) * w + reflect101(x + j, w)) * 3;
        const rv = im.d[q], gv = im.d[q + 1], bv = im.d[q + 2];
        const wgt = sw * cw[Math.abs(rv - r0) + Math.abs(gv - g0) + Math.abs(bv - b0)];
        sr += rv * wgt; sg += gv * wgt; sb += bv * wgt; ws += wgt;
      }
      out.d[p] = round8(sr / ws); out.d[p + 1] = round8(sg / ws); out.d[p + 2] = round8(sb / ws);
    }
  return out;
}

// Non-local means on planes (Float64Array per channel), OpenCV weighting:
// w = exp(-(mean template SSD) / (h^2 * channels)), zeroed below 0.001.
// Implemented offset-by-offset with integral images, so cost is O(search^2 * pixels).
function nlmPlanes(planes, w, h, hParam, tw = 7, sw = 21) {
  const tr = tw >> 1, sr = sw >> 1, P = tr + sr, W = w + 2 * P, H = h + 2 * P, cn = planes.length;
  const padPlane = (pl) => {
    const a = new Float64Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) a[y * W + x] = pl[reflect101(y - P, h) * w + reflect101(x - P, w)];
    return a;
  };
  const A = padPlane(planes[0]), B = cn > 1 ? padPlane(planes[1]) : null;
  const numA = new Float64Array(w * h), numB = new Float64Array(w * h), den = new Float64Array(w * h);
  // Only the region [sr, H-sr) x [sr, W-sr) is ever needed by an output pixel's template,
  // and its (dy,dx)-shifted partner always stays inside the padded image: no bounds checks.
  const RW = W - 2 * sr, RH = H - 2 * sr, IW = RW + 1;
  const integ = new Float64Array((RW + 1) * (RH + 1));
  const k = 1 / (tw * tw * hParam * hParam * cn), cutoff = -Math.log(0.001);  // w < 0.001 -> 0
  for (let dy = -sr; dy <= sr; dy++)
    for (let dx = -sr; dx <= sr; dx++) {
      const shift = dy * W + dx;
      // integral image of the squared difference between the image and its shifted copy
      for (let ry = 0; ry < RH; ry++) {
        let row = 0, p = (ry + sr) * W + sr;
        const o = (ry + 1) * IW + 1, op = ry * IW + 1;
        if (B) {
          for (let rx = 0; rx < RW; rx++, p++) {
            const u = A[p] - A[p + shift], v = B[p] - B[p + shift];
            row += u * u + v * v;
            integ[o + rx] = integ[op + rx] + row;
          }
        } else {
          for (let rx = 0; rx < RW; rx++, p++) {
            const u = A[p] - A[p + shift];
            row += u * u;
            integ[o + rx] = integ[op + rx] + row;
          }
        }
      }
      // template SSD for every output pixel -> weight -> accumulate the shifted pixel
      for (let y = 0; y < h; y++) {
        const r0 = y * IW, r1 = (y + tw) * IW;
        let q = (y + P + dy) * W + P + dx;
        for (let x = 0; x < w; x++, q++) {
          const s = integ[r1 + x + tw] - integ[r0 + x + tw] - integ[r1 + x] + integ[r0 + x];
          const e = s * k;
          if (e > cutoff) continue;
          const wgt = Math.exp(-e), i = y * w + x;
          den[i] += wgt;
          numA[i] += wgt * A[q];
          if (B) numB[i] += wgt * B[q];
        }
      }
    }
  const outA = numA.map((v, i) => v / den[i]);
  return B ? [outA, numB.map((v, i) => v / den[i])] : [outA];
}

// cv2.fastNlMeansDenoisingColored(img, None, 12, 12, 7, 21).
// OpenCV converts with COLOR_LBGR2Lab (linear, no gamma) and treats channel 0 as blue;
// the Python pipeline passes RGB, so channel 0 (red) plays "blue" here too.
export function nlmColored(im, hL = 12, hC = 12) {
  const { w, h } = im, n = w * h;
  const L = new Float64Array(n), A = new Float64Array(n), B = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lab = rgbToLab8(im.d[3 * i + 2], im.d[3 * i + 1], im.d[3 * i], false);
    L[i] = Math.round(lab[0]); A[i] = Math.round(lab[1]); B[i] = Math.round(lab[2]);
  }
  const [Ld] = nlmPlanes([L], w, h, hL);
  const [Ad, Bd] = nlmPlanes([A, B], w, h, hC);
  const out = image(w, h, 3);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = lab8ToRgb(round8(Ld[i]), round8(Ad[i]), round8(Bd[i]), false);
    out.d[3 * i] = b; out.d[3 * i + 1] = g; out.d[3 * i + 2] = r;
  }
  return out;
}

export const RESTORE = { none: (x) => x, gaussian: gaussian5, median: median5, bilateral, nlm: nlmColored };
export const restore = (im, m) => (RESTORE[m] || RESTORE.none)(im);

// ---------------------------------------------------------------------------
// 3. Enhancement
// ---------------------------------------------------------------------------
// cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply on one 8-bit plane
export function clahePlane(src, w, h, clip = 2.0, tiles = 8) {
  const tw = Math.floor(w / tiles), th = Math.floor(h / tiles), area = tw * th;
  const limit = Math.max(Math.floor((clip * area) / 256), 1);
  const luts = [];
  for (let ty = 0; ty < tiles; ty++)
    for (let tx = 0; tx < tiles; tx++) {
      const hist = new Int32Array(256);
      for (let y = ty * th; y < (ty + 1) * th; y++)
        for (let x = tx * tw; x < (tx + 1) * tw; x++) hist[src[y * w + x]]++;
      let clipped = 0;
      for (let i = 0; i < 256; i++) if (hist[i] > limit) { clipped += hist[i] - limit; hist[i] = limit; }
      const batch = Math.floor(clipped / 256);
      let residual = clipped - batch * 256;
      for (let i = 0; i < 256; i++) hist[i] += batch;
      if (residual) {
        const step = Math.max(Math.floor(256 / residual), 1);
        for (let i = 0; i < 256 && residual > 0; i += step, residual--) hist[i]++;
      }
      const lut = new Uint8Array(256), scale = 255 / area;
      let sum = 0;
      for (let i = 0; i < 256; i++) { sum += hist[i]; lut[i] = round8(sum * scale); }
      luts.push(lut);
    }
  const out = new Uint8Array(w * h), itw = 1 / tw, ith = 1 / th;
  for (let y = 0; y < h; y++) {
    const tyf = y * ith - 0.5;
    let ty1 = Math.floor(tyf), ty2 = ty1 + 1;
    const ya = tyf - ty1;
    ty1 = Math.max(ty1, 0); ty2 = Math.min(ty2, tiles - 1);
    for (let x = 0; x < w; x++) {
      const txf = x * itw - 0.5;
      let tx1 = Math.floor(txf), tx2 = tx1 + 1;
      const xa = txf - tx1;
      tx1 = Math.max(tx1, 0); tx2 = Math.min(tx2, tiles - 1);
      const v = src[y * w + x];
      const res = (luts[ty1 * tiles + tx1][v] * (1 - xa) + luts[ty1 * tiles + tx2][v] * xa) * (1 - ya) +
                  (luts[ty2 * tiles + tx1][v] * (1 - xa) + luts[ty2 * tiles + tx2][v] * xa) * ya;
      out[y * w + x] = round8(res);
    }
  }
  return out;
}

export function toLab(im) {                   // RGB -> 8-bit Lab planes (with sRGB gamma)
  const n = im.w * im.h, L = new Uint8Array(n), A = new Uint8Array(n), B = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const lab = rgbToLab8(im.d[3 * i], im.d[3 * i + 1], im.d[3 * i + 2], true);
    L[i] = round8(lab[0]); A[i] = round8(lab[1]); B[i] = round8(lab[2]);
  }
  return { L, A, B };
}

export function clahe(im) {                   // CLAHE on the L channel of LAB
  const { L, A, B } = toLab(im), L2 = clahePlane(L, im.w, im.h), out = image(im.w, im.h, 3);
  for (let i = 0; i < im.w * im.h; i++) {
    const [r, g, b] = lab8ToRgb(L2[i], A[i], B[i], true);
    out.d[3 * i] = r; out.d[3 * i + 1] = g; out.d[3 * i + 2] = b;
  }
  return out;
}

export function equalizePlane(src) {          // cv2.equalizeHist
  const hist = new Int32Array(256), n = src.length;
  for (const v of src) hist[v]++;
  let i = 0;
  while (!hist[i]) i++;
  const lut = new Uint8Array(256);
  if (hist[i] === n) { lut.fill(i); return src.map((v) => lut[v]); }
  const scale = 255 / (n - hist[i]);
  let sum = 0;
  for (lut[i++] = 0; i < 256; i++) { sum += hist[i]; lut[i] = round8(sum * scale); }
  return src.map((v) => lut[v]);
}

export function histeq(im) {                  // equalise Y of YCrCb
  const n = im.w * im.h, Y = new Uint8Array(n), Cr = new Float64Array(n), Cb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const r = im.d[3 * i], g = im.d[3 * i + 1], b = im.d[3 * i + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    Y[i] = round8(y); Cr[i] = round8((r - y) * 0.713 + 128); Cb[i] = round8((b - y) * 0.564 + 128);
  }
  const Y2 = equalizePlane(Y), out = image(im.w, im.h, 3);
  for (let i = 0; i < n; i++) {
    const y = Y2[i], cr = Cr[i] - 128, cb = Cb[i] - 128;
    out.d[3 * i] = round8(y + 1.403 * cr);
    out.d[3 * i + 1] = round8(y - 0.714 * cr - 0.344 * cb);
    out.d[3 * i + 2] = round8(y + 1.773 * cb);
  }
  return out;
}

const GAMMA_LUT = Uint8Array.from({ length: 256 }, (_, i) => Math.floor(Math.pow(i / 255, 0.7) * 255));
export const gamma = (im) => image(im.w, im.h, im.c, im.d.map((v) => GAMMA_LUT[v]));

export const ENHANCE = { none: (x) => x, clahe, histeq, gamma };
export const enhance = (im, m) => (ENHANCE[m] || ENHANCE.none)(im);

// ---------------------------------------------------------------------------
// 4. Segmentation + morphology
// ---------------------------------------------------------------------------
export function saturation(im) {              // HSV S (OpenCV fixed point), 0 where V < 25
  const n = im.w * im.h, out = image(im.w, im.h, 1);
  for (let i = 0; i < n; i++) {
    const r = im.d[3 * i], g = im.d[3 * i + 1], b = im.d[3 * i + 2];
    const v = Math.max(r, g, b), mn = Math.min(r, g, b);
    out.d[i] = v < 25 || v === 0 ? 0 : ((v - mn) * Math.round((255 << 12) / v) + 2048) >> 12;
  }
  return out;
}

export function otsu(plane) {                 // cv2 THRESH_OTSU threshold value
  const hist = new Float64Array(256), N = plane.length;
  for (const v of plane) hist[v]++;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += (i * hist[i]) / N;
  let q1 = 0, mu1 = 0, best = 0, t = 0;
  const eps = 1.1920929e-7;
  for (let i = 0; i < 256; i++) {
    const p = hist[i] / N;
    mu1 *= q1; q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < eps || Math.max(q1, q2) > 1 - eps) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2, s = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (s > best) { best = s; t = i; }
  }
  return t;
}

export function ellipseKernel(k) {            // cv2.getStructuringElement(MORPH_ELLIPSE, (k,k))
  const r = k >> 1, pts = [];
  for (let i = 0; i < k; i++) {
    const dy = i - r;
    if (Math.abs(dy) > r) continue;
    const dx = Math.round(r * Math.sqrt((r * r - dy * dy) / (r * r)));
    for (let j = Math.max(r - dx, 0); j < Math.min(r + dx + 1, k); j++) pts.push([dy, j - r]);
  }
  return pts;
}

// Binary erosion/dilation of a 0/1 mask; pixels outside the image never constrain the result
function morph(mask, w, h, pts, erodeOp) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v = erodeOp ? 1 : 0;
      for (const [dy, dx] of pts) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
        const m = mask[yy * w + xx];
        if (erodeOp && !m) { v = 0; break; }
        if (!erodeOp && m) { v = 1; break; }
      }
      out[y * w + x] = v;
    }
  return out;
}
export const erode = (m, w, h, pts, it = 1) => { for (let i = 0; i < it; i++) m = morph(m, w, h, pts, true); return m; };
export const dilate = (m, w, h, pts, it = 1) => { for (let i = 0; i < it; i++) m = morph(m, w, h, pts, false); return m; };

export function largestComponent(mask, w, h) {   // 8-connected, keep the biggest blob
  const lab = new Int32Array(w * h), stack = [];
  let best = 0, bestArea = 0, next = 0;
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || lab[s]) continue;
    next++; let area = 0;
    lab[s] = next; stack.push(s);
    while (stack.length) {
      const p = stack.pop(), y = (p / w) | 0, x = p - y * w;
      area++;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
          const q = yy * w + xx;
          if (mask[q] && !lab[q]) { lab[q] = next; stack.push(q); }
        }
    }
    if (area > bestArea) { bestArea = area; best = next; }
  }
  return { mask: lab.map((v) => (best && v === best ? 1 : 0)), count: next };
}

export function fillHoles(mask, w, h) {       // flood the background from outside; the rest is leaf
  const W = w + 2, H = h + 2, seen = new Uint8Array(W * H), stack = [0];
  const at = (x, y) => (x <= 0 || y <= 0 || x > w || y > h ? 0 : mask[(y - 1) * w + x - 1]);
  seen[0] = 1;
  while (stack.length) {
    const p = stack.pop(), y = (p / W) | 0, x = p - y * W;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
      const q = yy * W + xx;
      if (!seen[q] && !at(xx, yy)) { seen[q] = 1; stack.push(q); }
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = seen[(y + 1) * W + x + 1] ? 0 : 1;
  return out;
}

// Full segment_mask with every intermediate kept for display
export function segment(im) {
  const { w, h } = im, sat = saturation(im), blur = gaussian5(sat), t = otsu(blur.d);
  const binary = blur.d.map((v) => (v > t ? 1 : 0));
  const opened = dilate(erode(binary, w, h, ellipseKernel(5)), w, h, ellipseKernel(5));
  const closed = erode(dilate(opened, w, h, ellipseKernel(7), 2), w, h, ellipseKernel(7), 2);
  const cc = largestComponent(closed, w, h);
  const mask = fillHoles(cc.mask, w, h);
  let area = 0;
  for (const v of mask) area += v;
  return { sat, blur, threshold: t, binary, opened, closed, largest: cc.mask, components: cc.count,
           mask, area: area / (w * h) };
}

export function applyMask(im, mask, size, pad = 4) {   // zero background, crop bbox, resize
  const { w, h } = im;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < 0) return im;
  const m = image(w, h, 3);
  for (let i = 0; i < w * h; i++) if (mask[i]) for (let k = 0; k < 3; k++) m.d[3 * i + k] = im.d[3 * i + k];
  y0 = Math.max(0, y0 - pad); x0 = Math.max(0, x0 - pad);
  y1 = Math.min(h, y1 + pad + 1); x1 = Math.min(w, x1 + pad + 1);
  return resize(crop(m, x0, y0, x1 - x0, y1 - y0), size, size);
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------
export function psnr(ref, im) {
  let s = 0;
  for (let i = 0; i < ref.d.length; i++) { const v = ref.d[i] - im.d[i]; s += v * v; }
  const mse = s / ref.d.length;
  return mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse);
}

export function ssim(ref, im) {               // skimage structural_similarity, 7x7 uniform window
  const { w, h, c } = ref, win = 7, r = 3, NP = 49, cov = NP / (NP - 1);
  const C1 = (0.01 * 255) ** 2, C2 = (0.03 * 255) ** 2;
  let total = 0;
  for (let k = 0; k < c; k++) {
    const X = new Float64Array(w * h), Y = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) { X[i] = ref.d[i * c + k]; Y[i] = im.d[i * c + k]; }
    const filt = (f) => {
      const out = new Float64Array(w * h);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        let s = 0;
        for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
          const yy = reflectHalf(y + dy, h), xx = reflectHalf(x + dx, w);
          s += f(yy * w + xx);
        }
        out[y * w + x] = s / NP;
      }
      return out;
    };
    const ux = filt((i) => X[i]), uy = filt((i) => Y[i]);
    const uxx = filt((i) => X[i] * X[i]), uyy = filt((i) => Y[i] * Y[i]), uxy = filt((i) => X[i] * Y[i]);
    let s = 0, n = 0;
    for (let y = r; y < h - r; y++) for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const vx = cov * (uxx[i] - ux[i] * ux[i]), vy = cov * (uyy[i] - uy[i] * uy[i]), vxy = cov * (uxy[i] - ux[i] * uy[i]);
      s += ((2 * ux[i] * uy[i] + C1) * (2 * vxy + C2)) / ((ux[i] ** 2 + uy[i] ** 2 + C1) * (vx + vy + C2));
      n++;
    }
    total += s / n;
  }
  return total / c;
}

export function histogram(plane) {
  const h = new Float64Array(256);
  for (const v of plane) h[v]++;
  return h;
}

export function entropy(im) {                 // skimage shannon_entropy of the grey image (bits)
  const g = gray(im).d, h = histogram(g);
  let e = 0;
  for (const v of h) if (v) { const p = v / g.length; e -= p * Math.log2(p); }
  return e;
}

export function rmsContrast(im) {
  const g = gray(im).d;
  let m = 0;
  for (const v of g) m += v;
  m /= g.length;
  let s = 0;
  for (const v of g) s += (v - m) ** 2;
  return Math.sqrt(s / g.length) / 255;
}

export const meanGray = (im) => { const g = gray(im).d; let s = 0; for (const v of g) s += v; return s / g.length; };
