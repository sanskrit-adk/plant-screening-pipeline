// cnn.js - SmallCNN inference in plain JavaScript (no framework), exported from PyTorch by
// src/export_web.py. BatchNorm is folded into each convolution, so a block is
// conv3x3 (+bias) -> ReLU -> maxpool 2x2. Works in the browser and in Node.

export class SmallCNN {
  // manifest: parsed docs/model/model.json; buffer: ArrayBuffer of docs/model/weights.bin
  constructor(manifest, buffer) {
    this.meta = manifest;
    const all = new Float32Array(buffer);
    this.p = {};
    for (const l of manifest.layers) {
      const n = l.shape.reduce((a, b) => a * b, 1);
      this.p[l.name] = { shape: l.shape, data: all.subarray(l.offset, l.offset + n) };
    }
    this.blocks = [1, 2, 3, 4].map((k) => ({ w: this.p[`conv${k}.w`], b: this.p[`conv${k}.b`] }));
  }

  get nParams() { return Object.values(this.p).reduce((s, t) => s + t.data.length, 0); }

  // uint8 RGB HWC image -> float CHW tensor normalised to [-1, 1]: (x/255 - 0.5)/0.5
  static toTensor(im) {
    const { w, h } = im, t = new Float32Array(3 * w * h);
    for (let i = 0; i < w * h; i++) for (let k = 0; k < 3; k++) t[k * w * h + i] = (im.d[3 * i + k] / 255 - 0.5) / 0.5;
    return t;
  }

  // 3x3 convolution, zero padding 1, plus bias
  static conv(x, C, H, W, wt, bias) {
    const O = wt.shape[0], out = new Float32Array(O * H * W), wd = wt.data;
    for (let o = 0; o < O; o++) {
      const oo = out.subarray(o * H * W, (o + 1) * H * W);
      oo.fill(bias.data[o]);
      for (let c = 0; c < C; c++) {
        const xc = x.subarray(c * H * W, (c + 1) * H * W);
        for (let ky = 0; ky < 3; ky++)
          for (let kx = 0; kx < 3; kx++) {
            const wv = wd[((o * C + c) * 3 + ky) * 3 + kx];
            if (wv === 0) continue;
            const dy = ky - 1, dx = kx - 1;
            const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy), x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
            for (let y = y0; y < y1; y++) {
              const ro = y * W, ri = (y + dy) * W + dx;
              for (let xx = x0; xx < x1; xx++) oo[ro + xx] += wv * xc[ri + xx];
            }
          }
      }
    }
    return out;
  }

  static reluPool(x, C, H, W) {               // ReLU then 2x2 max pool, stride 2
    const h2 = H >> 1, w2 = W >> 1, out = new Float32Array(C * h2 * w2);
    for (let c = 0; c < C; c++)
      for (let y = 0; y < h2; y++)
        for (let xx = 0; xx < w2; xx++) {
          const b = c * H * W + 2 * y * W + 2 * xx;
          const m = Math.max(x[b], x[b + 1], x[b + W], x[b + W + 1], 0);
          out[c * h2 * w2 + y * w2 + xx] = m;
        }
    return out;
  }

  // Returns logits plus every block's activation map (for the feature-map display)
  forward(im) {
    let x = SmallCNN.toTensor(im), C = 3, H = im.h, W = im.w;
    const acts = [];
    for (const blk of this.blocks) {
      const conv = SmallCNN.conv(x, C, H, W, blk.w, blk.b);
      C = blk.w.shape[0];
      x = SmallCNN.reluPool(conv, C, H, W);
      H >>= 1; W >>= 1;
      acts.push({ C, H, W, data: x });
    }
    const gap = new Float32Array(C);             // global average pool
    for (let c = 0; c < C; c++) { let s = 0; for (let i = 0; i < H * W; i++) s += x[c * H * W + i]; gap[c] = s / (H * W); }
    const fw = this.p["fc.w"].data, fb = this.p["fc.b"].data, logits = [0, 0];
    for (let k = 0; k < 2; k++) { let s = fb[k]; for (let c = 0; c < C; c++) s += fw[k * C + c] * gap[c]; logits[k] = s; }
    return { logits, gap, acts };
  }
}

// Temperature-scaled softmax, P(diseased) = softmax(z / T)[1]
export function probDiseased(logits, T) {
  const a = logits[0] / T, b = logits[1] / T, m = Math.max(a, b);
  const ea = Math.exp(a - m), eb = Math.exp(b - m);
  return eb / (ea + eb);
}
