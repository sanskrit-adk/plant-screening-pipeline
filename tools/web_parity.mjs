// web_parity.mjs - checks that the browser port (docs/js) reproduces the Python pipeline.
//
//   python -m src.export_web          (writes results/web_check/ and docs/model/)
//   node tools/web_parity.mjs
//
// Every JS stage gets exactly the input the Python stage got, so differences are the port's
// own. Then the whole JS chain (restore -> enhance -> segment -> CNN, lossless) is run on the
// same degraded test images and its accuracy is compared with Python's.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as ip from "../docs/js/imgproc.js";
import { SmallCNN, probDiseased } from "../docs/js/cnn.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const chk = join(root, "results", "web_check");
const man = JSON.parse(readFileSync(join(chk, "manifest.json"), "utf8"));
const S = man.size, px = S * S;
const load = (name, c = 3) => {
  const buf = new Uint8Array(readFileSync(join(chk, `${name}.bin`)));
  const n = buf.length / (px * c);
  return Array.from({ length: n }, (_, i) => ip.image(S, S, c, buf.slice(i * px * c, (i + 1) * px * c)));
};
const deg = load("degraded");
const st = Object.fromEntries(Object.entries(man.stages).map(([k, sh]) => [k, load(k, sh.length === 4 ? 3 : 1)]));

function cmp(a, b) {
  let mad = 0, exact = 0, within1 = 0;
  for (let i = 0; i < a.d.length; i++) {
    const d = Math.abs(a.d[i] - b.d[i]);
    mad += d; if (d === 0) exact++; if (d <= 1) within1++;
  }
  const n = a.d.length;
  return { mad: mad / n, exact: exact / n, within1: within1 / n };
}
function report(name, pairs) {
  const r = pairs.map(([a, b]) => cmp(a, b));
  const avg = (k) => r.reduce((s, x) => s + x[k], 0) / r.length;
  console.log(`${name.padEnd(20)} mean|diff| ${avg("mad").toFixed(3).padStart(6)}   exact ${(100 * avg("exact")).toFixed(1).padStart(5)}%   within 1: ${(100 * avg("within1")).toFixed(1).padStart(5)}%`);
}

console.log(`\nStage parity on ${man.k} test images (each JS stage gets the Python stage's input)`);
const k = man.k;
for (const m of ["gaussian", "median", "bilateral", "nlm"])
  report(`restore ${m}`, deg.slice(0, k).map((d, i) => [ip.restore(d, m), st[`restore_${m}`][i]]));
for (const m of ["clahe", "histeq", "gamma"])
  report(`enhance ${m}`, st.restore_nlm.map((x, i) => [ip.enhance(x, m), st[`enhance_${m}`][i]]));
report("saturation", st.enhance_clahe.map((x, i) => [ip.saturation(x), st.saturation[i]]));
const ious = st.enhance_clahe.map((x, i) => {
  const a = ip.segment(x).mask, b = st.mask[i].d;
  let I = 0, U = 0;
  for (let j = 0; j < a.length; j++) { I += a[j] & b[j]; U += a[j] | b[j]; }
  return U ? I / U : 1;
});
console.log(`segment mask         mean IoU vs Python ${(ious.reduce((s, v) => s + v, 0) / ious.length).toFixed(4)}   min ${Math.min(...ious).toFixed(4)}`);

const model = new SmallCNN(JSON.parse(readFileSync(join(root, "docs/model/model.json"), "utf8")),
  readFileSync(join(root, "docs/model/weights.bin")).buffer.slice(0));
const T = model.meta.temperature;
let maxd = 0;
st.cnn_input.forEach((x, i) => { maxd = Math.max(maxd, Math.abs(probDiseased(model.forward(x).logits, T) - man.prob[i])); });
console.log(`CNN on Python input  max |P_js - P_py| = ${maxd.toExponential(2)}`);

// End to end: JS chain on the same degraded captures
const auc = (p, y) => {
  let s = 0, n = 0;
  for (let i = 0; i < p.length; i++) if (y[i] === 1) for (let j = 0; j < p.length; j++) if (y[j] === 0) { s += p[i] > p[j] ? 1 : p[i] === p[j] ? 0.5 : 0; n++; }
  return s / n;
};
const acc = (p, y) => p.reduce((s, v, i) => s + ((v > 0.5) === (y[i] === 1) ? 1 : 0), 0) / p.length;
const pj = [], t0 = Date.now();
for (const d of deg) {
  const pre = ip.enhance(ip.restore(d, "nlm"), "clahe"), seg = ip.segment(pre);
  pj.push(probDiseased(model.forward(ip.applyMask(pre, seg.mask, S)).logits, T));
}
const ms = (Date.now() - t0) / deg.length;
let agree = 0;
pj.forEach((p, i) => { if ((p > 0.5) === (man.prob[i] > 0.5)) agree++; });
console.log(`\nEnd to end on ${deg.length} degraded test images (lossless transmission)`);
console.log(`  Python: accuracy ${(100 * acc(man.prob, man.labels)).toFixed(2)}%  AUC ${auc(man.prob, man.labels).toFixed(4)}`);
console.log(`  JS    : accuracy ${(100 * acc(pj, man.labels)).toFixed(2)}%  AUC ${auc(pj, man.labels).toFixed(4)}  (${ms.toFixed(0)} ms/image in Node)`);
console.log(`  same 0.5-decision on ${(100 * agree / pj.length).toFixed(1)}% of images; mean |P_js - P_py| ${(pj.reduce((s, p, i) => s + Math.abs(p - man.prob[i]), 0) / pj.length).toFixed(4)}`);
