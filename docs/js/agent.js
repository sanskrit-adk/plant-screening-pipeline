// agent.js - the utility-based decision agent (port of src/decision.py).
//
// Actions: CLEAR, FLAG, RE-IMAGE. With calibrated p = P(diseased):
//   E[cost | CLEAR] = p * C_FN        E[cost | FLAG] = (1 - p) * C_FP
// so FLAG is the better bet once p > C_FP / (C_FP + C_FN), the Bayes threshold.
// RE-IMAGE costs C_RI and buys a second, independent look when p is in the uncertain band
// [t_low, t_high] or when a safety gate says the photo cannot be trusted.

export const bayesThreshold = (c) => c.fp / (c.fp + c.fn);

export function expectedCosts(p, c) {
  return { clear: p * c.fn, flag: (1 - p) * c.fp };
}

// First photo: gate failure or uncertain band -> RE-IMAGE, else FLAG above t_high, else CLEAR
export function decideFirst(p, gateOk, tLow, tHigh) {
  if (!gateOk) return { action: "RE-IMAGE", reason: "A safety gate failed, so this photo cannot be trusted." };
  if (p >= tLow && p <= tHigh) return { action: "RE-IMAGE", reason: "P(diseased) is inside the uncertain band." };
  if (p > tHigh) return { action: "FLAG", reason: "P(diseased) is above the upper threshold." };
  return { action: "CLEAR", reason: "P(diseased) is below the lower threshold." };
}

// Second photo: a failed gate never clears; otherwise use the Bayes threshold
export function decideSecond(p, gateOk, c) {
  if (!gateOk) return { action: "FLAG", reason: "The re-image also failed a safety gate. A failed gate never clears." };
  const tb = bayesThreshold(c);
  return p > tb
    ? { action: "FLAG", reason: `P(diseased) is above the Bayes threshold ${tb.toFixed(4)}.` }
    : { action: "CLEAR", reason: `P(diseased) is below the Bayes threshold ${tb.toFixed(4)}.` };
}

// numpy.quantile with linear interpolation
function quantile(sorted, q) {
  const pos = q * (sorted.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Exhaustive (t_low, t_high) search on validation data, minimising mean cost and breaking
// ties toward fewer re-images. val = { p0, p1, g0, g1, y } as exported in model.json.
export function searchThresholds(val, c) {
  const n = val.y.length, tb = bayesThreshold(c);
  const sorted = Float64Array.from(val.p0).sort();
  const cand = [0, tb, 1];
  for (let i = 0; i <= 100; i++) cand.push(quantile(sorted, i / 100));
  const grid = Array.from(new Set(cand)).sort((a, b) => a - b);
  const costRe = new Float64Array(n), costFlag = new Float64Array(n), costClear = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const flag2 = !val.g1[i] || val.p1[i] > tb;
    costRe[i] = c.reimage + (flag2 ? (val.y[i] === 0 ? c.fp : 0) : (val.y[i] === 1 ? c.fn : 0));
    costFlag[i] = val.y[i] === 0 ? c.fp : 0;
    costClear[i] = val.y[i] === 1 ? c.fn : 0;
  }
  let best = null;
  for (let a = 0; a < grid.length; a++)
    for (let b = a; b < grid.length; b++) {
      const lo = grid[a], hi = grid[b];
      let s = 0, r = 0;
      for (let i = 0; i < n; i++) {
        const p = val.p0[i], re = !val.g0[i] || (p >= lo && p <= hi);
        if (re) { s += costRe[i]; r++; } else s += p > hi ? costFlag[i] : costClear[i];
      }
      const cost = s / n, rate = r / n;
      if (!best || cost < best.cost - 1e-12 || (Math.abs(cost - best.cost) <= 1e-12 && rate < best.rate))
        best = { tLow: lo, tHigh: hi, cost, rate };
    }
  return best;
}
