"""Utility-based decision agent: CLEAR / FLAG / RE-IMAGE under asymmetric costs.

Costs (config `costs`):
    CLEAR a diseased leaf  -> C_FN   (missed disease; the expensive error)
    FLAG  a healthy leaf   -> C_FP   (wasted manual inspection)
    RE-IMAGE               -> C_RI + cost of the second decision

With a calibrated p = P(diseased | image):
    E[cost | CLEAR] = p * C_FN
    E[cost | FLAG]  = (1 - p) * C_FP
FLAG is cheaper whenever p > C_FP / (C_FP + C_FN)  -> the Bayes threshold (1/51 here).

Policy (two thresholds t_low <= t_high chosen on VALIDATION):
    attempt 1: gate failed or t_low <= p1 <= t_high  -> RE-IMAGE
               else p1 > t_high -> FLAG, otherwise CLEAR
    attempt 2: gate failed -> FLAG (a failed gate NEVER clears)
               else p2 > Bayes threshold -> FLAG, otherwise CLEAR
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bayes_threshold(costs: dict) -> float:
    """Probability above which FLAG has lower expected cost than CLEAR."""
    return float(costs["fp"]) / (float(costs["fp"]) + float(costs["fn"]))


def _outcome(flag: np.ndarray, y: np.ndarray, reimage: np.ndarray, costs: dict) -> dict:
    """Metrics for final flag decisions (and which samples were re-imaged)."""
    y = np.asarray(y).astype(int)
    fn_mask = (~flag) & (y == 1)
    fp_mask = flag & (y == 0)
    cost = fn_mask * float(costs["fn"]) + fp_mask * float(costs["fp"]) + reimage * float(costs["reimage"])
    P, N = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    return dict(
        cost_per_sample=float(cost.mean()) if len(y) else float("nan"),
        sensitivity=float((flag & (y == 1)).sum() / P),
        specificity=float(((~flag) & (y == 0)).sum() / N),
        decision_acc=float((flag == (y == 1)).mean()) if len(y) else float("nan"),
        flag_rate=float(flag.mean()) if len(y) else float("nan"),
        reimage_rate=float(reimage.mean()) if len(y) else float("nan"),
        FN=int(fn_mask.sum()),
        FP=int(fp_mask.sum()),
        _cost=cost.astype(np.float64),
    )


def simulate(p1, gate1, p2, gate2, y, t_low: float, t_high: float, costs: dict,
             use_gates: bool = True) -> dict:
    """Run the CLEAR / FLAG / RE-IMAGE policy on paired first/second captures.

    p1, p2: calibrated P(diseased) of the first photo and of the re-image.
    gate1, gate2: safety-gate results for each photo (ignored if use_gates=False).
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    g1 = np.asarray(gate1, bool) if use_gates else np.ones_like(p1, bool)
    g2 = np.asarray(gate2, bool) if use_gates else np.ones_like(p2, bool)
    tb = bayes_threshold(costs)

    reimage = (~g1) | ((p1 >= t_low) & (p1 <= t_high))
    flag_first = p1 > t_high
    flag_second = (~g2) | (p2 > tb)            # failed gate on the re-image -> FLAG
    flag = np.where(reimage, flag_second, flag_first)
    return _outcome(flag, y, reimage, costs)


def search_thresholds(p1, gate1, p2, gate2, y, costs: dict, use_gates: bool = True):
    """Grid-search (t_low, t_high) on VALIDATION minimising mean cost.

    Grid = unique({0, Bayes, 1} U 101 quantiles of p1). Every pair lo <= hi is tried;
    ties are broken toward the lower re-image rate. Returns (t_low, t_high, metrics).
    """
    p1 = np.asarray(p1, float)
    tb = bayes_threshold(costs)
    grid = np.unique(np.concatenate([[0.0, tb, 1.0], np.quantile(p1, np.linspace(0, 1, 101))]))

    # Precompute everything that does not depend on the thresholds (vectorised search)
    y = np.asarray(y).astype(int)
    g1 = np.asarray(gate1, bool) if use_gates else np.ones_like(p1, bool)
    g2 = np.asarray(gate2, bool) if use_gates else np.ones_like(p1, bool)
    flag2 = (~g2) | (np.asarray(p2, float) > tb)
    c_fn, c_fp, c_ri = float(costs["fn"]), float(costs["fp"]), float(costs["reimage"])
    cost_if_reimaged = c_ri + np.where(flag2, (y == 0) * c_fp, (y == 1) * c_fn)
    cost_flag1 = (y == 0) * c_fp
    cost_clear1 = (y == 1) * c_fn

    best = None
    for i, lo in enumerate(grid):
        in_lo = p1 >= lo
        for hi in grid[i:]:
            rei = (~g1) | (in_lo & (p1 <= hi))
            c = np.where(rei, cost_if_reimaged, np.where(p1 > hi, cost_flag1, cost_clear1)).mean()
            r = rei.mean()
            key = (round(float(c), 12), float(r))
            if best is None or key < best[0]:
                best = (key, float(lo), float(hi))
    _, lo, hi = best
    return lo, hi, simulate(p1, gate1, p2, gate2, y, lo, hi, costs, use_gates)


def baseline_policies(p, y, costs: dict) -> dict:
    """Single-shot baselines with no re-imaging: argmax (t=0.5) and Bayes threshold."""
    p = np.asarray(p, float)
    none = np.zeros_like(p, bool)
    out = {}
    for name, t in (("argmax_0.5", 0.5), ("bayes_threshold", bayes_threshold(costs))):
        m = _outcome(p > t, y, none, costs)
        m.pop("_cost")
        out[name] = dict(threshold=float(t), **m)
    return out


def classifier_metrics(p, y) -> dict:
    """CNN accuracy at 0.5 and ROC-AUC (NaN if only one class is present)."""
    p, y = np.asarray(p, float), np.asarray(y).astype(int)
    acc = float(((p > 0.5) == (y == 1)).mean()) if len(y) else float("nan")
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return dict(cnn_acc=acc, auc=auc)
