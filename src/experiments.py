"""Experiments: what is each pipeline stage worth?

    python -m src.experiments --exp {ablation,compression,denoise,figures,propagation,all}
                              [--config PATH] [--quick] [--retrain] [--source hf|folder|synthetic]

  ablation     switch each stage off in turn (retrain per config x seed), cost + bootstrap CI
  compression  JPEG quality sweep: accuracy/sensitivity vs compression ratio, knee, operating point
  denoise      objective PSNR/SSIM of each restorer vs the noise-free reference + end-to-end effect
  figures      illustrative figures (restoration vs enhancement, segmentation, JPEG artefacts)
  propagation  error propagation under harsh lighting, with and without safety gates
  all          figures, denoise, compression, ablation, propagation

Outputs: results/*.csv|json and figures/*.png
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import replace

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend: write PNGs, never open windows
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .data import load_data
from .decision import (baseline_policies, bayes_threshold, classifier_metrics,
                       search_thresholds, simulate)
from .degrade import degrade, rng_for
from .model import cnn_latency_ms, fit_temperature, predict_logits, probs_pos, train_model
from .pipeline import PipelineConfig, benchmark_latency, process
from .stages import (apply_mask, compress, enhance, iou, raw_bytes, restore,
                     saturation_channel, segment_mask)
from .utils import FIGURES, RESULTS, ensure_dirs, load_config, set_seed

# ----------------------------------------------------------------------------
# Plot style (reference palette; text in neutral ink, recessive axes)
# ----------------------------------------------------------------------------
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SERIES = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, "#6250d6"]
C_MUTED = "#a8a7a1"     # non-highlighted bars
C_INK2 = "#52514e"      # secondary text / reference lines

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a", "xtick.color": C_INK2,
    "ytick.color": C_INK2, "axes.grid": False, "font.size": 9, "axes.titlesize": 10,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})

METRIC_KEYS = ["cost_per_sample", "sensitivity", "specificity", "decision_acc",
               "flag_rate", "reimage_rate", "FN", "FP"]


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------
def _json_clean(o):
    """Make numpy / NaN / inf JSON-safe (NaN and inf become null)."""
    if isinstance(o, dict):
        return {str(k): _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _json_clean(o.tolist())
    return o


def save_json(obj, name: str) -> None:
    path = RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_clean(obj), f, indent=2)
    print(f"[save] {path}")


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = RESULTS / name
    df.to_csv(path, index=False)
    print(f"[save] {path}")


def save_fig(fig, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[save] {path}")


def idx_of(data: dict, split: str) -> np.ndarray:
    """Global indices of a split (respects --quick subsampling via data['idx'])."""
    return data["idx"][split]


def bootstrap_ci(diff: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Paired bootstrap of the mean of per-sample differences -> (mean, lo, hi)."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def rms_contrast(img: np.ndarray) -> float:
    """RMS contrast = std of grey levels scaled to [0,1]."""
    return float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).std() / 255.0)


def psnr(ref: np.ndarray, img: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def ssim(ref: np.ndarray, img: np.ndarray) -> float:
    return float(structural_similarity(ref, img, channel_axis=-1, data_range=255))


# ----------------------------------------------------------------------------
# 9.0 One full configuration: preprocess, (train), calibrate, choose thresholds, test
# ----------------------------------------------------------------------------
def run_config(name: str, data: dict, cfg: dict, p: PipelineConfig, seed: int, model=None,
               test_dcfg: dict | None = None, use_gates: bool = True, keep_mask: bool = False,
               verbose: bool = True):
    """Evaluate one pipeline configuration end to end.

    Thresholds and temperature are chosen on VALIDATION only; TEST is simulated once.
    Returns (row dict, model, extras dict).
    """
    X, y = data["X"], data["y"]
    tr, va, te = idx_of(data, "train"), idx_of(data, "val"), idx_of(data, "test")
    costs = cfg["costs"]
    t0 = time.time()
    if verbose:
        print(f"  [{name}] seed={seed} {p.label()} gates={'on' if use_gates else 'off'}")

    # 1. validation, first attempt
    Xv0, _, _, gv0, _ = process(X, va, p, cfg, seed, attempt=0)

    # 2. train if no model was supplied
    if model is None:
        Xtr, _, _, _, _ = process(X, tr, p, cfg, seed, attempt=0)
        set_seed(seed)
        model = train_model(Xtr, y[tr], Xv0, y[va], cfg, seed, verbose=verbose)

    # 3. validation, re-image attempt
    Xv1, _, _, gv1, _ = process(X, va, p, cfg, seed, attempt=1)

    # 4. temperature scaling on validation (first attempt), calibrated probabilities
    lv0 = predict_logits(model, Xv0)
    T = fit_temperature(lv0, y[va])
    pv0, pv1 = probs_pos(lv0, T), probs_pos(predict_logits(model, Xv1), T)

    # 5. choose (t_low, t_high) on validation
    t_low, t_high, _ = search_thresholds(pv0, gv0, pv1, gv1, y[va], costs, use_gates)

    # 6. test, both attempts (optionally under a stress degradation)
    km = keep_mask and p.segment
    Xt0, nb0, area0, gt0, m0 = process(X, te, p, cfg, seed, 0, dcfg=test_dcfg, keep_mask=km)
    Xt1, _, _, gt1, _ = process(X, te, p, cfg, seed, 1, dcfg=test_dcfg)
    pt0 = probs_pos(predict_logits(model, Xt0), T)
    pt1 = probs_pos(predict_logits(model, Xt1), T)

    # 7. simulate the decision policy on test exactly once
    res = simulate(pt0, gt0, pt1, gt1, y[te], t_low, t_high, costs, use_gates)
    per_cost = res.pop("_cost")

    raw = raw_bytes(X[0])
    bpi = float(nb0.mean())
    row = dict(config=name, seed=seed, restore=p.restore, enhance=p.enhance,
               jpeg_q="lossless" if p.jpeg_q is None else int(p.jpeg_q), segment=p.segment,
               use_gates=use_gates, T=T, t_low=t_low, t_high=t_high,
               **classifier_metrics(pt0, y[te]), **res,
               bytes_per_image=bpi, compression_ratio=raw / bpi,
               gate_fail_rate=float((~gt0).mean()))
    extras = dict(p_test0=pt0, p_test1=pt1, gate_test0=gt0, gate_test1=gt1, y_test=y[te],
                  cost=per_cost, area_test0=area0, masks_test0=m0,
                  baselines=baseline_policies(pt0, y[te], costs),
                  # validation arrays (used by src/export_web.py for in-browser threshold search)
                  p_val0=pv0, p_val1=pv1, gate_val0=gv0, gate_val1=gv1, y_val=y[va])
    if verbose:
        print(f"  [{name}] cost={row['cost_per_sample']:.3f} sens={row['sensitivity']:.3f} "
              f"spec={row['specificity']:.3f} acc={row['cnn_acc']:.3f} auc={row['auc']:.3f} "
              f"reimage={row['reimage_rate']:.3f} ({time.time() - t0:.0f}s)")
    return row, model, extras


def add_latency(row: dict, data: dict, cfg: dict, p: PipelineConfig, model, seed: int) -> dict:
    """Add per-stage ms, CNN ms and total ms (single-threaded, sequential) to a row."""
    lat = benchmark_latency(data["X"], idx_of(data, "test"), p, cfg, seed)
    lat["ms_cnn"] = cnn_latency_ms(model, int(cfg["data"]["img_size"]))
    lat["ms_total"] = sum(lat.values())
    row.update(lat)
    return row


# ----------------------------------------------------------------------------
# 9.1 Ablation
# ----------------------------------------------------------------------------
def ablation_configs(cfg: dict) -> dict:
    full = PipelineConfig.from_cfg(cfg)
    return {
        "full": full,
        "no_restoration": replace(full, restore="none"),
        "no_enhancement": replace(full, enhance="none"),
        "no_compression": replace(full, jpeg_q=None),
        "no_segmentation": replace(full, segment=False),
        "cnn_only": PipelineConfig(restore="none", enhance="none", jpeg_q=None, segment=False),
    }


def exp_ablation(data: dict, cfg: dict) -> pd.DataFrame:
    print("\n=== ABLATION: retrain per config x seed ===")
    configs = ablation_configs(cfg)
    seeds = list(cfg["experiments"]["seeds"])
    rows, costs, baselines = [], {}, {}
    for seed in seeds:
        for name, p in configs.items():
            row, model, ex = run_config(name, data, cfg, p, seed)
            rows.append(add_latency(row, data, cfg, p, model, seed))
            costs[(name, seed)] = ex["cost"]
            policy = {k: row[k] for k in METRIC_KEYS}
            baselines.setdefault(name, {})[f"seed_{seed}"] = dict(
                **ex["baselines"], cost_aware_reimage=dict(t_low=row["t_low"], t_high=row["t_high"], **policy))
    runs = pd.DataFrame(rows)
    save_csv(runs, "ablation_runs.csv")

    # Mean / std over seeds
    cols = ["cost_per_sample", "sensitivity", "specificity", "decision_acc", "cnn_acc", "auc",
            "reimage_rate", "FN", "bytes_per_image", "ms_total"]
    g = runs.groupby("config", sort=False)[cols]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=0).add_suffix("_std")).reset_index()

    # Paired bootstrap of per-sample cost (averaged across seeds) vs full
    full_cost = np.mean([costs[("full", s)] for s in seeds], axis=0)
    ci = []
    for name in configs:
        c = np.mean([costs[(name, s)] for s in seeds], axis=0)
        d, lo, hi = bootstrap_ci(c - full_cost, seed=int(cfg["seed"]))
        ci.append(dict(config=name, delta_cost_vs_full=d, delta_ci_lo=lo, delta_ci_hi=hi,
                       measurable_change=bool(name != "full" and (lo > 0 or hi < 0))))
    summary = summary.merge(pd.DataFrame(ci), on="config")
    save_csv(summary, "ablation_summary.csv")
    save_json(baselines, "baseline_policies.json")
    plot_ablation(summary)
    return summary


def plot_ablation(s: pd.DataFrame) -> None:
    names = list(s["config"])
    colors = [C_BLUE if n == "full" else C_MUTED for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("cost_per_sample", "Expected cost per leaf (lower is better)", 1.0),
              ("sensitivity", "Sensitivity, % (higher is better)", 100.0)]
    for ax, (m, title, k) in zip(axes, panels):
        mean, std = s[f"{m}_mean"] * k, s[f"{m}_std"] * k
        ax.bar(x, mean, yerr=std, color=colors, width=0.62, capsize=3,
               error_kw=dict(ecolor=C_INK2, lw=1), edgecolor="white", linewidth=2)
        for xi, v in zip(x, mean):
            ax.annotate(f"{v:.2f}" if k == 1 else f"{v:.1f}", (xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=C_INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
        ax.set_title(title, loc="left")
        if k == 100:
            lo = max(0.0, float((mean - std).min()) - 5)
            ax.set_ylim(lo, 101)
        ax.margins(y=0.15)
    fig.suptitle("Ablation: each stage switched off (mean ± std over seeds; full pipeline in blue)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "ablation.png")


# ----------------------------------------------------------------------------
# 9.2 Compression sweep
# ----------------------------------------------------------------------------
def jpeg_psnr(data: dict, cfg: dict, p: PipelineConfig, q, seed: int, n: int = 200) -> float:
    """Mean PSNR of JPEG(q)-decoded vs uncompressed preprocessed (restored+enhanced) images."""
    if q is None:
        return float("inf")
    vals = []
    for i in idx_of(data, "test")[:n]:
        deg = degrade(data["X"][i], cfg["degrade"], rng_for(seed, int(i), 0))
        pre = enhance(restore(deg, p.restore), p.enhance)
        dec, _ = compress(pre, q)
        vals.append(psnr(pre, dec))
    return float(np.mean(vals))


def exp_compression(data: dict, cfg: dict, retrain: bool = False) -> pd.DataFrame:
    print(f"\n=== COMPRESSION SWEEP ({'retrain per quality' if retrain else 'train once on lossless'}) ===")
    seed = int(cfg["seed"])
    base = PipelineConfig.from_cfg(cfg, jpeg_q=None)
    qualities = [None] + list(cfg["experiments"]["jpeg_qualities"])
    model = None
    rows = []
    for q in qualities:
        p = replace(base, jpeg_q=q)
        name = "lossless" if q is None else f"q{q}"
        row, m, _ = run_config(name, data, cfg, p, seed, model=None if retrain else model)
        if model is None and not retrain:
            model = m  # the lossless model is reused for every quality
        row["psnr_db"] = jpeg_psnr(data, cfg, base, q, seed)
        row["kb_per_image"] = row["bytes_per_image"] / 1024.0
        rows.append(row)
    df = pd.DataFrame(rows)
    save_csv(df, "compression_sweep.csv")

    # Knee: first ratio (ascending) where acc or sensitivity drops > tolerance below lossless
    tol = float(cfg["experiments"]["degrade_tolerance_pp"]) / 100.0
    ref = df[df.config == "lossless"].iloc[0]
    jp = df[df.config != "lossless"].sort_values("compression_ratio").reset_index(drop=True)
    ok = (jp.cnn_acc >= ref.cnn_acc - tol) & (jp.sensitivity >= ref.sensitivity - tol)
    bad = np.flatnonzero(~ok.values)
    knee = jp.iloc[bad[0]] if len(bad) else None
    # Recommended operating point: highest ratio that is within tolerance and below the knee
    safe = jp.iloc[: bad[0]] if len(bad) else jp
    rec = safe.iloc[-1] if len(safe) else ref
    summary = dict(
        tolerance_pp=tol * 100,
        lossless=dict(bytes_per_image=ref.bytes_per_image, compression_ratio=ref.compression_ratio,
                      cnn_acc=ref.cnn_acc, sensitivity=ref.sensitivity, cost=ref.cost_per_sample),
        knee=None if knee is None else dict(quality=knee.jpeg_q, compression_ratio=knee.compression_ratio,
                                            cnn_acc=knee.cnn_acc, sensitivity=knee.sensitivity),
        recommended=dict(quality=rec.jpeg_q, compression_ratio=rec.compression_ratio,
                         bytes_per_image=rec.bytes_per_image, cnn_acc=rec.cnn_acc,
                         sensitivity=rec.sensitivity, cost=rec.cost_per_sample, psnr_db=rec.psnr_db),
        retrained_per_quality=retrain,
    )
    save_json(summary, "compression_summary.json")
    plot_compression(df, ref, knee)
    return df


def plot_compression(df: pd.DataFrame, ref, knee) -> None:
    jp = df[df.config != "lossless"].sort_values("compression_ratio")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a) accuracy and sensitivity vs compression ratio (log x)
    a.semilogx(jp.compression_ratio, jp.cnn_acc * 100, "o-", color=C_BLUE, label="CNN accuracy")
    a.semilogx(jp.compression_ratio, jp.sensitivity * 100, "s-", color=C_ORANGE, label="Decision sensitivity")
    a.axhline(ref.cnn_acc * 100, ls="--", lw=1, color=C_BLUE, alpha=0.7, label="Lossless accuracy")
    a.axhline(ref.sensitivity * 100, ls=":", lw=1, color=C_ORANGE, alpha=0.9, label="Lossless sensitivity")
    for _, r in jp.iterrows():
        a.annotate(f"q{r.jpeg_q}", (r.compression_ratio, r.cnn_acc * 100), xytext=(0, -13),
                   textcoords="offset points", ha="center", fontsize=7, color=C_INK2)
    if knee is not None:
        a.axvline(knee.compression_ratio, color=C_INK2, lw=1)
        a.annotate(f"knee (q{knee.jpeg_q})", (knee.compression_ratio, 1.0),
                   xycoords=("data", "axes fraction"), xytext=(4, -10), textcoords="offset points",
                   fontsize=8, color=C_INK2, ha="left")
    a.set_xlabel("Compression ratio vs raw RGB (log scale)")
    a.set_ylabel("%")
    a.set_title("(a) Accuracy and sensitivity vs compression", loc="left")
    # plain "20:1"-style tick labels instead of 2x10^1
    a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}:1"))
    a.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}:1"))
    a.tick_params(axis="x", which="minor", labelsize=7)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
    a.margins(x=0.08, y=0.15)

    # (b) expected cost vs bandwidth
    d = df.sort_values("kb_per_image")
    b.plot(d.kb_per_image, d.cost_per_sample, "o-", color=C_AQUA)
    for _, r in d.iterrows():
        lab = "lossless" if r.config == "lossless" else f"q{r.jpeg_q}"
        b.annotate(lab, (r.kb_per_image, r.cost_per_sample), xytext=(4, 4),
                   textcoords="offset points", fontsize=7, color=C_INK2)
    b.set_xlabel("KB per transmitted image")
    b.set_ylabel("Expected cost per leaf")
    b.set_title("(b) Decision cost vs bandwidth", loc="left")
    b.margins(x=0.08, y=0.15)
    fig.tight_layout()
    save_fig(fig, "compression_sweep.png")


# ----------------------------------------------------------------------------
# 9.3 Denoising
# ----------------------------------------------------------------------------
def exp_denoise(data: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== DENOISING: objective (vs noise-free reference) + end to end ===")
    seed = int(cfg["seed"])
    methods = list(cfg["experiments"]["denoise_methods"])
    sigmas = list(cfg["experiments"]["noise_sigmas"])
    va = idx_of(data, "val")[:150]
    prev = cv2.getNumThreads()
    cv2.setNumThreads(1)  # fair single-thread timing
    obj = []
    try:
        for sigma in sigmas:
            dcfg = dict(cfg["degrade"], noise_sigma=sigma)
            pairs = [degrade(data["X"][i], dcfg, rng_for(seed, int(i), 0), return_reference=True) for i in va]
            for m in methods:
                ps, ss, ts = [], [], []
                for deg, ref in pairs:
                    t = time.perf_counter()
                    out = restore(deg, m)
                    ts.append((time.perf_counter() - t) * 1e3)
                    ps.append(psnr(ref, out))
                    ss.append(ssim(ref, out))
                obj.append(dict(sigma=sigma, method=m, psnr_db=np.mean(ps), ssim=np.mean(ss), ms=np.mean(ts)))
                print(f"  sigma={sigma:>3} {m:<10} PSNR {np.mean(ps):6.2f} dB  SSIM {np.mean(ss):.3f}  {np.mean(ts):6.2f} ms")
    finally:
        cv2.setNumThreads(prev)
    obj = pd.DataFrame(obj)
    save_csv(obj, "denoise_objective.csv")

    # End to end at the default sigma: retrain per method
    rows = []
    for m in methods:
        p = PipelineConfig.from_cfg(cfg, restore=m)
        row, model, _ = run_config(f"restore_{m}", data, cfg, p, seed)
        rows.append(add_latency(row, data, cfg, p, model, seed))
    e2e = pd.DataFrame(rows)
    save_csv(e2e, "denoise_end_to_end.csv")
    plot_denoise(obj, e2e, methods)
    return obj, e2e


def plot_denoise(obj: pd.DataFrame, e2e: pd.DataFrame, methods: list) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, m in enumerate(methods):
        d = obj[obj.method == m]
        a.plot(d.sigma, d.psnr_db, "o-", color=SERIES[k % len(SERIES)], label=m)
    a.set_xlabel("Noise sigma (grey levels)")
    a.set_ylabel("PSNR vs noise-free reference (dB)")
    a.set_title("(a) Objective restoration quality", loc="left")
    a.set_xticks(sorted(obj.sigma.unique()))
    a.legend(fontsize=7, ncol=2)

    x = np.arange(len(e2e))
    w = 0.38
    b.bar(x - w / 2, e2e.cnn_acc * 100, w, color=C_BLUE, label="CNN accuracy", edgecolor="white", linewidth=2)
    b.bar(x + w / 2, e2e.sensitivity * 100, w, color=C_ORANGE, label="Decision sensitivity",
          edgecolor="white", linewidth=2)
    b.set_xticks(x)
    b.set_xticklabels([c.replace("restore_", "") for c in e2e.config], fontsize=8)
    vals = np.concatenate([e2e.cnn_acc, e2e.sensitivity]) * 100
    b.set_ylim(max(0, vals.min() - 8), 101)
    b.set_ylabel("%")
    b.set_title("(b) End-to-end effect at default sigma (retrained)", loc="left")
    b.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    save_fig(fig, "denoise.png")


# ----------------------------------------------------------------------------
# 9.4 Illustrative figures
# ----------------------------------------------------------------------------
def pick_diseased_test(data: dict) -> int:
    te = idx_of(data, "test")
    pos = te[data["y"][te] == 1]
    return int(pos[0] if len(pos) else te[0])


def exp_figures(data: dict, cfg: dict) -> None:
    print("\n=== FIGURES ===")
    seed = int(cfg["seed"])
    i = pick_diseased_test(data)
    clean = data["X"][i]
    p = PipelineConfig.from_cfg(cfg)
    size = int(cfg["data"]["img_size"])

    # --- restoration vs enhancement ---
    noisy, ref = degrade(clean, cfg["degrade"], rng_for(seed, i, 0), return_reference=True)
    restored = restore(noisy, "nlm")
    dim_cfg = dict(cfg["degrade"], illum_min=0.15, illum_max=0.6, noise_sigma=0, sp_prob=0.0)
    dim = degrade(clean, dim_cfg, rng_for(seed, i, 0))
    enh = enhance(dim, "clahe")
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    panels = [
        (clean, "Clean"),
        (noisy, f"Noisy capture\nPSNR {psnr(ref, noisy):.1f} dB"),
        (restored, f"NLM restored\nPSNR {psnr(ref, restored):.1f} dB, SSIM {ssim(ref, restored):.2f}"),
        (dim, f"Dim, uneven, noise-free\nRMS contrast {rms_contrast(dim):.3f}"),
        (enh, f"CLAHE enhanced\nRMS {rms_contrast(enh):.3f}, entropy {shannon_entropy(gray(enh)):.2f}"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6))
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.suptitle("Restoration = objective (scored vs noise-free reference)  |  "
                 "Enhancement = subjective (no ground truth)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, "restoration_vs_enhancement.png")

    # --- segmentation steps ---
    pre = enhance(restore(noisy, p.restore), p.enhance)
    sat = saturation_channel(pre)
    mask = segment_mask(pre)
    crop = apply_mask(pre, mask, size)
    mtitle = f"Mask (area {mask.mean():.2f})"
    if data.get("gt_mask") is not None and data["gt_valid"][i]:
        mtitle += f", IoU {iou(mask, data['gt_mask'][i]):.2f}"
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))
    for ax, (im, t, cm) in zip(axes, [(pre, "Restored + enhanced", None), (sat, "HSV saturation", "gray"),
                                       (mask, mtitle, "gray"), (crop, "Masked crop (CNN input)", None)]):
        ax.imshow(im, cmap=cm, vmin=0 if cm else None, vmax=(1 if im is mask else 255) if cm else None)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.suptitle("Segmentation: Otsu on saturation -> open -> close -> largest component -> fill holes",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_fig(fig, "segmentation_steps.png")

    # --- JPEG artefacts ---
    qs = [95, 30, 10, 5]
    h, w = pre.shape[:2]
    c = max(16, h // 3)
    y0, x0 = (h - c) // 2, (w - c) // 2
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    im_err = None
    for k, q in enumerate(qs):
        dec, nb = compress(pre, q)
        err = np.abs(dec.astype(np.int16) - pre.astype(np.int16)).mean(axis=2)
        axes[0, k].imshow(dec[y0:y0 + c, x0:x0 + c], interpolation="nearest")
        axes[0, k].set_title(f"q={q}  ratio {raw_bytes(pre) / nb:.0f}:1", fontsize=9)
        im_err = axes[1, k].imshow(err[y0:y0 + c, x0:x0 + c], cmap="magma", vmin=0, vmax=40,
                                   interpolation="nearest")
        axes[1, k].set_title(f"|error|  PSNR {psnr(pre, dec):.1f} dB", fontsize=9)
        for r in (0, 1):
            axes[r, k].axis("off")
    fig.suptitle("JPEG artefacts on the zoomed centre crop (top) and absolute error vs uncompressed (bottom)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 0.92, 0.93], h_pad=2.5)
    cax = fig.add_axes([0.93, 0.08, 0.015, 0.36])
    fig.colorbar(im_err, cax=cax, label="mean |error| (grey levels)")
    save_fig(fig, "jpeg_artifacts.png")


# ----------------------------------------------------------------------------
# 9.5 Error propagation
# ----------------------------------------------------------------------------
def exp_propagation(data: dict, cfg: dict) -> pd.DataFrame:
    print("\n=== ERROR PROPAGATION: normal vs harsh lighting, gates off vs on ===")
    seed = int(cfg["seed"])
    p = PipelineConfig.from_cfg(cfg)
    has_gt = data.get("gt_mask") is not None
    _, model, _ = run_config("train_full", data, cfg, p, seed)  # trained under normal conditions
    te = idx_of(data, "test")
    conds = {"normal": None,
             "harsh_lighting": dict(cfg["degrade"], illum_min=float(cfg["experiments"]["stress_illum_min"]))}
    rows = []
    for cname, dcfg in conds.items():
        for use_gates in (False, True):
            row, _, ex = run_config(f"{cname}|gates_{'on' if use_gates else 'off'}", data, cfg, p, seed,
                                    model=model, test_dcfg=dcfg, use_gates=use_gates, keep_mask=has_gt)
            y, p0, g0 = ex["y_test"], ex["p_test0"], ex["gate_test0"]
            dis = y == 1
            r = dict(condition=cname, gates="on" if use_gates else "off",
                     cost_per_sample=row["cost_per_sample"], FN=row["FN"], FP=row["FP"],
                     sensitivity=row["sensitivity"], specificity=row["specificity"],
                     reimage_rate=row["reimage_rate"], gate_fail_rate=row["gate_fail_rate"],
                     cnn_acc=row["cnn_acc"], auc=row["auc"], t_low=row["t_low"], t_high=row["t_high"],
                     mean_p_diseased_gate_failed=float(p0[dis & ~g0].mean()) if (dis & ~g0).any() else float("nan"),
                     mean_p_diseased_gate_passed=float(p0[dis & g0].mean()) if (dis & g0).any() else float("nan"),
                     n_gate_failed=int((~g0).sum()),
                     misses_among_gate_failures=int(((ex["cost"] >= cfg["costs"]["fn"]) & ~g0 & dis).sum()))
            if has_gt and ex["masks_test0"] is not None:
                valid = data["gt_valid"][te]
                ious = [iou(m, gtm) for m, gtm, v in zip(ex["masks_test0"], data["gt_mask"][te], valid) if v]
                r["mean_seg_iou"] = float(np.mean(ious)) if ious else float("nan")
            rows.append(r)
    df = pd.DataFrame(rows)
    save_csv(df, "error_propagation.csv")
    return df


# ----------------------------------------------------------------------------
# 9.6 Quick mode + 9.7 main
# ----------------------------------------------------------------------------
def apply_quick(cfg: dict, data: dict) -> None:
    """Shrink everything for a fast smoke test (in place)."""
    rng = np.random.default_rng(int(cfg["seed"]))
    for s, n in (("train", 200), ("val", 80), ("test", 80)):
        idx = data["idx"][s]
        data["idx"][s] = np.sort(rng.permutation(idx)[:n])
    cfg["train"].update(epochs=2, patience=2)
    cfg["experiments"].update(seeds=[int(cfg["seed"])], jpeg_qualities=[75, 30, 5],
                              denoise_methods=["none", "median", "nlm"], noise_sigmas=[20], latency_n=10)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all",
                    choices=["ablation", "compression", "denoise", "figures", "propagation", "all"])
    ap.add_argument("--config", default=None, help="path to config.yaml (default: project root)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    ap.add_argument("--retrain", action="store_true", help="compression: retrain per JPEG quality")
    ap.add_argument("--source", default=None, choices=["hf", "folder", "synthetic"],
                    help="override data.source from the config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    if args.source:
        cfg["data"]["source"] = args.source
    set_seed(int(cfg["seed"]))
    ensure_dirs()

    data = load_data(cfg)
    data["idx"] = {s: np.flatnonzero(data["split"] == s) for s in ("train", "val", "test")}
    if args.quick:
        apply_quick(cfg, data)
        print("[quick] 200 train / 80 val / 80 test, 2 epochs, 1 seed")
    print(f"[decision] costs {cfg['costs']}  ->  Bayes threshold = {bayes_threshold(cfg['costs']):.4f}")

    order = ["figures", "denoise", "compression", "ablation", "propagation"] if args.exp == "all" else [args.exp]
    out = {}
    t0 = time.time()
    for e in order:
        if e == "figures":
            exp_figures(data, cfg)
        elif e == "denoise":
            out["denoise_objective"], out["denoise_end_to_end"] = exp_denoise(data, cfg)
        elif e == "compression":
            out["compression"] = exp_compression(data, cfg, retrain=args.retrain)
        elif e == "ablation":
            out["ablation"] = exp_ablation(data, cfg)
        elif e == "propagation":
            out["propagation"] = exp_propagation(data, cfg)

    # Summary tables
    show = {
        "denoise_objective": ["sigma", "method", "psnr_db", "ssim", "ms"],
        "denoise_end_to_end": ["config", "cnn_acc", "auc", "cost_per_sample", "sensitivity", "ms_total"],
        "compression": ["config", "compression_ratio", "kb_per_image", "psnr_db", "cnn_acc",
                        "sensitivity", "cost_per_sample"],
        "ablation": ["config", "cost_per_sample_mean", "cost_per_sample_std", "sensitivity_mean",
                     "cnn_acc_mean", "delta_cost_vs_full", "delta_ci_lo", "delta_ci_hi", "measurable_change"],
        "propagation": ["condition", "gates", "cost_per_sample", "FN", "sensitivity", "reimage_rate",
                        "gate_fail_rate", "misses_among_gate_failures"],
    }
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        for k, df in out.items():
            cols = [c for c in show[k] if c in df.columns]
            if k == "propagation" and "mean_seg_iou" in df.columns:
                cols.append("mean_seg_iou")
            print(f"\n--- {k} ---")
            print(df[cols].to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {RESULTS}, figures in {FIGURES}")


if __name__ == "__main__":
    main()
