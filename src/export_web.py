"""Train the full pipeline on real data and export everything the website (docs/) needs.

    python -m src.export_web [--config PATH] [--source hf|folder|synthetic] [--epochs N]

Writes
  docs/model/model.json     architecture, temperature, thresholds, costs, gates, test metrics,
                            validation probabilities (the site re-runs the threshold search
                            in the browser when the visitor changes the costs)
  docs/model/weights.bin    float32 little-endian conv/linear weights, BatchNorm folded in
  docs/samples/*.jpg        one held-out TEST photo per tomato class (+ samples.json)
  results/web_check/        Python reference outputs for tools/web_parity.mjs, which checks
                            that the JavaScript pipeline reproduces the Python one
  results/web_model.pt      the trained PyTorch weights
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

from .data import load_data, tomato_label
from .degrade import degrade, rng_for
from .experiments import run_config
from .model import predict_logits, probs_pos
from .pipeline import PipelineConfig, run_one
from .stages import enhance, restore, saturation_channel, segment_mask
from .utils import RESULTS, ROOT, ensure_dirs, load_config, set_seed

DOCS = ROOT / "docs"


# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------
def fold_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    """Fold eval-mode BatchNorm into the preceding bias-free conv: y = W'x + b'."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None, None]
    b = bn.bias - bn.running_mean * scale
    return w.detach().float().cpu().numpy(), b.detach().float().cpu().numpy()


def export_weights(model, out_dir) -> list[dict]:
    """Write weights.bin and return the layer manifest (name, shape, float offset)."""
    model = model.cpu().eval()
    arrays, layers, off = [], [], 0
    for k, block in enumerate(model.features):
        w, b = fold_bn(block[0], block[1])
        for name, a in ((f"conv{k + 1}.w", w), (f"conv{k + 1}.b", b)):
            layers.append(dict(name=name, shape=list(a.shape), offset=off))
            arrays.append(a.ravel())
            off += a.size
    lin = model.head[3]
    for name, a in (("fc.w", lin.weight.detach().numpy()), ("fc.b", lin.bias.detach().numpy())):
        layers.append(dict(name=name, shape=list(a.shape), offset=off))
        arrays.append(a.astype(np.float32).ravel())
        off += a.size
    np.concatenate(arrays).astype("<f4").tofile(out_dir / "weights.bin")
    return layers


# ----------------------------------------------------------------------------
# Sample photos (held-out test split, one per tomato class)
# ----------------------------------------------------------------------------
def pretty_class(name: str) -> str:
    d = name.split("___")[-1].replace("_", " ").strip()
    d = re.sub(r"\s+", " ", d)
    return "Healthy" if d.lower() == "healthy" else d[0].upper() + d[1:]


def export_samples(cfg: dict, out_dir, seed: int) -> list[dict]:
    """Copy one original 256x256 test photo per tomato class from the HF data.zip."""
    import zipfile
    from huggingface_hub import hf_hub_download

    repo = cfg["data"]["hf_name"]
    get = lambda f: hf_hub_download(repo, f, repo_type="dataset")
    with open(get("splits/color_test.txt"), "r", encoding="utf-8") as f:
        files = [ln.strip() for ln in f if ln.strip()]
    by_class: dict[str, list[str]] = {}
    for p in files:
        cls = p.split("/")[-2]
        if tomato_label(cls) is not None:
            by_class.setdefault(cls, []).append(p)
    rng = np.random.default_rng(seed)
    out = []
    with zipfile.ZipFile(get("data.zip")) as z:
        for cls in sorted(by_class, key=lambda c: (tomato_label(c), c)):
            rel = sorted(by_class[cls])[int(rng.integers(len(by_class[cls])))]
            slug = re.sub(r"[^a-z0-9]+", "-", cls.lower().replace("tomato___", "")).strip("-")
            (out_dir / f"{slug}.jpg").write_bytes(z.read(rel))
            out.append(dict(file=f"samples/{slug}.jpg", cls=cls, name=pretty_class(cls),
                            label=int(tomato_label(cls))))
    return out


# ----------------------------------------------------------------------------
# Parity data for the JavaScript port
# ----------------------------------------------------------------------------
def export_check(data, cfg, model, T, seed: int, n: int = 300) -> None:
    """Degraded test images + Python stage outputs/probabilities for tools/web_parity.mjs."""
    out = RESULTS / "web_check"
    out.mkdir(parents=True, exist_ok=True)
    te = data["idx"]["test"]
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(te, size=min(n, len(te)), replace=False))
    size = int(cfg["data"]["img_size"])
    deg = np.stack([degrade(data["X"][i], cfg["degrade"], rng_for(seed, int(i), 0)) for i in pick])

    # End-to-end probabilities with the full pipeline but lossless transmission
    # (the browser's JPEG encoder is not bit-identical to libjpeg, so parity skips it)
    p = PipelineConfig.from_cfg(cfg, jpeg_q=None)
    res = [run_one(d, p, size, cfg["gates"]) for d in deg]
    Xp = np.stack([r["image"] for r in res])
    prob = probs_pos(predict_logits(model, Xp), T)
    gate = np.array([r["gate_ok"] for r in res])

    # Per-stage outputs for the first 24 images
    k = min(24, len(deg))
    stages = {}
    for m in ("gaussian", "median", "bilateral", "nlm"):
        stages[f"restore_{m}"] = np.stack([restore(d, m) for d in deg[:k]])
    nl = stages["restore_nlm"]
    for m in ("clahe", "histeq", "gamma"):
        stages[f"enhance_{m}"] = np.stack([enhance(x, m) for x in nl])
    ce = stages["enhance_clahe"]
    stages["saturation"] = np.stack([saturation_channel(x) for x in ce])
    stages["mask"] = np.stack([segment_mask(x) for x in ce])
    stages["cnn_input"] = Xp[:k]

    manifest = dict(n=len(pick), k=k, size=size, labels=data["y"][pick].tolist(),
                    prob=prob.tolist(), gate=gate.tolist(), T=T, stages={})
    deg.astype(np.uint8).tofile(out / "degraded.bin")
    for name, arr in stages.items():
        arr.astype(np.uint8).tofile(out / f"{name}.bin")
        manifest["stages"][name] = list(arr.shape)
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    print(f"[web] parity data -> {out}")


# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, choices=["hf", "folder", "synthetic"])
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.source:
        cfg["data"]["source"] = args.source
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed = int(cfg["seed"])
    set_seed(seed)
    ensure_dirs()
    data = load_data(cfg)
    data["idx"] = {s: np.flatnonzero(data["split"] == s) for s in ("train", "val", "test")}

    p = PipelineConfig.from_cfg(cfg)
    row, model, ex = run_config("web_full", data, cfg, p, seed)
    torch.save(model.state_dict(), RESULTS / "web_model.pt")

    mdir = DOCS / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    layers = export_weights(model, mdir)
    r4 = lambda a: [round(float(v), 5) for v in a]
    meta = dict(
        architecture="SmallCNN: 4 x [conv3x3 -> BN (folded) -> ReLU -> maxpool2], 32-64-128-128, "
                     "global average pool, linear 128->2",
        img_size=int(cfg["data"]["img_size"]),
        normalise="(x/255 - 0.5) / 0.5",
        layers=layers,
        temperature=row["T"], t_low=row["t_low"], t_high=row["t_high"],
        costs=cfg["costs"], gates=cfg["gates"], degrade=cfg["degrade"],
        pipeline=p.as_dict(), data_source=cfg["data"]["source"],
        test_metrics={k: row[k] for k in ("cnn_acc", "auc", "cost_per_sample", "sensitivity",
                                          "specificity", "decision_acc", "reimage_rate", "FN", "FP")},
        n_test=int(len(ex["y_test"])),
        val=dict(p0=r4(ex["p_val0"]), p1=r4(ex["p_val1"]), g0=[bool(g) for g in ex["gate_val0"]],
                 g1=[bool(g) for g in ex["gate_val1"]], y=[int(v) for v in ex["y_val"]]),
    )
    (mdir / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[web] model -> {mdir}  ({sum(int(np.prod(l['shape'])) for l in layers)} parameters)")

    sdir = DOCS / "samples"
    sdir.mkdir(parents=True, exist_ok=True)
    if cfg["data"]["source"] == "hf":
        samples = export_samples(cfg, sdir, seed)
        (sdir / "samples.json").write_text(json.dumps(samples, indent=1), encoding="utf-8")
        print(f"[web] {len(samples)} sample photos -> {sdir}")
    export_check(data, cfg, model, row["T"], seed)


if __name__ == "__main__":
    main()
