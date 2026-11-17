"""Dataset loading: PlantVillage tomato leaves as a BINARY task (0 healthy / 1 diseased).

Sources (config data.source):
  hf        - Hugging Face `mohanty/PlantVillage`, config "color" (default)
              https://huggingface.co/datasets/mohanty/PlantVillage
  folder    - local copy of https://github.com/spMohanty/PlantVillage-Dataset
              data/raw/color/<ClassName>/*.jpg
              data/raw/segmented/<ClassName>/<stem>_final_masked.jpg  (optional, gives GT masks)
  synthetic - ~600 procedurally generated leaves with GT masks (smoke tests only)

Dataset references:
  Hugging Face (loaded by source=hf, includes segmented masks):
      https://huggingface.co/datasets/mohanty/PlantVillage
  Official GitHub repo:            https://github.com/spMohanty/PlantVillage-Dataset
  TensorFlow Datasets (class list and size info):
      https://www.tensorflow.org/datasets/catalog/plant_village
  Original paper (Hughes & Salathe, 2015):  https://arxiv.org/abs/1511.08060
  Mohanty, Hughes & Salathe (2016), Front. Plant Sci. 7:1419:
      https://doi.org/10.3389/fpls.2016.01419

Every source returns the same dict:
  X       [N,H,W,3] uint8 RGB      y      [N] int64 (0 healthy, 1 diseased)
  groups  [N] str (leaf id)        split  [N] str in {"train","val","test"}
  gt_mask [N,H,W] uint8 0/1 or None   gt_valid [N] bool or None (which masks exist)
and is cached to an .npz so later runs skip decoding.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .utils import resolve

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------
# Class-name mapping
# ----------------------------------------------------------------------------
def tomato_label(class_name: str) -> int | None:
    """Map a PlantVillage class name to 0/1, or None if it is not a tomato class.

    Case-insensitive: the name must start with "tomato"; "healthy" -> 0, any disease -> 1.
    """
    n = class_name.strip().lower()
    if not n.startswith("tomato"):
        return None
    return 0 if "healthy" in n else 1


# ----------------------------------------------------------------------------
# Small helpers shared by all sources
# ----------------------------------------------------------------------------
def to_rgb_resized(img, size: int) -> np.ndarray:
    """PIL image or numpy array -> RGB uint8 numpy array resized to size x size."""
    if not isinstance(img, np.ndarray):
        img = np.asarray(img.convert("RGB"))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def balance_indices(labels: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Keep ALL healthy (0) and at most ratio * n_healthy diseased (1), sampled at random."""
    neg = np.flatnonzero(labels == 0)
    pos = np.flatnonzero(labels == 1)
    n_pos = min(len(pos), int(round(ratio * len(neg)))) if len(neg) else len(pos)
    pos = rng.choice(pos, size=n_pos, replace=False) if n_pos < len(pos) else pos
    return np.sort(np.concatenate([neg, pos]))


def group_split(groups: np.ndarray, val_frac: float, test_frac: float, seed: int,
                with_test: bool = True) -> np.ndarray:
    """Assign train/val/(test) with GroupShuffleSplit so no group spans two splits."""
    n = len(groups)
    split = np.full(n, "train", dtype=object)
    idx = np.arange(n)
    if with_test and test_frac > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest, te = next(gss.split(idx, groups=groups))
        split[te] = "test"
        idx = idx[rest]
        # val fraction is relative to the whole dataset, so rescale for the remainder
        val_rel = val_frac / (1.0 - test_frac)
    else:
        val_rel = val_frac
    if val_frac > 0 and len(idx) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
        _, va = next(gss.split(idx, groups=groups[idx]))
        split[idx[va]] = "val"
    return split.astype(str)


def print_summary(d: dict, source: str) -> None:
    """Print healthy/diseased counts per split and whether GT masks exist."""
    print(f"[data] source={source}  N={len(d['y'])}  image={d['X'].shape[1:]}")
    for s in ("train", "val", "test"):
        m = d["split"] == s
        n0 = int(np.sum(d["y"][m] == 0))
        n1 = int(np.sum(d["y"][m] == 1))
        print(f"[data]   {s:<5}: {m.sum():5d}  (healthy {n0:5d} | diseased {n1:5d})")
    if d.get("gt_mask") is None:
        print("[data]   ground-truth leaf masks: NO")
    else:
        print(f"[data]   ground-truth leaf masks: YES ({int(d['gt_valid'].sum())} images)")


# ----------------------------------------------------------------------------
# (a) Hugging Face
# ----------------------------------------------------------------------------
def _detect_columns(features) -> tuple[str, str, str | None]:
    """Find the Image column, the ClassLabel column and a leaf-id column ("leaf" in the name)."""
    from datasets import ClassLabel, Image

    img_col = lab_col = leaf_col = None
    for name, feat in features.items():
        if isinstance(feat, Image) and img_col is None:
            img_col = name
        elif isinstance(feat, ClassLabel) and lab_col is None:
            lab_col = name
        if "leaf" in name.lower() and leaf_col is None:
            leaf_col = name
    if img_col is None or lab_col is None:
        raise RuntimeError(f"Could not detect image/label columns in features: {features}")
    return img_col, lab_col, leaf_col


def leaf_id_from_path(rel_path: str, leaf_map: dict | None) -> str:
    """Physical-leaf id for a PlantVillage file, replicating the HF `plant_village.py` script.

    Filenames look like "<uuid>___<photographer> <number>.JPG". The part after "___"
    identifies the photo session; leaf-map.json maps it to a physical leaf id, so
    several photos of the same leaf share one group.
    """
    parts = rel_path.replace("\\", "/").split("/")
    class_name, file_name = parts[-2], parts[-1]
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()
    sugg = (leaf_map or {}).get(ident.lower().strip())
    if sugg:
        if len(sugg) == 1:
            return sugg[0]
        for s in sugg:
            if class_name in s:
                return s
    return f"fallback_{ident}"


def _load_hf_zip(dcfg: dict, seed: int) -> dict:
    """Fallback when `load_dataset` cannot build the repo (it is a legacy loading script,
    which datasets>=4 no longer executes). Reads the same files the script uses:
    data.zip (~2 GB, downloaded once into the HF cache), splits/color_{train,test}.txt
    and leaf_grouping/leaf-map.json. The zip also has raw/segmented/, so GT masks are built.
    """
    import zipfile
    from huggingface_hub import hf_hub_download

    repo = dcfg["hf_name"]
    get = lambda f: hf_hub_download(repo, f, repo_type="dataset")
    print("[data] downloading data.zip (~2 GB, cached after the first run) ...")
    zpath = get("data.zip")
    with open(get("leaf_grouping/leaf-map.json"), "r", encoding="utf-8") as f:
        leaf_map = json.load(f)
    size = int(dcfg["img_size"])
    rng = np.random.default_rng(seed)

    X, y, groups, split, masks, valid = [], [], [], [], [], []
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        for part in ("train", "test"):
            with open(get(f"splits/color_{part}.txt"), "r", encoding="utf-8") as f:
                files = [ln.strip() for ln in f if ln.strip()]
            files = [p for p in files if tomato_label(p.split("/")[-2]) is not None]
            labels = np.array([tomato_label(p.split("/")[-2]) for p in files])
            for i in tqdm(balance_indices(labels, dcfg["pos_to_neg_ratio"], rng), desc=f"decode {part}"):
                rel = files[i]
                buf = np.frombuffer(z.read(rel), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                X.append(to_rgb_resized(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size))
                y.append(int(labels[i]))
                groups.append(leaf_id_from_path(rel, leaf_map))
                split.append(part)
                cls, fn = rel.split("/")[-2], rel.split("/")[-1]
                seg = f"raw/segmented/{cls}/{Path(fn).stem}_final_masked.jpg"
                m = cv2.imdecode(np.frombuffer(z.read(seg), np.uint8), cv2.IMREAD_GRAYSCALE) \
                    if seg in names else None
                if m is not None:
                    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
                    masks.append((m > 10).astype(np.uint8))
                    valid.append(True)
                else:
                    masks.append(np.zeros((size, size), np.uint8))
                    valid.append(False)

    split = np.asarray(split, dtype=object)
    groups = np.asarray(groups)
    tr = np.flatnonzero(split == "train")
    split[tr] = group_split(groups[tr], dcfg["val_frac"], 0.0, seed, with_test=False)
    valid = np.asarray(valid, bool)
    return dict(X=np.stack(X), y=np.asarray(y, np.int64), groups=groups, split=split.astype(str),
                gt_mask=np.stack(masks) if valid.any() else None,
                gt_valid=valid if valid.any() else None)


def load_hf(dcfg: dict, seed: int) -> dict:
    """Download/load PlantVillage from Hugging Face and build the binary tomato set."""
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install datasets  (or set data.source: folder/synthetic)") from e

    print(f"[data] loading Hugging Face dataset {dcfg['hf_name']!r} (config 'color') ...")
    try:
        ds = load_dataset(dcfg["hf_name"], "color")
    except Exception as e:  # noqa: BLE001 - any builder failure -> read the repo files directly
        print(f"[data] load_dataset failed ({type(e).__name__}: {str(e)[:120]}); "
              "reading the repository files directly instead.")
        return _load_hf_zip(dcfg, seed)
    rng = np.random.default_rng(seed)
    size = int(dcfg["img_size"])
    splits_available = list(ds.keys())
    print(f"[data] HF splits: {splits_available}")

    first = ds[splits_available[0]]
    img_col, lab_col, leaf_col = _detect_columns(first.features)
    names = first.features[lab_col].names
    print(f"[data] columns: image={img_col!r} label={lab_col!r} leaf_id={leaf_col!r}")
    if leaf_col is None:
        print("[data] WARNING: no leaf-id column found; grouping per image "
              "(validation split may contain near-duplicate leaves).")

    # Map each original class id -> 0/1/None
    class_map = {i: tomato_label(n) for i, n in enumerate(names)}

    # Use the dataset's own train/test split (leaf-grouped) when present
    if "train" in ds and "test" in ds:
        parts = [("train", ds["train"]), ("test", ds["test"])]
    else:
        print("[data] WARNING: no train/test split in the HF dataset; making a grouped split.")
        parts = [("all", ds[splits_available[0]])]

    X, y, groups, split = [], [], [], []
    for part_name, part in parts:
        raw_lab = np.asarray(part[lab_col])
        bin_lab = np.array([class_map[int(c)] if class_map[int(c)] is not None else -1
                            for c in raw_lab])
        tomato = np.flatnonzero(bin_lab >= 0)
        keep = tomato[balance_indices(bin_lab[tomato], dcfg["pos_to_neg_ratio"], rng)]
        leaf_ids = (np.asarray(part[leaf_col]).astype(str) if leaf_col
                    else np.array([f"{part_name}_{i}" for i in range(len(part))]))
        sub = part.select(keep.tolist())
        for j, row in enumerate(tqdm(sub, desc=f"decode {part_name}", total=len(keep))):
            X.append(to_rgb_resized(row[img_col], size))
            y.append(int(bin_lab[keep[j]]))
            groups.append(str(leaf_ids[keep[j]]))
            split.append(part_name)

    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    split = np.asarray(split, dtype=object)

    if parts[0][0] == "all":
        split = group_split(groups, dcfg["val_frac"], dcfg["test_frac"], seed)
    else:
        # Carve validation out of train, grouped by leaf id
        tr = np.flatnonzero(split == "train")
        sub_split = group_split(groups[tr], dcfg["val_frac"], 0.0, seed, with_test=False)
        split[tr] = sub_split
    return dict(X=X, y=y, groups=groups, split=split.astype(str), gt_mask=None, gt_valid=None)


# ----------------------------------------------------------------------------
# (b) Local folder (GitHub repo layout)
# ----------------------------------------------------------------------------
def load_folder(dcfg: dict, seed: int) -> dict:
    """Read data/raw/color/<Class>/*.jpg (+ optional segmented masks)."""
    root = resolve(dcfg["folder_root"])
    color = root / "color"
    seg = root / "segmented"
    if not color.is_dir():
        raise FileNotFoundError(
            f"{color} not found. Clone https://github.com/spMohanty/PlantVillage-Dataset "
            f"and copy raw/color (and optionally raw/segmented) into {root}.")
    size = int(dcfg["img_size"])
    rng = np.random.default_rng(seed)

    files, labels, classes = [], [], []
    for cdir in sorted(p for p in color.iterdir() if p.is_dir()):
        lab = tomato_label(cdir.name)
        if lab is None:
            continue
        for f in sorted(cdir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                files.append(f)
                labels.append(lab)
                classes.append(cdir.name)
    if not files:
        raise RuntimeError(f"No tomato images found under {color}")
    labels = np.asarray(labels)
    keep = balance_indices(labels, dcfg["pos_to_neg_ratio"], rng)

    # Optional: if the HF leaf map is present, group by physical leaf (stricter than the
    # default rule, whose "<uuid>" prefix is unique per photo).
    leaf_map = None
    for lm in (root / "leaf_grouping" / "leaf-map.json", root / "leaf-map.json"):
        if lm.exists():
            with open(lm, "r", encoding="utf-8") as f:
                leaf_map = json.load(f)
            print(f"[data] grouping by physical leaf using {lm}")
            break

    X, y, groups, masks, valid = [], [], [], [], []
    for i in tqdm(keep, desc="read folder"):
        f = files[i]
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        X.append(to_rgb_resized(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size))
        y.append(int(labels[i]))
        # Group = filename part before "___"; physical-leaf id if leaf-map.json is available
        groups.append(leaf_id_from_path(f"{classes[i]}/{f.name}", leaf_map) if leaf_map
                      else f.stem.split("___")[0])
        mpath = seg / classes[i] / f"{f.stem}_final_masked.jpg"
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) if mpath.exists() else None
        if m is not None:
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
            masks.append((m > 10).astype(np.uint8))
            valid.append(True)
        else:
            masks.append(np.zeros((size, size), np.uint8))
            valid.append(False)

    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups)
    split = group_split(groups, dcfg["val_frac"], dcfg["test_frac"], seed)
    valid = np.asarray(valid, bool)
    gt = np.stack(masks) if valid.any() else None
    return dict(X=X, y=y, groups=groups, split=split, gt_mask=gt,
                gt_valid=valid if gt is not None else None)


# ----------------------------------------------------------------------------
# (c) Synthetic leaves (smoke tests)
# ----------------------------------------------------------------------------
def make_synthetic_leaf(size: int, diseased: bool, rng: np.random.Generator):
    """One fake leaf: green ellipse on grey paper; diseased -> 3-7 brown lesion circles."""
    paper = int(rng.integers(150, 210))
    img = np.full((size, size, 3), paper, np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, (1, 1, 3)), 0, 255).astype(np.uint8)

    cx = int(size / 2 + rng.integers(-size // 10, size // 10 + 1))
    cy = int(size / 2 + rng.integers(-size // 10, size // 10 + 1))
    ax = int(size * rng.uniform(0.28, 0.40))      # semi-major axis
    ay = int(size * rng.uniform(0.15, 0.25))      # semi-minor axis
    ang = float(rng.uniform(0, 180))
    green = (int(rng.integers(30, 80)), int(rng.integers(110, 170)), int(rng.integers(30, 70)))

    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), ang, 0, 360, 1, -1)
    cv2.ellipse(img, (cx, cy), (ax, ay), ang, 0, 360, green, -1)
    # Midrib: a slightly lighter line along the major axis
    t = np.deg2rad(ang)
    p1 = (int(cx - 0.9 * ax * np.cos(t)), int(cy - 0.9 * ax * np.sin(t)))
    p2 = (int(cx + 0.9 * ax * np.cos(t)), int(cy + 0.9 * ax * np.sin(t)))
    cv2.line(img, p1, p2, tuple(min(255, c + 35) for c in green), 1)

    if diseased:
        ys, xs = np.nonzero(mask)
        for _ in range(int(rng.integers(3, 8))):
            k = int(rng.integers(len(xs)))
            r = int(rng.integers(max(2, size // 40), max(4, size // 14)))
            brown = (int(rng.integers(90, 140)), int(rng.integers(50, 80)), int(rng.integers(15, 40)))
            cv2.circle(img, (int(xs[k]), int(ys[k])), r, brown, -1)
        img[mask == 0] = paper  # keep lesions inside the leaf outline
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, mask


def load_synthetic(dcfg: dict, seed: int) -> dict:
    """~600 leaves, about 2:1 diseased:healthy, with GT masks; one group per image."""
    rng = np.random.default_rng(seed)
    n = int(dcfg.get("synthetic_n", 600))
    size = int(dcfg["img_size"])
    n_healthy = n // 3
    labels = np.array([0] * n_healthy + [1] * (n - n_healthy))
    rng.shuffle(labels)
    X, M = zip(*(make_synthetic_leaf(size, bool(l), rng) for l in labels))
    X = np.stack(X)
    M = np.stack(M)
    groups = np.array([f"syn_{i}" for i in range(n)])
    split = group_split(groups, dcfg["val_frac"], dcfg["test_frac"], seed)
    return dict(X=X, y=labels.astype(np.int64), groups=groups, split=split,
                gt_mask=M, gt_valid=np.ones(n, bool))


# ----------------------------------------------------------------------------
# Public entry point with caching
# ----------------------------------------------------------------------------
def _cache_path(dcfg: dict) -> Path:
    p = resolve(dcfg["cache"])
    if dcfg["source"] == "synthetic":
        p = p.with_name(p.stem + "_synthetic" + p.suffix)
    elif dcfg["source"] == "folder":
        p = p.with_name(p.stem + "_folder" + p.suffix)
    return p


def _cache_meta(dcfg: dict, seed: int) -> str:
    """Settings that change the cached arrays; a mismatch forces a rebuild."""
    keys = ["source", "hf_name", "folder_root", "img_size", "pos_to_neg_ratio",
            "val_frac", "test_frac", "synthetic_n"]
    return json.dumps({k: dcfg.get(k) for k in keys} | {"seed": seed}, sort_keys=True)


def load_data(cfg: dict) -> dict:
    """Load the configured source (from cache when possible) and print a summary."""
    dcfg = cfg["data"]
    seed = int(cfg["seed"])
    src = dcfg["source"]
    cache = _cache_path(dcfg)
    meta = _cache_meta(dcfg, seed)

    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if "meta" in z and str(z["meta"]) == meta:
            d = dict(X=z["X"], y=z["y"], groups=z["groups"].astype(str),
                     split=z["split"].astype(str),
                     gt_mask=z["gt_mask"] if "gt_mask" in z else None,
                     gt_valid=z["gt_valid"] if "gt_valid" in z else None)
            print(f"[data] loaded cache {cache}")
            print_summary(d, src)
            return d
        print("[data] cache settings changed; rebuilding")

    loaders = {"hf": load_hf, "folder": load_folder, "synthetic": load_synthetic}
    if src not in loaders:
        raise ValueError(f"data.source must be one of {list(loaders)}, got {src!r}")
    d = loaders[src](dcfg, seed)

    cache.parent.mkdir(parents=True, exist_ok=True)
    save = dict(X=d["X"], y=d["y"], groups=d["groups"].astype(str),
                split=d["split"].astype(str), meta=np.array(meta))
    if d["gt_mask"] is not None:
        save["gt_mask"] = d["gt_mask"]
        save["gt_valid"] = d["gt_valid"]
    np.savez_compressed(cache, **save)
    print(f"[data] cached to {cache}")
    print_summary(d, src)
    return d
