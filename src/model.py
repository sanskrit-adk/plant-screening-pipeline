"""Small CNN classifier, training loop with early stopping, temperature scaling."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .utils import get_device


# ----------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------
def _block(cin: int, cout: int) -> nn.Sequential:
    """Conv3x3 (no bias, BN follows) -> BatchNorm -> ReLU -> MaxPool2."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """4 conv blocks (32-64-128-128) -> global average pool -> dropout -> 2 logits."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def to_tensor(X: np.ndarray) -> torch.Tensor:
    """uint8 NHWC -> float NCHW normalised to [-1, 1]: (x/255 - 0.5) / 0.5."""
    t = torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).float()
    return (t / 255.0 - 0.5) / 0.5


class ArrayDataset(Dataset):
    """Numpy images in memory; optional flip/rot90 augmentation for training.

    Leaves have no canonical orientation, so the 8 dihedral transforms are label-preserving.
    Random draws use torch's global RNG (seeded) with num_workers=0 -> reproducible.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = to_tensor(X)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])     # horizontal
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[1])     # vertical
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                x = torch.rot90(x, k, dims=[1, 2])
        return x, self.y[i]


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def class_weights(y: np.ndarray) -> torch.Tensor:
    """Balanced weights N / (2 * count_c) so both classes contribute equally to the loss."""
    counts = np.bincount(y, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    return torch.tensor(len(y) / (2.0 * counts), dtype=torch.float32)


@torch.no_grad()
def _eval_loss(model, loader, crit, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        tot += crit(model(xb), yb).item() * len(yb)
        n += len(yb)
    return tot / max(n, 1)


def train_model(Xtr, ytr, Xva, yva, cfg: dict, seed: int, verbose: bool = True) -> SmallCNN:
    """Adam + class-weighted cross-entropy, early stopping on val loss, best state restored."""
    tc = cfg["train"]
    device = get_device(tc.get("device", "auto"))
    torch.manual_seed(seed)
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["lr"]))
    crit = nn.CrossEntropyLoss(weight=class_weights(np.asarray(ytr)).to(device))

    gen = torch.Generator().manual_seed(seed)  # seeded shuffling
    tr_loader = DataLoader(ArrayDataset(Xtr, ytr, augment=True), batch_size=int(tc["batch_size"]),
                           shuffle=True, generator=gen, num_workers=0)
    va_loader = DataLoader(ArrayDataset(Xva, yva), batch_size=256, shuffle=False, num_workers=0)

    best_loss, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for ep in range(1, int(tc["epochs"]) + 1):
        t0 = time.time()
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        vl = _eval_loss(model, va_loader, crit, device)
        improved = vl < best_loss - 1e-6
        if improved:
            best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose:
            print(f"    epoch {ep:2d}  val_loss {vl:.4f}{' *' if improved else '  '}  "
                  f"({time.time() - t0:.1f}s)")
        if bad >= int(tc["patience"]):
            if verbose:
                print(f"    early stop (best val_loss {best_loss:.4f})")
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Raw logits [N, 2] (numpy) for uint8 NHWC images."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch_size):
        out.append(model(to_tensor(X[i:i + batch_size]).to(device)).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 2), np.float32)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: minimise validation NLL of softmax(logits / T) over log T (LBFGS)."""
    lg = torch.as_tensor(logits, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_t.exp().item())
    if not np.isfinite(T) or T <= 0:
        return 1.0
    # On perfectly separable validation data NLL keeps falling as T -> 0, which would
    # produce hard 0/1 probabilities; clamp to a sane range.
    return float(np.clip(T, 0.05, 20.0))


def probs_pos(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Calibrated P(diseased) = softmax(logits / T)[:, 1]."""
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


@torch.no_grad()
def cnn_latency_ms(model: nn.Module, img_size: int, n: int = 50, warmup: int = 5) -> float:
    """Batch-1 inference latency in ms (5 warm-up runs, mean of 50)."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3 / n
