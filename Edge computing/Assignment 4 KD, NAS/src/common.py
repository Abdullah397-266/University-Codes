"""
Shared utilities for Assignment 4 (Knowledge Distillation on CIFAR-100 + Hardware-aware NAS on CIFAR-10).

Contains: paths, dataset loaders, a generic training loop, evaluation, CPU measurement helpers
(latency, size, peak memory, parameters, MACs), checkpoint helpers and common CLI arguments.
Every other script imports from here so that all models are trained/measured the same way.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
REPORT_DIR = ROOT / "report"
os.environ.setdefault("KINETO_LOG_LEVEL", "5")

# CIFAR is downloaded over HTTPS. Some Windows Python installs lack up-to-date root certificates and fail with
# "SSL: CERTIFICATE_VERIFY_FAILED"; point Python at certifi's CA bundle instead (if certifi is installed).
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass
for _d in (DATA_DIR, MODELS_DIR, ONNX_DIR, RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

STUDENTS = ("mobilenetv2", "squeezenet", "efficientnet_lite0")
TEACHER = "resnet50"

TEACHER_CKPT = MODELS_DIR / "resnet50_teacher.pth"
NAS_BEST_CONFIG = MODELS_DIR / "nas_best_config.json"
NAS_BEST_CKPT = MODELS_DIR / "nas_best.pth"


def rel(path) -> str:
    """Path relative to the project root, for readable console output."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def student_ckpt(arch: str, mode: str) -> Path:
    """mode is 'baseline' (trained alone) or 'kd' (distilled from the teacher)."""
    return MODELS_DIR / f"{arch}_{mode}.pth"


# --------------------------------------------------------------------------- dataset metadata
CIFAR100_CLASSES = (
    "apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy bridge bus butterfly "
    "camel can castle caterpillar cattle chair chimpanzee clock cloud cockroach couch crab crocodile cup "
    "dinosaur dolphin elephant flatfish forest fox girl hamster house kangaroo keyboard lamp lawn_mower "
    "leopard lion lizard lobster man maple_tree motorcycle mountain mouse mushroom oak_tree orange orchid "
    "otter palm_tree pear pickup_truck pine_tree plain plate poppy porcupine possum rabbit raccoon ray road "
    "rocket rose sea seal shark shrew skunk skyscraper snail snake spider squirrel streetcar sunflower "
    "sweet_pepper table tank telephone television tiger tractor train trout tulip turtle wardrobe whale "
    "willow_tree wolf woman worm"
).split()
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
assert len(CIFAR100_CLASSES) == 100

DATASETS = {
    "cifar100": dict(cls=torchvision.datasets.CIFAR100, num_classes=100, classes=CIFAR100_CLASSES,
                     mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    "cifar10": dict(cls=torchvision.datasets.CIFAR10, num_classes=10, classes=CIFAR10_CLASSES,
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
}


# --------------------------------------------------------------------------- misc helpers
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def banner(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n {title}\n{line}", flush=True)


def default_workers() -> int:
    # Windows spawns worker processes (slow start); 2 is a safe default there.
    return 2 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)


def add_common_args(p: argparse.ArgumentParser, epochs: int, batch_size: int = 128, lr: float = 0.1) -> None:
    """CLI options shared by every training script."""
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-subset", type=int, default=0, help="use only N training images (0 = all)")
    p.add_argument("--test-subset", type=int, default=0, help="evaluate on only N test images (0 = all)")
    p.add_argument("--synthetic", action="store_true",
                   help="use random FakeData with CIFAR shapes (pipeline smoke test without downloading)")


# --------------------------------------------------------------------------- data
def build_transforms(dataset: str, train: bool, resolution: int = 32) -> T.Compose:
    meta = DATASETS[dataset]
    ops = []
    if train:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if resolution != 32:
        ops.append(T.Resize((resolution, resolution), antialias=True))
    ops += [T.ToTensor(), T.Normalize(meta["mean"], meta["std"])]
    return T.Compose(ops)


def get_dataset(dataset: str, train: bool, transform=None, synthetic: bool = False):
    meta = DATASETS[dataset]
    if synthetic:
        size = 50000 if train else 10000
        return torchvision.datasets.FakeData(size=size, image_size=(3, 32, 32), num_classes=meta["num_classes"],
                                             transform=transform, random_offset=0 if train else 1)
    return meta["cls"](root=str(DATA_DIR), train=train, download=True, transform=transform)


def _subset(ds, n: int, seed: int = 0):
    if not n or n >= len(ds):
        return ds
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n].tolist()
    return Subset(ds, idx)


def get_loaders(dataset: str, batch_size: int = 128, workers: int = 2, resolution: int = 32,
                train_subset: int = 0, test_subset: int = 0, synthetic: bool = False,
                val_split: int = 0, seed: int = 0):
    """
    Returns (train_loader, eval_loader).
    * val_split == 0: eval_loader is the official test set (used for KD experiments).
    * val_split > 0 : the last `val_split` images of a shuffled training set become a validation set and the
      test set is NOT touched (used by NAS so that architecture selection never sees test data).
    """
    train_tf = build_transforms(dataset, True, resolution)
    eval_tf = build_transforms(dataset, False, resolution)
    pin = torch.cuda.is_available()
    kw = dict(num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)

    if val_split > 0:
        full_train = get_dataset(dataset, True, train_tf, synthetic)
        full_eval = get_dataset(dataset, True, eval_tf, synthetic)       # same images, no augmentation
        perm = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(seed)).tolist()
        val_idx, train_idx = perm[:val_split], perm[val_split:]
        if train_subset:
            train_idx = train_idx[:train_subset]
        train_ds, eval_ds = Subset(full_train, train_idx), Subset(full_eval, val_idx)
    else:
        train_ds = _subset(get_dataset(dataset, True, train_tf, synthetic), train_subset, seed)
        eval_ds = _subset(get_dataset(dataset, False, eval_tf, synthetic), test_subset, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, **kw)
    eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False, **kw)
    return train_loader, eval_loader


# --------------------------------------------------------------------------- training / evaluation
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    """Returns (top-1 accuracy %, mean cross-entropy)."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1), loss_sum / max(total, 1)


def make_optimizer(params, lr: float, weight_decay: float, epochs: int, steps_per_epoch: int):
    """SGD + Nesterov momentum with a one-epoch linear warm-up followed by cosine decay (per step)."""
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    total = max(1, epochs * steps_per_epoch)
    warm = min(steps_per_epoch, total // 2)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def fit(model: nn.Module, train_loader, eval_loader, device, epochs: int, lr: float, weight_decay: float,
        loss_fn=None, extra_params=None, ckpt_path: Path | None = None, ckpt_meta: dict | None = None,
        history_csv: Path | None = None, tag: str = "model", verbose: bool = True) -> dict:
    """
    Generic training loop used by teacher, baseline students, KD students and NAS candidates.

    loss_fn(x, y) -> (loss, logits, parts_dict). If None, plain cross-entropy on model(x) is used.
    extra_params: additional trainable parameters (e.g. KD feature adapters).
    The checkpoint with the best evaluation accuracy is saved to ckpt_path.
    """
    model.to(device)
    params = list(model.parameters()) + (list(extra_params) if extra_params else [])
    opt, sched = make_optimizer(params, lr, weight_decay, epochs, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    if loss_fn is None:
        def loss_fn(x, y):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            return loss, logits, {"ce": loss.item()}

    history, best_acc, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        seen, correct, loss_sum, parts_sum = 0, 0, 0.0, {}
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss, logits, parts = loss_fn(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = y.numel()
            seen += n
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == y).sum().item()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v * n
        val_acc, val_loss = evaluate(model, eval_loader, device)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / seen,
               "train_acc": 100.0 * correct / seen, "eval_loss": val_loss, "eval_acc": val_acc,
               "epoch_time_s": time.time() - t0}
        row.update({f"loss_{k}": v / seen for k, v in parts_sum.items()})
        history.append(row)
        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch
            if ckpt_path is not None:
                save_checkpoint(model, ckpt_path, {**(ckpt_meta or {}), "best_acc": val_acc, "epoch": epoch})
        if verbose:
            extra = " ".join(f"{k}={v / seen:.3f}" for k, v in parts_sum.items()) if len(parts_sum) > 1 else ""
            print(f"[{tag}] epoch {epoch:3d}/{epochs} | loss {row['train_loss']:.4f} | train {row['train_acc']:6.2f}% "
                  f"| eval {val_acc:6.2f}% {'*' if improved else ' '} | {row['epoch_time_s']:.1f}s {extra}", flush=True)
    if history_csv is not None:
        write_csv(history_csv, history)
    return {"best_acc": best_acc, "best_epoch": best_epoch, "history": history}


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(model: nn.Module, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)


def load_checkpoint(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run the corresponding training script first.")
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- measurement (CPU)
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Serialized FP32 state_dict size (what is actually written to disk)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2**20


@torch.no_grad()
def count_macs(model: nn.Module, input_shape=(1, 3, 32, 32)) -> int:
    """Multiply-accumulate count of Conv2d and Linear layers (measured with forward hooks)."""
    macs = []

    def conv_hook(m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
        macs.append(out.numel() * k)

    def lin_hook(m, inp, out):
        macs.append(out.numel() * m.in_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return int(sum(macs))


@torch.no_grad()
def measure_latency_ms(model: nn.Module, input_shape=(1, 3, 32, 32), runs: int = 100, warmup: int = 20,
                       threads: int | None = None) -> dict:
    """Real wall-clock CPU latency of one forward pass. Returns mean / median / p95 in milliseconds."""
    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    model = model.cpu().eval()
    x = torch.randn(input_shape)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(old)
    times.sort()
    return {"mean_ms": statistics.mean(times), "median_ms": statistics.median(times),
            "p95_ms": times[int(0.95 * (len(times) - 1))]}


@contextlib.contextmanager
def _silence_native_stderr():
    """The profiler's C++ backend (Kineto) prints status lines straight to file descriptor 2; hide them."""
    sys.stderr.flush()
    fd = sys.stderr.fileno()
    saved, devnull = os.dup(fd), os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    try:
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(devnull)


@torch.no_grad()
def measure_peak_memory_mb(model: nn.Module, input_shape=(1, 3, 32, 32)) -> float | None:
    """
    Peak CPU memory held by tensors during one inference = weights + input + the highest point of the
    activation/workspace allocations recorded by the PyTorch profiler. Allocations (attributed to the operator
    that made them) and frees ("[memory]" events) are replayed in time order and the running maximum is taken.
    Returns None if the profiler does not support memory tracking on this build.
    """
    model = model.cpu().eval()
    x = torch.randn(input_shape)
    model(x)  # warm-up so that one-time allocations are excluded
    try:
        from torch.profiler import ProfilerActivity, profile
        with _silence_native_stderr():
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
                model(x)
        timeline = []
        for e in prof.events():
            if e.name == "[memory]":
                timeline.append((e.time_range.start, e.cpu_memory_usage))      # frees (negative)
            elif e.self_cpu_memory_usage > 0:
                timeline.append((e.time_range.start, e.self_cpu_memory_usage))  # allocations by an operator
        if not timeline:
            return None
        cur = peak = 0
        for _, delta in sorted(timeline, key=lambda t: t[0]):
            cur += delta
            peak = max(peak, cur)
    except Exception:  # profiler unavailable / changed API
        return None
    weights = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    return (weights + peak + x.numel() * x.element_size()) / 2**20


def profile_model(model: nn.Module, input_shape=(1, 3, 32, 32), runs: int = 100, threads: int | None = 1) -> dict:
    lat = measure_latency_ms(model, input_shape, runs=runs, threads=threads)
    return {"params": count_params(model), "macs": count_macs(model, input_shape),
            "size_mb": model_size_mb(model), "latency_ms": lat["median_ms"], "latency_p95_ms": lat["p95_ms"],
            "peak_mem_mb": measure_peak_memory_mb(model, input_shape)}


# --------------------------------------------------------------------------- csv / json
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)
