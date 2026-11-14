"""
Shared utilities for Assignment 3 (quantization + pruning of ResNet-18 on CIFAR-10).

Contains: paths, data loading, the quantization-friendly CIFAR ResNet-18, a generic
training loop, evaluation, latency/size/MAC measurement helpers and checkpoint helpers.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.ao.quantization import (
    DeQuantStub,
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfig,
    QuantStub,
)
from torch.ao.quantization.fuse_modules import fuse_modules, fuse_modules_qat
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ONNX_DIR = MODELS_DIR / "onnx"
RESULTS_DIR = ROOT / "results"
for _d in (DATA_DIR, MODELS_DIR, ONNX_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The five deployable models. Keys are used everywhere (CLI, CSV columns, report).
MODEL_FILES = {
    "fp32": MODELS_DIR / "resnet18_fp32.pth",
    "ptq_int8": MODELS_DIR / "resnet18_ptq_int8.pt",              # TorchScript, INT8
    "qat_int8": MODELS_DIR / "resnet18_qat_int8.pt",              # TorchScript, INT8
    "unstructured_pruned": MODELS_DIR / "resnet18_unstructured_pruned.pth",
    "structured_pruned": MODELS_DIR / "resnet18_structured_pruned.pth",
}
ONNX_FILES = {k: ONNX_DIR / f"resnet18_{k}.onnx" for k in MODEL_FILES}
QUANT_META = MODELS_DIR / "quant_meta.json"

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------- misc helpers
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def default_backend() -> str:
    """fbgemm for x86 CPUs, qnnpack for ARM (e.g. Raspberry Pi / phones)."""
    return "fbgemm" if platform.machine().lower() in ("x86_64", "amd64") else "qnnpack"


def set_quant_backend(name: str) -> str:
    name = "onednn"
    supported = torch.backends.quantized.supported_engines
    if name not in supported:
        raise RuntimeError(f"Quantized backend '{name}' not supported here. Supported: {supported}")
    torch.backends.quantized.engine = name
    return name


def save_quant_meta(key: str, **info) -> None:
    meta = json.loads(QUANT_META.read_text()) if QUANT_META.exists() else {}
    meta[key] = info
    QUANT_META.write_text(json.dumps(meta, indent=2))


def load_quant_meta() -> dict:
    return json.loads(QUANT_META.read_text()) if QUANT_META.exists() else {}


def save_history(history: list, tag: str) -> Path:
    path = RESULTS_DIR / f"history_{tag}.csv"
    keys = sorted({k for row in history for k in row}, key=lambda k: (k != "epoch", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
    return path


# --------------------------------------------------------------------------- data
def get_datasets():
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10
    train = ds(str(DATA_DIR), train=True, transform=train_tf, download=True)
    train_plain = ds(str(DATA_DIR), train=True, transform=eval_tf, download=True)  # no augmentation (calibration)
    test = ds(str(DATA_DIR), train=False, transform=eval_tf, download=True)
    return train, train_plain, test


def make_loader(dataset, batch_size, shuffle=False, workers=None):
    workers = min(4, os.cpu_count() or 1) if workers is None else workers
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


# --------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    """ResNet basic block. `mid_c` (channels between conv1 and conv2) is configurable so that
    structured pruning can rebuild a physically smaller block."""

    def __init__(self, in_c: int, mid_c: int, out_c: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, mid_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
        else:
            self.shortcut = None
        # FloatFunctional makes the residual add quantizable in eager mode.
        self.add_relu = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            sc = self.shortcut(x)
        else:
            sc = x
        return self.add_relu.add_relu(out, sc)

    def fuse(self, is_qat: bool):
        fuse = fuse_modules_qat if is_qat else fuse_modules
        groups = [["conv1", "bn1", "relu1"], ["conv2", "bn2"]]
        if self.shortcut is not None:
            groups.append(["shortcut.0", "shortcut.1"])
        fuse(self, groups, inplace=True)


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18 (3x3 stem, no max-pool), ~11.2M parameters.

    mid_channels: optional list of 8 ints, the inner width of each BasicBlock.
                  None -> standard widths (64,64,128,128,256,256,512,512).
    """
    STAGES = ((64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2))  # (width, blocks, first stride)

    def __init__(self, num_classes: int = 10, mid_channels=None):
        super().__init__()
        widths = [w for w, n, _ in self.STAGES for _ in range(n)]
        self.mid_channels = list(mid_channels) if mid_channels is not None else widths
        assert len(self.mid_channels) == 8
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        blocks, in_c, idx = [], 64, 0
        for width, n, stride in self.STAGES:
            for j in range(n):
                blocks.append(BasicBlock(in_c, self.mid_channels[idx], width, stride if j == 0 else 1))
                in_c, idx = width, idx + 1
        self.layers = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = torch.flatten(self.pool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def fuse_model(self, is_qat: bool = False):
        """Fuse Conv+BN(+ReLU). PTQ: model must be in eval(). QAT: model must be in train()."""
        fuse = fuse_modules_qat if is_qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for blk in [m for m in self.modules() if isinstance(m, BasicBlock)]:
            blk.fuse(is_qat)


def qat_qconfig(backend: str) -> QConfig:
    """QAT qconfig using plain (non-fused) FakeQuantize so the model can also be exported to ONNX
    as QuantizeLinear/DequantizeLinear nodes. Per-tensor uint8 activations, per-channel int8 weights."""
    reduce = backend == "fbgemm"  # 7-bit activations avoid int16 overflow on x86 without VNNI
    act = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                 dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=reduce)
    wt = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-128, quant_max=127,
                                dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range=False,
                                ch_axis=0)
    return QConfig(activation=act, weight=wt)


# --------------------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, model: nn.Module, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), "mid_channels": getattr(model, "mid_channels", None), **meta},
               path)


def load_float_model(path: Path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = ResNet18(mid_channels=ck.get("mid_channels"))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def export_torchscript(model: nn.Module, path: Path) -> str:
    """Save a self-contained, executable TorchScript file. Returns 'script' or 'trace'."""
    model.eval()
    try:
        ts, how = torch.jit.script(model), "script"
    except Exception as e:  # fall back to tracing if scripting fails
        print(f"[warn] torch.jit.script failed ({e.__class__.__name__}); falling back to trace")
        ts, how = torch.jit.trace(model, torch.randn(1, 3, 32, 32)), "trace"
    torch.jit.save(ts, str(path))
    return how


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    preds, confs, labels = [], [], []
    for x, y in loader:
        prob = torch.softmax(model(x.to(device)).float(), dim=1)
        conf, pred = prob.max(1)
        preds.append(pred.cpu()); confs.append(conf.cpu()); labels.append(y)
    preds, confs, labels = (torch.cat(t).numpy() for t in (preds, confs, labels))
    return {"acc": float((preds == labels).mean() * 100), "preds": preds, "confs": confs, "labels": labels}


def train_epochs(model, train_loader, test_loader, device, epochs, lr, tag="train",
                 weight_decay=5e-4, amp=False, on_epoch_start=None):
    """SGD + cosine LR (per step). Returns a list of per-epoch dicts."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(train_loader)))
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        if on_epoch_start is not None:
            on_epoch_start(epoch, model)
        model.train()
        t0, loss_sum, correct, n = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
        test_acc = evaluate(model, test_loader, device)["acc"]
        row = {"epoch": epoch + 1, "train_loss": loss_sum / n, "train_acc": 100 * correct / n,
               "test_acc": test_acc, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        print(f"[{tag}] epoch {epoch + 1:3d}/{epochs}  loss {row['train_loss']:.4f}  "
              f"train {row['train_acc']:.2f}%  test {test_acc:.2f}%  ({row['seconds']:.0f}s)")
    return history


# --------------------------------------------------------------------------- measurements
def measure_latency(model, device, batch_size=1, warmup=20, iters=100) -> dict:
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        for _ in range(iters):
            t = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t) * 1000)
    a = np.array(times)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "img_per_s": float(batch_size * 1000 / a.mean())}


def count_params(model) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int(torch.count_nonzero(p)) for p in model.parameters())
    return total, nonzero


@torch.no_grad()
def count_macs(model, nonzero_only=False) -> int:
    """Multiply-accumulates for one 32x32 image (conv + linear layers)."""
    macs, hooks = [0], []
    dev = next(model.parameters()).device

    def w_count(w):
        return int(torch.count_nonzero(w)) if nonzero_only else w.numel()

    def conv_hook(m, i, o):
        macs[0] += w_count(m.weight) * o.shape[2] * o.shape[3]

    def lin_hook(m, i, o):
        macs[0] += w_count(m.weight)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    model(torch.zeros(1, 3, 32, 32, device=dev))
    for h in hooks:
        h.remove()
    return macs[0]


def file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / 1e6


def gzip_size_mb(path: Path) -> float:
    return len(gzip.compress(Path(path).read_bytes(), compresslevel=9)) / 1e6
