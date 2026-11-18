"""
Registry of every trained model, used by the unified inference pipeline, ONNX export and benchmarking.

Names: teacher, <student>_baseline, <student>_kd (student in mobilenetv2 / squeezenet / efficientnet_lite0),
and nas_best. load_trained(name) returns the model in eval mode on the CPU plus its dataset metadata.
"""
from __future__ import annotations

from common import (DATASETS, NAS_BEST_CKPT, ONNX_DIR, STUDENTS, TEACHER, TEACHER_CKPT, load_checkpoint,
                    student_ckpt)
from models import build_model

MODEL_NAMES = ["teacher"] + [f"{s}_{m}" for s in STUDENTS for m in ("baseline", "kd")] + ["nas_best"]
STUDENT_NAMES = [n for n in MODEL_NAMES if n != "teacher"]


def ckpt_path(name: str):
    if name == "teacher":
        return TEACHER_CKPT
    if name == "nas_best":
        return NAS_BEST_CKPT
    arch, mode = name.rsplit("_", 1)
    return student_ckpt(arch, mode)


def onnx_path(name: str):
    return ONNX_DIR / f"{name}.onnx"


def available(names=None) -> list[str]:
    return [n for n in (names or MODEL_NAMES) if ckpt_path(n).exists()]


def load_trained(name: str):
    """Returns (model, info) with info = dataset, classes, resolution, mean, std, meta."""
    ck = load_checkpoint(ckpt_path(name))
    meta = ck["meta"]
    if name == "nas_best":
        from nas_generator import build_network
        model = build_network(meta["config"], 10)
        dataset, res = "cifar10", meta["config"]["resolution"]
    else:
        model = build_model(TEACHER if name == "teacher" else name.rsplit("_", 1)[0], 100)
        dataset, res = "cifar100", 32
    model.load_state_dict(ck["state_dict"])
    model.eval()
    d = DATASETS[dataset]
    return model, {"dataset": dataset, "classes": d["classes"], "resolution": res, "mean": d["mean"],
                   "std": d["std"], "meta": meta}
