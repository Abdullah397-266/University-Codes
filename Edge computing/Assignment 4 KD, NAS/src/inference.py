"""
Unified inference pipeline (PyTorch) for every trained model: the three baseline students, the three
distilled students, the NAS model and (optionally) the teacher. Pre-processing (resize to the model's input
resolution + dataset normalisation) is chosen automatically from each checkpoint.

    python src/inference.py --model all --test-index 7          # every available model on one test image
    python src/inference.py --model mobilenetv2_kd --image cat.jpg --topk 3
    python src/inference.py --model students --test-index 0     # all students, no teacher
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from common import banner, get_dataset
from zoo import MODEL_NAMES, STUDENT_NAMES, available, load_trained


def preprocess(img: Image.Image, info: dict) -> torch.Tensor:
    tf = T.Compose([T.Resize((info["resolution"], info["resolution"]), antialias=True), T.ToTensor(),
                    T.Normalize(info["mean"], info["std"])])
    return tf(img.convert("RGB")).unsqueeze(0)


def load_input(args, dataset: str):
    """Returns (PIL image, true label or None)."""
    if args.image:
        return Image.open(args.image), None
    ds = get_dataset(dataset, train=False, transform=None, synthetic=args.synthetic)
    img, label = ds[args.test_index]
    return img, label


@torch.no_grad()
def predict(model, x: torch.Tensor, topk: int):
    t0 = time.perf_counter()
    probs = F.softmax(model(x), dim=1)[0]
    ms = (time.perf_counter() - t0) * 1000
    p, idx = probs.topk(topk)
    return list(zip(idx.tolist(), p.tolist())), ms


def resolve(names: str) -> list[str]:
    if names == "all":
        return available(MODEL_NAMES)
    if names == "students":
        return available(STUDENT_NAMES)
    return names.split(",")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="students", help=f"all | students | comma list of {MODEL_NAMES}")
    p.add_argument("--image", help="path to an image file (otherwise a test-set image is used)")
    p.add_argument("--test-index", type=int, default=0)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    names = resolve(args.model)
    if not names:
        raise SystemExit("No trained checkpoints found in models/ - train the models first.")
    banner(f"PyTorch CPU inference | models: {', '.join(names)}")
    inputs = {}
    for name in names:
        model, info = load_trained(name)
        if info["dataset"] not in inputs:
            inputs[info["dataset"]] = load_input(args, info["dataset"])
        img, label = inputs[info["dataset"]]
        preds, ms = predict(model, preprocess(img, info), args.topk)
        cls = info["classes"]
        truth = f" | true: {cls[label]}" if label is not None else ""
        top = ", ".join(f"{cls[i]} {pr * 100:.1f}%" for i, pr in preds)
        ok = "" if label is None else ("  [correct]" if preds[0][0] == label else "  [wrong]")
        print(f"{name:28s} ({info['dataset']}, {info['resolution']}px) {ms:6.2f} ms | top-{args.topk}: {top}{truth}{ok}")


if __name__ == "__main__":
    main()
