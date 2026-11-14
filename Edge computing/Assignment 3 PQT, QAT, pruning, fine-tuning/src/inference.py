"""
Deliverable 6: inference pipeline for all five models
(fp32, ptq_int8, qat_int8, unstructured_pruned, structured_pruned).

Examples
  python src/inference.py --model all --test-index 7           # compare every model on test image #7
  python src/inference.py --model qat_int8 --image cat.jpg     # classify your own image
  python src/inference.py --model structured_pruned --image cat.jpg --device cuda
INT8 models are CPU-only (fbgemm/qnnpack kernels) and are always run on the CPU.
"""
import argparse
import time

import torch
import torchvision.transforms as T
from PIL import Image

from common import (CLASSES, MEAN, MODEL_FILES, default_backend, get_datasets, load_float_model, load_quant_meta,
                    set_quant_backend)

INT8_MODELS = {"ptq_int8", "qat_int8"}
PREPROCESS = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(MEAN, (0.2470, 0.2435, 0.2616))])


def load_model(name: str, device="cpu"):
    """Return an eval-mode model ready for inference. INT8 models are TorchScript files on CPU."""
    path = MODEL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the pipeline first (see README).")
    if name in INT8_MODELS:
        meta = load_quant_meta().get(name, {})
        set_quant_backend(meta.get("backend", default_backend()))   # packed weights are backend-specific
        return torch.jit.load(str(path), map_location="cpu").eval()
    return load_float_model(path, device)[0]


def predict(model, x: torch.Tensor, device="cpu", topk=3):
    with torch.no_grad():
        prob = torch.softmax(model(x.to(device)).float(), dim=1)[0]
    conf, idx = prob.topk(topk)
    return [(CLASSES[i], float(c)) for c, i in zip(conf, idx)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="all", choices=["all"] + list(MODEL_FILES))
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", help="path to any RGB image (resized to 32x32)")
    src.add_argument("--test-index", type=int, default=0, help="index of a CIFAR-10 test image")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    if args.image:
        x, truth = PREPROCESS(Image.open(args.image).convert("RGB")).unsqueeze(0), None
    else:
        _, _, test = get_datasets()
        img, label = test[args.test_index]
        x, truth = img.unsqueeze(0), CLASSES[label]
        print(f"CIFAR-10 test image #{args.test_index}, ground truth: {truth}")

    names = list(MODEL_FILES) if args.model == "all" else [args.model]
    for name in names:
        dev = "cpu" if name in INT8_MODELS or not torch.cuda.is_available() else args.device
        try:
            model = load_model(name, dev)
        except FileNotFoundError as e:
            print(f"{name:20s} skipped: {e}")
            continue
        t = time.perf_counter()
        top = predict(model, x, dev)
        ms = (time.perf_counter() - t) * 1000
        print(f"{name:20s} [{dev}] {ms:7.1f} ms  " + " | ".join(f"{c} {p:.2%}" for c, p in top))


if __name__ == "__main__":
    main()
