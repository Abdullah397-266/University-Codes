"""
Deliverable 2: Post-Training Static Quantization (PTQ) of the FP32 ResNet-18.

Steps: load FP32 -> fuse Conv+BN+ReLU -> set qconfig/backend -> insert observers (prepare)
       -> calibrate on a subset of the TRAINING set -> convert to INT8 -> evaluate -> save.
INT8 inference runs on the CPU (fbgemm for x86, qnnpack for ARM).
"""
import argparse

import numpy as np
import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare
from torch.utils.data import Subset

from common import (MODEL_FILES, MODELS_DIR, default_backend, evaluate, export_torchscript, file_size_mb,
                    get_datasets, load_float_model, make_loader, save_quant_meta, seed_everything,
                    set_quant_backend)


def calibrate(model, loader):
    """Run representative data through the observed model so observers record activation ranges."""
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            model(x)
    print(f"Calibration finished ({len(loader.dataset)} images)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=default_backend(), choices=["fbgemm", "qnnpack", "x86"])
    ap.add_argument("--calib-samples", type=int, default=2000, help="size of the calibration subset")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    set_quant_backend(args.backend)

    model, ck = load_float_model(MODEL_FILES["fp32"], "cpu")       # PTQ always runs on CPU
    print(f"Loaded FP32 model (test acc {ck.get('test_acc', float('nan')):.2f}%)")

    # 1) fuse compatible layers (Conv+BN+ReLU, Conv+BN)
    model.eval()
    model.fuse_model(is_qat=False)
    # 2) configure the quantization backend / observers
    model.qconfig = get_default_qconfig(args.backend)
    print(f"qconfig: {model.qconfig}")
    # 3) insert observers
    prepare(model, inplace=True)
    # 4) calibration on a random subset of the training data (no augmentation)
    _, train_plain, test = get_datasets()
    idx = np.random.RandomState(args.seed).choice(len(train_plain), args.calib_samples, replace=False)
    calib_loader = make_loader(Subset(train_plain, idx.tolist()), args.batch_size, workers=args.workers)
    calibrate(model, calib_loader)
    # 5) convert to a real INT8 model
    int8 = convert(model, inplace=False)

    test_loader = make_loader(test, 256, workers=args.workers)
    acc = evaluate(int8, test_loader, "cpu")["acc"]
    print(f"PTQ INT8 test accuracy: {acc:.2f}%")

    # 6) save for deployment: TorchScript (executable) + state_dict (reference)
    how = export_torchscript(int8, MODEL_FILES["ptq_int8"])
    torch.save({"state_dict": int8.state_dict(), "backend": args.backend, "test_acc": acc},
               MODELS_DIR / "resnet18_ptq_int8_state.pth")
    save_quant_meta("ptq_int8", backend=args.backend, calib_samples=args.calib_samples,
                    torchscript=how, test_acc=acc)
    print(f"Saved -> {MODEL_FILES['ptq_int8']} ({file_size_mb(MODEL_FILES['ptq_int8']):.1f} MB)")


if __name__ == "__main__":
    main()
