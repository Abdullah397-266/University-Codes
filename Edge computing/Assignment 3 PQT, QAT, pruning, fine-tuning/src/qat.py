"""
Deliverable 3: Quantization-Aware Training (QAT).

Steps: load pretrained FP32 -> fuse (train mode) -> attach QAT qconfig -> prepare_qat (inserts
FakeQuantize modules) -> fine-tune with fake quantization on GPU -> freeze observers/BN stats near
the end -> convert to INT8 on CPU -> evaluate -> export (TorchScript INT8 + fake-quant state for ONNX).
"""
import argparse

import torch
import torch.ao.nn.intrinsic.qat as nniqat
from torch.ao.quantization import convert, disable_observer, prepare_qat

from common import (MODEL_FILES, MODELS_DIR, default_backend, evaluate, export_torchscript, file_size_mb,
                    get_datasets, get_device, load_float_model, make_loader, qat_qconfig, save_history,
                    save_quant_meta, seed_everything, set_quant_backend, train_epochs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default=default_backend(), choices=["fbgemm", "qnnpack", "x86"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--freeze-observer-epoch", type=int, default=None, help="epoch index to stop observers")
    ap.add_argument("--freeze-bn-epoch", type=int, default=None, help="epoch index to freeze BN statistics")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    set_quant_backend(args.backend)
    device = get_device()
    obs_epoch = args.freeze_observer_epoch if args.freeze_observer_epoch is not None else max(1, args.epochs - 3)
    bn_epoch = args.freeze_bn_epoch if args.freeze_bn_epoch is not None else max(1, args.epochs - 2)

    # 1) initialise from the pretrained FP32 model
    model, ck = load_float_model(MODEL_FILES["fp32"], "cpu")
    print(f"Loaded FP32 model (test acc {ck.get('test_acc', float('nan')):.2f}%)")
    # 2) fuse in TRAIN mode, attach QAT qconfig and insert FakeQuantize modules
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = qat_qconfig(args.backend)
    prepare_qat(model, inplace=True)
    model.to(device)

    train, _, test = get_datasets()
    train_loader = make_loader(train, args.batch_size, shuffle=True, workers=args.workers)
    test_loader = make_loader(test, 256, workers=args.workers)

    def schedule(epoch, m):
        if epoch >= obs_epoch:
            m.apply(disable_observer)          # freeze quantization ranges
        if epoch >= bn_epoch:
            m.apply(nniqat.freeze_bn_stats)    # freeze BN running statistics

    # 3) QAT fine-tuning (no AMP: fake-quant ops are kept in FP32)
    history = train_epochs(model, train_loader, test_loader, device, args.epochs, args.lr, tag="qat",
                           amp=False, on_epoch_start=schedule)
    model.apply(disable_observer)
    fq_acc = evaluate(model, test_loader, device)["acc"]
    print(f"Fake-quantized (simulated) accuracy: {fq_acc:.2f}%")

    # keep the fake-quant model for ONNX (QDQ) export
    model.cpu().eval()
    torch.save({"state_dict": model.state_dict(), "backend": args.backend, "test_acc": fq_acc},
               MODELS_DIR / "resnet18_qat_fakequant.pth")

    # 4) convert to a real INT8 model (CPU) and evaluate
    int8 = convert(model, inplace=False)
    cpu_loader = make_loader(test, 256, workers=args.workers)
    acc = evaluate(int8, cpu_loader, "cpu")["acc"]
    print(f"QAT INT8 test accuracy: {acc:.2f}%")

    # 5) export for inference
    how = export_torchscript(int8, MODEL_FILES["qat_int8"])
    torch.save({"state_dict": int8.state_dict(), "backend": args.backend, "test_acc": acc},
               MODELS_DIR / "resnet18_qat_int8_state.pth")
    save_quant_meta("qat_int8", backend=args.backend, epochs=args.epochs, torchscript=how,
                    test_acc=acc, fake_quant_acc=fq_acc)
    save_history(history, "qat")
    print(f"Saved -> {MODEL_FILES['qat_int8']} ({file_size_mb(MODEL_FILES['qat_int8']):.1f} MB)")


if __name__ == "__main__":
    main()
