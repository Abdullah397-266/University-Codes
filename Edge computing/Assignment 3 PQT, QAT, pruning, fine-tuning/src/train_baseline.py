"""Deliverable 1: train the FP32 ResNet-18 baseline on CIFAR-10 and save the checkpoint."""
import argparse

from common import (MODEL_FILES, evaluate, get_datasets, get_device, make_loader, save_checkpoint,
                    save_history, seed_everything, train_epochs, ResNet18)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")
    train, _, test = get_datasets()
    train_loader = make_loader(train, args.batch_size, shuffle=True, workers=args.workers)
    test_loader = make_loader(test, 256, workers=args.workers)

    model = ResNet18().to(device)
    history = train_epochs(model, train_loader, test_loader, device, args.epochs, args.lr,
                           tag="baseline", amp=not args.no_amp)
    acc = evaluate(model, test_loader, device)["acc"]
    save_checkpoint(MODEL_FILES["fp32"], model.cpu(), test_acc=acc, epochs=args.epochs, seed=args.seed)
    save_history(history, "baseline")
    print(f"Saved FP32 checkpoint -> {MODEL_FILES['fp32']}  (test acc {acc:.2f}%)")


if __name__ == "__main__":
    main()
