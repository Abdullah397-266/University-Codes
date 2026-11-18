"""
Q1 step 1 - train the ResNet-50 teacher on CIFAR-100.

    python src/train_teacher.py                 # default: 60 epochs, SGD + cosine, AMP on GPU
    python src/train_teacher.py --epochs 100    # longer run = stronger teacher
Output: models/resnet50_teacher.pth, results/history_teacher.csv
"""
import argparse

from common import (TEACHER, rel, TEACHER_CKPT, RESULTS_DIR, add_common_args, banner, count_params, fit,
                    get_device, get_loaders, seed_everything)
from models import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, epochs=60)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    banner(f"Teacher: ResNet-50 on CIFAR-100 | device={device} | epochs={args.epochs}"
           + (" | SYNTHETIC DATA (smoke test)" if args.synthetic else ""))
    train_loader, test_loader = get_loaders("cifar100", args.batch_size, args.workers, train_subset=args.train_subset,
                                            test_subset=args.test_subset, synthetic=args.synthetic)
    print(f"train images: {len(train_loader.dataset)} | test images: {len(test_loader.dataset)}")

    model = build_model(TEACHER, 100)
    print(f"parameters: {count_params(model) / 1e6:.2f} M")
    res = fit(model, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay,
              ckpt_path=TEACHER_CKPT, ckpt_meta={"arch": TEACHER, "dataset": "cifar100", "mode": "teacher"},
              history_csv=RESULTS_DIR / "history_teacher.csv", tag="teacher")
    print(f"\nBest teacher test accuracy: {res['best_acc']:.2f}% (epoch {res['best_epoch']}) -> {rel(TEACHER_CKPT)}")


if __name__ == "__main__":
    main()
