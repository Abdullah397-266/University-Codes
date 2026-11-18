"""
Q1 step 5 - baseline pipeline: train the three student networks on CIFAR-100 WITHOUT distillation
(plain cross-entropy on the hard labels). Same data, epochs and optimiser as the KD run, so the two are
directly comparable.

    python src/train_baseline.py --arch all
    python src/train_baseline.py --arch squeezenet --epochs 60
Output: models/<arch>_baseline.pth, results/history_<arch>_baseline.csv
"""
import argparse

from common import (STUDENTS, rel, RESULTS_DIR, add_common_args, banner, count_params, fit, get_device, get_loaders,
                    seed_everything, student_ckpt, write_csv)
from models import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="all", choices=("all",) + STUDENTS)
    add_common_args(p, epochs=40, lr=0.05)
    args = p.parse_args()

    device = get_device(args.device)
    archs = STUDENTS if args.arch == "all" else (args.arch,)
    train_loader, test_loader = get_loaders("cifar100", args.batch_size, args.workers, train_subset=args.train_subset,
                                            test_subset=args.test_subset, synthetic=args.synthetic)
    summary = []
    for arch in archs:
        seed_everything(args.seed)
        banner(f"Baseline student (no KD): {arch} | device={device} | epochs={args.epochs}"
               + (" | SYNTHETIC DATA (smoke test)" if args.synthetic else ""))
        model = build_model(arch, 100)
        print(f"parameters: {count_params(model) / 1e6:.2f} M")
        res = fit(model, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay,
                  ckpt_path=student_ckpt(arch, "baseline"),
                  ckpt_meta={"arch": arch, "dataset": "cifar100", "mode": "baseline"},
                  history_csv=RESULTS_DIR / f"history_{arch}_baseline.csv", tag=f"{arch}/baseline")
        summary.append({"arch": arch, "mode": "baseline", "best_test_acc": res["best_acc"],
                        "best_epoch": res["best_epoch"], "epochs": args.epochs})
        print(f"-> best {res['best_acc']:.2f}% saved to {rel(student_ckpt(arch, 'baseline'))}")
    write_csv(RESULTS_DIR / f"summary_baseline{'' if args.arch == 'all' else '_' + args.arch}.csv", summary)


if __name__ == "__main__":
    main()
