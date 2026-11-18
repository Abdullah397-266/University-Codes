"""
Q1 steps 3-4 - train the three students WITH knowledge distillation from the ResNet-50 teacher.

Loss = (1-alpha)*CE(hard labels) + alpha*T^2*KL(soft targets at temperature T) + beta*MSE(feature hint)
(see kd.py). Defaults: alpha=0.7, T=4, beta=1.0 (feature distillation on). Use --beta 0 for logit-only KD.

    python src/train_kd.py --arch all
    python src/train_kd.py --arch mobilenetv2 --temperature 6 --alpha 0.9 --beta 0
Output: models/<arch>_kd.pth, results/history_<arch>_kd.csv
"""
import argparse

from common import (STUDENTS, rel, TEACHER, TEACHER_CKPT, RESULTS_DIR, add_common_args, banner, count_params, evaluate,
                    fit, get_device, get_loaders, load_checkpoint, seed_everything, student_ckpt, write_csv)
from kd import DistillationLoss, Distiller
from models import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="all", choices=("all",) + STUDENTS)
    add_common_args(p, epochs=40, lr=0.05)
    p.add_argument("--alpha", type=float, default=0.7, help="weight of the soft-target term (1-alpha for hard labels)")
    p.add_argument("--temperature", type=float, default=4.0, help="softmax temperature T")
    p.add_argument("--beta", type=float, default=1.0, help="weight of the feature-distillation term (0 = off)")
    args = p.parse_args()

    device = get_device(args.device)
    archs = STUDENTS if args.arch == "all" else (args.arch,)
    train_loader, test_loader = get_loaders("cifar100", args.batch_size, args.workers, train_subset=args.train_subset,
                                            test_subset=args.test_subset, synthetic=args.synthetic)

    teacher = build_model(TEACHER, 100)
    teacher.load_state_dict(load_checkpoint(TEACHER_CKPT)["state_dict"])
    teacher.to(device)
    t_acc, _ = evaluate(teacher, test_loader, device)
    print(f"Loaded teacher {TEACHER_CKPT.name}: test accuracy {t_acc:.2f}%")

    summary = []
    for arch in archs:
        seed_everything(args.seed)
        banner(f"KD student: {arch} <- ResNet-50 | alpha={args.alpha} T={args.temperature} beta={args.beta} "
               f"| epochs={args.epochs}" + (" | SYNTHETIC DATA (smoke test)" if args.synthetic else ""))
        student = build_model(arch, 100)
        criterion = DistillationLoss(args.alpha, args.temperature, args.beta,
                                     student_ch=student.feature_channels, teacher_ch=teacher.feature_channels)
        distiller = Distiller(student, teacher, criterion).to(device)
        print(f"student parameters: {count_params(student) / 1e6:.2f} M | feature adapter: "
              f"{sum(q.numel() for q in distiller.trainable_extra()) / 1e6:.2f} M (training only)")
        res = fit(student, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay,
                  loss_fn=distiller.step, extra_params=distiller.trainable_extra(),
                  ckpt_path=student_ckpt(arch, "kd"),
                  ckpt_meta={"arch": arch, "dataset": "cifar100", "mode": "kd", "alpha": args.alpha,
                             "temperature": args.temperature, "beta": args.beta, "teacher_acc": t_acc},
                  history_csv=RESULTS_DIR / f"history_{arch}_kd.csv", tag=f"{arch}/kd")
        summary.append({"arch": arch, "mode": "kd", "best_test_acc": res["best_acc"], "best_epoch": res["best_epoch"],
                        "epochs": args.epochs, "alpha": args.alpha, "temperature": args.temperature,
                        "beta": args.beta, "teacher_acc": t_acc})
        print(f"-> best {res['best_acc']:.2f}% saved to {rel(student_ckpt(arch, 'kd'))}")
    write_csv(RESULTS_DIR / f"summary_kd{'' if args.arch == 'all' else '_' + args.arch}.csv", summary)


if __name__ == "__main__":
    main()
