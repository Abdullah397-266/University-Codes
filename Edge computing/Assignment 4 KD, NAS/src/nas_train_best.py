"""
Q2 step 7 - train the best discovered architecture FROM SCRATCH on the full CIFAR-10 training set
(new random weights; nothing is reused from the short search-time training) and evaluate on the test set.

    python src/nas_train_best.py                       # reads models/nas_best_config.json
    python src/nas_train_best.py --epochs 100
Output: models/nas_best.pth (config stored inside), results/history_nas_best.csv, results/nas_best_final.json
"""
import argparse

from common import (NAS_BEST_CKPT, rel, NAS_BEST_CONFIG, RESULTS_DIR, add_common_args, banner, fit, get_device,
                    get_loaders, profile_model, read_json, seed_everything, write_json)
from nas_generator import build_network
from nas_search_space import ArchConfig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, epochs=60, lr=0.05)
    p.add_argument("--config", default=str(NAS_BEST_CONFIG))
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    cfg = ArchConfig.from_dict(read_json(args.config))
    banner(f"Training best NAS architecture from scratch | device={device} | epochs={args.epochs}"
           + (" | SYNTHETIC DATA (smoke test)" if args.synthetic else ""))
    print(cfg.summary())
    train_loader, test_loader = get_loaders("cifar10", args.batch_size, args.workers, resolution=cfg.resolution,
                                            train_subset=args.train_subset, test_subset=args.test_subset,
                                            synthetic=args.synthetic)
    net = build_network(cfg, 10)
    res = fit(net, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay,
              ckpt_path=NAS_BEST_CKPT, ckpt_meta={"arch": "nas_best", "dataset": "cifar10", "config": cfg.to_dict()},
              history_csv=RESULTS_DIR / "history_nas_best.csv", tag="nas_best")
    prof = profile_model(build_network(cfg, 10), (1, 3, cfg.resolution, cfg.resolution), runs=200, threads=1)
    write_json(RESULTS_DIR / "nas_best_final.json", {"test_acc": res["best_acc"], "best_epoch": res["best_epoch"],
                                                     "epochs": args.epochs, **prof, "config": cfg.to_dict()})
    print(f"\nBest test accuracy {res['best_acc']:.2f}% (epoch {res['best_epoch']}) | {prof['params'] / 1e3:.0f} K params"
          f" | {prof['size_mb']:.3f} MB | {prof['latency_ms']:.2f} ms (1 thread) -> {rel(NAS_BEST_CKPT)}")


if __name__ == "__main__":
    main()
