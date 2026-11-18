"""
Automated benchmarking of every trained model (PyTorch):
test accuracy, parameters, MACs, FP32 storage size, CPU latency (1 thread and all threads) and peak
inference memory. Also produces the baseline-vs-distillation comparison table.

    python src/benchmark.py
    python src/benchmark.py --runs 300 --test-subset 2000
Output: results/benchmark_results.csv, results/kd_comparison.csv
"""
import argparse

import torch

from common import (RESULTS_DIR, STUDENTS, banner, default_workers, evaluate, get_device, get_loaders,
                    profile_model, measure_latency_ms, write_csv)
from zoo import MODEL_NAMES, available, load_trained


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--test-subset", type=int, default=0)
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--device", default="auto", help="device used for the accuracy pass")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    device = get_device(args.device)
    names = available(MODEL_NAMES)
    if not names:
        raise SystemExit("No trained checkpoints found in models/.")
    banner(f"Benchmark | accuracy on {device} | latency on CPU ({torch.get_num_threads()} threads available)")
    loaders = {}
    rows = []
    for name in names:
        model, info = load_trained(name)
        key = (info["dataset"], info["resolution"])
        if key not in loaders:
            loaders[key] = get_loaders(info["dataset"], 256, args.workers, resolution=info["resolution"],
                                       test_subset=args.test_subset, synthetic=args.synthetic)[1]
        acc, _ = evaluate(model.to(device), loaders[key], device)
        shape = (1, 3, info["resolution"], info["resolution"])
        prof = profile_model(model, shape, runs=args.runs, threads=1)
        multi = measure_latency_ms(model, shape, runs=args.runs, threads=None)
        row = {"model": name, "dataset": info["dataset"], "resolution": info["resolution"], "test_acc": round(acc, 2),
               "params_M": round(prof["params"] / 1e6, 4), "MMACs": round(prof["macs"] / 1e6, 2),
               "size_mb": round(prof["size_mb"], 3), "latency_1t_ms": round(prof["latency_ms"], 3),
               "latency_1t_p95_ms": round(prof["latency_p95_ms"], 3),
               "latency_allt_ms": round(multi["median_ms"], 3),
               "peak_mem_mb": None if prof["peak_mem_mb"] is None else round(prof["peak_mem_mb"], 2)}
        rows.append(row)
        print(f"{name:28s} acc {row['test_acc']:6.2f}% | {row['params_M']:7.3f} M params | {row['MMACs']:8.1f} MMACs | "
              f"{row['size_mb']:7.2f} MB | 1t {row['latency_1t_ms']:7.2f} ms | all-t {row['latency_allt_ms']:6.2f} ms "
              f"| peak {row['peak_mem_mb']} MB", flush=True)
    write_csv(RESULTS_DIR / "benchmark_results.csv", rows)

    by = {r["model"]: r for r in rows}
    teacher = by.get("teacher")
    comp = []
    for s in STUDENTS:
        b, k = by.get(f"{s}_baseline"), by.get(f"{s}_kd")
        if not (b and k):
            continue
        r = {"student": s, "baseline_acc": b["test_acc"], "kd_acc": k["test_acc"],
             "kd_gain_pts": round(k["test_acc"] - b["test_acc"], 2), "params_M": k["params_M"], "size_mb": k["size_mb"]}
        if teacher:
            r.update({"teacher_acc": teacher["test_acc"],
                      "compression_x": round(teacher["params_M"] / k["params_M"], 1),
                      "cpu_speedup_x": round(teacher["latency_1t_ms"] / k["latency_1t_ms"], 1)})
        comp.append(r)
    if comp:
        write_csv(RESULTS_DIR / "kd_comparison.csv", comp)
        banner("Baseline vs knowledge distillation (CIFAR-100 test accuracy)")
        for r in comp:
            print(f"{r['student']:20s} baseline {r['baseline_acc']:6.2f}% | KD {r['kd_acc']:6.2f}% | "
                  f"gain {r['kd_gain_pts']:+.2f} pts | {r.get('compression_x', '-')}x fewer params, "
                  f"{r.get('cpu_speedup_x', '-')}x faster than teacher")
    print("\nsaved results/benchmark_results.csv" + (" and results/kd_comparison.csv" if comp else ""))


if __name__ == "__main__":
    main()
