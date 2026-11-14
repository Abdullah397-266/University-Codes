"""
Deliverable 7: automated benchmark of all five PyTorch models on the CIFAR-10 test set.

Outputs
  results/benchmark_results.csv  one row per (model, device): accuracy, size, params, MACs, latency, throughput
  results/predictions.csv        per test image: true label + predicted class/confidence for every model

All models are benchmarked on the CPU (the edge scenario; INT8 kernels exist only on CPU). Use
--include-gpu to also time the float models on CUDA. Use --threads 1 to mimic a single-core edge device.
"""
import argparse
import csv

import pandas as pd
import torch

from common import (CLASSES, MODEL_FILES, RESULTS_DIR, count_macs, count_params, evaluate, file_size_mb,
                    get_datasets, gzip_size_mb, load_float_model, make_loader, measure_latency)
from inference import INT8_MODELS, load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch")
    ap.add_argument("--latency-iters", type=int, default=100)
    ap.add_argument("--include-gpu", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    _, _, test = get_datasets()
    loader = make_loader(test, 256, workers=args.workers)
    names = [n for n, p in MODEL_FILES.items() if p.exists()]
    missing = [n for n in MODEL_FILES if n not in names]
    if missing:
        print(f"[warn] skipping models without a checkpoint: {missing}")

    # FP32 architecture counts are reused for the INT8 models (same layers, 8-bit storage)
    ref_model, _ = load_float_model(MODEL_FILES["fp32"], "cpu")
    ref_params, ref_nonzero = count_params(ref_model)
    ref_macs = count_macs(ref_model)

    rows, preds = [], {}
    labels = None
    jobs = [(n, "cpu") for n in names]
    if args.include_gpu and torch.cuda.is_available():
        jobs += [(n, "cuda") for n in names if n not in INT8_MODELS]

    for name, dev in jobs:
        print(f"\n--- {name} on {dev} ---")
        model = load_model(name, dev)
        res = evaluate(model, loader, dev)
        lat1 = measure_latency(model, torch.device(dev), 1, 20, args.latency_iters)
        lat128 = measure_latency(model, torch.device(dev), 128, 2, 10)
        if name in INT8_MODELS:
            total, nonzero, macs, macs_nz = ref_params, ref_nonzero, ref_macs, ref_macs
        else:
            total, nonzero = count_params(model.cpu())
            macs, macs_nz = count_macs(model), count_macs(model, nonzero_only=True)
            model.to(dev)
        path = MODEL_FILES[name]
        rows.append({
            "model": name, "device": dev, "threads": args.threads if dev == "cpu" else "-",
            "test_acc_pct": round(res["acc"], 2),
            "params_total": total, "params_nonzero": nonzero,
            "sparsity_pct": round(100 * (1 - nonzero / total), 2) if name not in INT8_MODELS else 0.0,
            "macs_dense_M": round(macs / 1e6, 1), "macs_nonzero_M": round(macs_nz / 1e6, 1),
            "file_size_mb": round(file_size_mb(path), 2), "gzip_size_mb": round(gzip_size_mb(path), 2),
            "latency_bs1_mean_ms": round(lat1["mean_ms"], 3), "latency_bs1_p50_ms": round(lat1["p50_ms"], 3),
            "latency_bs1_p95_ms": round(lat1["p95_ms"], 3),
            "throughput_bs128_img_s": round(lat128["img_per_s"], 1),
        })
        print(f"acc {res['acc']:.2f}% | {lat1['mean_ms']:.2f} ms/img (bs1) | {lat128['img_per_s']:.0f} img/s (bs128)")
        if dev == "cpu":
            preds[name] = res
            labels = res["labels"]

    df = pd.DataFrame(rows)
    base = df[(df.model == "fp32") & (df.device == "cpu")].iloc[0] if "fp32" in names else None
    if base is not None:
        cpu_mask = df.device == "cpu"
        df["acc_delta_vs_fp32"] = (df.test_acc_pct - base.test_acc_pct).round(2)
        df["size_reduction_x"] = (base.file_size_mb / df.file_size_mb).round(2)
        df["gzip_reduction_x"] = (base.gzip_size_mb / df.gzip_size_mb).round(2)
        df["latency_speedup_x"] = None
        df.loc[cpu_mask, "latency_speedup_x"] = (base.latency_bs1_mean_ms / df.loc[cpu_mask, "latency_bs1_mean_ms"]).round(2)
    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)

    # per-image predictions of every model
    pred_df = pd.DataFrame({"index": range(len(labels)), "true_label": labels,
                            "true_class": [CLASSES[i] for i in labels]})
    for name, res in preds.items():
        pred_df[f"{name}_pred"] = res["preds"]
        pred_df[f"{name}_conf"] = res["confs"].round(4)
        pred_df[f"{name}_correct"] = (res["preds"] == labels).astype(int)
    pred_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)

    print("\n" + df.to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'benchmark_results.csv'} and {RESULTS_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
