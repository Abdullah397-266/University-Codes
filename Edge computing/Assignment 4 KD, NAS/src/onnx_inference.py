"""
CPU inference with ONNX Runtime for the exported student models and the NAS model.
Needs only numpy + Pillow + onnxruntime at deployment time (no PyTorch): pre-processing is re-implemented
in numpy below.

    python src/onnx_inference.py --model all --test-index 3           # predictions of every ONNX model
    python src/onnx_inference.py --model nas_best --image plane.png
    python src/onnx_inference.py --benchmark --threads 1              # latency + accuracy -> results/onnx_benchmark.csv
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from common import DATASETS, RESULTS_DIR, banner, get_dataset, read_json, write_csv
from zoo import STUDENT_NAMES, onnx_path


def model_info(name: str) -> dict:
    """Dataset / resolution of an exported model (NAS resolution comes from its saved config)."""
    if name == "nas_best":
        from common import NAS_BEST_CONFIG
        dataset, res = "cifar10", read_json(NAS_BEST_CONFIG)["resolution"]
    else:
        dataset, res = "cifar100", 32
    d = DATASETS[dataset]
    return {"dataset": dataset, "resolution": res, "classes": d["classes"],
            "mean": np.array(d["mean"], np.float32), "std": np.array(d["std"], np.float32)}


def preprocess(img: Image.Image, info: dict) -> np.ndarray:
    r = info["resolution"]
    img = img.convert("RGB")
    if img.size != (r, r):
        img = img.resize((r, r), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - info["mean"]) / info["std"]
    return x.transpose(2, 0, 1)[None].astype(np.float32)             # NCHW


def make_session(name: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path(name)), so, providers=["CPUExecutionProvider"])


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def benchmark(names, args):
    rows = []
    for name in names:
        info = model_info(name)
        sess = make_session(name, args.threads)
        ds = get_dataset(info["dataset"], train=False, transform=None, synthetic=args.synthetic)
        n = min(args.eval_images, len(ds))
        correct, batch, labels = 0, [], []
        for i in range(n):                                            # accuracy on the first n test images
            img, y = ds[i]
            batch.append(preprocess(img, info)[0])
            labels.append(y)
            if len(batch) == 256 or i == n - 1:
                logits = sess.run(None, {"input": np.stack(batch)})[0]
                correct += int((logits.argmax(1) == np.array(labels)).sum())
                batch, labels = [], []
        x = np.random.randn(1, 3, info["resolution"], info["resolution"]).astype(np.float32)
        for _ in range(args.warmup):
            sess.run(None, {"input": x})
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            sess.run(None, {"input": x})
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        row = {"model": name, "dataset": info["dataset"], "resolution": info["resolution"],
               "onnx_size_mb": round(onnx_path(name).stat().st_size / 2**20, 4),
               f"accuracy_{n}_imgs": round(100 * correct / n, 2),
               "latency_median_ms": round(statistics.median(times), 4),
               "latency_p95_ms": round(times[int(0.95 * (len(times) - 1))], 4),
               "throughput_img_s": round(1000 / statistics.mean(times), 1), "threads": args.threads}
        rows.append(row)
        print(f"{name:28s} acc {row[f'accuracy_{n}_imgs']:6.2f}% | median {row['latency_median_ms']:.3f} ms "
              f"| p95 {row['latency_p95_ms']:.3f} ms | {row['throughput_img_s']:.0f} img/s | {row['onnx_size_mb']:.2f} MB")
    write_csv(RESULTS_DIR / "onnx_benchmark.csv", rows)
    print("\nsaved results/onnx_benchmark.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", help=f"all | comma list of {STUDENT_NAMES}")
    p.add_argument("--image")
    p.add_argument("--test-index", type=int, default=0)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--eval-images", type=int, default=10000)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    candidates = STUDENT_NAMES if args.model == "all" else args.model.split(",")
    names = [n for n in candidates if onnx_path(n).exists()]
    if not names:
        raise SystemExit("No ONNX models found - run src/export_onnx.py first.")
    banner(f"ONNX Runtime {ort.__version__} | CPUExecutionProvider | {args.threads} thread(s)")
    if args.benchmark:
        return benchmark(names, args)
    for name in names:
        info = model_info(name)
        if args.image:
            img, label = Image.open(args.image), None
        else:
            img, label = get_dataset(info["dataset"], train=False, transform=None, synthetic=args.synthetic)[args.test_index]
        sess = make_session(name, args.threads)
        t0 = time.perf_counter()
        probs = softmax(sess.run(None, {"input": preprocess(img, info)})[0])[0]
        ms = (time.perf_counter() - t0) * 1000
        top = probs.argsort()[::-1][:args.topk]
        cls = info["classes"]
        txt = ", ".join(f"{cls[i]} {probs[i] * 100:.1f}%" for i in top)
        truth = "" if label is None else f" | true: {cls[label]} {'[correct]' if top[0] == label else '[wrong]'}"
        print(f"{name:28s} {ms:6.2f} ms | top-{args.topk}: {txt}{truth}")


if __name__ == "__main__":
    main()
