"""
Deliverable 8 (part 2): CPU inference + benchmarking of the exported ONNX models with ONNX Runtime
(no PyTorch needed for single-image inference except for CIFAR-10 loading in --benchmark mode).

  python src/onnx_inference.py --model qat_int8 --image cat.jpg
  python src/onnx_inference.py --benchmark            # -> results/onnx_benchmark.csv
"""
import argparse
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from common import CLASSES, MEAN, ONNX_FILES, RESULTS_DIR, STD, file_size_mb, get_datasets, make_loader


def make_session(path, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def preprocess(path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def benchmark(threads, iters):
    _, _, test = get_datasets()
    loader = make_loader(test, 256, workers=0)
    batches = [(x.numpy(), y.numpy()) for x, y in loader]
    rows = []
    for name, path in ONNX_FILES.items():
        if not path.exists():
            continue
        sess = make_session(path, threads)
        correct = total = 0
        for x, y in batches:
            pred = sess.run(None, {"input": x})[0].argmax(1)
            correct, total = correct + int((pred == y).sum()), total + len(y)
        x1 = batches[0][0][:1]
        for _ in range(20):
            sess.run(None, {"input": x1})
        t = []
        for _ in range(iters):
            s = time.perf_counter()
            sess.run(None, {"input": x1})
            t.append((time.perf_counter() - s) * 1000)
        xb = batches[0][0][:128]
        s = time.perf_counter()
        for _ in range(5):
            sess.run(None, {"input": xb})
        thr = 5 * len(xb) / (time.perf_counter() - s)
        rows.append({"model": name, "runtime": "onnxruntime-cpu", "threads": threads,
                     "test_acc_pct": round(100 * correct / total, 2), "file_size_mb": round(file_size_mb(path), 2),
                     "latency_bs1_mean_ms": round(float(np.mean(t)), 3),
                     "latency_bs1_p95_ms": round(float(np.percentile(t, 95)), 3),
                     "throughput_bs128_img_s": round(thr, 1)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "onnx_benchmark.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'onnx_benchmark.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(ONNX_FILES), default="fp32")
    ap.add_argument("--image")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--latency-iters", type=int, default=100)
    args = ap.parse_args()
    if args.benchmark:
        return benchmark(args.threads, args.latency_iters)
    if not args.image:
        ap.error("--image is required unless --benchmark is given")
    sess = make_session(ONNX_FILES[args.model], args.threads)
    prob = softmax(sess.run(None, {"input": preprocess(args.image)})[0])[0]
    for i in prob.argsort()[::-1][:3]:
        print(f"{CLASSES[i]:12s} {prob[i]:.2%}")


if __name__ == "__main__":
    main()
