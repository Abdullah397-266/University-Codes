"""One-command reproducible pipeline.  python run_all.py [--quick] [--section A]

--quick  runs every stage with tiny settings (1-2 epochs) as a smoke test of the whole pipeline (~10 min on GPU).
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--section", default="")
    ap.add_argument("--name", default="Abdullah")
    ap.add_argument("--roll", default="23i-0089")
    args = ap.parse_args()
    q = args.quick
    steps = [
        ("1. baseline FP32", "train_baseline.py", ["--epochs", "1"] if q else []),
        ("2. PTQ", "ptq.py", ["--calib-samples", "256"] if q else []),
        ("3. QAT", "qat.py", ["--epochs", "1"] if q else []),
        ("4. unstructured pruning", "prune_unstructured.py",
         ["--iterations", "2", "--ft-epochs", "1", "--final-ft-epochs", "1"] if q else []),
        ("5. structured pruning", "prune_structured.py", ["--ft-epochs", "1"] if q else []),
        ("6. ONNX export", "export_onnx.py", []),
        ("7. PyTorch benchmark", "benchmark.py", ["--latency-iters", "20"] if q else []),
        ("8. ONNX benchmark", "onnx_inference.py", ["--benchmark"] + (["--latency-iters", "20"] if q else [])),
        ("9. report", "generate_report.py", ["--name", args.name, "--roll", args.roll, "--section", args.section]),
    ]
    for title, script, extra in steps:
        print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, str(SRC / script)] + extra, check=True)
    print("\nAll stages finished. See models/, results/ and report/.")


if __name__ == "__main__":
    main()
