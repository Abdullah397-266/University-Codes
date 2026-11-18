"""
Run the complete Assignment 4 pipeline (Q1 knowledge distillation + Q2 hardware-aware NAS) step by step.
Each step's console output is saved to logs/NN_<step>.log, rendered as a screenshot into
report/screenshots/, and finally the PDF technical report is generated.

    python run_all.py --quick            # smoke test of the whole pipeline with tiny settings (~10-20 min on a GPU)
    python run_all.py                    # full run (all defaults; several hours on a laptop GPU)
    python run_all.py --from-step 5      # resume from step 5 (e.g. after an interruption)
    python run_all.py --only 8           # run a single step
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"

# (name, script, full-run args, extra args for --quick)
STEPS = [
    ("teacher", "src/train_teacher.py", [], ["--epochs", "1", "--train-subset", "1024", "--test-subset", "512"]),
    ("baseline_students", "src/train_baseline.py", ["--arch", "all"],
     ["--epochs", "1", "--train-subset", "1024", "--test-subset", "512"]),
    ("kd_students", "src/train_kd.py", ["--arch", "all"],
     ["--epochs", "1", "--train-subset", "1024", "--test-subset", "512"]),
    ("nas_search", "src/nas_search.py", [],
     ["--population", "3", "--generations", "2", "--children", "2", "--epochs-per-candidate", "1",
      "--train-subset", "1024", "--val-split", "512"]),
    ("nas_train_best", "src/nas_train_best.py", [], ["--epochs", "1", "--train-subset", "1024", "--test-subset", "512"]),
    ("inference", "src/inference.py", ["--model", "all", "--test-index", "7"], []),
    ("export_onnx", "src/export_onnx.py", [], []),
    ("onnx_inference", "src/onnx_inference.py", ["--model", "all", "--test-index", "7"], []),
    ("onnx_benchmark", "src/onnx_inference.py", ["--benchmark", "--threads", "1"],
     ["--runs", "50", "--eval-images", "512"]),
    ("benchmark", "src/benchmark.py", [], ["--runs", "30", "--test-subset", "512"]),
]
DATA_STEPS = {"teacher", "baseline_students", "kd_students", "nas_search", "nas_train_best", "inference",
              "onnx_inference", "onnx_benchmark", "benchmark"}


def run_step(i: int, name: str, script: str, args: list[str], python: str) -> None:
    cmd = [python, "-u", script] + args
    shown = "python " + " ".join([script] + args)
    log = LOGS / f"{i:02d}_{name}.log"
    print(f"\n>>> [{i}/{len(STEPS)}] {shown}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"PS {ROOT.name}> {shown}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            if "Warning" in line and ("onnx" in line.lower() or "deprecat" in line.lower()):
                continue                                   # keep exporter deprecation noise out of the screenshots
            sys.stdout.write(line)
            f.write(line)
        code = proc.wait()
        f.write(f"\n[step finished in {(time.time() - t0) / 60:.1f} min, exit code {code}]\n")
    if code != 0:
        raise SystemExit(f"Step {i} ({name}) failed with exit code {code}; see {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny settings: checks that every step runs end-to-end")
    p.add_argument("--synthetic", action="store_true", help="random data instead of CIFAR (no download needed)")
    p.add_argument("--from-step", type=int, default=1)
    p.add_argument("--only", type=int)
    p.add_argument("--clean", action="store_true", help="delete models/, results/, logs/ and screenshots first")
    args = p.parse_args()

    if args.clean:
        for d in ("models", "results", "logs", "report/screenshots"):
            shutil.rmtree(ROOT / d, ignore_errors=True)
    LOGS.mkdir(exist_ok=True)
    t0 = time.time()
    for i, (name, script, full, quick) in enumerate(STEPS, 1):
        if (args.only and i != args.only) or i < args.from_step:
            continue
        a = full + (quick if args.quick else []) + (["--synthetic"] if args.synthetic and name in DATA_STEPS else [])
        run_step(i, name, script, a, sys.executable)
    subprocess.run([sys.executable, "src/make_screenshots.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "src/generate_report.py"], cwd=ROOT, check=True)
    print(f"\nPipeline finished in {(time.time() - t0) / 60:.1f} min. Report: report/Technical_Report.pdf")


if __name__ == "__main__":
    main()
