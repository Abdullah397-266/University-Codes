"""
Export trained models to ONNX (opset 17, dynamic batch dimension) and verify every file:
  1. onnx.checker validates the graph,
  2. ONNX Runtime (CPU) output is compared with the PyTorch output on the same random input.

    python src/export_onnx.py                    # all trained students + NAS model
    python src/export_onnx.py --include-teacher
Output: models/onnx/<name>.onnx, results/onnx_export.csv
"""
import argparse
import warnings

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import RESULTS_DIR, banner, write_csv
from zoo import STUDENT_NAMES, available, load_trained, onnx_path


def export(name: str, opset: int) -> dict:
    model, info = load_trained(name)
    r = info["resolution"]
    x = torch.randn(1, 3, r, r)
    path = onnx_path(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, x, str(path), input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
                          do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    xb = torch.randn(4, 3, r, r)                                      # batch 4 also checks the dynamic axis
    with torch.no_grad():
        ref = model(xb).numpy()
    out = sess.run(None, {"input": xb.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    same_top1 = bool((ref.argmax(1) == out.argmax(1)).all())
    size = path.stat().st_size / 2**20
    print(f"{name:28s} -> {path.name:32s} {size:7.2f} MB | input 1x3x{r}x{r} | max |torch-ort| = {diff:.2e} "
          f"| top-1 match: {same_top1}")
    return {"model": name, "onnx_file": path.name, "onnx_size_mb": round(size, 4), "resolution": r,
            "max_abs_diff": diff, "top1_match": same_top1, "opset": opset}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-teacher", action="store_true")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    names = available((["teacher"] if args.include_teacher else []) + STUDENT_NAMES)
    if not names:
        raise SystemExit("No trained checkpoints found in models/.")
    banner(f"ONNX export (opset {args.opset}) + verification with ONNX Runtime {ort.__version__}")
    rows = [export(n, args.opset) for n in names]
    write_csv(RESULTS_DIR / "onnx_export.csv", rows)
    print(f"\n{len(rows)} models exported -> models/onnx/, summary in results/onnx_export.csv")


if __name__ == "__main__":
    main()
