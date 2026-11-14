"""
Deliverable 8 (part 1): export all five models to ONNX for CPU deployment with ONNX Runtime.

* fp32 / unstructured_pruned / structured_pruned : plain FP32 graphs (dynamic batch size).
* qat_int8 : the QAT model exported with FakeQuantize -> QuantizeLinear/DequantizeLinear (QDQ) nodes, i.e. the
             learned scales/zero-points travel with the model and ONNX Runtime executes it as INT8.
             (Fallback if that export fails: QAT-trained FP32 weights quantized by ONNX Runtime.)
* ptq_int8 : PyTorch eager-mode INT8 modules cannot be exported to ONNX directly, so the FP32 ONNX graph is
             statically quantized by ONNX Runtime with the SAME calibration protocol (random training subset).
"""
import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from torch.ao.quantization import disable_fake_quant, disable_observer, prepare_qat

from common import (MODEL_FILES, MODELS_DIR, ONNX_FILES, ResNet18, default_backend, get_datasets,
                    load_float_model, load_quant_meta, qat_qconfig, set_quant_backend)


def export(model, path, opset):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=opset,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)   # classic TorchScript-based exporter
    except TypeError:                                                      # older torch without `dynamo` kwarg
        torch.onnx.export(model, dummy, str(path), **kw)
    onnx.checker.check_model(onnx.load(str(path)))
    # numerical sanity check against PyTorch
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    out = sess.run(None, {"input": dummy.numpy()})[0]
    print(f"  exported {path.name}  max|torch-onnx| = {np.abs(ref - out).max():.2e}")


class CifarReader(CalibrationDataReader):
    def __init__(self, n, seed=42):
        _, train_plain, _ = get_datasets()
        idx = np.random.RandomState(seed).choice(len(train_plain), n, replace=False)
        self.data = iter([{"input": train_plain[int(i)][0].numpy()[None]} for i in idx])

    def get_next(self):
        return next(self.data, None)


def ort_quantize(fp32_onnx, out_path, calib):
    quantize_static(str(fp32_onnx), str(out_path), CifarReader(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    print(f"  quantized with ONNX Runtime -> {out_path.name}")


def export_qat(opset, calib):
    path = ONNX_FILES["qat_int8"]
    meta = load_quant_meta().get("qat_int8", {})
    backend = set_quant_backend(meta.get("backend", default_backend()))
    ck = torch.load(MODELS_DIR / "resnet18_qat_fakequant.pth", map_location="cpu")
    m = ResNet18()
    m.train()
    m.fuse_model(is_qat=True)
    m.qconfig = qat_qconfig(backend)
    prepare_qat(m, inplace=True)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.apply(disable_observer)
    try:
        export(m, path, opset)
        print("  QAT exported as QDQ graph (true INT8 in ONNX Runtime)")
    except Exception as e:
        print(f"  [warn] QDQ export failed ({e.__class__.__name__}: {str(e)[:120]}).")
        print("  Falling back: QAT-trained FP32 weights -> ONNX -> ONNX Runtime static quantization")
        m.apply(disable_fake_quant)
        tmp = MODELS_DIR / "onnx" / "_qat_fp32_tmp.onnx"
        export(m, tmp, opset)
        ort_quantize(tmp, path, calib)
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--calib-samples", type=int, default=500)
    args = ap.parse_args()

    for name in ("fp32", "unstructured_pruned", "structured_pruned"):
        if MODEL_FILES[name].exists():
            print(f"[{name}]")
            export(load_float_model(MODEL_FILES[name], "cpu")[0], ONNX_FILES[name], args.opset)
    if ONNX_FILES["fp32"].exists():
        print("[ptq_int8]")
        ort_quantize(ONNX_FILES["fp32"], ONNX_FILES["ptq_int8"], args.calib_samples)
    if (MODELS_DIR / "resnet18_qat_fakequant.pth").exists():
        print("[qat_int8]")
        export_qat(args.opset, args.calib_samples)


if __name__ == "__main__":
    main()
