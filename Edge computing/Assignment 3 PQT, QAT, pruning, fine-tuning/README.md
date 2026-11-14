# Edge Computing & Intelligent Systems - Assignment 3
**Quantization (PTQ + QAT) and Pruning of ResNet-18 on CIFAR-10**
Abdullah - Roll no. 23i-0089

Everything runs from one codebase: one ResNet-18 definition (`src/common.py`) is trained, quantized (PTQ, QAT),
pruned (unstructured, structured), exported (TorchScript, ONNX) and benchmarked.

## 1. Installation
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```
Requirements: Python 3.10+, PyTorch >= 2.3, NVIDIA GPU recommended for training (CPU works, but slowly).
CIFAR-10 is downloaded automatically into `data/` on first run.

## 2. Quick start
```bash
python run_all.py --quick      # smoke test of the WHOLE pipeline with tiny settings (about 10 minutes)
python run_all.py --section A  # full run: all checkpoints, ONNX files, CSVs and the PDF report
```
Do the `--quick` run first to catch environment problems early, then run the full pipeline (it overwrites the quick results).

## 3. Step-by-step (what `run_all.py` executes)
| Step | Command | Output |
|---|---|---|
| 1 Baseline | `python src/train_baseline.py` | `models/resnet18_fp32.pth` |
| 2 PTQ | `python src/ptq.py` | `models/resnet18_ptq_int8.pt` (+ `_state.pth`) |
| 3 QAT | `python src/qat.py` | `models/resnet18_qat_int8.pt`, `resnet18_qat_fakequant.pth` |
| 4 Unstructured pruning | `python src/prune_unstructured.py` | `models/resnet18_unstructured_pruned.pth` |
| 5 Structured pruning | `python src/prune_structured.py` | `models/resnet18_structured_pruned.pth` |
| 6 ONNX export | `python src/export_onnx.py` | `models/onnx/*.onnx` (5 files) |
| 7 Benchmark | `python src/benchmark.py` | `results/benchmark_results.csv`, `results/predictions.csv` |
| 8 ONNX benchmark | `python src/onnx_inference.py --benchmark` | `results/onnx_benchmark.csv` |
| 9 Report | `python src/generate_report.py --section A` | `report/Technical_Report.pdf` |

Useful options: `--epochs`, `--target-sparsity 0.9`, `--iterations`, `--prune-ratio 0.6`, `--calib-samples`,
`--backend fbgemm|qnnpack`, `--threads 1` (benchmark as a single-core edge device), `--include-gpu`.

## 4. Inference
```bash
python src/inference.py --model all --test-index 7              # all five models on CIFAR-10 test image 7
python src/inference.py --model qat_int8 --image my_photo.jpg   # your own image
python src/onnx_inference.py --model structured_pruned --image my_photo.jpg   # ONNX Runtime, CPU
```

## 5. Deliverable map
| # | Deliverable | Where |
|---|---|---|
| 1 | Baseline model | `src/train_baseline.py`, `models/resnet18_fp32.pth` |
| 2 | PTQ | `src/ptq.py` (fusion, backend, calibration, convert), `models/resnet18_ptq_int8.pt` |
| 3 | QAT | `src/qat.py` (prepare_qat, fine-tune, convert), `models/resnet18_qat_int8.pt` |
| 4 | Unstructured pruning | `src/prune_unstructured.py`, `models/resnet18_unstructured_pruned.pth` |
| 5 | Structured pruning | `src/prune_structured.py`, `models/resnet18_structured_pruned.pth` |
| 6 | Inference pipeline | `src/inference.py` (all 5 models) |
| 7 | Benchmarking | `src/benchmark.py`, `results/*.csv` |
| 8 | ONNX deployment | `src/export_onnx.py`, `src/onnx_inference.py`, `models/onnx/` |
| 9 | Technical report | `src/generate_report.py` -> `report/Technical_Report.pdf` (add screenshots first) |
| 10 | README | this file |

## 6. Design notes (read before your viva)
* **One model, five uses.** `ResNet18` has Quant/DeQuant stubs and `FloatFunctional` for the residual add, so the same class is trained in FP32, fused/quantized, and rebuilt with smaller widths for structured pruning.
* **INT8 runs on CPU.** PyTorch eager-mode INT8 kernels (fbgemm on x86, qnnpack on ARM) exist for CPU only. QAT *training* uses the GPU (fake quantization), then the model is converted on CPU. INT8 TorchScript files are backend-specific: `models/quant_meta.json` records which one was used.
* **PTQ vs QAT.** PTQ only calibrates ranges; QAT fine-tunes the weights with simulated quantization noise, which usually recovers more accuracy.
* **Unstructured pruning** (global L1, cubic schedule, iterative with fine-tuning, `prune.remove` to make masks permanent) lowers non-zero parameters and gzip size, but dense kernels do not get faster because tensor shapes are unchanged.
* **Structured pruning** removes inner channels of each BasicBlock (conv1 filters, matching BN entries, conv2 input channels) and builds a physically smaller network. The residual stream width is kept unchanged so shortcut additions remain valid; this is the standard safe approach for ResNets.
* **ONNX.** FP32 and both pruned models export directly. `qat_int8` is exported with QuantizeLinear/DequantizeLinear nodes (true INT8 in ONNX Runtime). PyTorch eager INT8 modules cannot be exported to ONNX directly, so `ptq_int8.onnx` is produced by ONNX Runtime static quantization of the FP32 graph with the same calibration protocol. `export_onnx.py` prints which route was used.
* **INT8 rows in the benchmark CSV** reuse the FP32 architecture's parameter and MAC counts (same layers, 8-bit storage).

## 7. Troubleshooting
* `Quantized backend 'fbgemm' not supported` -> pass `--backend qnnpack` (ARM) or `--backend x86`.
* `CUDA out of memory` -> `--batch-size 64`. Windows data-loader errors -> `--workers 0`.
* Latency numbers vary between machines/threads: always compare models measured in the same run.

## 8. Submission checklist
* Rename the folder/zip to `23i-0089_<SECTION>` (e.g. `23i-0089_A`) and upload to Google Classroom only.
* Remove `data/` (CIFAR-10 download) and `__pycache__/` before zipping; keep `models/`, `results/`, `report/Technical_Report.pdf`.
* Push the repo to GitHub (the checkpoints are < 50 MB each, under GitHub's 100 MB limit).
* Add terminal screenshots of every script to `report/screenshots/`, then re-run `python src/generate_report.py --section <X>`.
