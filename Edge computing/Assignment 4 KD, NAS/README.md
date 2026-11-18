# Edge Computing & Intelligent Systems - Assignment 4
**Q1: Knowledge Distillation (ResNet-50 -> MobileNetV2 / SqueezeNet / EfficientNet-Lite0, CIFAR-100)**
**Q2: Hardware-aware Evolutionary Neural Architecture Search (CIFAR-10)**
Abdullah Qadri - Roll no. 23i-0089 - Section B

One code base: a shared utility module (`src/common.py`) provides data loading, one training loop for every
model, evaluation and real CPU measurements (latency, size, peak memory, params, MACs). `run_all.py` runs the
whole pipeline, saves every step's console output to `logs/`, turns it into screenshots and builds the PDF report.

## 1. Installation
```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (Linux/macOS: source .venv/bin/activate)
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
# e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
Requirements: Python 3.10+, PyTorch >= 2.3, an NVIDIA GPU for training (CPU works but is far too slow for the
ResNet-50 teacher). CIFAR-100 and CIFAR-10 are downloaded automatically into `data/` on first use.
Check that the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.

## 2. Quick start
```bash
python run_all.py --quick          # smoke test of the WHOLE pipeline with tiny settings (~10-20 min)
python run_all.py --clean          # full run: all checkpoints, ONNX files, CSVs, screenshots and the PDF report
```
Run `--quick` first to catch environment problems early; the full run (`--clean`) wipes the quick results first.
If the full run is interrupted, continue with `python run_all.py --from-step N` (step numbers below).
`--synthetic` replaces CIFAR with random images of the same shape (pipeline test without a download).

**Smoke-test evidence.** `logs/smoke_test/` and `report/screenshots/smoke_test/` hold the console output of a
`python run_all.py --quick --synthetic` run that verified every step end-to-end (random data, 1 epoch, so the
accuracies there are chance level). The full run writes the real logs/screenshots one level up and the report
uses those; `--clean` removes the smoke-test folders too.

## 3. Step-by-step (what `run_all.py` executes)
| Step | Command | Output |
|---|---|---|
| 1 Teacher | `python src/train_teacher.py` | `models/resnet50_teacher.pth` |
| 2 Baseline students | `python src/train_baseline.py --arch all` | `models/<student>_baseline.pth` |
| 3 KD students | `python src/train_kd.py --arch all` | `models/<student>_kd.pth` |
| 4 NAS search | `python src/nas_search.py` | `results/nas_history.csv/.json`, `models/nas_best_config.json`, `results/nas_best_architecture.json` |
| 5 Train best NAS model | `python src/nas_train_best.py` | `models/nas_best.pth`, `results/nas_best_final.json` |
| 6 Inference (PyTorch) | `python src/inference.py --model all --test-index 7` | console |
| 7 ONNX export | `python src/export_onnx.py` | `models/onnx/*.onnx` (7 files), `results/onnx_export.csv` |
| 8 ONNX inference | `python src/onnx_inference.py --model all --test-index 7` | console |
| 9 ONNX benchmark | `python src/onnx_inference.py --benchmark --threads 1` | `results/onnx_benchmark.csv` |
| 10 Benchmark | `python src/benchmark.py` | `results/benchmark_results.csv`, `results/kd_comparison.csv` |
| - Screenshots | `python src/make_screenshots.py` | `report/screenshots/*.png` (from `logs/*.log`) |
| - Report | `python src/generate_report.py --section B` | `report/Technical_Report.pdf` |

Default budgets: teacher 60 epochs, each student 40 epochs (baseline and KD), NAS search 10 + 8x6 candidates
x 5 epochs on 20k images, best NAS model 60 epochs. Useful options: `--epochs`, `--batch-size`, `--workers`,
`--alpha`, `--temperature`, `--beta` (KD), `--population`, `--generations`, `--max-latency-ms`, `--max-size-mb`,
`--max-peak-mem-mb` (NAS). Every script prints its options with `--help`.

To shorten the run (e.g. close to the deadline), pass smaller budgets to individual steps, for example
`python src/train_teacher.py --epochs 30` and `python src/train_kd.py --epochs 20`.

## 4. Inference
```bash
python src/inference.py --model all --test-index 7                 # every trained model on one test image
python src/inference.py --model mobilenetv2_kd --image my_photo.jpg   # your own image
python src/onnx_inference.py --model nas_best --image plane.png       # ONNX Runtime on the CPU (no PyTorch needed)
```
Model names: `teacher`, `mobilenetv2_baseline`, `mobilenetv2_kd`, `squeezenet_baseline`, `squeezenet_kd`,
`efficientnet_lite0_baseline`, `efficientnet_lite0_kd`, `nas_best`.

## 5. Deliverable map
| # | Deliverable | Where |
|---|---|---|
| 1 | Teacher model | `src/train_teacher.py`, `models/resnet50_teacher.pth` |
| 2 | Student architectures | `src/models.py` |
| 3 | Baseline students | `src/train_baseline.py`, `models/*_baseline.pth` |
| 4 | Knowledge distillation | `src/kd.py`, `src/train_kd.py`, `models/*_kd.pth` |
| 5 | NAS search space | `src/nas_search_space.py`, `src/nas_generator.py` |
| 6 | NAS algorithm | `src/nas_search.py` (evolutionary), `src/nas_hardware.py` |
| 7 | NAS results | `results/nas_history.csv`, `results/nas_history.json`, `results/nas_best_architecture.json` |
| 8 | Optimized architecture | `src/nas_train_best.py`, `models/nas_best.pth` |
| 9 | Inference pipeline | `src/inference.py`, `src/zoo.py` |
| 10 | ONNX deployment | `src/export_onnx.py`, `src/onnx_inference.py`, `models/onnx/` |
| 11 | Benchmarking | `src/benchmark.py`, `results/*.csv` |
| 12 | Technical report | `report/Technical_Report.pdf` (generated by `src/generate_report.py`) |
| 13 | README | this file |

## 6. Design notes
* **CIFAR adaptation.** ImageNet networks downsample 32x; on 32x32 images that leaves 1x1 maps. Every model
  removes its first two stride-2 operations so all final feature maps are 4x4 - teacher and students are then
  spatially aligned for feature distillation.
* **Students.** MobileNetV2 is torchvision's implementation with the strides changed. SqueezeNet 1.1 and
  EfficientNet-Lite0 are implemented in `models.py`: SqueezeNet with BatchNorm in the Fire modules (the original
  has none and trains poorly from scratch) and without the ReLU on the logits (non-negative logits distort the
  softened distribution); EfficientNet-Lite0 follows the Lite recipe - no squeeze-and-excitation, ReLU6 instead of
  Swish, fixed 32-ch stem and 1280-ch head, B0 stage layout.
* **KD loss** (`kd.py`): `(1-alpha)*CE + alpha*T^2*KL(softmax(t/T) || softmax(s/T)) + beta*MSE(adapter(F_s), F_t)`.
  Temperature softens both distributions so the student learns which wrong classes the teacher finds similar;
  T^2 keeps gradient magnitudes independent of T. The 1x1 conv adapter maps student channels to the teacher's
  2048 and exists only during training (it is not part of the exported student).
* **Fair baseline.** Baseline and KD students use the same data, seed, epochs and optimiser; only the loss differs.
* **NAS.** (mu+lambda) evolution: tournament selection, uniform crossover, per-gene mutation, elitism. Every
  candidate is measured on the CPU *before* training; those violating latency / size / peak-memory limits are
  rejected (and logged) without training. Selection uses a validation split carved from the training set, never
  the test set. Fitness `val_acc * (latency/target)^-0.07` (MnasNet soft reward) prefers faster nets at equal
  accuracy. The winner is re-initialised and trained from scratch on all 50k images.
* **Peak memory** comes from the PyTorch profiler's allocation trace (weights + activations); the code returns
  `None` where the profiler cannot track memory, and the constraint is then skipped ("where supported").
* **Latency is machine-specific.** The NAS limits (3 ms, 1.5 MB, 2.5 MB at 1 thread) were tuned so that some
  candidates are rejected on a typical laptop CPU; if almost everything is rejected (or nothing), adjust
  `--max-latency-ms`.

## 7. Troubleshooting
* `SSL: CERTIFICATE_VERIFY_FAILED` while downloading CIFAR -> `pip install --upgrade certifi` (the code uses
  certifi's certificates automatically). If it still fails, download
  https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz and https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  in the browser and put both files (still compressed) in `data/`; the scripts verify and extract them.
* `CUDA out of memory` -> `--batch-size 64`. Windows data-loader errors -> `--workers 0`.
* `Could not find feasible architectures` -> relax `--max-latency-ms` / `--max-size-mb`.
* Compare latencies only within one run on one machine.
