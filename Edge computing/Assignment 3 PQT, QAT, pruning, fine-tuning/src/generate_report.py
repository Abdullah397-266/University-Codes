"""
Deliverable 9: build the technical report PDF from the real experiment outputs.

Reads results/*.csv and models/quant_meta.json, draws charts, and embeds any screenshots found in
report/screenshots/ (PNG/JPG - add terminal/benchmark screenshots there before generating).
  python src/generate_report.py --name Abdullah --roll 23i-0089 --section A
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # no torch import needed to build the report
RESULTS_DIR = ROOT / "results"
QUANT_META = ROOT / "models" / "quant_meta.json"

REPORT_DIR = ROOT / "report"
SHOTS = REPORT_DIR / "screenshots"
LABELS = {"fp32": "FP32", "ptq_int8": "PTQ INT8", "qat_int8": "QAT INT8",
          "unstructured_pruned": "Unstructured", "structured_pruned": "Structured"}


def read(name):
    p = RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def charts(bench):
    out = []
    if bench is not None:
        b = bench[bench.device == "cpu"]
        names = [LABELS.get(m, m) for m in b.model]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        for a, col, title in zip(ax, ["test_acc_pct", "latency_bs1_mean_ms", "file_size_mb"],
                                 ["Test accuracy (%)", "CPU latency, batch 1 (ms)", "Checkpoint size (MB)"]):
            a.bar(names, b[col], color="#4c72b0")
            a.set_title(title, fontsize=10)
            a.tick_params(axis="x", rotation=30, labelsize=8)
            if col == "test_acc_pct":
                a.set_ylim(max(0, b[col].min() - 3), min(100, b[col].max() + 1.5))
        fig.tight_layout()
        p = REPORT_DIR / "chart_benchmark.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        out.append(("Figure 1: accuracy, latency and size of the five models (CPU).", p))
    curves = [(t, read(f"history_{t}.csv")) for t in ("baseline", "qat", "unstructured", "structured")]
    curves = [(t, h) for t, h in curves if h is not None]
    if curves:
        fig, ax = plt.subplots(1, len(curves), figsize=(3.2 * len(curves), 3), squeeze=False)
        for a, (t, h) in zip(ax[0], curves):
            a.plot(range(1, len(h) + 1), h.test_acc, label="test")
            a.plot(range(1, len(h) + 1), h.train_acc, label="train")
            a.set_title(t, fontsize=10)
            a.set_xlabel("epoch", fontsize=8)
            a.legend(fontsize=7)
        fig.tight_layout()
        p = REPORT_DIR / "chart_training.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        out.append(("Figure 2: accuracy during training / fine-tuning.", p))
    return out


def table(df, cols, header, widths=None):
    data = [header] + [[str(r[c]) for c in cols] for _, r in df.iterrows()]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde5f0")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7),
                           ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Abdullah")
    ap.add_argument("--roll", default="23i-0089")
    ap.add_argument("--section", default="")
    ap.add_argument("--out", default=str(REPORT_DIR / "Technical_Report.pdf"))
    args = ap.parse_args()

    S = getSampleStyleSheet()
    H1, H2, B = S["Heading1"], S["Heading2"], S["BodyText"]
    story = []
    P = lambda t, st=B: story.append(Paragraph(t, st))
    gap = lambda h=6: story.append(Spacer(1, h))

    bench, onnxb = read("benchmark_results.csv"), read("onnx_benchmark.csv")
    meta = json.loads(QUANT_META.read_text()) if QUANT_META.exists() else {}

    P("Edge Computing &amp; Intelligent Systems - Assignment 3", S["Title"])
    P("Model Optimization for Edge Deployment: Quantization (PTQ, QAT) and Pruning of ResNet-18 on CIFAR-10",
      S["Heading3"])
    P(f"{args.name} &nbsp;|&nbsp; Roll no. {args.roll} &nbsp;|&nbsp; Section {args.section or '-'}")
    gap(10)

    P("1. Introduction", H1)
    P("This report documents a PyTorch optimization pipeline that compresses a CIFAR-10 ResNet-18 in four ways: "
      "post-training static quantization (PTQ), quantization-aware training (QAT), iterative unstructured "
      "magnitude pruning and structured channel pruning. All five resulting models are benchmarked on the "
      "CIFAR-10 test set (10,000 images) and exported to ONNX for CPU deployment.")

    P("2. Setup and baseline", H1)
    P("<b>Model.</b> CIFAR-style ResNet-18 (3x3 stem, no max-pool, 8 BasicBlocks, 11.2M parameters). The residual "
      "add uses <i>FloatFunctional</i> and the network has Quant/DeQuant stubs so one definition serves FP32, "
      "PTQ, QAT and pruning.")
    P("<b>Training.</b> SGD (momentum 0.9, Nesterov, weight decay 5e-4), cosine learning-rate schedule, random crop "
      "+ horizontal flip, mixed precision on GPU. The FP32 checkpoint is <i>models/resnet18_fp32.pth</i>.")
    if bench is not None and "fp32" in set(bench.model):
        r = bench[(bench.model == "fp32") & (bench.device == "cpu")].iloc[0]
        P(f"Baseline test accuracy: <b>{r.test_acc_pct:.2f}%</b>, size {r.file_size_mb:.1f} MB.")

    P("3. Post-training static quantization (PTQ)", H1)
    P("Conv+BN+ReLU groups are fused, observers are inserted (backend qconfig: per-channel INT8 weights, "
      "per-tensor UINT8 activations), a calibration pass over a random subset of the <b>training</b> set records "
      "activation ranges, and the model is converted to INT8 (scale/zero-point per tensor). The result is saved as "
      "an executable TorchScript file. "
      + (f"Backend: {meta['ptq_int8']['backend']}, calibration images: {meta['ptq_int8']['calib_samples']}."
         if "ptq_int8" in meta else ""))

    P("4. Quantization-aware training (QAT)", H1)
    P("Starting from the FP32 weights, the model is fused in training mode and <i>prepare_qat</i> inserts "
      "FakeQuantize modules. Fine-tuning with simulated quantization lets the weights adapt to rounding error; "
      "gradients pass through the rounding with the straight-through estimator. Observers and BatchNorm "
      "statistics are frozen in the last epochs, then the model is converted to real INT8 on the CPU. "
      + (f"Fake-quant accuracy {meta['qat_int8']['fake_quant_acc']:.2f}%, INT8 accuracy "
         f"{meta['qat_int8']['test_acc']:.2f}%." if "qat_int8" in meta else ""))

    P("5. Unstructured pruning", H1)
    P("Global L1-magnitude pruning over all Conv2d and Linear weights, in several iterations following a cubic "
      "sparsity schedule; every iteration is followed by fine-tuning with the mask active. At the end the masks are "
      "made permanent with <i>prune.remove</i>. Weights are set to zero but tensor shapes are unchanged, so dense "
      "kernels gain no speed; the benefit is the non-zero parameter count and compressed (gzip) size.")

    P("6. Structured channel pruning", H1)
    P("For every BasicBlock the filters of conv1 are ranked by L1 norm and the weakest are removed together with the "
      "matching BatchNorm entries and the corresponding input channels of conv2. The residual stream width is "
      "kept so that the shortcut addition stays valid. A new, physically smaller ResNet18 is built "
      "(<i>mid_channels</i> stored in the checkpoint) and the surviving weights are copied in, so inference uses "
      "smaller dense tensors with no masked channels. The reconstructed model is then fine-tuned.")

    P("7. Results", H1)
    if bench is not None:
        cols = ["model", "device", "test_acc_pct", "sparsity_pct", "macs_dense_M", "file_size_mb", "gzip_size_mb",
                "latency_bs1_mean_ms", "throughput_bs128_img_s"]
        hdr = ["Model", "Dev", "Acc %", "Sparsity %", "MACs (M)", "Size MB", "gzip MB", "Lat. ms", "img/s"]
        P("Table 1: PyTorch benchmark (results/benchmark_results.csv).")
        story.append(table(bench, cols, hdr))
        gap()
        cpu = bench[bench.device == "cpu"].set_index("model")
        notes = []
        if {"fp32", "ptq_int8"} <= set(cpu.index):
            notes.append(f"PTQ changes accuracy by {cpu.loc['ptq_int8', 'test_acc_pct'] - cpu.loc['fp32', 'test_acc_pct']:+.2f} "
                         f"points and shrinks the file {cpu.loc['fp32', 'file_size_mb'] / cpu.loc['ptq_int8', 'file_size_mb']:.1f}x.")
        if {"ptq_int8", "qat_int8"} <= set(cpu.index):
            notes.append(f"QAT INT8 vs PTQ INT8 accuracy: {cpu.loc['qat_int8', 'test_acc_pct'] - cpu.loc['ptq_int8', 'test_acc_pct']:+.2f} points.")
        if {"fp32", "structured_pruned"} <= set(cpu.index):
            notes.append(f"Structured pruning cuts MACs from {cpu.loc['fp32', 'macs_dense_M']:.0f}M to "
                         f"{cpu.loc['structured_pruned', 'macs_dense_M']:.0f}M; batch-1 latency changes by "
                         f"{cpu.loc['fp32', 'latency_bs1_mean_ms'] / cpu.loc['structured_pruned', 'latency_bs1_mean_ms']:.2f}x.")
        if {"fp32", "unstructured_pruned"} <= set(cpu.index):
            notes.append(f"Unstructured pruning reaches {cpu.loc['unstructured_pruned', 'sparsity_pct']:.1f}% sparsity "
                         f"(gzip size {cpu.loc['fp32', 'gzip_size_mb'] / cpu.loc['unstructured_pruned', 'gzip_size_mb']:.1f}x smaller) "
                         f"while dense latency changes by {cpu.loc['fp32', 'latency_bs1_mean_ms'] / cpu.loc['unstructured_pruned', 'latency_bs1_mean_ms']:.2f}x.")
        for n in notes:
            P("- " + n)
    else:
        P("[benchmark_results.csv not found - run src/benchmark.py first]")
    for cap, p in charts(bench):
        story.append(Image(str(p), width=17 * cm, height=17 * cm * (3.4 / 12 if "benchmark" in str(p) else 0.33)))
        P(cap, S["Italic"])
    if onnxb is not None:
        P("Table 2: ONNX Runtime CPU benchmark (results/onnx_benchmark.csv).")
        story.append(table(onnxb, ["model", "test_acc_pct", "file_size_mb", "latency_bs1_mean_ms", "throughput_bs128_img_s"],
                           ["Model", "Acc %", "Size MB", "Lat. ms", "img/s"]))
    gap()

    P("8. Analysis and discussion", H1)
    P("<i>Write your own interpretation here (edit this paragraph in src/generate_report.py or in the PDF source): "
      "why did QAT recover accuracy compared with PTQ? Why does unstructured pruning not speed up dense inference "
      "while structured pruning does? What trade-off would you choose for an edge device?</i>")

    shots = sorted(glob.glob(str(SHOTS / "*.png")) + glob.glob(str(SHOTS / "*.jpg")))
    P("9. Screenshots", H1)
    if shots:
        for s in shots:
            story.append(Image(s, width=16 * cm, height=9 * cm, kind="proportional"))
            P(os.path.basename(s), S["Italic"])
    else:
        P("[Add terminal screenshots of each script's output to report/screenshots/ and regenerate this report.]")

    P("10. Reproducibility", H1)
    P("Environment: see requirements.txt. Seeds are fixed (42). Run <i>python run_all.py</i> to regenerate all "
      "checkpoints, ONNX files, CSV results and this report (README.md lists each step).")

    SimpleDocTemplate(args.out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.8 * cm,
                      bottomMargin=1.8 * cm).build(story)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
