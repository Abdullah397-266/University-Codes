"""
Generate the technical report (PDF) from the code, the CSV/JSON results and the screenshots.
All numbers and charts are read from results/ - re-run this after any experiment to refresh the report.

    python src/generate_report.py --section B
Output: report/Technical_Report.pdf (+ charts in report/figures/)
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

from common import rel, LOGS_DIR, REPORT_DIR, RESULTS_DIR, STUDENTS, count_macs, count_params  # noqa: E402

FIG = REPORT_DIR / "figures"
SHOTS = REPORT_DIR / "screenshots"
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a2", "#0b0b0b", "#52514e"
NICE = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet 1.1", "efficientnet_lite0": "EfficientNet-Lite0"}

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6,
                    textColor=colors.HexColor("#1f3b5c"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor(MUTED))
CODE = ParagraphStyle("C", parent=BODY, fontName="Courier", fontSize=8.5, leading=11, leftIndent=10,
                      backColor=colors.HexColor("#f3f3f1"))
CENTER = ParagraphStyle("CT", parent=BODY, alignment=TA_CENTER)


# --------------------------------------------------------------------------- data helpers
def read_csv(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def read_json(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def fnum(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-" if v in (None, "") else str(v)


def table(rows, header, widths=None, font=8.5):
    t = Table([header] + rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font), ("FONT", (0, 1), (-1, -1), "Helvetica", font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b5c")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c8ccd2")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def img(path, width=16 * cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- charts
def chart_kd(comp):
    if not comp:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    x = range(len(comp))
    b = [float(r["baseline_acc"]) for r in comp]
    k = [float(r["kd_acc"]) for r in comp]
    ax.bar([i - 0.2 for i in x], b, 0.38, color=BLUE, label="Baseline (no KD)")
    ax.bar([i + 0.2 for i in x], k, 0.38, color=ORANGE, label="Distilled (KD)")
    for i, (bb, kk) in enumerate(zip(b, k)):
        ax.text(i - 0.2, bb, f"{bb:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.text(i + 0.2, kk, f"{kk:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    if comp[0].get("teacher_acc"):
        t = float(comp[0]["teacher_acc"])
        ax.axhline(t, color=MUTED, linestyle="--", linewidth=1)
        ax.text(len(comp) - 0.5, t, f"teacher ResNet-50 {t:.1f}%", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(list(x), [NICE.get(r["student"], r["student"]) for r in comp])
    ax.set_ylabel("CIFAR-100 test accuracy (%)", fontsize=8, color=MUTED)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    out = FIG / "kd_vs_baseline.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_curves():
    have = [s for s in STUDENTS if (RESULTS_DIR / f"history_{s}_baseline.csv").exists()
            and (RESULTS_DIR / f"history_{s}_kd.csv").exists()]
    if not have:
        return None
    fig, axes = plt.subplots(1, len(have), figsize=(7.2, 2.6), dpi=150, sharey=True, squeeze=False)
    for ax, s in zip(axes[0], have):
        for mode, col, lab in (("baseline", BLUE, "Baseline"), ("kd", ORANGE, "KD")):
            h = read_csv(f"history_{s}_{mode}.csv")
            ax.plot([int(r["epoch"]) for r in h], [float(r["eval_acc"]) for r in h], color=col, linewidth=2,
                    marker="o" if len(h) < 3 else None, label=lab)
        ax.set_title(NICE[s], fontsize=9, color=INK)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
        style_axes(ax)
    axes[0][0].set_ylabel("test accuracy (%)", fontsize=8, color=MUTED)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIG / "student_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def chart_nas(hist, best):
    if not hist:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=150)
    acc = [h for h in hist if h["status"] == "accepted"]
    rej = [h for h in hist if h["status"] != "accepted"]
    ax[0].scatter([float(h["latency_ms"]) for h in rej], [float(h["size_mb"]) for h in rej], s=22, color=GRAY,
                  label=f"rejected ({len(rej)})", edgecolors="white", linewidths=1)
    ax[0].scatter([float(h["latency_ms"]) for h in acc], [float(h["size_mb"]) for h in acc], s=26, color=BLUE,
                  label=f"accepted ({len(acc)})", edgecolors="white", linewidths=1)
    if best:
        c = best["constraints"]
        ax[0].axvline(c["max_latency_ms"], color=MUTED, linestyle="--", linewidth=1)
        ax[0].axhline(c["max_size_mb"], color=MUTED, linestyle="--", linewidth=1)
        ax[0].scatter([best["latency_ms"]], [best["size_mb"]], s=70, color=ORANGE, marker="*", label="best",
                      edgecolors="white", linewidths=1, zorder=5)
    ax[0].set_xlabel("CPU latency, 1 thread (ms)", fontsize=8, color=MUTED)
    ax[0].set_ylabel("model size (MB)", fontsize=8, color=MUTED)
    ax[0].set_title("Measured candidates vs. constraints", fontsize=9)
    ax[0].legend(frameon=False, fontsize=7)
    style_axes(ax[0])
    gens = sorted({int(h["generation"]) for h in acc})
    best_so_far, run = [], -1
    for g in gens:
        run = max([run] + [float(h["val_acc"]) for h in acc if int(h["generation"]) == g])
        best_so_far.append(run)
    ax[1].scatter([int(h["generation"]) for h in acc], [float(h["val_acc"]) for h in acc], s=18, color=BLUE,
                  alpha=0.6, label="candidate", edgecolors="white", linewidths=0.8)
    ax[1].plot(gens, best_so_far, color=ORANGE, linewidth=2, label="best so far")
    ax[1].set_xlabel("generation", fontsize=8, color=MUTED)
    ax[1].set_ylabel("validation accuracy (%)", fontsize=8, color=MUTED)
    ax[1].set_title("Search progress", fontsize=9)
    ax[1].legend(frameon=False, fontsize=7)
    style_axes(ax[1])
    fig.tight_layout()
    out = FIG / "nas_search.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- report
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="Abdullah Qadri")
    p.add_argument("--roll", default="23i-0089")
    p.add_argument("--section", default="B")
    p.add_argument("--repo", default="https://github.com/Abdullah397-266/University-Codes/tree/main/"
                                      "Edge%20computing/Assignment%204%20KD%2C%20NAS")
    args = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    synthetic = any("SYNTHETIC DATA" in f.read_text(errors="ignore") for f in LOGS_DIR.glob("*.log"))
    comp, bench = read_csv("kd_comparison.csv"), read_csv("benchmark_results.csv")
    hist, best = read_json("nas_history.json"), read_json("nas_best_architecture.json")
    final = read_json("nas_best_final.json")
    onnx_exp, onnx_b = read_csv("onnx_export.csv"), read_csv("onnx_benchmark.csv")
    kd_sum = read_csv("summary_kd.csv")

    s = []
    s += [Spacer(1, 1.5 * cm), Paragraph("<b>Edge Computing &amp; Intelligent Systems</b>", ParagraphStyle(
        "T", parent=CENTER, fontSize=18, leading=22)), Spacer(1, 6),
          Paragraph("Assignment 4 - Technical Report", ParagraphStyle("T2", parent=CENTER, fontSize=14, leading=18)),
          Spacer(1, 4), Paragraph("Knowledge Distillation (CIFAR-100) and Hardware-Aware Neural Architecture Search "
                                  "(CIFAR-10)", CENTER), Spacer(1, 0.8 * cm),
          Paragraph(f"{args.name} &nbsp;|&nbsp; {args.roll} &nbsp;|&nbsp; Section {args.section}", CENTER),
          Paragraph(date.today().strftime("%d %B %Y"), CENTER), Spacer(1, 0.4 * cm),
          Paragraph(f"GitHub repository: <link href='{args.repo}' color='#2a78d6'><u>{args.repo}</u></link>",
                    ParagraphStyle("RL", parent=CENTER, fontSize=8.5, leading=11)), Spacer(1, 0.6 * cm)]
    if synthetic:
        s.append(Paragraph("<font color='#c0392b'><b>Note:</b> the numbers in this version come from a pipeline smoke "
                           "test on synthetic random data (no CIFAR download), so accuracies are at chance level. "
                           "Re-run <i>python run_all.py</i> on the real datasets to regenerate every table, chart and "
                           "screenshot.</font>", BODY))
    s.append(Paragraph("<b>Contents.</b> 1 Overview - 2 Q1 Knowledge distillation (teacher, students, KD framework, "
                       "baseline, results) - 3 Q2 Hardware-aware NAS (search space, generator, evolutionary search, "
                       "constraints, results) - 4 Inference and ONNX deployment - 5 Benchmarking - 6 Screenshots - "
                       "7 Deliverables and how to run", BODY))

    # ---------------------------------------------------------------- 1 overview
    s += [Paragraph("1. Overview", H1), Paragraph(
        "The code base (<i>src/</i>) is a single PyTorch project with one shared utility module (<i>common.py</i>) "
        "that provides data loading, one training loop used by every model, evaluation and on-device CPU "
        "measurement (latency, storage size, peak memory, parameters, MACs). <i>run_all.py</i> executes the whole "
        "pipeline, saves each step's console output in <i>logs/</i>, renders it as a screenshot and regenerates this "
        "report from the CSV/JSON files in <i>results/</i>.", BODY)]

    # ---------------------------------------------------------------- 2 KD
    s += [Paragraph("2. Question 1 - Knowledge distillation", H1), Paragraph("2.1 Teacher and student networks", H2),
          Paragraph("All networks take 32x32 CIFAR images. ImageNet-style networks shrink the input 32x, which would "
                    "leave a 1x1 map; each model therefore removes its first two stride-2 operations, so every final "
                    "feature map is 4x4. This keeps teacher and student feature maps spatially aligned for feature "
                    "distillation.", BODY)]
    from models import build_model
    arch_rows, notes = [], {
        "resnet50": "torchvision bottleneck ResNet-50; 3x3 stride-1 stem, max-pool removed",
        "mobilenetv2": "torchvision MobileNetV2; stem and first stride-2 block set to stride 1",
        "squeezenet": "own SqueezeNet 1.1 (Fire modules) with BatchNorm, no ReLU on logits",
        "efficientnet_lite0": "own EfficientNet-Lite0: MBConv, no SE, ReLU6, fixed stem/head"}
    for a in ("resnet50",) + STUDENTS:
        m = build_model(a)
        arch_rows.append([NICE.get(a, "ResNet-50 (teacher)"), f"{count_params(m) / 1e6:.2f}",
                          f"{count_macs(m) / 1e6:.0f}", m.feature_channels, Paragraph(notes[a], SMALL)])
    s.append(table(arch_rows, ["Model", "Params (M)", "MMACs", "Feat. ch.", "Implementation"],
                   [3.2 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 8.3 * cm]))
    s += [Paragraph("2.2 Distillation framework (kd.py)", H2), Paragraph(
        "The <i>Distiller</i> wraps a frozen teacher (always in eval mode, no gradients) and a trainable student. "
        "For every batch both networks return their logits and last feature map, and <i>DistillationLoss</i> "
        "combines three terms:", BODY),
          Paragraph("L = (1 - alpha) * CE(z_s, y) + alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) "
                    "+ beta * MSE(r(F_s), F_t)", CODE),
          Paragraph("<b>Soft targets with temperature:</b> dividing logits by T &gt; 1 flattens the teacher's "
                    "distribution so that the relative probabilities of wrong classes (which classes the teacher "
                    "finds similar) carry signal; the T^2 factor keeps the gradient magnitude independent of T. "
                    "<b>Hard + soft:</b> alpha balances ground-truth cross-entropy against imitation of the teacher. "
                    "<b>Feature distillation (FitNets hint):</b> a learnable 1x1 conv + BN regressor r maps the "
                    "student's last feature map to the teacher's channel count (e.g. 1280 to 2048) and the MSE to the "
                    "teacher's layer-4 map is minimised; the regressor is used only during training.", BODY)]
    if kd_sum:
        r0 = kd_sum[0]
        s.append(Paragraph(f"Hyper-parameters used: alpha = {r0['alpha']}, T = {r0['temperature']}, "
                           f"beta = {r0['beta']}, {r0['epochs']} epochs.", BODY))
    s += [Paragraph("2.3 Training setup", H2), Paragraph(
        "Teacher, baseline students and distilled students share the same loop: SGD with Nesterov momentum 0.9, "
        "weight decay 5e-4, one-epoch linear warm-up then cosine decay, random-crop + horizontal-flip augmentation, "
        "mixed precision on GPU, gradient clipping at 5, and the best-test-accuracy checkpoint is kept. The baseline "
        "pipeline (<i>train_baseline.py</i>) and the KD pipeline (<i>train_kd.py</i>) use identical data, epochs, "
        "seed and optimiser, so the only difference is the loss.", BODY)]
    s.append(Paragraph("2.4 Results", H2))
    if comp:
        rows = [[NICE[r["student"]], fnum(r["baseline_acc"]), fnum(r["kd_acc"]),
                 f"{float(r['kd_gain_pts']):+.2f}", fnum(r["params_M"], 3), r.get("compression_x", "-"),
                 r.get("cpu_speedup_x", "-")] for r in comp]
        s.append(table(rows, ["Student", "Baseline acc %", "KD acc %", "Gain (pts)", "Params (M)",
                              "Smaller than teacher", "CPU speed-up"]))
        gains = [float(r["kd_gain_pts"]) for r in comp]
        best_r = comp[gains.index(max(gains))]
        s.append(Paragraph(f"Distillation changed test accuracy by {', '.join(f'{g:+.2f}' for g in gains)} points "
                           f"for {', '.join(NICE[r['student']] for r in comp)} respectively; the largest effect is on "
                           f"{NICE[best_r['student']]} ({float(best_r['kd_gain_pts']):+.2f} points).", BODY))
        for c in (chart_kd(comp), chart_curves()):
            if c:
                s.append(img(c, 15.5 * cm))
    else:
        s.append(Paragraph("Results not available yet - run the training and benchmark steps.", BODY))

    # ---------------------------------------------------------------- 3 NAS
    s += [PageBreak(), Paragraph("3. Question 2 - Hardware-aware Neural Architecture Search", H1),
          Paragraph("3.1 Search space (nas_search_space.py)", H2)]
    from nas_search_space import (ACTIVATIONS, BLOCKS, DEPTHS, DROPOUTS, EXPANDS, HEAD_CHANNELS, KERNELS,
                                  NUM_STAGES, RESOLUTIONS, STAGE_CHANNELS, STEM_CHANNELS)
    space = [["Input resolution", str(RESOLUTIONS)], ["Stem channels", str(STEM_CHANNELS)],
             ["Number of stages", str(NUM_STAGES)], ["Blocks per stage (depth)", str(DEPTHS)],
             ["Filters per stage", " / ".join(str(c) for c in STAGE_CHANNELS)], ["Kernel size", str(KERNELS)],
             ["Block type", ", ".join(BLOCKS) + "  (dwsep = depthwise separable)"],
             ["MBConv expansion", str(EXPANDS)], ["Activation", ", ".join(ACTIVATIONS)],
             ["Head channels", str(HEAD_CHANNELS)], ["Dropout", str(DROPOUTS)]]
    s.append(table(space, ["Dimension", "Choices"], [5 * cm, 11.5 * cm]))
    s += [Paragraph("Depth, filters, kernel, block type and activation are chosen independently for every stage. "
                    "<i>nas_generator.py</i> turns a configuration into a network: stem conv, stages (stride 2 at the "
                    "start of every stage after the first), 1x1 head conv, global average pooling, dropout and a "
                    "linear classifier; residual connections are inserted automatically wherever a block keeps its "
                    "shape.", BODY),
          Paragraph("3.2 Hardware-aware evolutionary search (nas_search.py, nas_hardware.py)", H2),
          Paragraph("(mu + lambda) evolution with elitism: a random initial population, then in each generation "
                    "parents are chosen by tournament selection, children are created by uniform crossover and "
                    "per-gene mutation, and the best <i>population</i> individuals survive. Before any training, each "
                    "candidate is built and measured on the CPU: median batch-1 latency with a fixed thread count, "
                    "serialized FP32 size, and peak inference memory from the PyTorch profiler's allocation trace "
                    "(weights + activations; recorded where the profiler supports it). Candidates that break a "
                    "constraint are rejected and logged without being trained. Accepted candidates are trained for a "
                    "few epochs on a CIFAR-10 training subset and scored on a validation split held out from the "
                    "training set (the test set is never used for selection). Fitness = val_acc x (latency / target)"
                    "^-0.07 (MnasNet soft reward), so faster networks win ties.", BODY)]
    if best:
        c, a = best["constraints"], best["search_args"]
        s.append(Paragraph(f"Constraints: latency &lt;= {c['max_latency_ms']} ms ({c['threads']} thread), size &lt;= "
                           f"{c['max_size_mb']} MB, peak memory &lt;= {c['max_peak_mem_mb']} MB. Search: population "
                           f"{a['population']}, {a['generations']} generations x {a['children']} children, "
                           f"{a['epochs_per_candidate']} epochs per candidate on {a['train_subset']} images, "
                           f"validation on {a['val_split']} images.", BODY))
        s += [Paragraph("3.3 Search results", H2), Paragraph(
            f"{best['evaluated']} architectures were measured; {best['accepted']} satisfied all constraints and were "
            f"trained, {best['rejected']} were rejected by the hardware check (search time "
            f"{best['search_time_min']} min). Best architecture:", BODY), Paragraph(best["summary"], CODE)]
        s.append(table([[fnum(best["val_acc"]), fnum(best["latency_ms"], 3), fnum(best["size_mb"], 3),
                         fnum(best["peak_mem_mb"], 2), f"{best['params'] / 1e3:.1f}", f"{best['macs'] / 1e6:.2f}"]],
                       ["Val acc %", "Latency ms", "Size MB", "Peak mem MB", "Params (K)", "MMACs"]))
        ch = chart_nas(hist, best)
        if ch:
            s.append(img(ch, 15.5 * cm))
        if hist:
            top = sorted([h for h in hist if h["status"] == "accepted"], key=lambda h: -float(h["fitness"]))[:5]
            s.append(Paragraph("Top-5 candidates by fitness:", BODY))
            s.append(table([[h["id"], h["generation"], fnum(h["val_acc"]), fnum(h["latency_ms"], 2),
                             fnum(h["size_mb"], 2), Paragraph(h["config_summary"], SMALL)] for h in top],
                           ["id", "gen", "val %", "ms", "MB", "architecture"],
                           [0.8 * cm, 0.9 * cm, 1.3 * cm, 1.2 * cm, 1.2 * cm, 11.1 * cm], font=7.5))
            rejs = [h for h in hist if h["status"] != "accepted"][:4]
            if rejs:
                s.append(Paragraph("Examples of rejected candidates: " + "; ".join(
                    f"#{h['id']} ({h['reject_reasons']})" for h in rejs) + ".", SMALL))
    if final:
        s += [Paragraph("3.4 Best architecture trained from scratch (nas_train_best.py)", H2), Paragraph(
            f"Re-initialised and trained on the full CIFAR-10 training set for {final['epochs']} epochs: test accuracy "
            f"<b>{final['test_acc']:.2f}%</b>, {final['params'] / 1e3:.1f} K parameters, {final['size_mb']:.3f} MB, "
            f"{final['latency_ms']:.2f} ms on one CPU thread.", BODY)]

    # ---------------------------------------------------------------- 4 deployment
    s += [Paragraph("4. Inference pipeline and ONNX deployment", H1), Paragraph(
        "<i>zoo.py</i> registers every checkpoint (teacher, 3 baseline students, 3 KD students, NAS model) together "
        "with its dataset, class names and input resolution, so <i>inference.py</i> runs any of them with the "
        "correct pre-processing. <i>export_onnx.py</i> exports the students and the NAS model (opset 17, dynamic "
        "batch), validates each graph with onnx.checker and compares ONNX Runtime outputs to PyTorch. "
        "<i>onnx_inference.py</i> runs the models with ONNX Runtime's CPUExecutionProvider using numpy-only "
        "pre-processing (no PyTorch needed on the device) and benchmarks latency and accuracy.", BODY)]
    if onnx_exp:
        s.append(table([[r["model"], fnum(r["onnx_size_mb"], 2), r["resolution"], f"{float(r['max_abs_diff']):.1e}",
                         r["top1_match"]] for r in onnx_exp],
                       ["Model", "ONNX MB", "Input px", "max |torch - ORT|", "Top-1 identical"]))
    if onnx_b:
        acc_key = [k for k in onnx_b[0] if k.startswith("accuracy_")][0]
        s += [Spacer(1, 6), table([[r["model"], fnum(r[acc_key]), fnum(r["latency_median_ms"], 3),
                                    fnum(r["latency_p95_ms"], 3), r["throughput_img_s"]] for r in onnx_b],
                                  ["Model (ONNX Runtime, CPU)", f"Acc % ({acc_key.split('_')[1]} imgs)",
                                   "Median ms", "p95 ms", "img/s"])]

    # ---------------------------------------------------------------- 5 benchmark
    s += [Paragraph("5. Benchmarking (benchmark.py)", H1)]
    if bench:
        s.append(table([[r["model"], fnum(r["test_acc"]), fnum(r["params_M"], 3), fnum(r["MMACs"], 1),
                         fnum(r["size_mb"], 2), fnum(r["latency_1t_ms"], 2), fnum(r["latency_allt_ms"], 2),
                         fnum(r["peak_mem_mb"], 1)] for r in bench],
                       ["Model", "Test acc %", "Params M", "MMACs", "Size MB", "1-thr ms", "all-thr ms",
                        "Peak MB"], font=8))
        s.append(Paragraph("Latency is the median of repeated batch-1 forward passes on the CPU after warm-up. "
                           "Teacher and students are evaluated on CIFAR-100, the NAS model on CIFAR-10.", SMALL))
    else:
        s.append(Paragraph("Run src/benchmark.py to fill this section.", BODY))

    # ---------------------------------------------------------------- 6 screenshots
    shots = sorted(SHOTS.glob("*.png"))
    if shots:
        s += [PageBreak(), Paragraph("6. Screenshots", H1),
              Paragraph("Terminal screenshots of every pipeline step from the actual run (training on the laptop GPU with CUDA, latency on the CPU). Figures are in pipeline order.", BODY)]
        for i, sh in enumerate(shots, 1):
            s.append(KeepTogether([img(sh, 17 * cm), Paragraph(f"Figure S{i}: {sh.stem.split('_', 1)[-1].replace('_', ' ')}", SMALL), Spacer(1, 6)]))

    # ---------------------------------------------------------------- 7 deliverables
    s += [PageBreak(), Paragraph("7. Deliverables and how to run", H1)]
    deliv = [["1 Teacher model", "src/train_teacher.py, models/resnet50_teacher.pth"],
             ["2 Student architectures", "src/models.py (MobileNetV2, SqueezeNet, EfficientNet-Lite0)"],
             ["3 Baseline students", "src/train_baseline.py, models/*_baseline.pth"],
             ["4 Knowledge distillation", "src/kd.py, src/train_kd.py, models/*_kd.pth"],
             ["5 NAS search space", "src/nas_search_space.py, src/nas_generator.py"],
             ["6 NAS algorithm", "src/nas_search.py (evolutionary), src/nas_hardware.py"],
             ["7 NAS results", "results/nas_history.csv/.json, results/nas_best_architecture.json"],
             ["8 Optimized architecture", "src/nas_train_best.py, models/nas_best.pth"],
             ["9 Inference pipeline", "src/inference.py, src/zoo.py"],
             ["10 ONNX deployment", "src/export_onnx.py, src/onnx_inference.py, models/onnx/*.onnx"],
             ["11 Benchmarking", "src/benchmark.py, results/benchmark_results.csv, kd_comparison.csv, "
                                 "onnx_benchmark.csv"],
             ["12 Technical report", "src/generate_report.py -> report/Technical_Report.pdf"],
             ["13 README", "README.md"]]
    s.append(table(deliv, ["Deliverable", "Files"], [4.5 * cm, 12 * cm]))
    s += [Spacer(1, 8), Paragraph("Complete pipeline:", BODY),
          Paragraph("python -m venv .venv &amp;&amp; .venv\\Scripts\\activate &amp;&amp; pip install -r requirements.txt"
                    "<br/>python run_all.py --quick &nbsp;&nbsp;# smoke test<br/>python run_all.py "
                    "--clean &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# full run, regenerates this report", CODE)]

    out = REPORT_DIR / "Technical_Report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Assignment 4 Technical Report", author=args.name)

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.1 * cm, f"{args.roll} - Section {args.section} - Assignment 4")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"Report written to {rel(out)}" + (" (smoke-test data)" if synthetic else ""))


if __name__ == "__main__":
    main()
