"""
Render the console logs written by run_all.py (logs/*.log) as terminal-style PNG screenshots in
report/screenshots/, so every pipeline step has an image of its real console output for the report.
Each log starts with the command that produced it. Long logs keep the first lines (banner/settings)
and the last lines (final results), with a marker showing how many lines were skipped.

    python src/make_screenshots.py                # all logs
    python src/make_screenshots.py --max-lines 50
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import LOGS_DIR, REPORT_DIR, rel  # noqa: E402

BG, FG, PROMPT, DIM = "#0c0c0c", "#cccccc", "#3a96dd", "#767676"


def render(log: Path, out: Path, max_lines: int, width: int) -> None:
    raw = log.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = []
    for ln in raw:
        lines += textwrap.wrap(ln, width, subsequent_indent="    ") or [""]
    if len(lines) > max_lines:
        head, tail = 12, max_lines - 13
        lines = lines[:head] + [f"... ({len(lines) - head - tail} lines omitted, full output in logs/{log.name}) ..."] \
            + lines[-tail:]
    h = 0.6 + 0.19 * len(lines)
    fig = plt.figure(figsize=(13.5, h), dpi=110)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(BG)
    y = 1 - 0.3 / h
    step = 0.19 / h
    for ln in lines:
        color = PROMPT if ln.startswith("PS ") or ln.startswith("$ ") else (DIM if ln.startswith("...") else FG)
        ax.text(0.01, y, ln, family="DejaVu Sans Mono", fontsize=9, color=color, va="top", transform=ax.transAxes)
        y -= step
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default=str(LOGS_DIR))
    p.add_argument("--out", default=str(REPORT_DIR / "screenshots"))
    p.add_argument("--max-lines", type=int, default=46)
    p.add_argument("--width", type=int, default=150)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs = sorted(Path(args.logs).glob("*.log"))
    for log in logs:
        render(log, out / f"{log.stem}.png", args.max_lines, args.width)
        print(f"{log.name} -> {rel(out / (log.stem + '.png'))}")
    print(f"{len(logs)} screenshots written")


if __name__ == "__main__":
    main()
