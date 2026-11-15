"""Build the summary figures shown in the README from results/*.json."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"
CLASS_COLORS = {"negative": "#2a78d6", "neutral": "#eb6834", "positive": "#1baf7a"}


def short_name(rec):
    key = rec["key"]
    if key.startswith("tfidf"):
        return "TF-IDF + logistic regression"
    if key.startswith("finetuned"):
        return "Fine-tuned XLM-RoBERTa"
    shots = rec.get("shots_per_class", 0)
    return "LLM, zero-shot" if shots == 0 else f"LLM, {shots} examples per class"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()
    records = [json.load(open(p, encoding="utf-8"))
               for p in glob.glob(os.path.join(args.results_dir, "*.json"))]
    records.sort(key=lambda r: r["macro_f1"])
    names = [short_name(r) for r in records]

    # Figure 1: macro-F1 per method (one measure, one hue, sorted)
    fig, ax = plt.subplots(figsize=(7.5, 3.2), facecolor=SURFACE)
    style(ax)
    values = [r["macro_f1"] for r in records]
    ax.barh(names, values, color=BLUE, height=0.55)
    for y, v in enumerate(values):
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, 1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlabel("Macro-F1 on the test set", color=MUTED, fontsize=10)
    ax.set_title("Fine-tuning beats the untrained LLM on Nepali sentiment", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig_macro_f1.png"), dpi=200)
    plt.close(fig)

    # Figure 2: per-class F1, grouped by method
    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor=SURFACE)
    style(ax)
    x, width = np.arange(len(records)), 0.24
    for i, cls in enumerate(CLASS_COLORS):
        vals = [r["per_class"][cls]["f1"] for r in records]
        bars = ax.bar(x + (i - 1) * (width + 0.02), vals, width, color=CLASS_COLORS[cls], label=cls)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                    color=INK, fontsize=8.5)
    ax.set_xticks(x, [n.replace(" + ", "\n+ ").replace(", ", ",\n") for n in names])
    ax.tick_params(axis="x", labelcolor=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("F1 on the test set", color=MUTED, fontsize=10)
    ax.set_title("The neutral class is the weakest for every method", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    ax.legend(title="Sentiment class", frameon=False, ncol=3, loc="upper left",
              labelcolor=INK, fontsize=9, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig_per_class_f1.png"), dpi=200)
    plt.close(fig)
    print("Figures written to", args.results_dir)


if __name__ == "__main__":
    main()
