"""Redraw every report figure from the saved metrics (no retraining, no GPU needed).
Covers TF-IDF and the Colab-trained RoBERTa: confusion matrices, model comparison, summary table,
RoBERTa learning curves, LIME examples and the class distribution.   Run: python src/replot_figures.py"""
import sys, json, runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import ROOT, BINARY_LABELS, STAGE2_LABELS, PALETTE, plot_confusion

M, F = ROOT / "results/metrics", ROOT / "results/figures"
F.mkdir(parents=True, exist_ok=True)
NAMES = {"tfidf": "TF-IDF + LR", "roberta": "RoBERTa-base (fine-tuned)"}
TASKS = {"binary": ("Stage 1", BINARY_LABELS), "category": ("Stage 2", STAGE2_LABELS),
         "end_to_end": ("End-to-end", ["Not Dark"] + STAGE2_LABELS)}
load = lambda p: json.loads(p.read_text(encoding="utf-8"))

rows, table = {}, []
for key, name in NAMES.items():
    for task, (title, labels) in TASKS.items():
        p = M / f"{key}_{task}.json"
        if not p.exists():
            continue
        m = load(p)
        plot_confusion(m["confusion_matrix"], labels, f"{name}: {title} (test set, n={m.get('n_test', sum(map(sum, m['confusion_matrix'])))})",
                       F / f"cm_{key}_{task}.png")
        rows.setdefault(name, {})[title] = m
        table.append({"model": name, "task": title, "accuracy": round(m["accuracy"], 4), "f1_macro": round(m["f1_macro"], 4),
                      "f1_weighted": round(m["f1_weighted"], 4),
                      "roc_auc": round(m.get("roc_auc") or m.get("roc_auc_macro_ovr") or float("nan"), 4)})
pd.DataFrame(table).to_csv(M / "summary_table.csv", index=False)
print(pd.DataFrame(table).to_string(index=False))

# model comparison
tasks = ["Stage 1", "Stage 2", "End-to-end"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
colours = [PALETTE["grey"], PALETTE["red"]]
for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "f1_macro", "Macro-F1")]:
    w = 0.38
    for i, (name, r) in enumerate(rows.items()):
        vals = [r.get(t, {}).get(metric, np.nan) for t in tasks]
        xs = np.arange(3) + (i - (len(rows) - 1) / 2) * w
        ax.bar(xs, vals, w, label=name, color=colours[i % 2])
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=PALETTE["ink"])
    ax.set_xticks(np.arange(3), tasks); ax.set_ylim(0.6, 1.03); ax.set_title(title)
    ax.yaxis.grid(True, color=PALETTE["grid"]); ax.set_axisbelow(True)
axes[0].legend(loc="lower left", fontsize=9)
fig.tight_layout(); fig.savefig(F / "model_comparison.png", dpi=200); plt.close(fig)

# RoBERTa learning curves (saved by the Colab notebook)
hist = {t: load(M / f"roberta_{t}.json").get("history") for t in ["binary", "category"] if (M / f"roberta_{t}.json").exists()}
if hist:
    fig, axes = plt.subplots(1, len(hist), figsize=(5 * len(hist), 3.4))
    axes = np.atleast_1d(axes)
    for ax, (t, h) in zip(axes, hist.items()):
        e = [x["epoch"] for x in h]
        ax.plot(e, [x["train_loss"] for x in h], "o-", color=PALETTE["grey"], label="Training loss")
        ax2 = ax.twinx(); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(PALETTE["grid"])
        ax2.plot(e, [x["val_macro_f1"] for x in h], "s-", color=PALETTE["red"], label="Validation macro-F1")
        ax.set_title(TASKS[t][0]); ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss"); ax2.set_ylabel("Validation macro-F1")
        ax.set_xticks(e)
        lines = ax.get_lines() + ax2.get_lines(); ax.legend(lines, [l.get_label() for l in lines], fontsize=8.5, loc="center right")
    fig.tight_layout(); fig.savefig(F / "roberta_learning_curves.png", dpi=200); plt.close(fig)

# LIME examples (word weights saved by the Colab notebook)
lp = M / "roberta_lime_examples.json"
if lp.exists():
    for i, ex in enumerate(load(lp), 1):
        w = sorted(ex["weights"], key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(6, 0.36 * len(w) + 1.1))
        ax.barh([x[0] for x in w], [x[1] for x in w], color=[PALETTE["red"] if x[1] > 0 else PALETTE["grey"] for x in w])
        ax.axvline(0, color=PALETTE["muted"], lw=0.8)
        txt = ex["text"] if len(ex["text"]) <= 60 else ex["text"][:57] + "..."
        ax.set_title(f"“{txt}”  ({100 * ex['p_dark']:.0f}% dark)", fontsize=10)
        ax.set_xlabel("LIME weight (red pushes towards dark pattern)")
        fig.tight_layout(); fig.savefig(F / f"lime_roberta_{i}.png", dpi=200); plt.close(fig)

runpy.run_path(str(ROOT / "src/plot_data.py"))   # class distribution
print("figures written to", F)
