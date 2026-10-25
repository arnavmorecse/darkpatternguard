"""Shared constants, metric helpers and plotting style (warm academic palette)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BINARY_LABELS = ["Not Dark Pattern", "Dark Pattern"]
STAGE2_LABELS = ["Scarcity", "Social Proof", "Urgency", "Misdirection", "Other (rare)"]

PALETTE = {"ink": "#1C1F24", "muted": "#5A616B", "grid": "#E3E5E8", "grey": "#C4C9D0",
           "red": "#D6293E", "orange": "#E8710A", "mustard": "#C99A06", "violet": "#7A4FD0", "green": "#1E8E5A"}
CMAP = LinearSegmentedColormap.from_list("signal", ["#FFFFFF", "#F6C9CF", "#D6293E", "#8E1426"])
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": PALETTE["grid"], "axes.labelcolor": PALETTE["ink"], "axes.titleweight": "semibold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"], "axes.grid": False, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def compute_metrics(y_true, y_pred, proba, labels):
    """Accuracy, macro/weighted P/R/F1, per-class table, confusion matrix, ROC-AUC."""
    idx = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": float(np.mean(p)), "recall_macro": float(np.mean(r)),
        "f1_macro": f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=idx, average="weighted", zero_division=0),
        "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in idx},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
        "n_test": int(len(y_true)),
    }
    try:
        if len(labels) == 2:
            m["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        else:
            m["roc_auc_macro_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=idx)
    except ValueError as e:
        m["roc_auc_note"] = str(e)
    return m


def plot_confusion(cm, labels, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2.5, 1.0 * len(labels) + 2))
    im = ax.imshow(cm, cmap=CMAP)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    thr = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thr else PALETTE["ink"], fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.outline.set_visible(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
