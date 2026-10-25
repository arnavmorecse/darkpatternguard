"""Class-distribution figure for the report (Stage 1 and Stage 2, per split)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from common import ROOT, PALETTE, STAGE2_LABELS
S = ["train", "val", "test"]; col = [PALETTE["grey"], PALETTE["red"], PALETTE["ink"]]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
b = {s: pd.read_csv(ROOT / f"data/splits/binary_{s}.csv").label.value_counts().sort_index() for s in S}
c = {s: pd.read_csv(ROOT / f"data/splits/category_{s}.csv").category.value_counts().reindex(STAGE2_LABELS, fill_value=0) for s in S}
for k, (d, labels, a, t) in enumerate([(b, ["Not dark", "Dark"], ax[0], "Stage 1 - binary"),
                                       (c, STAGE2_LABELS, ax[1], "Stage 2 - category (dark texts only)")]):
    x = np.arange(len(labels))
    for i, s in enumerate(S):
        bars = a.bar(x + i * 0.27, d[s].values, 0.27, label=s, color=col[i], edgecolor="white")
        a.bar_label(bars, fontsize=7)
    a.set_xticks(x + 0.27, labels, rotation=15 if k else 0); a.set_title(t); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(ROOT / "results/figures/class_distribution.png", dpi=200)
print("saved")
