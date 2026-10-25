"""Step 6 - End-to-end evaluation + comparison figures + sample audit reports.

End-to-end = run the WHOLE pipeline on the binary test set (353 texts) and score a
6-way outcome: Not Dark / Scarcity / Social Proof / Urgency / Misdirection / Other.
A dark text counts as correct only if Stage 1 fires AND Stage 2 picks the right class.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STAGE2_LABELS, PALETTE, compute_metrics, plot_confusion, save_json
from pipeline import DarkPatternGuard, RobertaBackend, TfidfBackend, roberta_available
import audit_engine

E2E = ["Not Dark"] + STAGE2_LABELS


def end_to_end(backend):
    t = pd.read_csv(ROOT / "data/splits/binary_test.csv")
    c = pd.read_csv(ROOT / "data/splits/category_test.csv").drop_duplicates("raw_text").set_index("raw_text")
    X = t.raw_text.tolist()
    p1 = backend.proba_binary(X); p2 = backend.proba_category(X)
    pred = np.where(p1[:, 1] >= 0.5, 1 + p2.argmax(1), 0)
    true = np.array([0 if l == 0 else 1 + int(c.loc[x, "category_id"]) for x, l in zip(X, t.label)])
    proba = np.hstack([p1[:, [0]], p1[:, [1]] * p2])  # P(not dark), P(dark & class k)
    m = compute_metrics(true, pred, proba, E2E)
    return m


def comparison_plot(rows, path):
    tasks = ["Stage 1", "Stage 2", "End-to-end"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = [PALETTE["grey"], PALETTE["red"], PALETTE["ink"]]
    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"), (axes[1], "f1_macro", "Macro-F1")]:
        models = list(rows)
        w = 0.8 / len(models)
        for i, mdl in enumerate(models):
            vals = [rows[mdl].get(t, {}).get(metric, np.nan) for t in tasks]
            b = ax.bar(np.arange(3) + i * w, vals, w, label=mdl, color=colors[i % 3], edgecolor="white")
            for x, v in zip(np.arange(3) + i * w, vals):
                if not np.isnan(v): ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(np.arange(3) + w * (len(models) - 1) / 2, tasks); ax.set_ylim(0.6, 1.02)
        ax.set_title(title); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    rows = {}
    backs = [("TF-IDF + LR", TfidfBackend())]
    if roberta_available():
        backs.append(("RoBERTa-base (fine-tuned)", RobertaBackend()))
    for name, b in backs:
        key = "tfidf" if "TF" in name else "roberta"
        m = end_to_end(b)
        if key == "roberta":
            # the official RoBERTa numbers come from the Colab GPU run; a CPU re-run can differ by a
            # prediction or two (fp16 weights run in fp32), so keep it separate instead of overwriting
            save_json(m, ROOT / "results/metrics/roberta_end_to_end_cpu.json")
            print(f"{name} (laptop CPU re-run): E2E acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f}")
            continue
        save_json(m, ROOT / f"results/metrics/{key}_end_to_end.json")
        plot_confusion(m["confusion_matrix"], E2E, f"{name} - end-to-end (test, n={m['n_test']})",
                       ROOT / f"results/figures/cm_{key}_end_to_end.png")
        s1 = json.loads((ROOT / f"results/metrics/{key}_binary.json").read_text(encoding="utf-8"))
        s2 = json.loads((ROOT / f"results/metrics/{key}_category.json").read_text(encoding="utf-8"))
        rows[name] = {"Stage 1": s1, "Stage 2": s2, "End-to-end": m}
        print(f"{name}: E2E acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f}")
    import runpy
    runpy.run_path(str(ROOT / "src/replot_figures.py"))   # all figures + summary table, TF-IDF and Colab RoBERTa
    # sample audit reports for the report / demo
    g = DarkPatternGuard()
    samples = {"urgency_scarcity": "Hurry! Only 3 left in stock - offer ends in 02:14:55",
               "confirmshaming": "No thanks, I don't want to save money",
               "social_proof": "Michael in Pontiac, United States purchased a Classic Tee 4 minutes ago",
               "cookie_banner": "We use cookies to personalise ads. Accept All | Ask me later",
               "benign": "Free returns within 30 days of delivery"}
    out = ROOT / "results/audit_reports"; out.mkdir(parents=True, exist_ok=True)
    for k, t in samples.items():
        r = g.analyse(t, num_samples=300)
        (out / f"{k}.md").write_text(audit_engine.render(r, "md"), encoding="utf-8")
        (out / f"{k}.html").write_text(audit_engine.render(r, "html"), encoding="utf-8")
    print("sample audit reports ->", out)


if __name__ == "__main__":
    main()
