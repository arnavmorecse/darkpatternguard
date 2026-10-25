"""Step 2 - TF-IDF + Logistic Regression baselines for Stage 1 and Stage 2.

* Vectoriser = word 1-2 grams + character 2-5 grams (char n-grams catch "0nly", "4 left" etc.)
* Hyper-parameter C chosen on the VALIDATION split by macro-F1; test is touched once.
* class_weight="balanced" so rare classes are not ignored.
"""
import sys, joblib
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import ROOT, SEED, BINARY_LABELS, STAGE2_LABELS, compute_metrics, plot_confusion, save_json


def make_pipeline(C):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    return Pipeline([("tfidf", feats), ("clf", clf)])


def run(stage, label_col, labels):
    d = {s: pd.read_csv(ROOT / f"data/splits/{stage}_{s}.csv") for s in ["train", "val", "test"]}
    search = {}
    for C in [0.25, 0.5, 1, 2, 4, 8, 16, 32]:
        pipe = make_pipeline(C).fit(d["train"].text, d["train"][label_col])
        search[C] = f1_score(d["val"][label_col], pipe.predict(d["val"].text), average="macro")
    best_C = max(search, key=search.get)
    pipe = make_pipeline(best_C).fit(d["train"].text, d["train"][label_col])
    t = d["test"]
    m = compute_metrics(t[label_col].values, pipe.predict(t.text), pipe.predict_proba(t.text), labels)
    m.update({"model": "TF-IDF (word 1-2 + char 2-5) + LogisticRegression", "best_C": best_C,
              "val_macro_f1_by_C": {str(k): v for k, v in search.items()},
              "n_train": len(d["train"]), "n_val": len(d["val"])})
    (ROOT / "models/tfidf").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ROOT / f"models/tfidf/{stage}.joblib")
    save_json(m, ROOT / f"results/metrics/tfidf_{stage}.json")
    plot_confusion(m["confusion_matrix"], labels, f"TF-IDF + LR - Stage {'1' if stage=='binary' else '2'} (test)",
                   ROOT / f"results/figures/cm_tfidf_{stage}.png")
    print(f"[{stage}] best C={best_C}  acc={m['accuracy']:.4f}  macroF1={m['f1_macro']:.4f}  "
          f"weightedF1={m['f1_weighted']:.4f}  AUC={m.get('roc_auc', m.get('roc_auc_macro_ovr'))}")
    for k, v in m["per_class"].items():
        print(f"     {k:18s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")


if __name__ == "__main__":
    run("binary", "label", BINARY_LABELS)
    run("category", "category_id", STAGE2_LABELS)
