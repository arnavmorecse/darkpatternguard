"""Compare the extension's decision rules on the shared Stage 1 test set (353 texts):
TF-IDF alone, RoBERTa alone, and the agreement rule min(TF-IDF, RoBERTa) used by server.py.
Writes results/metrics/agreement_rule.json."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from common import ROOT, save_json
from pipeline import TfidfBackend, RobertaBackend

t = pd.read_csv(ROOT / "data/splits/binary_test.csv")
X, y = t.raw_text.tolist(), t.label.values
pt = TfidfBackend().proba_binary(X)[:, 1]
pr = RobertaBackend().proba_binary(X)[:, 1]
rules = {"TF-IDF alone": pt, "RoBERTa alone": pr, "Agreement: min(TF-IDF, RoBERTa)": np.minimum(pt, pr)}
out = {}
for name, p in rules.items():
    for thr in (0.5, 0.8):
        yh = (p >= thr).astype(int)
        neg = y == 0
        out[f"{name} @ {thr}"] = {"accuracy": accuracy_score(y, yh), "precision_dark": precision_score(y, yh, zero_division=0),
                                  "recall_dark": recall_score(y, yh), "f1_macro": f1_score(y, yh, average="macro"),
                                  "false_alarm_rate": float(yh[neg].mean()), "false_alarms": int(yh[neg].sum()),
                                  "missed_dark": int(((yh == 0) & (y == 1)).sum())}
save_json(out, ROOT / "results/metrics/agreement_rule.json")
print(f"{'rule':42s} {'acc':>6s} {'prec':>6s} {'recall':>6s} {'FA':>4s} {'miss':>4s}")
for k, v in out.items():
    print(f"{k:42s} {v['accuracy']:.4f} {v['precision_dark']:.4f} {v['recall_dark']:.4f} {v['false_alarms']:4d} {v['missed_dark']:4d}")
