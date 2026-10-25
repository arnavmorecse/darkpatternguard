"""Step 5 - The full DarkPatternGuard pipeline.

text -> Stage 1 (dark?) -> Stage 2 (category, only if dark) -> LIME -> audit report

Two interchangeable backends:
  * "roberta" - fine-tuned RoBERTa-base (models/roberta/{binary,category}, produced in Colab)
  * "tfidf"   - TF-IDF + Logistic Regression (models/tfidf/*.joblib, trains in seconds)
backend="auto" uses RoBERTa when its weights + torch are present, else TF-IDF.
"""
import sys
from pathlib import Path
import numpy as np
import joblib

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from common import ROOT, BINARY_LABELS, STAGE2_LABELS
from teammate_data_utils.preprocessor import preprocess_text
import audit_engine


class TfidfBackend:
    name = "TF-IDF + Logistic Regression"

    def __init__(self):
        self.m1 = joblib.load(ROOT / "models/tfidf/binary.joblib")
        self.m2 = joblib.load(ROOT / "models/tfidf/category.joblib")

    def proba_binary(self, texts):
        return self.m1.predict_proba([preprocess_text(t) for t in texts])

    def proba_category(self, texts):
        return self.m2.predict_proba([preprocess_text(t) for t in texts])


class RobertaBackend:
    name = "Fine-tuned RoBERTa-base"

    def __init__(self, max_len=64):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        d1, d2 = ROOT / "models/roberta/binary", ROOT / "models/roberta/category"
        self.tok = AutoTokenizer.from_pretrained(d1)
        # Online: keep the weights in half precision (the format they were saved in). On the test sets this gives
        # the same accuracy as float32 while using about 600 MB less memory. DPG_FP32=1 restores float32.
        import os
        dt = torch.float32 if os.environ.get("DPG_FP32") == "1" else torch.float16
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        self.m1 = AutoModelForSequenceClassification.from_pretrained(d1, dtype=dt).to(self.dev).eval()
        self.m2 = AutoModelForSequenceClassification.from_pretrained(d2, dtype=dt).to(self.dev).eval()
        self.max_len = max_len

    def _proba(self, model, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok([" ".join(str(t).split()) for t in texts[i:i + bs]], truncation=True,
                           max_length=self.max_len, padding=True, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                out.append(self.torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
        return np.vstack(out)

    def proba_binary(self, texts):
        return self._proba(self.m1, list(texts))

    def proba_category(self, texts):
        return self._proba(self.m2, list(texts))


def roberta_available():
    try:
        import torch, transformers  # noqa
    except ImportError:
        return False
    return (ROOT / "models/roberta/binary/config.json").exists() and (ROOT / "models/roberta/category/config.json").exists()


class DarkPatternGuard:
    def __init__(self, backend="auto", threshold=0.5):
        if backend == "auto":
            backend = "roberta" if roberta_available() else "tfidf"
        self.backend = RobertaBackend() if backend == "roberta" else TfidfBackend()
        self.threshold = threshold

    def predict(self, text):
        p1 = self.backend.proba_binary([text])[0]
        s1 = {"label": int(p1[1] >= self.threshold), "p_dark": float(p1[1]), "backend": self.backend.name}
        s2 = None
        if s1["label"] == 1:
            p2 = self.backend.proba_category([text])[0]
            k = int(np.argmax(p2))
            s2 = {"category": STAGE2_LABELS[k], "confidence": float(p2[k]),
                  "probs": {STAGE2_LABELS[i]: float(v) for i, v in enumerate(p2)}}
        return s1, s2

    def explain(self, text, num_features=8, num_samples=500, stage=1):
        """LIME (Ribeiro et al., 2016): perturb the text by removing words, fit a
        local linear model to the classifier's outputs, report word weights."""
        from lime.lime_text import LimeTextExplainer
        if stage == 1:
            fn, names, target = self.backend.proba_binary, BINARY_LABELS, 1
        else:
            fn, names = self.backend.proba_category, STAGE2_LABELS
            target = int(np.argmax(fn([text])[0]))
        exp = LimeTextExplainer(class_names=names, random_state=42).explain_instance(
            text, lambda xs: fn(list(xs)), labels=(target,), num_features=num_features, num_samples=num_samples)
        return {"class_name": names[target], "stage": stage,
                "weights": [(str(w), float(v)) for w, v in exp.as_list(label=target)], "_lime": exp}

    def analyse(self, text, with_lime=True, num_samples=500):
        s1, s2 = self.predict(text)
        expl = None
        if with_lime and len(text.split()) > 1:
            e = self.explain(text, num_samples=num_samples)
            expl = {k: v for k, v in e.items() if k != "_lime"}
        return audit_engine.audit(text, s1, s2, expl)


if __name__ == "__main__":
    g = DarkPatternGuard()
    for t in ["Hurry! Only 3 left in stock - offer ends in 02:14:55",
              "No thanks, I don't want to save money",
              "Free shipping on orders over $50",
              "We use cookies. Accept All | Ask me later",
              "Michael in Pontiac, United States purchased a Classic Tee 4 minutes ago"]:
        r = g.analyse(t, num_samples=300)
        print(f"\n{t}\n  -> dark={r['stage1']['p_dark']:.2f}  cat={r['stage2']['category'] if r['stage2'] else '-'}"
              f"  risk={r['risk_level']}  rules={[h['id'] for h in r['rule_hits']]}")
        print("  LIME:", r["explanation"]["weights"][:4] if r["explanation"] else None)
