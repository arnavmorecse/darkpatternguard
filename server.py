"""DarkPatternGuard local API - backend for the browser extension.

Run:  python server.py        (listens on http://127.0.0.1:8765, this computer only)

Hybrid "screen then confirm" design:
  1. TF-IDF + LR scores EVERY text segment on the page (milliseconds for hundreds).
  2. Segments TF-IDF finds even mildly suspicious (p >= SCREEN) are re-scored by the
     fine-tuned RoBERTa (if installed).
  3. Agreement rule: the final score is min(TF-IDF, RoBERTa), i.e. BOTH models must call it dark.
     On live pages almost every segment is benign, so false alarms matter more than on the
     balanced test set; fine-tuned RoBERTa is over-confident (~0.999) on some benign promo text.
  4. Stage 2 category (RoBERTa's when available) + the transparent audit rules are attached.
Page text never leaves this computer.
"""
import json, sys, time, re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np
from common import STAGE2_LABELS
from pipeline import TfidfBackend, RobertaBackend, roberta_available
from audit_engine import match_rules, risk_level

HOST, PORT = "127.0.0.1", 8765
SCREEN = 0.30      # TF-IDF probability above which RoBERTa double-checks
FLAG = 0.50        # final probability needed to flag a segment
MAX_SEGMENTS = 600
MAX_LEN = 400

print("Loading TF-IDF models...", flush=True)
TFIDF = TfidfBackend()
ROBERTA = None
if roberta_available():
    try:
        print("Loading fine-tuned RoBERTa (first load takes ~20 s)...", flush=True)
        ROBERTA = RobertaBackend()
    except Exception as e:  # torch missing / broken -> TF-IDF only
        print("RoBERTa unavailable, using TF-IDF only:", e, flush=True)
CACHE = {}


def classify(texts):
    out = [None] * len(texts)
    todo = [i for i, t in enumerate(texts) if t not in CACHE]
    if todo:
        X = [texts[i] for i in todo]
        p1 = TFIDF.proba_binary(X)[:, 1]
        p2 = TFIDF.proba_category(X)
        final_p, final_c, model = p1.copy(), p2.copy(), np.array(["tfidf"] * len(X), dtype=object)
        rob_p = [None] * len(X)
        if ROBERTA is not None:
            idx = [k for k in range(len(X)) if p1[k] >= SCREEN]
            if idx:
                sub = [X[k] for k in idx]
                rp1 = ROBERTA.proba_binary(sub)[:, 1]
                rp2 = ROBERTA.proba_category(sub)
                for j, k in enumerate(idx):
                    rob_p[k] = float(rp1[j])
                    final_p[k], final_c[k], model[k] = min(p1[k], rp1[j]), rp2[j], "both"
        for j, i in enumerate(todo):
            t = X[j]
            hits = match_rules(t)
            p = float(final_p[j])
            flagged = p >= FLAG or any(h.get("ccpa") for h in hits)
            k = int(np.argmax(final_c[j]))
            level, score = risk_level(p, hits)
            CACHE[t] = {
                "p_dark": round(p, 4), "flagged": bool(flagged), "model": model[j],
                "p_tfidf": round(float(p1[j]), 4), "p_roberta": None if rob_p[j] is None else round(rob_p[j], 4),
                # privacy/consent wording is outside the shopping training data, so the rule's label wins
                "category": ("Privacy choice / consent" if any(h.get("ccpa") for h in hits) else
                             STAGE2_LABELS[k] if p >= FLAG else (hits[0]["category"] if hits else None)),
                "category_conf": round(float(final_c[j][k]), 3) if p >= FLAG else None,
                "risk": level, "rules": [{"id": h["id"], "name": h["name"], "evidence": h["evidence"],
                                          "question": h["question"]} for h in hits],
            }
    for i, t in enumerate(texts):
        out[i] = CACHE[t]
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"ok": True, "backend": "TF-IDF + RoBERTa (both must agree)" if ROBERTA else "TF-IDF only",
                             "roberta": ROBERTA is not None, "threshold": FLAG})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/classify"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            texts = [re.sub(r"\s+", " ", str(t)).strip()[:MAX_LEN] for t in data.get("texts", [])][:MAX_SEGMENTS]
            t0 = time.time()
            res = classify(texts)
            self._send(200, {"results": res, "ms": int(1000 * (time.time() - t0)),
                             "roberta": ROBERTA is not None})
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        if "POST" in (args[0] if args else ""):
            sys.stdout.write("  %s\n" % (fmt % args))


if __name__ == "__main__":
    print(f"DarkPatternGuard API ready on http://{HOST}:{PORT}  "
          f"({'TF-IDF + RoBERTa, both must agree' if ROBERTA else 'TF-IDF only'})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
