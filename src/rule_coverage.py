"""Rule coverage on the test split: how often rules fire on dark vs not-dark text."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
from common import ROOT, save_json
from audit_engine import match_rules
t = pd.read_csv(ROOT / "data/splits/binary_test.csv")
t["hits"] = t.raw_text.map(lambda x: [h["id"] for h in match_rules(x)])
t["any"] = t.hits.map(bool)
res = {"fire_rate_on_dark": float(t[t.label == 1]["any"].mean()),
       "fire_rate_on_not_dark": float(t[t.label == 0]["any"].mean()),
       "rule_precision_for_dark": float(t[t["any"]].label.mean()),
       "per_rule_fires": t.explode("hits").dropna(subset=["hits"]).groupby(["hits", "label"]).size().unstack(fill_value=0).to_dict("index")}
save_json(res, ROOT / "results/metrics/rule_coverage.json")
print(res)
print(t[(t.label == 0) & t["any"]][["raw_text", "hits"]].head(10).to_string())
