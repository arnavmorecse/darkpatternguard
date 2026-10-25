"""Step 4 - Rule-based audit engine.

Combines (a) the model's Stage 1 probability and Stage 2 category with
(b) transparent regex rules, and maps findings to verified regulatory
references. Output wording is deliberately cautious: a 'finding' is a
candidate for human review, never a legal conclusion.
"""
import json, re
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
RULES = json.loads((ROOT / "src/audit_rules.json").read_text(encoding="utf-8"))
REG = json.loads((ROOT / "src/regulatory_mapping.json").read_text(encoding="utf-8"))
for r in RULES:
    r["_compiled"] = [re.compile(p, re.IGNORECASE) for p in r["patterns"]]


def match_rules(text):
    hits = []
    for r in RULES:
        spans = [m.group(0) for c in r["_compiled"] for m in [c.search(text)] if m]
        if spans:
            hits.append({"id": r["id"], "name": r["name"], "category": r["category"],
                         "weight": r["weight"], "evidence": spans[:3], "question": r["question"],
                         "ccpa": r.get("ccpa", False)})
    return hits


def risk_level(p_dark, hits):
    """Transparent scoring: model probability (0-4 pts) + rule weights."""
    score = 4 * p_dark + sum(h["weight"] for h in hits)
    if p_dark < 0.5 and not hits:
        return "No indicators", score
    if score >= 7:
        return "High - prioritise human review", score
    if any(h.get("ccpa") for h in hits):  # privacy-choice indicators are reviewed even if the
        return "Medium - review recommended", score  # shopping-trained model scores them low
    if score >= 4:
        return "Medium - review recommended", score
    return "Low - minor indicators", score


def audit(text, stage1, stage2=None, explanation=None):
    """stage1 = {'label','p_dark'}; stage2 = {'category','confidence','probs'} or None."""
    hits = match_rules(text)
    level, score = risk_level(stage1["p_dark"], hits)
    cats = []
    if stage2 and stage1["label"] == 1:
        cats.append(stage2["category"])
    cats += [h["category"] for h in hits if h["category"] not in cats]
    refs, seen = [], set()
    for c in cats:
        for k in REG["category_to_frameworks"].get(c, []):
            if k not in seen:
                seen.add(k); refs.append({"key": k, "for_category": c, **REG["frameworks"][k]})
    agreement = None
    if stage2 and stage1["label"] == 1 and hits:
        agreement = any(h["category"] == stage2["category"] for h in hits)
    return {
        "text": text, "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "stage1": stage1, "stage2": stage2 if stage1["label"] == 1 else None,
        "rule_hits": hits, "risk_level": level, "risk_score": round(score, 2),
        "model_rule_agreement": agreement, "regulatory_references": refs,
        "ccpa_relevant": any(h["ccpa"] for h in hits),
        "explanation": explanation, "disclaimer": REG["_note"],
    }


_env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=select_autoescape(["html"]))


def render(report, fmt="md"):
    w = (report.get("explanation") or {}).get("weights") or []
    lime_max = max((abs(v) for _, v in w), default=0)   # scales the LIME bars in the HTML report
    return _env.get_template(f"audit_report.{fmt}.j2").render(r=report, lime_max=lime_max)
