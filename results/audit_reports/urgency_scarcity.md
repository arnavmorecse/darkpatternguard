# DarkPatternGuard - Audit Report
*Generated 25 Oct 2026, 22:26 · automated screening aid, not legal advice*

**Text analysed:** "Hurry! Only 3 left in stock - offer ends in 02:14:55"

## 1. Model output
| Stage | Result | Confidence |
|---|---|---|
| Stage 1 - detection | Dark pattern | 99.9% dark |
| Stage 2 - category | Urgency | 58.7% |

Backend: Fine-tuned RoBERTa-base

## 2. Overall risk: **High - prioritise human review** (score 8.0)
Model and rules agree on the category.

## 3. Rule findings
- **URG-01 Deadline or countdown pressure** (Urgency) - evidence: offer ends, ends in, 02:14:55
  - Reviewer question: Is the deadline real, and does the offer actually end when stated?
- **SCA-01 Low-stock or high-demand claim** (Scarcity) - evidence: Only 3 left, 3 left in stock
  - Reviewer question: Does the stated stock level reflect real inventory?


## 4. LIME explanation (top words)
| Word | Weight (towards "Dark Pattern") |
|---|---|
| 55 | +0.018 |
| left | +0.017 |
| ends | +0.016 |
| 02 | +0.014 |
| 14 | +0.013 |
| Hurry | +0.011 |
| offer | +0.011 |
| Only | +0.011 |

## 5. Regulatory references for review
- **15 U.S.C. § 45 (Section 5: unfair or deceptive acts or practices)** - Federal Trade Commission Act *(relevant to Urgency)*: The FTC's 2022 staff report 'Bringing Dark Patterns to Light' groups false scarcity claims and fake countdown timers under design elements that induce false beliefs. [source](https://www.ftc.gov/reports/bringing-dark-patterns-light)
- **Cal. Bus. & Prof. Code §§ 17200, 17500** - California Unfair Competition Law / False Advertising Law *(relevant to Urgency)*: State-level prohibition of unfair, deceptive or misleading business practices and advertising; relevant if a scarcity, urgency or popularity claim is untrue. [source](https://leginfo.legislature.ca.gov/)

*CCPA/CPRA dark-pattern rules (Civ. Code § 1798.140(h),(l); 11 CCR § 7004) apply to privacy choices and consent. No privacy-choice indicator was found in this text.*

---
Screening references only - NOT legal advice. Verified against primary or official sources on 25 Oct 2026. CCPA/CPRA dark-pattern provisions govern privacy choices and consent; general shopping-pressure tactics fall under consumer-protection law instead. A text flagged here is a candidate for human review, never a finding of violation.