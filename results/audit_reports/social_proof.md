# DarkPatternGuard - Audit Report
*Generated 25 Oct 2026, 22:26 · automated screening aid, not legal advice*

**Text analysed:** "Michael in Pontiac, United States purchased a Classic Tee 4 minutes ago"

## 1. Model output
| Stage | Result | Confidence |
|---|---|---|
| Stage 1 - detection | Dark pattern | 99.9% dark |
| Stage 2 - category | Social Proof | 99.3% |

Backend: Fine-tuned RoBERTa-base

## 2. Overall risk: **Medium - review recommended** (score 5.0)
Model and rules agree on the category.

## 3. Rule findings
- **SOC-01 Activity or popularity notification** (Social Proof) - evidence: purchased a Classic Tee 4 minutes ago, in Pontiac, United States purchased
  - Reviewer question: Are the activity counts and testimonials genuine and current?


## 4. LIME explanation (top words)
| Word | Weight (towards "Dark Pattern") |
|---|---|
| purchased | +0.201 |
| minutes | +0.124 |
| 4 | +0.108 |
| ago | +0.095 |
| States | +0.078 |
| Classic | -0.070 |
| Michael | +0.052 |
| in | +0.048 |

## 5. Regulatory references for review
- **15 U.S.C. § 45 (Section 5: unfair or deceptive acts or practices)** - Federal Trade Commission Act *(relevant to Social Proof)*: The FTC's 2022 staff report 'Bringing Dark Patterns to Light' groups false scarcity claims and fake countdown timers under design elements that induce false beliefs. [source](https://www.ftc.gov/reports/bringing-dark-patterns-light)
- **Cal. Bus. & Prof. Code §§ 17200, 17500** - California Unfair Competition Law / False Advertising Law *(relevant to Social Proof)*: State-level prohibition of unfair, deceptive or misleading business practices and advertising; relevant if a scarcity, urgency or popularity claim is untrue. [source](https://leginfo.legislature.ca.gov/)

*CCPA/CPRA dark-pattern rules (Civ. Code § 1798.140(h),(l); 11 CCR § 7004) apply to privacy choices and consent. No privacy-choice indicator was found in this text.*

---
Screening references only - NOT legal advice. Verified against primary or official sources on 25 Oct 2026. CCPA/CPRA dark-pattern provisions govern privacy choices and consent; general shopping-pressure tactics fall under consumer-protection law instead. A text flagged here is a candidate for human review, never a finding of violation.