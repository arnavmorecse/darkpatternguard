# DarkPatternGuard - Audit Report
*Generated 25 Oct 2026, 22:26 · automated screening aid, not legal advice*

**Text analysed:** "No thanks, I don't want to save money"

## 1. Model output
| Stage | Result | Confidence |
|---|---|---|
| Stage 1 - detection | Dark pattern | 99.9% dark |
| Stage 2 - category | Misdirection | 99.1% |

Backend: Fine-tuned RoBERTa-base

## 2. Overall risk: **Medium - review recommended** (score 7.0)
Model and rules agree on the category.

## 3. Rule findings
- **MIS-01 Confirmshaming / guilt-laden decline option** (Misdirection) - evidence: No thanks, I, I don't want
  - Reviewer question: Does the decline option use guilt or shame to steer the choice?


## 4. LIME explanation (top words)
| Word | Weight (towards "Dark Pattern") |
|---|---|
| save | +0.098 |
| No | +0.086 |
| want | +0.084 |
| I | +0.061 |
| money | +0.059 |
| don | +0.057 |
| thanks | +0.044 |
| t | +0.017 |

## 5. Regulatory references for review
- **15 U.S.C. § 45 (Section 5: unfair or deceptive acts or practices)** - Federal Trade Commission Act *(relevant to Misdirection)*: The FTC's 2022 staff report 'Bringing Dark Patterns to Light' groups false scarcity claims and fake countdown timers under design elements that induce false beliefs. [source](https://www.ftc.gov/reports/bringing-dark-patterns-light)
- **Cal. Code Regs. tit. 11, § 7004(a)-(c)** - CCPA Regulations (CPPA) *(relevant to Misdirection)*: Methods for CCPA requests and consent must be easy to understand, offer symmetry in choice, avoid confusing language, avoid manipulative language or choice architecture, and be easy to execute; non-compliant methods may be dark patterns and consent obtained through them is invalid. [source](https://law.cornell.edu/regulations/california/11-CCR-7004)

*CCPA/CPRA dark-pattern rules (Civ. Code § 1798.140(h),(l); 11 CCR § 7004) apply to privacy choices and consent. No privacy-choice indicator was found in this text.*

---
Screening references only - NOT legal advice. Verified against primary or official sources on 25 Oct 2026. CCPA/CPRA dark-pattern provisions govern privacy choices and consent; general shopping-pressure tactics fall under consumer-protection law instead. A text flagged here is a candidate for human review, never a finding of violation.