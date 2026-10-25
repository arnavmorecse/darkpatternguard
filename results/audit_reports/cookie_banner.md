# DarkPatternGuard - Audit Report
*Generated 25 Oct 2026, 22:26 · automated screening aid, not legal advice*

**Text analysed:** "We use cookies to personalise ads. Accept All | Ask me later"

## 1. Model output
| Stage | Result | Confidence |
|---|---|---|
| Stage 1 - detection | Not a dark pattern | 0.3% dark |

Backend: Fine-tuned RoBERTa-base

## 2. Overall risk: **Medium - review recommended** (score 3.01)


## 3. Rule findings
- **PRV-01 Asymmetric privacy/consent choice** (Privacy choice / consent) - evidence: Ask me later, Accept All
  - Reviewer question: Is declining as easy as accepting (e.g. 'Accept All' paired with 'Decline All')?


## 4. LIME explanation (top words)
| Word | Weight (towards "Dark Pattern") |
|---|---|
| personalise | -0.001 |
| Ask | +0.001 |
| Accept | +0.000 |
| We | +0.000 |
| later | +0.000 |
| ads | +0.000 |
| cookies | +0.000 |
| me | +0.000 |

## 5. Regulatory references for review
- **Cal. Civ. Code § 1798.140(l)** - CCPA as amended by CPRA *(relevant to Privacy choice / consent)*: Defines a dark pattern as a user interface designed or manipulated with the substantial effect of subverting or impairing user autonomy, decision-making, or choice. [source](https://www.cppa.ca.gov/pdf/enfadvisory202402.pdf)
- **Cal. Civ. Code § 1798.140(h)** - CCPA as amended by CPRA *(relevant to Privacy choice / consent)*: Agreement obtained through the use of dark patterns does not constitute consent. [source](https://www.cppa.ca.gov/pdf/enfadvisory202402.pdf)
- **Cal. Code Regs. tit. 11, § 7004(a)-(c)** - CCPA Regulations (CPPA) *(relevant to Privacy choice / consent)*: Methods for CCPA requests and consent must be easy to understand, offer symmetry in choice, avoid confusing language, avoid manipulative language or choice architecture, and be easy to execute; non-compliant methods may be dark patterns and consent obtained through them is invalid. [source](https://law.cornell.edu/regulations/california/11-CCR-7004)
- **Advisory 2024-02 (4 Sep 2024)** - CPPA Enforcement Advisory No. 2024-02 (non-binding guidance) *(relevant to Privacy choice / consent)*: Guidance on avoiding dark patterns in privacy requests and consent: clear language and symmetrical choices; effect, not intent, is what matters. [source](https://www.cppa.ca.gov/pdf/enfadvisory202402.pdf)

---
Screening references only - NOT legal advice. Verified against primary or official sources on 25 Oct 2026. CCPA/CPRA dark-pattern provisions govern privacy choices and consent; general shopping-pressure tactics fall under consumer-protection law instead. A text flagged here is a candidate for human review, never a finding of violation.