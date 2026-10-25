<p align="center"><img src="extension/icons/icon128.png" width="72" alt=""></p>

<h1 align="center">DarkPatternGuard</h1>

<p align="center">Finds manipulative interface text on websites, says what kind of trick it is, and shows why.</p>

<p align="center"><a href="https://darkpatternguard.streamlit.app"><b>Try it online →</b></a></p>

---

Countdown timers that reset, "only 2 left" banners, decline buttons that read *"No thanks, I don't want to save money"*,
cookie pop-ups where accepting takes one click and refusing takes five. These are **dark patterns**: design choices
that push people into decisions they wouldn't otherwise make.

DarkPatternGuard reads a piece of website text and answers three questions:

1. **Is this a dark pattern?** (Stage 1, yes or no)
2. **If so, which kind?** (Stage 2: Scarcity, Social Proof, Urgency, Misdirection, or Other)
3. **Why does the model think so?** (LIME word-level explanation, plus readable rules and the regulations a reviewer may want to check)

It comes as a **web app** for checking text or whole files, and a **browser extension** that highlights possible
dark patterns on any page as you browse.

> This is a screening aid for human review. A finding is not a legal conclusion.

## Results

All models are evaluated on the same held-out test set: 353 texts for Stage 1 (164 not dark, 189 dark) and the
189 dark texts for Stage 2.

| Model | Stage 1 accuracy | Stage 2 accuracy | Stage 2 macro-F1 | End-to-end accuracy |
|---|---|---|---|---|
| TF-IDF + logistic regression | 94.05% | 94.18% | 0.895 | 92.92% |
| RoBERTa-base, fine-tuned | **96.32%** | **96.30%** | **0.937** | **95.47%** |

*End-to-end* is the full six-way decision (not dark, or one of the five types) made by chaining both stages.
RoBERTa scores 96.60% at Stage 1 when run on a CPU instead of the Colab GPU; one prediction differs.

The 11 audit rules fire on 65% of dark test texts and on none of the 164 not-dark ones.

## How it works

```
website text
   │
   ├─▶ Stage 1  dark or not?           TF-IDF + LR  or  fine-tuned RoBERTa
   ├─▶ Stage 2  which kind?            (only if Stage 1 says dark)
   ├─▶ LIME     which words mattered?
   ├─▶ Rules    11 readable checks (countdowns, low-stock claims, confirmshaming, consent asymmetry, …)
   └─▶ Report   risk level + relevant regulations, as HTML or Markdown
```

**Data.** Two public datasets treated as one corpus: the
[ec-darkpattern dataset](https://github.com/yamanalab/ec-darkpattern) (2,356 texts) and the
[Mathur et al. (2019)](https://github.com/aruneshmathur/dark-patterns) dataset of dark patterns found on
11,000 shopping sites. Splits are 70 / 15 / 15, grouped on normalised text so that near-duplicates never land on
both sides of the split (seed 42).

**Stage 2 classes.** Obstruction, Sneaking and Forced Action have too few examples to learn on their own, so they
are merged into *Other*.

**RoBERTa.** `roberta-base`, fully fine-tuned on a Colab T4 GPU (AdamW, learning rate 2e-5, batch 16, 5 epochs,
max length 64, class-weighted loss for Stage 2, best epoch picked on validation macro-F1).

**Regulations.** Privacy-choice patterns are mapped to the CCPA/CPRA dark-pattern provisions
(Cal. Civ. Code § 1798.140(h), (l); 11 CCR § 7004). Shopping-pressure patterns are mapped to the FTC Act § 5,
California's UCL/FAL, SB 478 and the Automatic Renewal Law. Every reference has a source link in
`src/regulatory_mapping.json`.

## Run it locally (Windows)

Needs Python 3.10 or newer.

```bat
git clone https://github.com/arnavmorecse/darkpatternguard.git
cd darkpatternguard
RUN_DEMO.bat
```

`RUN_DEMO.bat` installs the packages, trains the TF-IDF models once with your library versions, and opens the app at
http://localhost:8501. On other systems:

```bash
pip install -r requirements-core.txt
python src/train_baselines.py
streamlit run app.py
```

**Adding RoBERTa.** The fine-tuned weights are about 480 MB, so they are not in the repository. Either download
`roberta_models.zip` from the [Releases](../../releases) page and unzip it into `models/` (you should end up with
`models/roberta/binary` and `models/roberta/category`), or train your own: open
`notebooks/02_RoBERTa_finetune_Colab.ipynb` in Colab, choose a T4 GPU, run all cells, upload `colab_upload.zip` when
asked, and unzip the downloaded `roberta_outputs.zip` here. Then run `pip install -r requirements-roberta.txt`.

**Reproducing the evaluation.**

```bat
python src\prepare_splits.py     &:: build the splits from data/raw
python src\train_baselines.py    &:: TF-IDF + LR, both stages
python src\evaluate.py           &:: end-to-end metrics, figures, sample audit reports
python src\eval_agreement.py     &:: the extension's "both models agree" rule
```

## Browser extension

The extension splits each page into text segments, sends them to a small model server on your own computer, and
outlines anything flagged. Hover a highlight to see the type and why it was flagged; the toolbar popup lists every
finding, and **Analyse** opens any of them in the web app for the full explanation.

1. Double-click `RUN_EXTENSION_SERVER.bat` and leave the window open.
2. **Chrome, Edge, Brave:** open `chrome://extensions`, turn on Developer mode, click **Load unpacked**, pick the `extension` folder.
3. **Firefox, Zen:** open `about:debugging#/runtime/this-firefox`, click **Load Temporary Add-on**, pick `extension/manifest.json`.

On a live page almost every segment is harmless, so the extension only flags text when **both** models agree
(the lower of the two scores must pass the threshold). On the test set that cuts false alarms from 2 to 1 at the
cost of some recall:

| Rule at 0.5 | Accuracy | False alarms | Missed dark |
|---|---|---|---|
| TF-IDF alone | 94.05% | 2 | 19 |
| RoBERTa alone | 96.60% | 2 | 10 |
| Both must agree (extension) | 93.77% | 1 | 21 |

Scanning happens on your computer. Text leaves it only when you press **Analyse** or **Open full report**, and then
only the text you chose goes to the web app.

## Deploying the web app

The live app at **https://darkpatternguard.streamlit.app** runs on [Streamlit Community Cloud](https://share.streamlit.io) straight from this repository:

1. Attach `roberta_models.zip` to a GitHub release (`MAKE_MODELS_ZIP.bat` builds it from `models/roberta`).
2. Create an app from this repo with `app.py` as the main file and Python 3.12.
3. Under *Secrets*, add `MODELS_URL = "<link to roberta_models.zip in the release>"`.

On first start the app downloads the weights and runs RoBERTa in half precision, which gave the same test-set
accuracy as full precision with about 600 MB less memory. Without `MODELS_URL` it falls back to TF-IDF.

## Project layout

```
app.py                       Streamlit web app
server.py                    local model server for the extension (127.0.0.1:8765)
extension/                   Chrome / Firefox extension
src/pipeline.py              DarkPatternGuard: both stages, LIME, audit
src/audit_engine.py          rules, risk level, report rendering
src/audit_rules.json         the 11 audit rules
src/regulatory_mapping.json  regulations with sources
src/prepare_splits.py        builds the train / validation / test splits
src/train_baselines.py       TF-IDF + logistic regression
src/evaluate.py              end-to-end evaluation, figures, sample reports
src/teammate_data_utils/     dataset loading, text normalisation, grouped splitting
notebooks/                   RoBERTa fine-tuning notebook for Colab
data/                        raw datasets and splits
models/tfidf/                trained TF-IDF models (RoBERTa goes in models/roberta/)
results/                     metrics (JSON, CSV), figures, sample audit reports
templates/                   audit report templates (HTML, Markdown)
```

## Limitations

- **Text only.** Visual tricks such as greyed-out buttons, tiny fine print or pre-ticked boxes are invisible to it.
- **Shopping-site data.** Other kinds of site (news, banking, social media) produce more false alarms.
- **Overconfidence.** RoBERTa can score harmless promotional text, such as a free-returns notice, as almost certainly dark. The extension's agreement rule exists for this reason.
- **Dataset bias.** 15 of the 19 texts in the data that mention "free shipping" are labelled dark, so the models lean that way.
- **Small classes.** *Other* has only 7 test examples, so its scores are unreliable.

## Team

NLP Lab ESE project, B.Tech Computer Science and Engineering, Symbiosis Institute of Technology, Pune.

- **Arnav More** ([@arnavmorecse](https://github.com/arnavmorecse))
- **Abhyudaya Dixit** ([@Abhyudaya1612](https://github.com/Abhyudaya1612)), dataset inspection and data utilities

## References

- Mathur, A. et al. (2019). *Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites.* Proc. ACM HCI 3 (CSCW).
- Yada, Y. et al. (2022). *Dark patterns in e-commerce: a dataset and its baseline evaluations.* IEEE BigData. ([ec-darkpattern](https://github.com/yamanalab/ec-darkpattern))
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- Ribeiro, M. T., Singh, S., Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD.
