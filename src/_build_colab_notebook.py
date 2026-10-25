import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# DarkPatternGuard – RoBERTa-base fine-tuning (Stage 1 + Stage 2)
**Run on Google Colab with a GPU:** Runtime → Change runtime type → **T4 GPU** → Save. Then Runtime → **Run all**.

What this notebook does:
1. Loads the leakage-checked splits from `colab_upload.zip` (same splits as the TF-IDF baselines).
2. Fine-tunes **RoBERTa-base** end-to-end for Stage 1 (dark / not dark) and Stage 2 (5 classes).
3. Picks the best epoch on the **validation** set by macro-F1; the test set is used once.
4. Saves metrics, confusion matrices, LIME explanations, and the trained models.
5. Packs everything into `roberta_outputs.zip` to download and unzip into the project folder.

Expected time on a T4: about 10–15 minutes in total.
""")
code("""
!pip -q install lime
import torch, transformers, sys, platform
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch the runtime to T4 GPU!")
print("torch", torch.__version__, "| transformers", transformers.__version__, "| python", platform.python_version())
""")
md("## 1. Upload the data\nWhen the upload button appears, choose **`colab_upload.zip`** from the `dark_pattern_guard_v2` folder.")
code("""
import os, zipfile
if not os.path.exists("data/splits/binary_train.csv"):
    from google.colab import files
    up = files.upload()
    zipfile.ZipFile(next(iter(up))).extractall(".")
for f in sorted(os.listdir("data/splits")): print(f)
""")
md("## 2. Settings, metrics and plotting helpers")
code("""
import json, random, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             roc_auc_score, confusion_matrix)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SEED = 42
MODEL_NAME = "roberta-base"
MAX_LEN = 64          # 99%+ of UI strings are shorter than this (checked below)
BATCH = 16
LR = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP = 0.1
BINARY_LABELS = ["Not Dark Pattern", "Dark Pattern"]
STAGE2_LABELS = ["Scarcity", "Social Proof", "Urgency", "Misdirection", "Other (rare)"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
for d in ["models/roberta/binary", "models/roberta/category", "results/metrics", "results/figures"]:
    Path(d).mkdir(parents=True, exist_ok=True)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

CMAP = LinearSegmentedColormap.from_list("warm", ["#FFFFFF", "#F6C9CF", "#D6293E", "#8E1426"])

def compute_metrics(y_true, y_pred, proba, labels):
    idx = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "precision_macro": float(np.mean(p)), "recall_macro": float(np.mean(r)),
         "f1_macro": f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0),
         "f1_weighted": f1_score(y_true, y_pred, labels=idx, average="weighted", zero_division=0),
         "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in idx},
         "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(), "n_test": int(len(y_true))}
    try:
        m["roc_auc"] = roc_auc_score(y_true, proba[:, 1]) if len(labels) == 2 else None
        if len(labels) > 2:
            m["roc_auc_macro_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=idx); m.pop("roc_auc")
    except ValueError as e:
        m["roc_auc_note"] = str(e)
    return m

def plot_confusion(cm, labels, title, path):
    cm = np.array(cm); fig, ax = plt.subplots(figsize=(1.2*len(labels)+2.5, 1.0*len(labels)+2))
    im = ax.imshow(cm, cmap=CMAP)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right"); ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "#1C1F24")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=200); plt.show()

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
lens = [len(tok(t)["input_ids"]) for t in pd.read_csv("data/splits/binary_train.csv").raw_text]
print(f"token length: median {np.median(lens):.0f}, 99th pct {np.percentile(lens, 99):.0f}, max {max(lens)}  (MAX_LEN={MAX_LEN})")
""")
md("## 3. Training loop\nPlain PyTorch loop (robust across `transformers` versions). AdamW + linear warm-up/decay, mixed precision, best epoch kept by validation macro-F1. Stage 2 uses class-weighted loss so the rare bucket is not ignored.")
code("""
def batches(texts, labels, shuffle):
    data = list(zip(texts, labels))
    def collate(b):
        enc = tok([x for x, _ in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in b]); return enc
    return DataLoader(data, batch_size=BATCH, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(SEED))

@torch.no_grad()
def predict_proba(model, texts, bs=64):
    model.eval(); out = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i+bs]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEV)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.vstack(out)

def train_stage(stage, label_col, labels, class_weighted):
    set_seed()
    d = {s: pd.read_csv(f"data/splits/{stage}_{s}.csv") for s in ["train", "val", "test"]}
    ytr = d["train"][label_col].astype(int).values
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels),
              id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)}, ignore_mismatched_sizes=True).to(DEV)
    w = None
    if class_weighted:
        cnt = np.bincount(ytr, minlength=len(labels))
        w = torch.tensor(len(ytr) / (len(labels) * cnt), dtype=torch.float).to(DEV)
        print("class weights:", dict(zip(labels, w.cpu().numpy().round(2))))
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    dl = batches(d["train"].raw_text.tolist(), ytr.tolist(), True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = len(dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    best, history, t0 = -1, [], time.time()
    out_dir = f"models/roberta/{stage}"
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0
        for b in dl:
            b = {k: v.to(DEV) for k, v in b.items()}; y = b.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                loss = loss_fn(model(**b).logits.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
        pv = predict_proba(model, d["val"].raw_text.tolist())
        vf1 = f1_score(d["val"][label_col], pv.argmax(1), average="macro")
        history.append({"epoch": ep, "train_loss": tot / len(dl), "val_macro_f1": vf1})
        print(f"[{stage}] epoch {ep}: train loss {tot/len(dl):.4f} | val macro-F1 {vf1:.4f}")
        if vf1 > best:
            best = vf1; model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    train_min = (time.time() - t0) / 60
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(DEV)   # best checkpoint
    pt = predict_proba(model, d["test"].raw_text.tolist())
    m = compute_metrics(d["test"][label_col].astype(int).values, pt.argmax(1), pt, labels)
    m.update({"model": "RoBERTa-base fine-tuned", "best_val_macro_f1": best, "history": history,
              "hyperparameters": {"lr": LR, "batch": BATCH, "epochs": EPOCHS, "max_len": MAX_LEN, "weight_decay": WEIGHT_DECAY,
                                  "warmup_ratio": WARMUP, "class_weighted_loss": class_weighted, "seed": SEED},
              "train_minutes": round(train_min, 1), "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
              "n_train": len(d["train"]), "n_val": len(d["val"])})
    json.dump(m, open(f"results/metrics/roberta_{stage}.json", "w"), indent=2, default=float)
    plot_confusion(m["confusion_matrix"], labels, f"RoBERTa-base - Stage {'1' if stage=='binary' else '2'} (test)",
                   f"results/figures/cm_roberta_{stage}.png")
    print(f"TEST  acc={m['accuracy']:.4f}  macro-F1={m['f1_macro']:.4f}  weighted-F1={m['f1_weighted']:.4f}")
    for k, v in m["per_class"].items():
        print(f"   {k:18s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
    return model, history
""")
md("## 4. Stage 1 – binary detection")
code("""m1, h1 = train_stage("binary", "label", BINARY_LABELS, class_weighted=False)""")
md("## 5. Stage 2 – category classification (5 classes)")
code("""m2, h2 = train_stage("category", "category_id", STAGE2_LABELS, class_weighted=True)""")
md("## 6. Learning curves")
code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, h, t in [(axes[0], h1, "Stage 1"), (axes[1], h2, "Stage 2")]:
    e = [x["epoch"] for x in h]
    ax.plot(e, [x["train_loss"] for x in h], "o-", color="#C4C9D0", label="train loss")
    ax2 = ax.twinx(); ax2.plot(e, [x["val_macro_f1"] for x in h], "s--", color="#D6293E", label="val macro-F1")
    ax.set_title(t); ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax2.set_ylabel("val macro-F1")
fig.tight_layout(); fig.savefig("results/figures/roberta_learning_curves.png", dpi=200); plt.show()
""")
md("## 7. End-to-end evaluation on the shared test set\nThe whole pipeline on the 353 binary-test texts, scored on a 6-way outcome (Not Dark + 5 categories). A dark text is correct only if Stage 1 fires **and** Stage 2 picks the right class.")
code("""
E2E = ["Not Dark"] + STAGE2_LABELS
t = pd.read_csv("data/splits/binary_test.csv")
c = pd.read_csv("data/splits/category_test.csv").drop_duplicates("raw_text").set_index("raw_text")
X = t.raw_text.tolist()
p1 = predict_proba(m1, X); p2 = predict_proba(m2, X)
pred = np.where(p1[:, 1] >= 0.5, 1 + p2.argmax(1), 0)
true = np.array([0 if l == 0 else 1 + int(c.loc[x, "category_id"]) for x, l in zip(X, t.label)])
me = compute_metrics(true, pred, np.hstack([p1[:, [0]], p1[:, [1]] * p2]), E2E)
json.dump(me, open("results/metrics/roberta_end_to_end.json", "w"), indent=2, default=float)
plot_confusion(me["confusion_matrix"], E2E, f"RoBERTa-base - end-to-end (test, n={me['n_test']})", "results/figures/cm_roberta_end_to_end.png")
print(f"END-TO-END acc={me['accuracy']:.4f} macro-F1={me['f1_macro']:.4f}")
errs = t.assign(pred=[E2E[i] for i in pred], true=[E2E[i] for i in true])
errs = errs[errs.pred != errs.true][["raw_text", "true", "pred"]]
errs.to_csv("results/metrics/roberta_errors.csv", index=False); errs.head(15)
""")
md("## 8. LIME explanations from the fine-tuned model")
code("""
from lime.lime_text import LimeTextExplainer
examples = ["Hurry! Only 3 left in stock - offer ends in 02:14:55",
            "No thanks, I don't want to save money",
            "Michael in Pontiac, United States purchased a Classic Tee 4 minutes ago",
            "Free returns within 30 days of delivery"]
lime_out = []
for ex in examples:
    e = LimeTextExplainer(class_names=BINARY_LABELS, random_state=42).explain_instance(
        ex, lambda xs: predict_proba(m1, list(xs)), labels=(1,), num_features=8, num_samples=1000)
    lime_out.append({"text": ex, "p_dark": float(predict_proba(m1, [ex])[0, 1]),
                     "weights": [(str(w), float(v)) for w, v in e.as_list(label=1)]})
    fig = e.as_pyplot_figure(label=1); fig.set_size_inches(6, 3)
    for b in fig.axes[0].patches: b.set_color("#D6293E" if b.get_width() > 0 else "#C4C9D0")
    plt.title(f"LIME: '{ex[:45]}...'", fontsize=9); plt.tight_layout()
    plt.savefig(f"results/figures/lime_roberta_{len(lime_out)}.png", dpi=200); plt.show()
json.dump(lime_out, open("results/metrics/roberta_lime_examples.json", "w"), indent=2)
""")
md("## 9. Package and download\nModels are stored in half precision (fp16) to halve the download (~500 MB total). Unzip `roberta_outputs.zip` **inside the `dark_pattern_guard_v2` folder**, so that `models/roberta/...` and `results/...` land in place.")
code("""
for stage in ["binary", "category"]:
    m = AutoModelForSequenceClassification.from_pretrained(f"models/roberta/{stage}").half()
    m.save_pretrained(f"models/roberta/{stage}", safe_serialization=True)
!rm -f roberta_outputs.zip && zip -qr roberta_outputs.zip models/roberta results && ls -lh roberta_outputs.zip
from google.colab import files
files.download("roberta_outputs.zip")
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
