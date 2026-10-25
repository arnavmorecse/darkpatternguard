"""Step 1 - Build the final train/val/test splits used by every model.

Starts from the teammate's leakage-aware binary splits (grouped on normalised
text, seed 42) and derives the Stage 2 splits FROM them, so one test set is
shared by the whole pipeline (no category-train text can appear in binary-test).

Outputs data/splits/{binary,category}_{train,val,test}.csv with two text columns:
  text      - normalised (lower-cased) text, used by TF-IDF
  raw_text  - original casing/punctuation from dataset.tsv, used by RoBERTa
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from teammate_data_utils.preprocessor import preprocess_text  # same normaliser as the teammate

RARE = {"Obstruction", "Sneaking", "Forced Action"}
STAGE2_LABELS = ["Scarcity", "Social Proof", "Urgency", "Misdirection", "Other (rare)"]


def main():
    raw = pd.read_csv(ROOT / "data/raw/ec_dataset.tsv", sep="\t")
    raw["norm"] = raw["text"].astype(str).map(preprocess_text)
    # norm -> original text / category (first occurrence; categories agree 100%)
    first = raw.drop_duplicates("norm").set_index("norm")
    out = ROOT / "data/splits"
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for split in ["train", "val", "test"]:
        b = pd.read_csv(ROOT / f"data/teammate_splits/binary/{split}.csv")
        b["raw_text"] = b["text"].map(first["text"])
        b["pattern_category"] = b["text"].map(first["Pattern Category"])
        miss = b["raw_text"].isna().sum()
        if miss:  # fall back to normalised text if the normaliser drifted
            b["raw_text"] = b["raw_text"].fillna(b["text"])
        b[["text", "raw_text", "label", "pattern_category"]].to_csv(out / f"binary_{split}.csv", index=False)

        c = b[b.label == 1].copy()
        c["category"] = c["pattern_category"].where(~c["pattern_category"].isin(RARE), "Other (rare)")
        c["category_id"] = c["category"].map({k: i for i, k in enumerate(STAGE2_LABELS)})
        assert c["category_id"].notna().all(), c[c.category_id.isna()].head()
        c[["text", "raw_text", "category", "category_id", "pattern_category"]].to_csv(out / f"category_{split}.csv", index=False)
        report.append((split, len(b), int(b.label.sum()), miss, c["category"].value_counts().to_dict()))

    for split, n, pos, miss, dist in report:
        print(f"{split:5s} binary={n} (dark={pos}, unmatched raw={miss}) | stage2={dist}")
    # leakage checks
    S = {s: pd.read_csv(out / f"binary_{s}.csv") for s in ["train", "val", "test"]}
    print("overlap train/test:", len(set(S["train"].text) & set(S["test"].text)),
          "train/val:", len(set(S["train"].text) & set(S["val"].text)))


if __name__ == "__main__":
    main()
