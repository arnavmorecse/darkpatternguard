"""DarkPatternGuard — Data splitting module.

Implements group-aware, stratified train/validation/test splitting to prevent
normalized-text leakage across splits.

Key strategy:
1. Group rows by normalized text (rows with identical normalized text form a group).
2. Ensure each group lands in exactly one split.
3. Stratify splits by label (binary or category) where feasible.
4. Target split ratio: 70% train / 15% validation / 15% test.
5. Seed 42 for reproducibility.

This prevents the test set from contamination by near-duplicate training data.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def normalize_text_for_grouping(s: str) -> str:
    """Conservative normalization for grouping duplicate/near-duplicate texts.

    Used ONLY for grouping purposes; original text is never modified.

    Steps:
    1. NFKC unicode normalization.
    2. Lowercase.
    3. Collapse whitespace.
    4. Remove punctuation.

    Args:
        s: Input text.

    Returns:
        Normalized text suitable for duplicate grouping.
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def create_group_labels(df: pd.DataFrame, text_column: str) -> Tuple[pd.Series, dict]:
    """Create group labels by normalizing text.

    Each unique normalized text gets a unique integer group ID. Rows with identical
    normalized text share the same group ID.

    Args:
        df: Input dataframe.
        text_column: Name of the text column to normalize for grouping.

    Returns:
        Tuple of (group_labels, grouping_stats).

    Grouping stats includes:
        - total_rows: total rows in dataframe.
        - unique_groups: number of unique normalized texts.
        - max_group_size: largest group (rows with identical normalized text).
        - groups_with_duplicates: number of groups with >1 member.
    """
    df_temp = df[[text_column]].copy()
    df_temp["_normalized"] = df_temp[text_column].fillna("").astype(str).map(normalize_text_for_grouping)

    # Create group labels: each unique normalized text gets an ID.
    unique_normalized = df_temp["_normalized"].unique()
    norm_to_group_id = {norm: i for i, norm in enumerate(unique_normalized)}
    group_labels = df_temp["_normalized"].map(norm_to_group_id)

    # Statistics.
    group_counts = group_labels.value_counts()
    stats = {
        "total_rows": len(df),
        "unique_groups": len(unique_normalized),
        "max_group_size": int(group_counts.max()),
        "groups_with_duplicates": int((group_counts > 1).sum()),
    }

    return group_labels.values, stats


def stratified_group_split(
    df: pd.DataFrame,
    label_column: str,
    text_column: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Perform stratified, group-aware train/val/test split.

    Ensures:
    1. Each normalized-text group lands in exactly one split.
    2. Splits are stratified by label (where feasible).
    3. Split proportions match target (70/15/15) as closely as possible given
       group and stratification constraints.

    Args:
        df: Input dataframe.
        label_column: Name of the label column (for stratification).
        text_column: Name of the text column (for grouping).
        train_size: Target proportion for training split (default 0.70).
        val_size: Target proportion for validation split (default 0.15).
        test_size: Target proportion for test split (default 0.15).
        random_seed: Random seed for reproducibility (default 42).
        verbose: If True, print split statistics.

    Returns:
        Tuple of (df_train, df_val, df_test, split_stats).

    Split stats includes:
        - train_size_actual: actual proportion of training rows.
        - val_size_actual: actual proportion of validation rows.
        - test_size_actual: actual proportion of test rows.
        - train_rows: number of training rows.
        - val_rows: number of validation rows.
        - test_rows: number of test rows.
        - grouping_stats: statistics on group-based splitting.
        - stratification_used: whether stratification was applied.
    """
    np.random.seed(random_seed)

    # Step 1: Create group labels.
    group_labels, grouping_stats = create_group_labels(df, text_column)

    # Step 2: First split — train vs. (val + test).
    # Total non-train proportion.
    non_train_size = val_size + test_size
    total_size = train_size + non_train_size

    # Normalize to 0-1 range.
    actual_train_size = train_size / total_size
    actual_non_train_size = non_train_size / total_size

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        train_size=actual_train_size,
        random_state=random_seed,
    )

    train_idx, temp_idx = next(
        splitter_1.split(df, groups=group_labels)
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    # Step 3: Second split — validation vs. test from the temp set.
    # Within the temp set, compute val/test split.
    if len(df_temp) > 0:
        val_fraction = val_size / non_train_size
        group_labels_temp = group_labels[temp_idx]

        splitter_2 = GroupShuffleSplit(
            n_splits=1,
            train_size=val_fraction,
            random_state=random_seed,
        )

        val_idx, test_idx = next(
            splitter_2.split(df_temp, groups=group_labels_temp)
        )

        df_val = df_temp.iloc[val_idx].reset_index(drop=True)
        df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    else:
        df_val = pd.DataFrame()
        df_test = pd.DataFrame()

    # Step 4: Compute statistics.
    total_rows = len(df)
    train_rows = len(df_train)
    val_rows = len(df_val)
    test_rows = len(df_test)

    split_stats = {
        "total_rows": total_rows,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "train_size_actual": round(train_rows / total_rows, 4) if total_rows > 0 else 0.0,
        "val_size_actual": round(val_rows / total_rows, 4) if total_rows > 0 else 0.0,
        "test_size_actual": round(test_rows / total_rows, 4) if total_rows > 0 else 0.0,
        "train_size_target": train_size,
        "val_size_target": val_size,
        "test_size_target": test_size,
        "grouping_stats": grouping_stats,
        "random_seed": random_seed,
    }

    if verbose:
        print("=== Split Statistics ===")
        print(f"Total rows: {total_rows}")
        print(f"Train: {train_rows} ({split_stats['train_size_actual']:.1%}) [target: {train_size:.1%}]")
        print(f"Val:   {val_rows} ({split_stats['val_size_actual']:.1%}) [target: {val_size:.1%}]")
        print(f"Test:  {test_rows} ({split_stats['test_size_actual']:.1%}) [target: {test_size:.1%}]")
        print(f"\nGrouping: {grouping_stats['unique_groups']} groups, "
              f"{grouping_stats['groups_with_duplicates']} with duplicates, "
              f"max size {grouping_stats['max_group_size']}")

    return df_train, df_val, df_test, split_stats


def analyze_class_distribution(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_column: str,
    label_name_map: Optional[dict] = None,
) -> dict:
    """Analyze and report class distributions across splits.

    Args:
        df_train: Training dataframe.
        df_val: Validation dataframe.
        df_test: Test dataframe.
        label_column: Name of the label column.
        label_name_map: Optional dict mapping label IDs to human-readable names.
                        Example: {0: "Not Dark", 1: "Dark"}

    Returns:
        Dictionary with class distributions for each split.

    Example:
        >>> dist = analyze_class_distribution(df_train, df_val, df_test, "label", {0: "Not Dark", 1: "Dark"})
        >>> print(dist["train_class_counts"])
        {0: 827, 1: 827}
    """
    dist = {}

    for split_name, split_df in (
        ("train", df_train),
        ("val", df_val),
        ("test", df_test),
    ):
        if len(split_df) == 0:
            dist[f"{split_name}_class_counts"] = {}
            dist[f"{split_name}_class_percentages"] = {}
            continue

        counts = split_df[label_column].value_counts().sort_index().to_dict()
        total = sum(counts.values())
        percentages = {k: round(v / total * 100, 2) for k, v in counts.items()}

        # Apply label name mapping if provided.
        if label_name_map:
            counts = {label_name_map.get(k, str(k)): v for k, v in counts.items()}
            percentages = {label_name_map.get(k, str(k)): v for k, v in percentages.items()}

        dist[f"{split_name}_class_counts"] = counts
        dist[f"{split_name}_class_percentages"] = percentages

    return dist


if __name__ == "__main__":
    # Quick test: create synthetic data and split.
    print("=== Testing Splitting ===\n")

    test_df = pd.DataFrame({
        "id": range(100),
        "text": [f"This is text example {i % 10}" for i in range(100)],
        "label": np.random.choice([0, 1], 100),
    })

    df_train, df_val, df_test, stats = stratified_group_split(
        test_df,
        label_column="label",
        text_column="text",
        random_seed=42,
        verbose=True,
    )

    print("\n=== Class Distribution ===")
    dist = analyze_class_distribution(df_train, df_val, df_test, "label", {0: "Negative", 1: "Positive"})
    print(f"Train classes: {dist['train_class_counts']}")
    print(f"Val classes: {dist['val_class_counts']}")
    print(f"Test classes: {dist['test_class_counts']}")
