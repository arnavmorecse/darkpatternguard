"""DarkPatternGuard — Data loading module.

Loads raw datasets (A and B) from their official locations, verifies integrity,
and provides them as preprocessed or raw pandas dataframes for downstream processing.

Uses verified file paths from configs/config.yaml.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load project configuration from YAML.

    Args:
        config_path: Path to config.yaml (relative to project root).

    Returns:
        Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            f"Are you running from the project root?"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_dataset_paths(config: dict) -> Tuple[Path, Path]:
    """Extract and validate dataset paths from config.

    Args:
        config: Configuration dictionary from load_config().

    Returns:
        Tuple of (path_to_dataset_a, path_to_dataset_b).

    Raises:
        FileNotFoundError: If dataset paths don't exist.
    """
    paths = config["paths"]
    path_a = Path(paths["dataset_a"])
    path_b = Path(paths["dataset_b"])

    if not path_a.exists():
        raise FileNotFoundError(f"Dataset A not found at {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Dataset B not found at {path_b}")

    return path_a, path_b


def verify_file_integrity(file_path: Path, expected_sha256: Optional[str] = None) -> str:
    """Compute SHA-256 hash of a file for integrity verification.

    Args:
        file_path: Path to file.
        expected_sha256: If provided, verify hash matches this value.

    Returns:
        SHA-256 hash as hex string.

    Raises:
        ValueError: If computed hash does not match expected hash.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    computed_hash = h.hexdigest()

    if expected_sha256 and computed_hash != expected_sha256:
        raise ValueError(
            f"File integrity check failed for {file_path.name}.\n"
            f"Expected: {expected_sha256}\n"
            f"Got: {computed_hash}"
        )

    return computed_hash


def load_dataset_a(
    path: Optional[Path] = None,
    config: Optional[dict] = None,
    verify_hash: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Load Dataset A (ec-darkpattern).

    Args:
        path: Path to dataset.tsv. If None, inferred from config.
        config: Configuration dict. If None, loaded from default path.
        verify_hash: If True, verify file SHA-256 against known value.

    Returns:
        Tuple of (dataframe, metadata_dict).

    Metadata includes:
        - path: file path used.
        - sha256: file hash (if verified).
        - shape: (rows, columns).
        - columns: list of column names.
        - missing_values: dict of missing counts per column.
    """
    if config is None:
        config = load_config()
    if path is None:
        path = Path(config["paths"]["dataset_a"])

    # Optional hash verification against Phase 0 reported hash.
    expected_hash = "bd7bc7d93f5bf08591b10886d7aa3a7ba8ac266f5ce8206c00cd4daf7868508a"
    if verify_hash:
        actual_hash = verify_file_integrity(path, expected_hash)
        metadata = {"sha256": actual_hash}
    else:
        metadata = {}

    # Load using official loader semantics (tab-separated, quoted fields).
    df = pd.read_csv(path, sep="\t")

    metadata.update(
        {
            "path": str(path),
            "shape": (int(df.shape[0]), int(df.shape[1])),
            "columns": list(df.columns),
            "missing_values": df.isna().sum().to_dict(),
        }
    )

    return df, metadata


def load_dataset_b(
    path: Optional[Path] = None,
    config: Optional[dict] = None,
    verify_hash: bool = True,
    drop_missing_text: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Load Dataset B (Mathur dark-patterns).

    Args:
        path: Path to dark-patterns.csv. If None, inferred from config.
        config: Configuration dict. If None, loaded from default path.
        verify_hash: If True, verify file SHA-256 against known value.
        drop_missing_text: If True, drop rows with missing Pattern String.
                           If False, keep all rows (will be filtered in splitting).

    Returns:
        Tuple of (dataframe, metadata_dict).

    Metadata includes:
        - path: file path used.
        - sha256: file hash (if verified).
        - shape: (rows, columns) — before any filtering.
        - rows_after_filtering: rows after dropping missing text (if applied).
        - columns: list of column names.
        - missing_values: dict of missing counts per column.
    """
    if config is None:
        config = load_config()
    if path is None:
        path = Path(config["paths"]["dataset_b"])

    # Optional hash verification against Phase 0 reported hash.
    expected_hash = "fb2877af3b7bef359abcfb415f3a477df02481300d84f5eb9762b3ca2523bc9b"
    if verify_hash:
        actual_hash = verify_file_integrity(path, expected_hash)
        metadata = {"sha256": actual_hash}
    else:
        metadata = {}

    df = pd.read_csv(path)
    original_shape = df.shape

    metadata.update(
        {
            "path": str(path),
            "shape_original": (int(original_shape[0]), int(original_shape[1])),
            "columns": list(df.columns),
            "missing_values": df.isna().sum().to_dict(),
        }
    )

    if drop_missing_text:
        df = df[df["Pattern String"].notna()].reset_index(drop=True)
        metadata["shape_after_filtering"] = (int(df.shape[0]), int(df.shape[1]))

    return df, metadata


def load_inspection_summary() -> dict:
    """Load the Phase 0 inspection summary (machine-readable results).

    Returns:
        Dictionary of inspection results.

    Raises:
        FileNotFoundError: If reports/inspection_summary.json not found.
    """
    summary_path = Path("reports/inspection_summary.json")
    if not summary_path.exists():
        raise FileNotFoundError(
            "Phase 0 inspection summary not found at reports/inspection_summary.json. "
            "Run: python src/inspect_datasets.py"
        )

    with open(summary_path) as f:
        return json.load(f)


if __name__ == "__main__":
    # Quick test: load both datasets and print metadata.
    print("=== Loading Dataset A ===")
    df_a, meta_a = load_dataset_a()
    print(f"Loaded {meta_a['shape'][0]} rows, {meta_a['shape'][1]} columns")
    print(f"SHA-256 (verified): {meta_a.get('sha256', 'N/A')[:16]}...")
    print(f"Columns: {meta_a['columns']}")
    print(f"Missing values: {meta_a['missing_values']}\n")

    print("=== Loading Dataset B ===")
    df_b, meta_b = load_dataset_b()
    print(f"Loaded {meta_b['shape_original'][0]} rows, {meta_b['shape_original'][1]} columns")
    print(f"SHA-256 (verified): {meta_b.get('sha256', 'N/A')[:16]}...")
    print(f"Columns: {meta_b['columns']}")
    print(f"Missing Pattern String: {meta_b['missing_values']['Pattern String']}\n")

    print("=== Loading Phase 0 Inspection Summary ===")
    summary = load_inspection_summary()
    print(f"Random seed: {summary['random_seed']}")
    print(f"Cross-dataset overlap (normalized): {summary['cross_dataset_overlap']['normalized_text_overlap_count']}")
