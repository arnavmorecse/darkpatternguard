"""DarkPatternGuard data module.

Provides data loading, preprocessing, and splitting functionality.
"""

from .loader import load_config, load_dataset_a, load_dataset_b, load_inspection_summary
from .preprocessor import preprocess_text, preprocess_dataframe
from .splitter import (
    stratified_group_split,
    analyze_class_distribution,
)

__all__ = [
    "load_config",
    "load_dataset_a",
    "load_dataset_b",
    "load_inspection_summary",
    "preprocess_text",
    "preprocess_dataframe",
    "stratified_group_split",
    "analyze_class_distribution",
]
