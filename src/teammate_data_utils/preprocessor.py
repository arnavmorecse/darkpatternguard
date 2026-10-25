"""DarkPatternGuard — Text preprocessing pipeline.

Reusable functions for text normalization and cleaning, preserving meaningful
signals for dark-pattern detection while removing noise and encoding artifacts.

Used by both TF-IDF (baseline) and RoBERTa (Stage 1 & 2) models.

Key design decisions:
- Conservative normalization: preserve case when meaningful, lowercase only for
  consistency across models.
- Stop words are NOT removed indiscriminately; they carry meaning in UI text.
- Stemming/lemmatization is avoided; they can remove context clues like "cannot"
  vs. "can" (critical for cancellation barriers).
- URLs, emails, and HTML artifacts are standardized but preserved, as their
  presence may signal deception.
- Whitespace is normalized; multiple spaces, tabs, newlines are collapsed to single space.
- Original raw datasets are never modified.

All preprocessing runs in memory on loaded dataframes. The raw files remain
untouched on disk.

Random seed: 42
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, tabs, and newlines into a single space.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.

    Example:
        >>> normalize_whitespace("Hello   world  \n  test")
        'Hello world test'
    """
    text = re.sub(r"[\s\n\t\r]+", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Apply NFKC unicode normalization.

    NFKC folds characters to their canonical form and replaces compatibility
    characters (e.g., full-width spaces, ligatures) with their ASCII equivalents.
    Safe for text analysis; does not remove meaningful content.

    Args:
        text: Input text.

    Returns:
        Unicode-normalized text.

    Example:
        >>> normalize_unicode("ﬁnal")  # ligature
        'final'
    """
    return unicodedata.normalize("NFKC", text)


def standardize_urls(text: str) -> str:
    """Replace URLs with a placeholder token [URL].

    Rationale: URLs themselves carry little semantic content for dark-pattern
    detection. Replacing them removes noise while preserving the signal that a
    link was present.

    Args:
        text: Input text.

    Returns:
        Text with URLs replaced by [URL].

    Example:
        >>> standardize_urls("Visit https://example.com now")
        'Visit [URL] now'
    """
    # Match http(s), ftp, and www-style URLs.
    text = re.sub(
        r"https?://[^\s]+|www\.[^\s]+|ftp://[^\s]+",
        "[URL]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def standardize_emails(text: str) -> str:
    """Replace email addresses with a placeholder token [EMAIL].

    Rationale: Specific email addresses carry no semantic meaning for dark-pattern
    detection. The presence of contact information, however, may be relevant.

    Args:
        text: Input text.

    Returns:
        Text with emails replaced by [EMAIL].

    Example:
        >>> standardize_emails("Contact us at support@example.com now")
        'Contact us at [EMAIL] now'
    """
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)
    return text


def remove_html_tags(text: str) -> str:
    """Remove HTML/XML tags and decode common HTML entities.

    Args:
        text: Input text (may contain HTML).

    Returns:
        Text with tags removed and entities decoded.

    Example:
        >>> remove_html_tags("<b>Save &amp; buy</b>")
        'Save & buy'
    """
    # Remove HTML/XML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities.
    replacements = {
        "&nbsp;": " ",
        "&lt;": "<",
        "&gt;": ">",
        "&amp;": "&",
        "&quot;": '"',
        "&apos;": "'",
        "&#39;": "'",
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)

    return text


def lowercase_text(text: str) -> str:
    """Convert text to lowercase.

    Args:
        text: Input text.

    Returns:
        Lowercase text.
    """
    return text.lower()


def remove_repeated_characters(text: str, max_repeat: int = 2) -> str:
    """Remove excessive character repetition (e.g., 'Looooove' -> 'Love').

    Rationale: Intentional repetition in UI text (e.g., "Hurrrry!") can signal
    manipulation, but excessive repetition (>2 chars) is usually encoding noise.

    Args:
        text: Input text.
        max_repeat: Maximum allowed consecutive character repetitions.

    Returns:
        Text with excessive repetitions reduced.

    Example:
        >>> remove_repeated_characters("Looooove this!!!")
        'Loove this!!'
    """
    pattern = r"(.)\1{" + str(max_repeat) + ",}"
    return re.sub(pattern, r"\1" * max_repeat, text)


def standardize_abbreviations(text: str) -> str:
    """Standardize common abbreviations for consistency.

    Rationale: Abbreviations like "w/" (with), "w/o" (without), "b/c" (because)
    are common in UI text and should be expanded for clarity.

    Args:
        text: Input text.

    Returns:
        Text with standardized abbreviations.

    Example:
        >>> standardize_abbreviations("Ships w/ tracking number")
        'Ships with tracking number'
    """
    abbrev_map = {
        r"\bw/\b": "with",
        r"\bw/o\b": "without",
        r"\bb/c\b": "because",
        r"\basap\b": "as soon as possible",
        r"\bpromo\b": "promotion",
        r"\bdiscount\b": "discount",
    }
    for pattern, replacement in abbrev_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def preprocess_text(
    text: str,
    lowercase: bool = True,
    remove_repeated: bool = True,
    standardize_urls_flag: bool = True,
    standardize_emails_flag: bool = True,
    remove_html: bool = True,
    standardize_abbrev: bool = True,
) -> str:
    """Apply full preprocessing pipeline to a single text.

    Pipeline order (important):
    1. Handle missing/empty: return empty string if input is None/empty.
    2. Unicode normalization (compatibility forms).
    3. Remove HTML tags and decode entities.
    4. Standardize URLs and emails.
    5. Standardize abbreviations.
    6. Remove excessive character repetition.
    7. Normalize whitespace.
    8. Lowercase (optional, but recommended for consistency).

    Args:
        text: Input text. If None or empty string, returns empty string.
        lowercase: If True, convert to lowercase.
        remove_repeated: If True, reduce excessive character repetition.
        standardize_urls_flag: If True, replace URLs with [URL].
        standardize_emails_flag: If True, replace emails with [EMAIL].
        remove_html: If True, remove HTML tags and decode entities.
        standardize_abbrev: If True, standardize common abbreviations.

    Returns:
        Preprocessed text (empty string if input was None/empty).

    Example:
        >>> text = "Don't MISS  https://example.com — Limited Time!!!"
        >>> preprocess_text(text)
        "don't miss [url] limited time"
    """
    # Handle missing/empty input.
    if text is None or (isinstance(text, str) and text.strip() == ""):
        return ""

    text = str(text).strip()

    # Step 1: Unicode normalization.
    text = normalize_unicode(text)

    # Step 2: Remove HTML tags and decode entities.
    if remove_html:
        text = remove_html_tags(text)

    # Step 3: Standardize URLs and emails.
    if standardize_urls_flag:
        text = standardize_urls(text)
    if standardize_emails_flag:
        text = standardize_emails(text)

    # Step 4: Standardize abbreviations.
    if standardize_abbrev:
        text = standardize_abbreviations(text)

    # Step 5: Remove excessive character repetition.
    if remove_repeated:
        text = remove_repeated_characters(text)

    # Step 6: Normalize whitespace.
    text = normalize_whitespace(text)

    # Step 7: Lowercase.
    if lowercase:
        text = lowercase_text(text)

    return text


def preprocess_dataframe(
    df,
    text_column: str,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple:
    """Apply preprocessing to all text in a dataframe column.

    Args:
        df: Input dataframe.
        text_column: Name of the text column to preprocess.
        inplace: If False, return a new dataframe; if True, modify in place.
        verbose: If True, print statistics about preprocessing.

    Returns:
        Tuple of (processed_dataframe, preprocessing_stats).

    Preprocessing stats includes:
        - original_rows: original dataframe row count.
        - rows_with_text: rows where text_column is not None/empty.
        - rows_missing_or_empty: rows where text was missing/empty.
        - avg_original_length: average character length before preprocessing.
        - avg_processed_length: average character length after preprocessing.

    Example:
        >>> df_processed, stats = preprocess_dataframe(df, 'text')
        >>> print(stats)
    """
    if not inplace:
        df = df.copy()

    original_rows = len(df)
    texts_before = df[text_column].copy()

    # Count missing/empty before processing.
    missing_or_empty = (
        df[text_column].isna() | (df[text_column].astype(str).str.strip() == "")
    ).sum()

    # Apply preprocessing.
    df[text_column] = df[text_column].fillna("").astype(str).map(preprocess_text)

    # Statistics.
    avg_original = texts_before.astype(str).str.len().mean()
    avg_processed = df[text_column].str.len().mean()

    stats = {
        "original_rows": original_rows,
        "rows_with_text": original_rows - missing_or_empty,
        "rows_missing_or_empty": missing_or_empty,
        "avg_original_length": float(avg_original),
        "avg_processed_length": float(avg_processed),
    }

    if verbose:
        print(f"Preprocessing stats for column '{text_column}':")
        print(f"  Original rows: {stats['original_rows']}")
        print(f"  Rows with text: {stats['rows_with_text']}")
        print(f"  Rows missing/empty: {stats['rows_missing_or_empty']}")
        print(f"  Avg original length: {stats['avg_original_length']:.1f}")
        print(f"  Avg processed length: {stats['avg_processed_length']:.1f}")

    return df, stats


if __name__ == "__main__":
    # Quick self-test.
    test_texts = [
        "Don't MISS https://example.com — Limited Time!!! Only 2 left in stock.",
        None,
        "",
        "<b>Save &amp; BUY NOW</b> w/ free shipping",
        "Contact support@example.com for help",
    ]

    print("=== Preprocessing Examples ===\n")
    for i, text in enumerate(test_texts):
        print(f"Original [{i}]: {repr(text)}")
        processed = preprocess_text(text) if text else preprocess_text("")
        print(f"Processed [{i}]: {repr(processed)}")
        print()
