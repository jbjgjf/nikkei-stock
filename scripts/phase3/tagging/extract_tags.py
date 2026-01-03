"""Tag extraction (YES/NO/UNK) for Phase3.

TODO:
- Load tag schema from tags_config_path.
- Parse corpus per candidate and extract evidence spans.
- Return per-candidate tag values and evidence records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def extract_tags(
    candidates_df: pd.DataFrame,
    corpus_dir: Path,
    tags_config_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract tags and evidence.

    Returns
    -------
    tags_df
        Columns: entity_id, tag_id, tag_value, tag_source
    evidence_df
        Columns: entity_id, tag_id, source_file, quote, confidence
    """

    # Placeholder: return empty outputs with the expected schema.
    tags_df = pd.DataFrame(
        columns=["entity_id", "tag_id", "tag_value", "tag_source"]
    )
    evidence_df = pd.DataFrame(
        columns=["entity_id", "tag_id", "source_file", "quote", "confidence"]
    )
    return tags_df, evidence_df
