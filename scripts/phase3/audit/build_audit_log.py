"""Audit log builder for Phase3.

TODO:
- Join evidence with rule outcomes.
- Emit human-readable citations for each decision.
"""

from __future__ import annotations

import pandas as pd


def build_audit_log(
    tags_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    rules_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build audit log.

    Returns
    -------
    audit_df
        Columns: entity_id, axis_id, tag_id, tag_value, source_file, quote, rule_id
    """

    # Placeholder: return empty outputs with the expected schema.
    audit_df = pd.DataFrame(
        columns=[
            "entity_id",
            "axis_id",
            "tag_id",
            "tag_value",
            "source_file",
            "quote",
            "rule_id",
        ]
    )
    return audit_df
