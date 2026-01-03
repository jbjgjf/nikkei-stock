"""Rule-based judgments for Phase3.

TODO:
- Load rules from rules_config_path.
- Evaluate if-else rules per axis using tag values.
- Output per-axis labels and an overall decision if needed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def apply_rules(tags_df: pd.DataFrame, rules_config_path: Path) -> pd.DataFrame:
    """Apply rule-based judgments.

    Returns
    -------
    rules_df
        Columns: entity_id, axis_id, axis_label, rule_id, rationale
    """

    # Placeholder: return empty outputs with the expected schema.
    rules_df = pd.DataFrame(
        columns=["entity_id", "axis_id", "axis_label", "rule_id", "rationale"]
    )
    return rules_df
