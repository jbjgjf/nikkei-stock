import pandas as pd
import yaml
from pathlib import Path

def apply_rules(tags_df: pd.DataFrame, rules_config_path: Path) -> pd.DataFrame:
    tags_yaml = rules_config_path.parent / "tags.yaml"
    meta = yaml.safe_load(tags_yaml.read_text())

    core_tags = [t["tag"] for t in meta["tags"] if t["in_score"]]
    tag2struct = {t["tag"]: t["structure"] for t in meta["tags"]}

    rows = []
    for _, r in tags_df.iterrows():
        eid = r["entity_id"]
        core_yes = sum(1 for t in core_tags if r[t] == "yes")

        rows.append({
            "entity_id": eid,
            "axis_id": "core_score",
            "axis_label": f"{core_yes}/{len(core_tags)}",
            "rule_id": "core_yes_count",
            "rationale": "count of core yes tags",
        })

        for struct in sorted(set(tag2struct.values())):
            ys = sum(1 for t in core_tags if tag2struct[t] == struct and r[t] == "yes")
            ts = sum(1 for t in core_tags if tag2struct[t] == struct)
            rows.append({
                "entity_id": eid,
                "axis_id": f"struct::{struct}",
                "axis_label": f"{ys}/{ts}",
                "rule_id": "structure_core_yes",
                "rationale": "core tags by structure",
            })

    return pd.DataFrame(rows)