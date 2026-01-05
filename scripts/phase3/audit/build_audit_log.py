import pandas as pd

def build_audit_log(tags_df, evidence_df, rules_df):
    long = tags_df.melt(
        id_vars=["entity_id"],
        var_name="tag_id",
        value_name="tag_value"
    )
    merged = long.merge(evidence_df, on=["entity_id", "tag_id"], how="left")
    merged = merged.merge(
        rules_df[["entity_id", "axis_id", "rule_id"]],
        on="entity_id",
        how="left"
    )
    if "source_file" not in merged.columns:
        merged["source_file"] = ""
    else:
        merged["source_file"] = merged["source_file"].fillna("")
    cols = ["entity_id", "axis_id", "tag_id", "tag_value", "source_file", "evidence", "rule_id"]
    if "snippet" in merged.columns:
        cols.append("snippet")
    return merged[cols]