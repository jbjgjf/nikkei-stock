#!/usr/bin/env python3
"""
Phase3 pipeline entrypoint.

Steps
-----
1) Tag extraction (YES/NO/UNK)
2) Rule-based judgments
3) Audit log (evidence trace)
4) CSV outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.phase3.audit.build_audit_log import build_audit_log
from scripts.phase3.rules.apply_rules import apply_rules
from scripts.phase3.tagging.extract_tags import extract_tags


DEFAULT_CORPUS_DIR = Path("/Users/shou/hobby/CPX/nikkei-stock/data/source/phase3/corpus")
DEFAULT_TAGS_CONFIG = Path("/Users/shou/hobby/CPX/nikkei-stock/config/phase3/tags.yaml")
DEFAULT_RULES_CONFIG = Path("/Users/shou/hobby/CPX/nikkei-stock/config/phase3/rules.yaml")
DEFAULT_SCORES_BASE = Path("/Users/shou/hobby/CPX/nikkei-stock/data/scores/p3")
DEFAULT_INTERIM_BASE = Path("/Users/shou/hobby/CPX/nikkei-stock/data/interim/phase3")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase3 screening: tagging + rules + audit")
    p.add_argument("--run_id", required=True, help="Run ID used under data/scores/p3 and data/interim/phase3")
    p.add_argument("--candidates", required=True, help="Path to phaseE_selected.csv")
    p.add_argument("--corpus_dir", type=Path, default=DEFAULT_CORPUS_DIR)
    p.add_argument("--config_tags", type=Path, default=DEFAULT_TAGS_CONFIG)
    p.add_argument("--config_rules", type=Path, default=DEFAULT_RULES_CONFIG)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    scores_dir = DEFAULT_SCORES_BASE / args.run_id
    interim_dir = DEFAULT_INTERIM_BASE / args.run_id

    scores_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = args.corpus_dir
    # Allow passing a root folder that contains corpus_text/ and block.csv
    if (corpus_dir / "corpus_text").exists():
        corpus_dir = corpus_dir / "corpus_text"

    txt_files = sorted(corpus_dir.glob("*.txt"))
    print(f"[INFO] Found txt files: {len(txt_files)} in {corpus_dir}")
    candidates_df = pd.DataFrame({"entity_id": [p.stem for p in txt_files]})

    blocks_csv_path = None
    for fname in ("block.csv", "blocks.csv"):
        p = args.corpus_dir / fname
        if p.exists():
            blocks_csv_path = p
            break

    # 1) Tag extraction
    tags_df, evidence_df = extract_tags(
        candidates_df=candidates_df,
        corpus_dir=corpus_dir,
        tags_config_path=args.config_tags,
        rules_config_path=args.config_rules,
        blocks_csv_path=blocks_csv_path,
    )
    print(f"[INFO] Tags rows: {len(tags_df)} | Evidence rows: {len(evidence_df)}")

    tags_path = interim_dir / "phase3_tags.csv"
    tags_long_path = interim_dir / "phase3_tags_long.csv"
    evidence_path = interim_dir / "phase3_evidence.csv"
    tags_df.to_csv(tags_path, index=False)
    tags_df.to_csv(tags_long_path, index=False)
    evidence_df.to_csv(evidence_path, index=False)

    # 2) Rule-based judgments
    rules_df = apply_rules(tags_df=tags_df, rules_config_path=args.config_rules)
    rules_path = interim_dir / "phase3_rules.csv"
    rules_df.to_csv(rules_path, index=False)

    # 3) Audit log
    audit_df = build_audit_log(tags_df=tags_df, evidence_df=evidence_df, rules_df=rules_df)
    audit_path = interim_dir / "phase3_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    # 4) Score/pass outputs (placeholders; implement aggregation in rules module)
    scores_path = scores_dir / "phase3_scores.csv"
    pass_path = scores_dir / "phase3_pass.csv"
    # NOTE: For now, we output rule judgments as-is.
    # You can later implement aggregation/scoring to build true scores/pass tables.
    rules_df.to_csv(scores_path, index=False)
    rules_df.to_csv(pass_path, index=False)

    print(f"[OK] tags:     {tags_path}")
    print(f"[OK] tags_long:{tags_long_path}")
    print(f"[OK] evidence: {evidence_path}")
    print(f"[OK] rules:    {rules_path}")
    print(f"[OK] audit:    {audit_path}")
    print(f"[OK] scores:   {scores_path}")
    print(f"[OK] pass:     {pass_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
