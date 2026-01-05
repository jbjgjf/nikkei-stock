from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import yaml

TagValue = Literal["yes", "no", "unk"]

@dataclass
class TagDef:
    tag: str
    in_score: bool
    structure: str

@dataclass
class Rule:
    tag: str
    any_match: list[str]
    none_match: list[str]
    value: TagValue
    # optional note for debugging / documentation
    note: str | None = None

def safe(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", name).strip()

def load_tags(path: Path) -> dict[str, TagDef]:
    data = yaml.safe_load(path.read_text())
    return {
        t["tag"]: TagDef(
            tag=t["tag"],
            in_score=t["in_score"],
            structure=t["structure"],
        )
        for t in data["tags"]
    }

def load_rules(path: Path) -> dict[str, Rule]:
    data = yaml.safe_load(path.read_text())
    return {
        r["tag"]: Rule(
            tag=r["tag"],
            any_match=r.get("any_match", []),
            none_match=r.get("none_match", []),
            value=r.get("value", "yes"),
            note=r.get("note"),
        )
        for r in data.get("rules", [])
    }


# Helper to pick a column from a list of candidates
def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_blocks(blocks_csv_path: Path) -> dict[str, list[dict]]:
    """Load block-level corpus, grouped by entity_id.

    Expected (flexible) columns:
      - entity: entity_id / company / company_name / name
      - text: text / block_text / content / body
      - source: source_file / file / pdf / path
      - block id: block_id / id / block
      - page: page / pageno / page_no (optional)

    Returns:
      dict[entity_id] -> list of blocks (dict with keys: text, source_file, block_id, page)
    """
    df = pd.read_csv(blocks_csv_path)

    ent_col = _pick_col(df, ["entity_id", "entity", "company", "company_name", "name"])
    text_col = _pick_col(df, ["text", "block_text", "content", "body"])
    src_col = _pick_col(df, ["source_file", "source", "file", "pdf", "path"])
    bid_col = _pick_col(df, ["block_id", "block", "id"])
    page_col = _pick_col(df, ["page", "pageno", "page_no"])

    if ent_col is None or text_col is None:
        # Cannot use block mode; fall back to raw .txt corpus
        return {}

    out: dict[str, list[dict]] = {}
    for _, r in df.iterrows():
        raw_ent = str(r[ent_col])
        ent = safe(raw_ent)
        text = "" if pd.isna(r[text_col]) else str(r[text_col])
        source_file = "" if (src_col is None or pd.isna(r[src_col])) else str(r[src_col])
        block_id = "" if (bid_col is None or pd.isna(r[bid_col])) else str(r[bid_col])
        page = "" if (page_col is None or pd.isna(r[page_col])) else str(r[page_col])
        out.setdefault(ent, []).append({
            "text": text,
            "source_file": source_file,
            "block_id": block_id,
            "page": page,
        })
    return out


def _search_patterns_in_blocks(patterns: list[str], blocks: list[dict]) -> tuple[bool, str, str, str]:
    """Return (hit, source_file, snippet, matched_pattern) for the first match found."""
    for p in patterns:
        cre = re.compile(p, re.I | re.S)
        for b in blocks:
            m = cre.search(b.get("text", "") or "")
            if not m:
                continue
            s = b.get("text", "") or ""
            # Small evidence snippet around match
            a = max(m.start() - 80, 0)
            z = min(m.end() + 80, len(s))
            snippet = s[a:z].replace("\n", " ").strip()
            return True, (b.get("source_file") or ""), snippet, p
    return False, "", "", ""

def extract_tags(
    *,
    candidates_df: pd.DataFrame,
    corpus_dir: Path,
    tags_config_path: Path,
    rules_config_path: Path,
    blocks_csv_path: Optional[Path] = None,
):
    tags = load_tags(tags_config_path)
    rules = load_rules(rules_config_path)

    company_col = _pick_col(
        candidates_df,
        [
            "entity_id",
            "entity",
            "company",
            "company_name",
            "name",
            "company_name_ja",
            "company_name_jp",
            "company_ja",
            "company_jp",
            "銘柄名",
            "会社名",
        ],
    )
    if company_col is None:
        company_col = candidates_df.columns[0]
        print(f"[WARN] Company column not found by name. Falling back to first column: {company_col}")
    rows, evidences = [], []

    blocks_by_entity: dict[str, list[dict]] = {}
    if blocks_csv_path is not None and blocks_csv_path.exists():
        blocks_by_entity = load_blocks(blocks_csv_path)

    processed = 0
    for raw in candidates_df[company_col]:
        cid = safe(str(raw))
        print(f"[INFO] Processing entity_id: {cid}")
        blocks = blocks_by_entity.get(cid, [])
        if blocks:
            # Join blocks for fallback full-text search
            text = "\n\n".join((b.get("text") or "") for b in blocks)
        else:
            corpus_path = corpus_dir / f"{cid}.txt"
            if not corpus_path.exists():
                print(f"[WARN] Corpus txt not found for entity_id={cid} at {corpus_path}")
                continue
            text = corpus_path.read_text(errors="ignore")

        row = {"entity_id": cid}
        for tag, tdef in tags.items():
            rule = rules.get(tag)
            value: TagValue = "unk"
            ev_source_file = ""
            ev_snippet = ""
            ev_pattern = ""
            ev_kind = ""  # 'none' or 'any'

            if rule:
                if rule.none_match:
                    if blocks:
                        hit, sf, snip, pat = _search_patterns_in_blocks(rule.none_match, blocks)
                    else:
                        hit = any(re.search(p, text, re.I | re.S) for p in rule.none_match)
                        sf, snip, pat = "", "", ""
                    if hit:
                        value = "no"
                        ev_source_file, ev_snippet, ev_pattern, ev_kind = sf, snip, pat, "none"

                if value == "unk" and rule.any_match:
                    if blocks:
                        hit, sf, snip, pat = _search_patterns_in_blocks(rule.any_match, blocks)
                    else:
                        hit = any(re.search(p, text, re.I | re.S) for p in rule.any_match)
                        sf, snip, pat = "", "", ""
                    if hit:
                        value = rule.value
                        ev_source_file, ev_snippet, ev_pattern, ev_kind = sf, snip, pat, "any"

            row[tag] = value
            if value != "unk":
                evidences.append({
                    "entity_id": cid,
                    "tag_id": tag,
                    "evidence": value,
                    "source_file": ev_source_file,
                    "snippet": ev_snippet,
                    "matched_pattern": ev_pattern,
                    "match_kind": ev_kind,
                })
        rows.append(row)
        processed += 1

    print(f"[INFO] Processed entities with corpus: {processed}")
    return pd.DataFrame(rows), pd.DataFrame(evidences)
