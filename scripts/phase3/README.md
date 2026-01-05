# Phase3: 第三次スクリーニング（タグ抽出 + ルール判定）

## 目的
- Phase3入口で選ばれた候補に対して、NLP/タグ抽出 → ルール判定 → 監査ログ → 集計CSVを出力する。
- 学習モデルは使わず、if-elseルールで判定する。

## 構成（分離）
- tagging/ : タグ抽出（YES/NO/UNK）
- rules/   : ルール判定（◎○× / 高中低 / ○△×など）
- audit/   : 根拠引用ログ（どの文/根拠で判定したか）
- run_phase3.py : エントリポイント

## 入力
- 候補リスト: data/scores/p3/<run_id>/phaseE_selected.csv
- 参照コーパス: data/source/phase3/corpus/
- タグ定義: config/phase3/tags.yaml
- ルール定義: config/phase3/rules.yaml

## 中間生成物
- data/interim/phase3/<run_id>/
  - phase3_tags.csv
  - phase3_evidence.csv
  - phase3_rules.csv
  - phase3_audit.csv

## 出力
- data/scores/p3/<run_id>/
  - phase3_scores.csv
  - phase3_pass.csv

## 実行例
```bash
python scripts/phase3/run_phase3.py \
  --run_id 20260103_p3_screening_v1 \
  --candidates /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/20260103_p3_start_phaseE_gap_from_p2_75/phaseE_selected.csv \
  --corpus_dir /Users/shou/hobby/CPX/nikkei-stock/data/source/phase3/corpus \
  --config_tags /Users/shou/hobby/CPX/nikkei-stock/config/phase3/tags.yaml \
  --config_rules /Users/shou/hobby/CPX/nikkei-stock/config/phase3/rules.yaml
```

## メモ
- run_id は data/interim/phase3/<run_id>/ と data/scores/p3/<run_id>/ の両方に使う。
- 現状は雛形のみ。各モジュール内の TODO を埋めて実装する。


## Phase3: NLP-based Tagging & Ranking

Phase3 evaluates companies using ESG / climate-related disclosures by converting qualitative information
(integrated reports, disclosures) into structured, explainable signals.

### Overview

Phase3 is intentionally split into **two clearly separated steps**:

1. **Tagging & Scoring (Phase3 core)**
2. **Ranking & Shortlisting (rank_top)**

This separation is critical for explainability, reproducibility, and future extensibility.

---

### 1. Tagging & Scoring (Phase3 core)

**Input**
- Text extracted from integrated reports (PDF → TXT), one file per company.

**Tag system**
- Total tags: **55**
  - Core tags: 27 (used for scoring)
  - Support tags: 18
  - Observe tags: 10
- Each tag is evaluated as:
  - `yes` / `no` / `unk`

**Configuration**
- Tag definitions: `config/phase3/tags.yaml`
- Rule-based detection (regex-based): `config/phase3/rules.yaml`

**Processing flow**
- `extract_tags.py`  
  → Detects tag-level signals from text using rules and patterns.
- `apply_rules.py`  
  → Aggregates tag results into structural axes (core / governance / Scope3, etc.).
- `build_audit_log.py`  
  → Records why each decision was made (traceability and auditability).

**Output**
- `phase3_scores.csv`
  - Rows represent aggregated axes such as:
    - `core_score` (e.g. `13/27`)
    - `struct::ガバナンス`
    - `struct::Scope3`
    - `struct::サプライチェーン`
    - `struct::透明性`
    - etc.
  - This file is the **single source of truth** for Phase3 results.

---

### 2. Ranking & Shortlisting (`rank_top.py`)

**Purpose**
- Convert Phase3 scoring results into an explicit, explainable ranking.
- No NLP or text processing is performed at this stage.

**Script**
- `scripts/phase3/tagging/rank_top.py`

**Input**
- `phase3_scores.csv`

**Ranking logic**

Primary metric:
- `core_yes`  
  → Numerator of `core_score` (e.g. `13/27 → 13`)

Tie-breakers (descending order):
1. `structure_coverage`  
   Number of `struct::*` axes with at least one core-yes signal.
2. `impl_strength`  
   Sum of `struct::ガバナンス` + `struct::Scope3`.
3. `struct::透明性`
4. `struct::サプライチェーン`
5. `struct::事業転換`
6. `struct::イノベーション`
7. `struct::規制対応`

**Output**
- `phase3_ranktop_full.csv`  
  → Full ranked list of companies.
- `phase3_ranktop_top20.csv`  
  → Top-20 shortlist used for subsequent analysis.

These outputs are designed to be:
- Directly reusable in Notion or reports
- Extendable with external data (e.g. J-PlatPat, ICP) without modifying Phase3 core logic

---

### Design Rationale

- **Explainability**  
  Every score and ranking can be traced back to tags and rules.
- **Separation of concerns**  
  - Phase3 core = evaluation
  - rank_top = decision support
- **Extensibility**  
  External quantitative data can be layered on top at the ranking stage.

This structure is intentionally aligned with competition-level evaluation standards,
emphasizing reproducibility, transparency, and auditability.