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
