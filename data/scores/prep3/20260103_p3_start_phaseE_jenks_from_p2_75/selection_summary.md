# PhaseE Selection Summary
- method: jenks
- input: /Users/shou/hobby/CPX/nikkei-stock/data/scores/p2_75/phaseE_rank_panel.csv
- rows_used: 183
- year_filter: 2023
- k_range: 3..7
- chosen_k: 7
- chosen_gvf: 0.118590
- target_selected: 40
- min_selected: 20
- boundary_class: 0
- selected_count: 40

## Rationale
PhaseEスコア（score__phaseE）に対してFisher–Jenks自然分類（Natural Breaks）を適用し、スコア分布の自然なまとまり（クラス）を抽出した。
上位クラスから順に第三次候補に含め、目標社数（target_selected）に到達するまで累積した。
目標到達時に境界クラスが大きい場合は、境界クラス内で開示の信頼性指標（n_scored_metrics, quality_weight, score__total）を優先して必要数のみを追加し、恣意性を抑えつつ候補数を調整した。

## Outputs
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/20260103_p3_start_phaseE_jenks_from_p2_75/phaseE_gap_rank_panel.csv
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/20260103_p3_start_phaseE_jenks_from_p2_75/phaseE_selected.csv
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/20260103_p3_start_phaseE_jenks_from_p2_75/selection_summary.md
