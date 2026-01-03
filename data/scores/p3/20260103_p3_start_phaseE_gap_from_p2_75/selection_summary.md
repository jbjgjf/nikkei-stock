# PhaseE Gap-based Selection Summary

- input: /Users/shou/hobby/CPX/nikkei-stock/data/scores/p2_75/phaseE_rank_panel.csv
- rows_used: 183
- year_filter: 2023
- top_fraction_search: 0.30 (top 54 rows)
- cut_index (0-based): 9
- threshold(score__phaseE): 0.482642
- selected_count: 10

## Rationale
PhaseEスコアを降順に並べ、隣接スコア差（gap）の最大点を『不連続点』として採用した。
ただし下位側のノイズで巨大ギャップが出ることを避けるため、上位top_fractionの範囲内で最大ギャップを探索した。

## Outputs
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/20260103_p3_start_phaseE_gap_from_p2_75/phaseE_gap_rank_panel.csv
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/20260103_p3_start_phaseE_gap_from_p2_75/phaseE_selected.csv
- /Users/shou/hobby/CPX/nikkei-stock/data/scores/p3/20260103_p3_start_phaseE_gap_from_p2_75/selection_summary.md
