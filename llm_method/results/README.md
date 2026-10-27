# Aggregate results

These files are the metric tables from the paper runs. They contain scores
only and do not contain the comments themselves or the LLM reasoning traces.

`*.csv` is the point-estimate table from that run. For an ordinal evidence
metric, precision, recall, and F1 are the 4-class scores weighted by the
number of true examples in each class. Precision, recall, and F1 are
computed for each class and then averaged (scikit-learn `average='weighted'`).
Weighted recall equals accuracy. Columns ending in `_macro` are the
unweighted average of those same per-class scores (scikit-learn
`average='macro'`). For a binary metric, precision, recall, and F1 are the
positive-class scores.

`*_ci.csv` is the 1,000-draw bootstrap. There is no interval file for the
in-context run.

| File prefix | What the table uses it for |
| --- | --- |
| `criteria_ordinal` | Criteria block, evidence ordinal |
| `criteria_binary` | Criteria block, evidence binary, suggestion, connection |
| `steps_ordinal` | Evaluation-steps block, evidence ordinal |
| `steps_binary` | Evaluation-steps block, evidence binary, suggestion, connection |
| `temp_0_0` | Temperature 0.0 |
| `temp_0_5` | Temperature 0.5 |
| `temp_1_0` | Temperature 1.0 |
| `in_context_steps` | In-context binary run, not in the main table |

Rows are `r1`, `r2`, `both` (2,418 comparisons), and `consensus` (748 comments).
