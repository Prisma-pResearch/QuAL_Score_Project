# Data

The comments scored for the paper are not in this folder and are not public.
`example_input.json` is five made-up comments of the same shape as the real ones, for checking
that the scripts run.

## Schema

A JSON list. Each object:

| Field | Meaning |
| --- | --- |
| `id` | Integer id. The full set uses 0 through 1508. |
| `input` | Empty string. The comment itself is `actual_output`. |
| `actual_output` | The feedback comment. |
| `evidence_r1`, `evidence_r2` | Rater evidence score, integer 0-3. |
| `suggestion_r1`, `suggestion_r2` | Rater suggestion score, integer 0 or 1. |
| `connection_r1`, `connection_r2` | Rater connection score. Stored as 0.0 or 1.0. |
| `total_r1`, `total_r2` | Sum of that rater's three scores, 0-5. |

After a judge run, the same objects also have `<metric>_llm` (mapped integer),
`<metric>_llm_raw` (DeepEval's 0-1 score), and `<metric>_llm_reason`.

## Split

The spreadsheet was read in order into one JSON list of 1,509 records.
Development is the first 300 records, ids 0-299. Test is the remaining
1,209, ids 300-1508. Paper numbers use the test split. The in-context run
reads the development split as examples and scores the test split.

Consensus is the test records where the two raters agree on evidence,
suggestion, and connection. That subset has 748 comments.

Binary evidence, used only when comparing to a binary evidence prediction,
maps rater scores 0 and 1 to 0 and scores 2 and 3 to 1. Suggestion and
connection are already binary.
