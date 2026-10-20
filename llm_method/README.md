# QuAL LLM judge

This folder contains code and summary results relating to the Large Language Model (LLM) approach.

Each feedback comment is scored by gpt-oss-120b through the GEval
metric from the DeepEval Python library. One model call per dimension:

- Evidence: 0-3, or binary 0/1 when that run says so
- Suggestion: 0/1
- Connection: 0/1

The client always sends `judges/system_prompt.py` as the system message,
including on the evaluation-steps runs. Prompt text for each paper setup
lives in `prompts/`.

## Setup

This project uses the [uv](https://docs.astral.sh/uv/) Python package manager.

```bash
cd llm_method
uv sync
cp .env.example .env
./scripts/serve_vllm.sh
```

`scripts/serve_vllm.sh` starts a local vLLM server for gpt-oss-120b.
`vllm` must already be on `PATH`. The server listens on port 8000.

There is one client, which is set up for the OpenAI chat-completions API.
It reads `LLM_API_KEY`, `LLM_BASE_URL`, and optional `LLM_MODEL`. Unset, those are key `dummy`,
`http://127.0.0.1:8000/v1`, and `gpt-oss-120b`.
`--max-tokens` defaults to 2048.

## Score a file

To run on your own data:

```bash
uv run python run_judge.py \
  --input path/to/test.json \
  --prompt eval_steps_binary \
  --bin-evidence
```

New runs are written under `runs/` in this folder. Files placed there are not committed.

Bootstrap confidence intervals (CI):

```bash
uv run python -m evaluation.bootstrap_ci \
  --input runs/<run>/evaluated_test_cases.json \
  --seed 42
```

`results/` holds the saved point-estimate and CI CSVs from the paper runs, named after the config id (`<id>.csv` and `<id>_ci.csv`), which is the name you can choose for each run.

Creating figures:

```bash
uv run python -m evaluation.figures combo \
  --ordinal-csv results/steps_ordinal.csv \
  --binary-csv results/steps_binary.csv \
  --ordinal-ci results/steps_ordinal_ci.csv \
  --binary-ci results/steps_binary_ci.csv \
  --title "Evaluation steps" \
  --output /tmp/eval_steps.png
```

Temperature figures:

```bash
uv run python -m evaluation.figures temperature \
  --label T=0.0 --points results/temp_0_0.csv --ci results/temp_0_0_ci.csv \
  --label T=0.5 --points results/temp_0_5.csv --ci results/temp_0_5_ci.csv \
  --label T=1.0 --points results/temp_1_0.csv --ci results/temp_1_0_ci.csv \
  --output /tmp/temperature.png
```

## Paper runs

Test split only (1,209 comments). Suggestion and connection in the main
comparison come from the binary run of that family, not the ordinal run.

Commands in this folder call the local vLLM server from
`scripts/serve_vllm.sh`. Output tokens default to 2048.

| Config id | Prompt | Temp | Paper row |
| --- | --- | --- | --- |
| `criteria_ordinal` | criteria, ordinal evidence | 0.0 | Criteria block, evidence ordinal |
| `criteria_binary` | criteria, binary evidence | 0.0 | Criteria block, evidence binary, suggestion, connection |
| `steps_ordinal` | written steps, ordinal evidence | 0.0 | Steps block, evidence ordinal |
| `steps_binary` | written steps, binary evidence | 0.0 | Steps block, evidence binary, suggestion, connection |
| `temp_0_0`, `temp_0_5`, `temp_1_0` | steps, both evidence scales | 0.0 / 0.5 / 1.0 | Temperature figure |
| `in_context_steps` | steps plus dev-set examples | 0.0 | Not in the main table |

For ordinal evidence, precision, recall, and F1 are the 4-class scores
weighted by the number of true examples in each class. Weighted recall
equals accuracy. `*_macro` columns, where present, are the unweighted
average of the same per-class scores. For a binary metric, precision,
recall, and F1 are the positive-class scores.

`results/<id>.csv` and `results/<id>_ci.csv` are the saved tables for each config id.

## Data

The real comments are not in this folder. Schema, the 300/1,209 split, and the
consensus rule are in `data/README.md`. `data/example_input.json` is five
made-up comments for a dry run of the metric scripts.
