"""Binary evaluation-steps prompt with development-set examples in context.

Examples are not stored in this file. Pass a JSON file of development
records to ``load_shared_context``. The runner appends that text to each
test case's context.
"""

import json
from pathlib import Path
from typing import Any, List, Optional, Union

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

PROMPT_ID = "in_context_binary"
PREPEND_DECLARED_PROMPT = False
USES_DEVELOPMENT_EXAMPLES = True

DECLARED_SYSTEM_PROMPT = """You are evaluating attending physician feedback given to medical residents during surgical training.

Use the in-context examples in the test case context as calibration examples.
Follow the evaluation steps exactly and score only the provided feedback snippet.
"""

CONTEXT_HEADER = (
    "In-context examples from the 300 development set. "
    "Each item has `feedback` and both raters' labels."
)


def load_shared_context(examples_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Format development records as the shared context payload.

    Args:
        examples_path: JSON list of records with ``id``, ``actual_output``,
            and the rater label fields. Required.

    Returns:
        Two strings: a short header, then a compact JSON list.

    Raises:
        ValueError: If ``examples_path`` is missing.
        FileNotFoundError: If the file does not exist.
    """
    if examples_path is None:
        raise ValueError("in_context_binary requires --examples pointing at a development JSON file")
    path = Path(examples_path)
    if not path.is_file():
        raise FileNotFoundError(f"Examples file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list):
        raise ValueError(f"Examples file must be a JSON list: {path}")

    compact = []
    for sample in samples:
        compact.append(
            {
                "id": sample["id"],
                "feedback": sample.get("actual_output", ""),
                "labels": {
                    "evidence_r1": sample.get("evidence_r1"),
                    "evidence_r2": sample.get("evidence_r2"),
                    "suggestion_r1": sample.get("suggestion_r1"),
                    "suggestion_r2": sample.get("suggestion_r2"),
                    "connection_r1": sample.get("connection_r1"),
                    "connection_r2": sample.get("connection_r2"),
                },
            }
        )
    examples_json = json.dumps(compact, ensure_ascii=True, separators=(",", ":"))
    return [CONTEXT_HEADER, examples_json]


def get_feedback_metrics(model: Optional[Any] = None) -> List[GEval]:
    """Build binary Evidence, Suggestion, and Connection, using context.

    Args:
        model: DeepEval judge.

    Returns:
        The three strict-mode GEval metrics.
    """
    evaluation_params = [LLMTestCaseParams.CONTEXT, LLMTestCaseParams.ACTUAL_OUTPUT]
    evidence_metric = GEval(
        name="Evidence",
        evaluation_steps=[
            "Decide whether the feedback contains enough observed-performance evidence to support judging resident behavior.",
            "Treat specific observed details as stronger evidence than generic statements.",
            "Score 1 if sufficient evidence is present; otherwise score 0.",
        ],
        evaluation_params=evaluation_params,
        model=model,
        strict_mode=True,
    )
    suggestion_metric = GEval(
        name="Suggestion",
        evaluation_steps=[
            "Decide whether the feedback gives actionable advice for what the resident should do differently or better.",
            "Score 1 if actionable improvement advice is present; otherwise score 0.",
        ],
        evaluation_params=evaluation_params,
        model=model,
        strict_mode=True,
    )
    connection_metric = GEval(
        name="Connection",
        evaluation_steps=[
            "First determine whether a suggestion is present.",
            "If no suggestion is present, score 0.",
            "If a suggestion is present, score 1 only if it directly links to behavior/evidence described in the same feedback.",
        ],
        evaluation_params=evaluation_params,
        model=model,
        strict_mode=True,
    )
    return [evidence_metric, suggestion_metric, connection_metric]
