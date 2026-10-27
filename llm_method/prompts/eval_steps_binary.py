"""Evaluation-steps prompt, binary evidence.

Each metric is strict 0/1 and is given evaluation steps only. There is
no criteria string and no rubric.
"""

from typing import Any, List, Optional

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

PROMPT_ID = "eval_steps_binary"
PREPEND_DECLARED_PROMPT = False
USES_DEVELOPMENT_EXAMPLES = False

EVIDENCE_STEPS = [
    "Check if there is any comment at all about the resident's performance",
    "If a comment exists, assess whether it provides evidence of observed behavior",
    "Evaluate if the evidence includes details of what was observed",
]
SUGGESTION_STEPS = [
    "Identify if the feedback contains a suggestion for how the resident can improve",
    "Score 1 if a suggestion for improvement is present, 0 if no suggestion",
]
CONNECTION_STEPS = [
    "Identify if the feedback contains a suggestion for how the resident can improve",
    "If there is no suggestion, score 0",
    "If there is a suggestion, assess whether the suggestion directly addresses or relates to the described behavior",
    "Score 1 if the suggestion is clearly connected to the described behavior",
    "Score 0 if there is no suggestion, no described behavior, or they are not connected",
]


def load_shared_context(examples_path: Optional[str] = None) -> List[str]:
    """This setup does not add in-context examples.

    Args:
        examples_path: Unused.

    Returns:
        An empty list.
    """
    del examples_path
    return []


def get_feedback_metrics(model: Optional[Any] = None) -> List[GEval]:
    """Build binary Evidence, Suggestion, and Connection.

    Args:
        model: DeepEval judge.

    Returns:
        The three strict-mode GEval metrics.
    """
    evidence_metric = GEval(
        name="Evidence",
        evaluation_steps=EVIDENCE_STEPS,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    suggestion_metric = GEval(
        name="Suggestion",
        evaluation_steps=SUGGESTION_STEPS,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    connection_metric = GEval(
        name="Connection",
        evaluation_steps=CONNECTION_STEPS,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    return [evidence_metric, suggestion_metric, connection_metric]
