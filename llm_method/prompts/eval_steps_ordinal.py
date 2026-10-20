"""Evaluation-steps prompt, ordinal evidence.

Evidence has both a criteria question and a 0-10 rubric. Suggestion and
Connection are strict 0/1 with written steps.
"""

from typing import Any, List, Optional

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCaseParams

PROMPT_ID = "eval_steps_ordinal"
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
EVIDENCE_RUBRIC = [
    Rubric(score_range=(0, 0), expected_outcome="No comment provided at all"),
    Rubric(score_range=(1, 3), expected_outcome="Comment present but no or minimal evidence of performance"),
    Rubric(score_range=(4, 6), expected_outcome="Comment present, brief mention of evidence partial details/description"),
    Rubric(score_range=(7, 10), expected_outcome="Description with evidence of resident performance"),
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
    """Build Evidence (0-3), Suggestion (0/1), and Connection (0/1).

    Args:
        model: DeepEval judge.

    Returns:
        The three GEval metrics.
    """
    evidence_metric = GEval(
        name="Evidence",
        criteria="Does the rater provide sufficient evidence about resident performance?",
        evaluation_steps=EVIDENCE_STEPS,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        rubric=EVIDENCE_RUBRIC,
    )
    suggestion_metric = GEval(
        name="Suggestion",
        criteria="Does the rater provide a suggestion for improvement?",
        evaluation_steps=SUGGESTION_STEPS,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    connection_metric = GEval(
        name="Connection",
        criteria="Is the rater’s suggestion linked to the behavior described?",
        evaluation_steps=CONNECTION_STEPS,
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    return [evidence_metric, suggestion_metric, connection_metric]
