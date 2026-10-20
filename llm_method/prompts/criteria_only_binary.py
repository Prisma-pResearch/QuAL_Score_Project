"""Criteria-only prompt, binary evidence.

No evaluation steps are passed.

Unlike the ordinal criteria-only snapshot, this one pastes
``DECLARED_SYSTEM_PROMPT`` into every metric's criteria string, ahead of
the question. The judge client also sends the tuned system prompt.
"""

from typing import Any, List, Optional

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

PROMPT_ID = "criteria_only_binary"
PREPEND_DECLARED_PROMPT = True
USES_DEVELOPMENT_EXAMPLES = False

DECLARED_SYSTEM_PROMPT = """You are evaluating attending physician feedback given to medical residents during surgical training.

EVIDENCE: does the rater provide sufficient evidence about resident performance?
- Score 0: No comment.
- Score 1: yes/full description.

SUGGESTION: does the rater provide a clear, actionable suggestion for improvement? (0= no, 1= yes).
- Score 1: Yes
- Score 0: No

CONNECTION: is the suggestion explicitly linked to the specific evidence described? (0= no, 1= yes).
- Score 1: Yes
- Score 0: No

NOTES: If the feedback only gives a suggestion (i.e: you should a better job retracting), it automatically gets a score of 2 (1 point for evidence because comment was present but no evidence, and 1 point for suggestion, 0 points for connection)

    """


def _criteria(question: str) -> str:
    """Paste the short rubric in front of a criteria question.

    Args:
        question: The yes/no question for this metric.

    Returns:
        The criteria string.
    """
    return "SYSTEM_PROMPT:\n" + DECLARED_SYSTEM_PROMPT + "\n\n\n" + question


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
        criteria=_criteria("Does the rater provide sufficient evidence about resident performance?"),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    suggestion_metric = GEval(
        name="Suggestion",
        criteria=_criteria("Does the rater provide a suggestion for improvement?"),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    connection_metric = GEval(
        name="Connection",
        criteria=_criteria("Is the rater’s suggestion linked to the behavior described?"),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    return [evidence_metric, suggestion_metric, connection_metric]
