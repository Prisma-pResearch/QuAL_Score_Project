"""Criteria-only prompt, ordinal evidence.

GEval is given a criteria string and a 0-10 rubric, and no evaluation
steps.
"""

from typing import Any, List, Optional

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCaseParams

PROMPT_ID = "criteria_only_ordinal"
PREPEND_DECLARED_PROMPT = False
USES_DEVELOPMENT_EXAMPLES = False

DECLARED_SYSTEM_PROMPT = """You are evaluating attending physician feedback given to medical residents during surgical training.

EVIDENCE: does the rater provide sufficient evidence about resident performance?
- Score 0: No comment.
- Score 1: no sufficient evidence, but comment present.
- Score 2: comment present, evidence somewhat sufficient but without details/full description.
- Score 3: yes/full description.

SUGGESTION: does the rater provide a clear, actionable suggestion for improvement? (0= no, 1= yes).
- Score 1: Yes
- Score 0: No

CONNECTION: is the suggestion explicitly linked to the specific evidence described? (0= no, 1= yes).
- Score 1: Yes
- Score 0: No

NOTES: If the feedback only gives a suggestion (i.e: you should a better job retracting), it automatically gets a score of 2 (1 point for evidence because comment was present but no evidence, and 1 point for suggestion, 0 points for connection)

    """

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
        The three GEval metrics, in that order.
    """
    evidence_metric = GEval(
        name="Evidence",
        criteria="Does the rater provide sufficient evidence about resident performance?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        rubric=EVIDENCE_RUBRIC,
    )
    suggestion_metric = GEval(
        name="Suggestion",
        criteria="Does the rater provide a suggestion for improvement?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    connection_metric = GEval(
        name="Connection",
        criteria="Is the rater’s suggestion linked to the behavior described?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model,
        strict_mode=True,
    )
    return [evidence_metric, suggestion_metric, connection_metric]
