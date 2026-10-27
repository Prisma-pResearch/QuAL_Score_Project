"""Map DeepEval's 0-1 scores back onto the QuAL scales."""

from typing import Any, Dict, List, Tuple

# Inclusive upper bounds on the normalized score. A value at the boundary
# stays in the lower bin.
EVIDENCE_SCORE_UPPER_BOUNDS: Tuple[Tuple[float, int], ...] = (
    (0.10, 0),
    (0.35, 1),
    (0.65, 2),
)
BINARY_THRESHOLD = 0.5


def normalized_to_evidence_score(normalized_score: float) -> int:
    """Convert a normalized 0-1 score to an evidence score of 0, 1, 2, or 3.

    GEval reports an integer on a 0-10 rubric and divides by 10. The bins
    below line up with that rubric:

    - 0: about 0.0 (rubric point 0)
    - 1: about 0.1-0.3 (rubric points 1-3)
    - 2: about 0.4-0.6 (rubric points 4-6)
    - 3: about 0.7-1.0 (rubric points 7-10)

    Args:
        normalized_score: DeepEval score in 0-1.

    Returns:
        Evidence score on the 0-3 scale.
    """
    for upper_bound, score in EVIDENCE_SCORE_UPPER_BOUNDS:
        if normalized_score <= upper_bound:
            return score
    return 3


def normalized_to_binary_score(normalized_score: float) -> int:
    """Convert a normalized 0-1 score to 0 or 1.

    Strict-mode GEval metrics already return 0 or 1. The threshold still
    applies so a non-strict score is mapped the same way.

    Args:
        normalized_score: DeepEval score in 0-1.

    Returns:
        1 when the score is at least 0.5, otherwise 0.
    """
    return 1 if normalized_score >= BINARY_THRESHOLD else 0


def binarize_evidence_label(score: int) -> int:
    """Map an ordinal evidence label to low (0) or high (1).

    Scores 0 and 1 become 0. Scores 2 and 3 become 1.

    Args:
        score: Evidence label on the 0-3 scale.

    Returns:
        Binary evidence label.
    """
    return 0 if score <= 1 else 1


def convert_score(
    metric_name: str,
    normalized_score: float,
    binarize_evidence: bool = False,
) -> int:
    """Convert one DeepEval score to the scale used in the result files.

    Args:
        metric_name: Metric name, such as evidence, evidence_bin, or suggestion.
        normalized_score: DeepEval score in 0-1.
        binarize_evidence: When true, a metric named exactly "evidence" uses
            the binary mapping. evidence_bin and evidence_ord ignore this flag
            and follow their names.

    Returns:
        Integer score on the metric's original scale.
    """
    name = metric_name.lower()
    if "evidence_ord" in name:
        return normalized_to_evidence_score(normalized_score)
    if "evidence_bin" in name:
        return normalized_to_binary_score(normalized_score)
    if name == "evidence" and not binarize_evidence:
        return normalized_to_evidence_score(normalized_score)
    return normalized_to_binary_score(normalized_score)


def metric_config(metric: Any) -> Dict[str, Any]:
    """Serialize one GEval metric for the run folder.

    Args:
        metric: GEval metric instance.

    Returns:
        Name, criteria, steps, strict mode, and rubric, when present.
    """
    config: Dict[str, Any] = {
        "name": metric.name,
        "criteria": metric.criteria,
        "evaluation_steps": metric.evaluation_steps,
        "strict_mode": getattr(metric, "strict_mode", False),
    }
    rubric = getattr(metric, "rubric", None)
    if rubric:
        config["rubric"] = [
            {
                "score_range": list(item.score_range),
                "expected_outcome": item.expected_outcome,
            }
            for item in rubric
        ]
    return config


def get_metrics_config(metrics: List[Any]) -> Dict[str, Any]:
    """Serialize the metrics used in a run.

    Args:
        metrics: GEval metrics passed to DeepEval.

    Returns:
        A JSON-ready dict.
    """
    return {"metrics": [metric_config(metric) for metric in metrics]}
