"""Judge client and score mapping for the QuAL LLM method."""

from judges.openai_compatible import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    openai_compatible_judge,
)
from judges.scores import (
    convert_score,
    get_metrics_config,
    normalized_to_binary_score,
    normalized_to_evidence_score,
)
from judges.system_prompt import FEEDBACK_EVALUATION_SYSTEM_PROMPT


def get_judge(temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS):
    """Return the OpenAI-compatible judge.

    The default server is local vLLM (``scripts/serve_vllm.sh``). Override
    with ``LLM_BASE_URL`` and ``LLM_API_KEY``. ``max_tokens`` defaults to 2048.

    Args:
        temperature: Sampling temperature.
        max_tokens: Completion token cap.

    Returns:
        A DeepEval model wrapper.
    """
    return openai_compatible_judge(temperature=temperature, max_tokens=max_tokens)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL_NAME",
    "FEEDBACK_EVALUATION_SYSTEM_PROMPT",
    "convert_score",
    "get_judge",
    "get_metrics_config",
    "normalized_to_binary_score",
    "normalized_to_evidence_score",
    "openai_compatible_judge",
]
