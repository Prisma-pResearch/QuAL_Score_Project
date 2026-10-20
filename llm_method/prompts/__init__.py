"""Paper prompt setups. Each module builds the GEval metrics for one run."""

import importlib
from types import ModuleType

PROMPT_IDS = (
    "criteria_only_ordinal",
    "criteria_only_binary",
    "eval_steps_ordinal",
    "eval_steps_binary",
    "eval_steps_combo",
    "in_context_binary",
)


def load_prompt(name: str) -> ModuleType:
    """Import a prompt module by id.

    Args:
        name: One of ``PROMPT_IDS``.

    Returns:
        The prompt module.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name not in PROMPT_IDS:
        available = ", ".join(PROMPT_IDS)
        raise ValueError(f"Unknown prompt: {name}. Available: {available}")
    return importlib.import_module(f"prompts.{name}")


__all__ = ["PROMPT_IDS", "load_prompt"]
