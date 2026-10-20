"""OpenAI-compatible chat client.

vLLM can output to this API, and it is the default server: ``LLM_BASE_URL`` falls
back to the local URL printed by ``scripts/serve_vllm.sh``. Set the same
variables to point the same client at any other OpenAI-compatible server.

The client does not request token log probabilities. GEval can weight the
score token by those probabilities, but only when the client returns them.
The paper runs' raw scores are whole tenths (or exact 0/1 in strict mode),
which is what you get from the parsed integer score alone.
"""

import os
from typing import Optional

from deepeval.models import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

from judges.system_prompt import FEEDBACK_EVALUATION_SYSTEM_PROMPT

# Local server started by scripts/serve_vllm.sh (port 8000, OpenAI /v1 route).
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
# fake API key; any value works for the local vLLM server but replace with real API key for cloud inference
DEFAULT_API_KEY = "dummy"
DEFAULT_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MAX_TOKENS = 2048


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    """DeepEval judge that speaks the OpenAI chat-completions API.

    Every call sends ``system_prompt`` as the system message, then the
    GEval user prompt. ``generate`` returns the message text only.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str = FEEDBACK_EVALUATION_SYSTEM_PROMPT,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    def load_model(self) -> OpenAI:
        """Return the sync client."""
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _async_openai(self) -> AsyncOpenAI:
        """Return the async client."""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._async_client

    def _messages(self, prompt: str) -> list:
        """Build the system-plus-user message list.

        Args:
            prompt: GEval user prompt.

        Returns:
            Chat messages.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(self, prompt: str, *args: object, **kwargs: object) -> str:
        """Generate a completion.

        Extra arguments from newer DeepEval releases are ignored. Schema
        objects are not filled in here; DeepEval parses the returned text.

        Args:
            prompt: User prompt produced by GEval.

        Returns:
            Assistant message text.
        """
        del args, kwargs
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(str(prompt)),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str, *args: object, **kwargs: object) -> str:
        """Generate a completion asynchronously.

        Args:
            prompt: User prompt produced by GEval.

        Returns:
            Assistant message text.
        """
        del args, kwargs
        client = self._async_openai()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=self._messages(str(prompt)),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def get_model_name(self) -> str:
        """Return the served model name."""
        return self.model_name


def openai_compatible_judge(
    temperature: float = 0.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> OpenAICompatibleJudge:
    """Build the judge from ``LLM_API_KEY``, ``LLM_BASE_URL``, and ``LLM_MODEL``.

    Missing values fall back to a local vLLM server on port 8000. The base
    URL is not rewritten: a trailing slash is stripped, and ``/v1`` is not
    added.

    Args:
        temperature: Sampling temperature.
        max_tokens: Completion token cap.

    Returns:
        Configured judge.
    """
    api_key = os.getenv("LLM_API_KEY") or DEFAULT_API_KEY
    base_url = (os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_name = os.getenv("LLM_MODEL") or DEFAULT_MODEL_NAME
    return OpenAICompatibleJudge(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
