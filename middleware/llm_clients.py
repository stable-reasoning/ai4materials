# llm_clients.py
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Type

# --- Interface ---------------------------------------------------------------
from utils.common import ModelConfig


@dataclass(frozen=True)
class ModelConfig1:
    name: str
    model: str
    temperature: float


class LLMError(RuntimeError):
    pass


class BaseLLM(ABC):
    """Minimal, uniform interface for text generation."""

    @abstractmethod
    def from_config(self, conf: ModelConfig) -> "BaseLLM":
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        """Return the model's response text for a single-turn prompt."""
        raise NotImplementedError


# --- OpenAI implementation ---------------------------------------------------
# Docs (Responses + Chat Completions shown in SDK README). We use Chat Completions,
# which OpenAI states remains supported; you can swap to Responses if you prefer. :contentReference[oaicite:2]{index=2}
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class OpenAIClient(BaseLLM):
    model: str = "gpt-4o-mini"  # pick any available text model
    api_key: Optional[str] = None
    _client: OpenAI = None

    def from_config(self, conf: ModelConfig) -> "OpenAIClient":
        return OpenAIClient(
            model=conf.model,
        )

    def __post_init__(self):
        if OpenAI is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is missing.")
        self._client = OpenAI(api_key=key)

    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 512),
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # surface a consistent error
            raise LLMError(f"OpenAI error: {e}") from e


# --- Gemini implementation ---------------------------------------------------
# Uses the Google GenAI SDK (official & GA). It auto-picks GEMINI_API_KEY from env. :contentReference[oaicite:3]{index=3}
try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore


@dataclass
class GeminiClient(BaseLLM):
    model: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    _client: Any = None

    def __post_init__(self):
        if genai is None:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        # If api_key=None, the SDK reads GEMINI_API_KEY automatically. :contentReference[oaicite:4]{index=4}
        self._client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, *, system: Optional[str] = None, **kwargs) -> str:
        try:
            # Gemini accepts a list of "contents" parts; a single string works for simple cases. :contentReference[oaicite:5]{index=5}
            instruct = f"{system}\n\n{prompt}" if system else prompt
            resp = self._client.models.generate_content(
                model=self.model,
                contents=instruct,
                config={
                    "temperature": kwargs.get("temperature", 0.2),
                    "max_output_tokens": kwargs.get("max_tokens", 512),
                },
            )
            # The GenAI SDK provides .text for the aggregated text output. :contentReference[oaicite:6]{index=6}
            return getattr(resp, "text", "") or ""
        except Exception as e:
            raise LLMError(f"Gemini error: {e}") from e


# --- Simple registry / factory ----------------------------------------------

PROVIDERS: Dict[str, Type[BaseLLM]] = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def make_client(provider: str, **kwargs) -> BaseLLM:
    """Create a client by name: 'openai' or 'gemini'."""
    try:
        cls = PROVIDERS[provider.lower()]
    except KeyError:
        raise ValueError(f"Unknown provider '{provider}'. Choices: {list(PROVIDERS)}")
    return cls.from_config(**kwargs)


# --- Example usage -----------------------------------------------------------

if __name__ == "__main__":
    system_msg = "You are a concise, helpful assistant."
    prompt = "List three bullet points on why adapters are useful in Python."

    oa = make_client("openai", model="gpt-4o-mini")
    ge = make_client("gemini", model="gemini-2.5-flash")

    print("\n--- OpenAI ---")
    print(oa.generate(prompt, system=system_msg))

    print("\n--- Gemini ---")
    print(ge.generate(prompt, system=system_msg))
