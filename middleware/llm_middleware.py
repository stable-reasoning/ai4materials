import json
import re
from typing import List, Dict, Any, TypeVar, Callable, Protocol, Coroutine

from openai import OpenAI

from utils.common import ModelConfig

T_Coerced = TypeVar("T_Coerced")


def coerce_to_json_list(content: Any) -> List[Dict[str, Any]]:
    # Extract first [...] to avoid stray characters
    m = re.search(r"\[.*\]", content, flags=re.DOTALL)
    raw_json = m.group(0) if m else content
    data = json.loads(raw_json)
    if isinstance(data, list):
        return data
    raise ValueError("LLM did not return a JSON list.")


def coerce_to_json(content: Any) -> Dict[str, Any]:
    """A simple transformer that just cleans up a string."""
    return json.loads(content)


def coerce_to_float(content: Any) -> float:
    return float(content)


def coerce_to_simple_string(content: Any) -> str:
    """A simple transformer that just cleans up a string."""
    return str(content).strip()


def get_client(config: ModelConfig) -> OpenAI:
    return OpenAI()


class GenericLLMCallable(Protocol):
    """A protocol for a generic, async LLM calling function."""

    def __call__(
            self,
            messages: List[Any],
            config: ModelConfig,
            transformer: Callable[[Any], T_Coerced],
            metadata: Dict[str, Any],
    ) -> Coroutine[Any, Any, T_Coerced]:
        ...


async def call_llm(messages: List[Any],
                   config: ModelConfig,
                   transformer: Callable[[Any], T_Coerced],
                   metadata: Dict[str, Any]) -> T_Coerced:
    last_err = None
    client = get_client(config=config)
    for _ in range(config.retries):
        try:
            # Responses API (multi-modal)
            resp = client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                messages=messages,
                reasoning_effort="high",
                verbosity="medium"
            )
            # Extracting and printing the response content
            length = len(str(messages))
            print(f"prompt_tokens: {resp.usage.prompt_tokens}")
            print(f"completion_tokens: {resp.usage.completion_tokens}")
            print(f"input bytes: {length}")

            content = resp.choices[0].message.content.strip()
            # print(content)
            return transformer(content)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"OpenAI parse failed for {last_err}")


class LLMClient:
    pass


async def call_openai_parse(
        client: OpenAI,
        messages: List[Any],
        config: ModelConfig,
        max_retries: int = 1) -> str:
    last_err = None

    for _ in range(max_retries):
        try:
            # Responses API (multi-modal)
            resp = client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                messages=messages,
            )
            # Extracting and printing the response content
            length = len(str(messages))
            print(f"prompt_tokens: {resp.usage.prompt_tokens}")
            print(f"completion_tokens: {resp.usage.completion_tokens}")
            print(f"input bytes: {length}")

            content = resp.choices[0].message.content.strip()
            # print(content)
            return str(content).strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"OpenAI parse failed for {last_err}")
