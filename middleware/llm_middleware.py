import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, TypeVar, Callable, Protocol, Coroutine, Tuple, Optional

from openai import OpenAI

from utils.common import ModelConfig
from time import perf_counter_ns

T_Coerced = TypeVar("T_Coerced")


@dataclass
class LLMMetrics:
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


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
            metrics: Optional[LLMMetrics] = None,
    ) -> Coroutine[Any, Any, T_Coerced]:
        ...


async def call_llm(messages: List[Any],
                   config: ModelConfig,
                   transformer: Callable[[Any], T_Coerced],
                   metadata: Dict[str, Any],
                   metrics: Optional[LLMMetrics] = None) -> T_Coerced:
    last_err = None
    client = get_client(config=config)
    for _ in range(config.retries):
        try:
            start = perf_counter_ns()
            # Responses API (multi-modal)
            resp = client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                messages=messages,
                reasoning_effort="high",
                verbosity="medium"
            )

            if metrics is not None:
                elapsed_ns = perf_counter_ns() - start
                metrics.latency_ms = elapsed_ns // 1_000_000
                metrics.prompt_tokens = resp.usage.prompt_tokens
                metrics.completion_tokens = resp.usage.completion_tokens

            content = resp.choices[0].message.content.strip()
            # print(content)
            return transformer(content)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"OpenAI parse failed for {last_err}")


class LLMClient:
    pass
