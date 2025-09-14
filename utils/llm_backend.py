import json
import re
from time import perf_counter_ns
from typing import List, Dict, Any

from openai import OpenAI


def coerce_blocks(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    raise ValueError("LLM did not return a JSON list.")


async def call_openai_parse(
        client: OpenAI,
        messages: List[Any],
        model: str = "o4-mini",
        temperature: float = 1.0,
        max_retries: int = 1) -> List[Dict[str, Any]]:
    last_err = None

    for _ in range(max_retries):
        try:
            start = perf_counter_ns()
            # Responses API (multi-modal)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
            latency = (perf_counter_ns() - start) // 1_000_000  # ms
            # Extracting and printing the response content
            length = len(str(messages))
            print(f"prompt_tokens: {resp.usage.prompt_tokens}")
            print(f"completion_tokens: {resp.usage.completion_tokens}")
            print(f"input bytes: {length}")

            content = resp.choices[0].message.content.strip()
            # print(content)
            # Extract first [...] to avoid stray characters
            m = re.search(r"\[.*\]", content, flags=re.DOTALL)
            raw_json = m.group(0) if m else content
            data = json.loads(raw_json)
            return coerce_blocks(data)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"OpenAI parse failed for {last_err}")
