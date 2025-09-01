from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI

from utils.common import DocumentBundle, SourceTxtBlock, prune_and_validate, to_jsonl
from utils.ioutils import get_keys_from_json_file
from utils.llm_backend import call_openai_parse, test_call_openai_parse
from utils.prompt_manager import PromptManager
from utils.settings import global_config


class QABaseTest:

    def __init__(self, prompts: PromptManager, doc_id: str, llm_hook: Any,
                 excluded_types: Iterable[str] = ("unknown", "reference_entry")):
        self._excluded_types = {t.lower() for t in excluded_types}
        self._raw_records: List[Dict[str, Any]] = []
        self._blocks: List[SourceTxtBlock] = []
        self.prompts = prompts
        self.doc_bundle: DocumentBundle = DocumentBundle(doc_id)
        self.out_dir = Path(global_config.runs_path)
        self.llm_hook = llm_hook

    def load_file(self) -> List[Dict[str, Any]]:

        path = self.doc_bundle.get_records_path()
        text = path.read_text(encoding="utf-8").strip()
        records: List[Dict[str, Any]]

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                records = parsed
            else:
                raise ValueError("Top-level JSON must be an array for non-JSONL mode.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        self._raw_records = records
        print(f"Loaded {len(records)} records from {path}")
        return records

    def filter_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:

        filtered: List[Dict[str, Any]] = []
        for rec in records:
            t = str(rec.get("type", "")).lower()
            if t in self._excluded_types:
                continue
            filtered.append(rec)
        return filtered

    def build_jsonl_from_file(self) -> str:
        self.load_file()
        filtered = self.filter_records(self._raw_records)
        validated = prune_and_validate(filtered)
        return to_jsonl(validated)

    async def run_batch(self):
        json_lines = self.build_jsonl_from_file()

        run_id = str(uuid.uuid4())
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        results = []

        question = ""
        user_prompt = self.prompts.compose_prompt(
            "qa_dataset_answer_llm.j2",
            question=question,
            context=json_lines
        )
        # print(user_prompt)
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        page_blocks = await test_call_openai_parse(self.llm_hook, messages, temperature=1.0)

        if not page_blocks:
            print(f"WARNING: No LLM response for {self.doc_bundle.doc_id}. Skipping.")
            return

        logging.info(f"generated {len(page_blocks)} questions")

        with open(self.doc_bundle.get_qa_path(), "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(page_blocks, ensure_ascii=False))


async def main():
    client = OpenAI()
    prompt_man = PromptManager()
    doc_processor = QABaseTest(prompts=prompt_man, doc_id="0001", llm_hook=client)
    await doc_processor.run_batch()


if __name__ == "__main__":
    asyncio.run(main())
