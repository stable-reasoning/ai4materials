from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI

from utils.common import DocumentBundle, SourceTxtBlock
from utils.ioutils import get_keys_from_json_file
from utils.llm_backend import call_openai_parse
from utils.prompt_manager import PromptManager


class QADatasetGenerator:

    def __init__(self, prompts: PromptManager, doc_id: str, llm_hook: Any,
                 excluded_types: Iterable[str] = ("unknown", "reference_entry")):
        self._excluded_types = {t.lower() for t in excluded_types}
        self._raw_records: List[Dict[str, Any]] = []
        self._blocks: List[SourceTxtBlock] = []
        self.prompts = prompts
        self.doc_bundle: DocumentBundle = DocumentBundle(doc_id)
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

    def prune_and_validate(self, records: Sequence[Dict[str, Any]]) -> List[SourceTxtBlock]:
        blocks: List[SourceTxtBlock] = []
        for rec in records:
            if "idx" not in rec or "type" not in rec or "text" not in rec:
                continue

            block = SourceTxtBlock(
                idx=int(rec["idx"]),
                type=str(rec["type"]),
                text=str(rec["text"])
            )
            blocks.append(block)
        return blocks

    def to_jsonl(self, blocks: Sequence[SourceTxtBlock]) -> str:
        lines = [json.dumps(asdict(b), ensure_ascii=False) for b in blocks]
        print(f"Compiled JSONL with {len(lines)} lines")
        return "\n".join(lines)

    def build_jsonl_from_file(self) -> str:
        self.load_file()
        filtered = self.filter_records(self._raw_records)
        validated = self.prune_and_validate(filtered)
        return self.to_jsonl(validated)

    async def process_document(self):
        json_lines = self.build_jsonl_from_file()
        figure_labels = get_keys_from_json_file(self.doc_bundle.get_figures_path())
        table_labels = get_keys_from_json_file(self.doc_bundle.get_tables_path())
        system_prompt = self.prompts.compose_prompt("level_1_reasoning_sys_v1.j2")
        user_prompt = self.prompts.compose_prompt(
            "level_1_reasoning_user_v1.j2",
            figures=figure_labels,
            tables=table_labels,
            claims=json_lines
        )
        # print(user_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        page_blocks = await call_openai_parse(self.llm_hook, messages, temperature=1.0)

        if not page_blocks:
            print(f"WARNING: No LLM response for {self.doc_bundle.doc_id}. Skipping.")
            return

        logging.info(f"generated {len(page_blocks)} questions")

        with open(self.doc_bundle.get_qa_path(), "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(page_blocks, ensure_ascii=False))


async def main():
    client = OpenAI()
    prompt_man = PromptManager()
    doc_processor = QADatasetGenerator(prompts=prompt_man, doc_id="0001", llm_hook=client)
    await doc_processor.process_document()


if __name__ == "__main__":
    asyncio.run(main())
