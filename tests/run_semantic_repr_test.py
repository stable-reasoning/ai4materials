from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI

from utils.common import DocumentBundle, SourceTxtBlock, prune_and_validate, to_jsonl, pick_keys, Answer, ModelConfig
from utils.llm_backend import test_call_openai_parse
from utils.prompt_manager import PromptManager
from utils.settings import global_config


class QASemanticReprTest:

    def __init__(self, prompts: PromptManager, doc_id: str, llm_hook: Any,
                 excluded_types: Iterable[str] = ("unknown", "reference_entry")):
        self._excluded_types = {t.lower() for t in excluded_types}
        self._raw_records: List[Dict[str, Any]] = []
        self._blocks: List[SourceTxtBlock] = []
        self.prompts = prompts
        self.doc_bundle: DocumentBundle = DocumentBundle(doc_id)
        self.out_dir = Path(global_config.runs_path)
        self.llm_hook = llm_hook

    def load_records(self, from_path: Path) -> List[Dict[str, Any]]:

        text = from_path.read_text(encoding="utf-8").strip()
        records: List[Dict[str, Any]]

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                records = parsed
            else:
                raise ValueError("Top-level JSON must be an array for non-JSONL mode.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        print(f"Loaded {len(records)} records from {from_path}")
        return records

    def filter_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:

        filtered: List[Dict[str, Any]] = []
        for rec in records:
            t = str(rec.get("type", "")).lower()
            if t in self._excluded_types:
                continue
            filtered.append(rec)
        return filtered

    def build_jsonl_from_file(self, path: Path) -> str:
        self._raw_records = self.load_records(path)
        filtered = self.filter_records(self._raw_records)
        validated = prune_and_validate(filtered)
        return to_jsonl(validated)

    async def run_batch(self, questions: List[Dict[str, Any]], config: ModelConfig):
        source_text_jsonl = self.build_jsonl_from_file(self.doc_bundle.get_records_path())
        semantic_repr_jsonl = self.load_records(self.doc_bundle.get_semantic_layer_path())
        print(f"{len(semantic_repr_jsonl)}")
        run_id = str(uuid.uuid4())
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        results = []
        for question in questions:
            q = pick_keys(question, ['id', 'type', 'options'])

            user_prompt = self.prompts.compose_prompt(
                "qa_dataset_answer_semantic_repr.j2",
                question=str(q),
                source_text_blocks=source_text_jsonl,
                semantic_repr=semantic_repr_jsonl
            )
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            pred_answer = await test_call_openai_parse(self.llm_hook, messages, config=config)
            answer = Answer(
                    question_id=question['id'],
                    run_id=run_id,
                    config_name=config.name,
                    question_type=question['type'],
                    gold_answer=question['answer'],
                    pred_answer=pred_answer
                )
            print(answer)
            results.append(dataclasses.asdict(answer))

        logging.info(f"generated {len(results)} answers")

        results_file_name = self.out_dir / f"{self.doc_bundle.doc_id}-{ts}"
        with open(results_file_name, "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(results, ensure_ascii=False))


async def main():
    client = OpenAI()
    prompt_man = PromptManager()
    doc_bundle = DocumentBundle("0001")
    with open(doc_bundle.get_qa_path(), 'r', encoding='utf-8') as f:
        data = json.load(f)
    for idx, r in enumerate(data):
        r['id'] = f"{doc_bundle.doc_id}-{idx}"
    config = ModelConfig(
        name = 'test1',
        model='o4-mini',
        temperature=1.0
    )
    doc_processor = QASemanticReprTest(prompts=prompt_man, doc_id="0001", llm_hook=client)
    await doc_processor.run_batch(questions=data, config=config)


if __name__ == "__main__":
    asyncio.run(main())
