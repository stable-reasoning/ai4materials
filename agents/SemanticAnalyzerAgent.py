from dataclasses import dataclass
from typing import Dict, Any, List, Iterable, Sequence

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_json_list
from utils.common import ModelConfig, DocumentBundle, load_file, prune_and_validate, to_jsonl
from utils.extraction_utils import get_keys_from_json_file
from utils.prompt_manager import PromptManager
from utils.settings import logger


class SemanticAnalyzerAgent(Agent):
    """An agent to fetch posts from a public API."""

    config: "SemanticAnalyzerAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig
        excluded_types: Iterable[str] = ("unknown", "reference_entry")

    def _filter_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            rec for rec in records
            if str(rec.get("type", "")).lower() not in self.config.excluded_types
        ]

    async def run(self, doc_ids: List[int]) -> Dict[str, Any]:

        processed_docs = []
        for doc_id in doc_ids:
            try:
                doc_bundle=DocumentBundle(str(doc_id))
                raw_records = load_file(doc_bundle.get_records_path())
                filtered = self._filter_records(raw_records)
                pruned = prune_and_validate(filtered)
                json_lines = to_jsonl(pruned)
                figure_labels = get_keys_from_json_file(doc_bundle.get_figures_path())
                table_labels = get_keys_from_json_file(doc_bundle.get_tables_path())
                system_prompt = self.config.pm.compose_prompt("level_1_reasoning_sys_v1.j2")
                user_prompt = self.config.pm.compose_prompt(
                    "level_1_reasoning_user_v1.j2",
                    figures=figure_labels,
                    tables=table_labels,
                    claims=json_lines
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                page_blocks = await call_llm(messages, self.config.model_config, coerce_to_json_list, metadata={})
                res_path = self.save_locally(f"semantic_{doc_id}.json", page_blocks)
                processed_docs.append({"document_id": doc_id, "path": str(res_path)})
            except Exception as e:
                logger.error(e)

        return {
            "processed_document_ids.json": processed_docs
        }
