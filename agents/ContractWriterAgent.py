from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Iterable, Sequence

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_json_list
from utils.common import ModelConfig, DocumentBundle, load_file
from utils.prompt_manager import PromptManager
from utils.settings import logger


class ContractWriterAgent(Agent):
    """An agent to fetch posts from a public API."""

    config: "ContractWriterAgent.Config"

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

    async def run(self, semantic_documents: List[Dict[str,Any]]) -> Dict[str, Any]:

        processed_docs = []
        for doc in semantic_documents:
            try:
                doc_id = doc['document_id']
                doc_bundle=DocumentBundle(str(doc_id))
                semantic_repr = load_file(Path(doc['path']))
                user_prompt = self.config.pm.compose_prompt(
                    "level_2_reasoning_v1.j2",
                    semantic_repr=semantic_repr
                )
                messages = [
                    {"role": "user", "content": user_prompt}
                ]

                page_blocks = await call_llm(messages, self.config.model_config, coerce_to_json_list, metadata={})
                res_path = self.save_locally(f"contract_{doc_id}.json", page_blocks)
                processed_docs.append({"document_id": doc_id, "path": str(res_path)})
            except Exception as e:
                logger.error(e)

        return {
            "contracts.json": processed_docs
        }
