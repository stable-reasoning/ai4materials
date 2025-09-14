from dataclasses import dataclass
from typing import Dict, Any, List

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm
from utils.DocumentAnalyser import DocumentProcessor
from utils.common import ModelConfig, DocumentBundle
from utils.prompt_manager import PromptManager
from utils.settings import logger


class DocumentAnalyzerAgent(Agent):

    config: "DocumentAnalyzerAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig

    async def run(self, doc_ids: List[int]) -> Dict[str, Any]:

        processed_docs = []
        for doc_id in doc_ids:
            try:
                doc_processor = DocumentProcessor(
                    doc_bundle=DocumentBundle(str(doc_id)),
                    pm=self.config.pm,
                    model_config=self.config.model_config,
                    im_store=self.config.image_store,
                    llm_hook=call_llm
                )
                await doc_processor.process_document()
                processed_docs.append(doc_id)
            except Exception as e:
                logger.error(e)

        return {
            "processed_document_ids.json": processed_docs
        }

