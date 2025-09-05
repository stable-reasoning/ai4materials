from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Set

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_json_list
from utils.common import ModelConfig, DocumentBundle, load_file
from utils.prompt_manager import PromptManager
from utils.settings import logger


class QAAnswerAgent(Agent):
    """An agent to fetch posts from a public API."""

    config: "QAAnswerAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig
        metadata: Dict[str, Any]

    async def run(self, qa_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"answering {len(qa_dataset)} questions")
        processed_qs = []

        return {
            "answers.json": processed_qs
        }


