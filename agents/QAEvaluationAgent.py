from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Set

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_json_list
from utils.common import ModelConfig, DocumentBundle, load_file
from utils.prompt_manager import PromptManager
from utils.settings import logger


class QAEvaluationAgent(Agent):
    """An agent to fetch posts from a public API."""

    config: "QAEvaluationAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig
        metadata: Dict[str, Any]

    async def run(self, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"answering {len(answers)} questions")
        processed_qs = []

        return {
            "results.json": processed_qs
        }


