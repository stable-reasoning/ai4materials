import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, Any, List

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_float, coerce_to_simple_string, identity
from utils.common import ModelConfig, Answer
from utils.prompt_manager import PromptManager
from utils.settings import logger


class QADesignerAgent(Agent):
    config: "QADesignerAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig
        metadata: Dict[str, Any]

    async def run(self, contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"designing new materials")
        # compiling the contract
        logger.info(f"loaded {len(contracts)} list")

        contract_id = self.config.metadata.get('contract_id')
        if not contract_id:
            raise ValueError(f"missing argument: contract_id")

        try:
            contract = next(c for c in contracts if c.get('contract_id') == contract_id)
        except StopIteration as e:
            raise ValueError(f"contract with id={contract_id} not found")

        user_prompt = self.config.pm.compose_prompt(
            "material_design_v1.j2",
            contract=json.dumps(contract)
        )
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        ans = await call_llm(messages, self.config.model_config, identity, metadata={})
        path = self.save_raw_ouput_locally(f"mat_design_{contract_id}.csv", ans)
        return {
            "designs_paths.json": str(path)
        }
