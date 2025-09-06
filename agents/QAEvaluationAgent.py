import dataclasses
from dataclasses import dataclass
from typing import Dict, Any, List

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_float
from utils.common import ModelConfig, Answer
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
        logger.info(f"evaluating {len(answers)} answers")
        processed_qs = []
        ans_set = [Answer(**a) for a in answers]
        batch = int(self.config.metadata.get('batch', 1))

        for a in ans_set:
            user_prompt = self.config.pm.compose_prompt(
                "qa_answer_evaluation_v1.j2",
                answer=dataclasses.asdict(a)
            )
            messages = [
                {"role": "user", "content": user_prompt}
            ]

            ans = await call_llm(messages, self.config.model_config, coerce_to_float, metadata={})
            if ans > -1:
                a.eval_score = ans
                print(a)
                processed_qs.append(dataclasses.asdict(a))
            else:
                logger.error(f"answer is corrupted: {ans}")

            logger.info(f"got {len(processed_qs)} answers")

        return {
            "eval_answers.json": processed_qs
        }
