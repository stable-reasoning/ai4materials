import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_json
from utils.common import ModelConfig, DocumentBundle, load_file, Question, Answer
from utils.prompt_manager import PromptManager
from utils.settings import logger

# tqdm for progress bar; logging-aware redirection to avoid garbled logs
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def format_records(recs: List[Dict[str, Any]]) -> str:
    return '\n'.join([f"\n type: {r['type']}\n {r['text']}" for r in recs])


class QAAnswerAgent(Agent):
    """An agent to fetch posts from a public API."""

    config: "QAAnswerAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig
        metadata: Dict[str, Any]

    async def run(self, qa_dataset: List[Dict[str, Any]], contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"answering {len(qa_dataset)} questions")
        processed_qs = []
        qs = [Question(**item) for item in qa_dataset]
        contract_map = {c['document_id']: Path(c['path']) for c in contracts}
        batch = int(self.config.metadata.get('batch', 1))
        flags = self.config.metadata.get('context_flags', '')
        print(flags)

        # Progress bar around the main loop, while keeping logging clean
        with logging_redirect_tqdm():
            for q in tqdm(qs, total=len(qs), desc="Answering", unit="q"):
                doc_id = q.question_id.split('-')[0]
                raw_text = format_records(load_file(DocumentBundle(str(doc_id)).get_records_path()))
                contract = load_file(contract_map.get(doc_id))
                context = []
                if 'CC' in flags:
                    context.append(f"\n[CONTRACT]\n {contract}")
                if 'RAW_TEXT' in flags:
                    context.append(f"\n[RAW_TEXT]\n {raw_text}")
                user_prompt = self.config.pm.compose_prompt(
                    "qa_answering_llm.j2",
                    question=q.question,
                    context=context
                )
                messages = [
                    {"role": "user", "content": user_prompt}
                ]

                ans = await call_llm(messages, self.config.model_config, coerce_to_json, metadata={})
                if ans.get('answer') and ans.get('explanation'):
                    answer = Answer(
                        question_id=q.question_id,
                        question=q.question,
                        experiment_id=self.env.get('experiment_id', ''),
                        config_name=self.config.model_config.name,
                        question_type=q.question_type,
                        gold_answer=q.gold_answer,
                        gold_trace=q.gold_trace,
                        pred_answer=ans.get('answer'),
                        pred_trace=ans.get('explanation')
                    )
                    processed_qs.append(dataclasses.asdict(answer))
                else:
                    logger.error(f"answer is corrupted: {ans}")

        logger.info(f"got {len(processed_qs)} answers")

        return {
            "answers.json": processed_qs
        }
