import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, LLMMetrics, coerce_to_json_list
from utils.common import ModelConfig, DocumentBundle, load_file, Question, Answer
from utils.prompt_manager import PromptManager
from utils.settings import logger


def format_records(recs: List[Dict[str, Any]]) -> str:
    return '\n'.join([f"\n type: {r['type']}\n {r['text']}" for r in recs])


def mask(q: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in q.items() if k in ['question_id', 'question_type', 'question']}


class QAAnswerAgent(Agent):

    config: "QAAnswerAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig
        metadata: Dict[str, Any]

    async def run(self, qa_dataset: List[Dict[str, Any]], contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"answering {len(qa_dataset)} batch(es) of questions")
        processed_qs = []
        contract_map = {c['document_id']: Path(c['path']) for c in contracts}
        questions_map = {c['document_id']: Path(c['path']) for c in qa_dataset}
        docs = questions_map.keys()
        flags = self.config.metadata.get('context_flags', '')
        print(flags)

        # Progress bar around the main loop, while keeping logging clean
        with logging_redirect_tqdm():
            for doc_id in tqdm(docs, total=len(docs), desc="Answering", unit="q"):
                raw_text = format_records(load_file(DocumentBundle(str(doc_id)).get_records_path()))
                questions = load_file(questions_map[doc_id])
                logger.info(f"loaded {len(questions)} questions")
                q_map = {q['question_id']: q for q in questions}
                contract = load_file(contract_map.get(doc_id))
                masked_qs = [mask(q) for q in questions]
                context = []
                if 'CC' in flags:
                    context.append(f"\n[CONTRACT]\n {contract}")
                if 'RAW_TEXT' in flags:
                    context.append(f"\n[RAW_TEXT]\n {raw_text}")
                context = context or ['-']
                user_prompt = self.config.pm.compose_prompt(
                    "qa_answering_llm.j2",
                    questions=masked_qs,
                    context=context
                )
                messages = [
                    {"role": "user", "content": user_prompt}
                ]
                metrics = LLMMetrics()
                ans_li = await call_llm(messages, self.config.model_config, coerce_to_json_list, metadata={},
                                        metrics=metrics)
                for ans in ans_li:
                    if ans.get('answer') and ans.get('explanation') and ans.get('question_id'):
                        qid = ans.get('question_id')
                        if not q_map.get(qid):
                            continue
                        q = Question(**q_map.get(qid))
                        if q:
                            answer = Answer(
                                question_id=q.question_id,
                                question=q.question,
                                experiment_id=self.env.get('experiment_id', ''),
                                config_name=self.config.model_config.name,
                                question_type=q.question_type,
                                gold_answer=q.gold_answer,
                                gold_trace=q.gold_trace,
                                pred_answer=ans.get('answer'),
                                pred_trace=ans.get('explanation'),
                                notes=q.notes,
                                prompt_tokens=metrics.prompt_tokens,
                                completion_tokens=metrics.completion_tokens,
                                latency=metrics.latency_ms
                            )
                            processed_qs.append(dataclasses.asdict(answer))
                        else:
                            logger.error(f"answer is corrupted: {ans}")

        logger.info(f"got {len(processed_qs)} answers")

        return {
            "answers.json": processed_qs
        }
