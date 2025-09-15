import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from core import Agent
from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import call_llm, coerce_to_json_list, LLMMetrics
from utils.common import ModelConfig, DocumentBundle, load_file, Question
from utils.prompt_manager import PromptManager
from utils.settings import logger

VALID_TYPES: Set[str] = {"counterfactual"}


def is_record_valid(record: Any) -> bool:
    """
    Validates a single record against the specified schema.

    """
    if not isinstance(record, dict):
        return False

    required_keys = {"category", "question", "contract_id", "gold_answer"}
    if not required_keys.issubset(record.keys()):
        return False

    try:
#        if not (isinstance(record['category'], str) and record['category'].lower().strip() in VALID_TYPES):
#            return False
        if not (isinstance(record['question'], str) and record['question']):
            return False  # Must be a non-empty string
        if not (isinstance(record['contract_id'], str)):
            return False
        if not (isinstance(record['gold_answer'], dict) and record['gold_answer']):
            return False  # Must be a non-empty object
    except (TypeError, KeyError):
        # Catch potential errors if data structure is unexpectedly malformed.
        return False

    return True


def form_question_sketch(q: Dict[str, Any]) -> Optional[Question]:
    gold_answer = q.get('gold_answer')
    answer = gold_answer.get('answer')
    explanation = gold_answer.get('explanation')
    notes = gold_answer.get('notes')
    if gold_answer and answer and explanation:
        trace = '\n'.join([explanation] + gold_answer.get('paths', []))
        return Question(
            question_id=q['id'],
            question_type=q['category'],
            question=q['question'],
            gold_answer=answer,
            gold_trace=trace,
            notes=notes
        )
    return None


class QADatasetGeneratorAgent(Agent):

    config: "QADatasetGeneratorAgent.Config"

    @dataclass
    class Config:
        pm: PromptManager
        image_store: ImageStorage
        model_config: ModelConfig
        metadata: Dict[str, Any]

    async def run(self, contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"processing {len(contracts)} documents")
        processed_docs = []
        error_count = 0
        dataset_size = 0
        for c in contracts:
            try:
                doc_id = c['document_id']
                doc_bundle = DocumentBundle(str(doc_id))
                raw_text = load_file(doc_bundle.get_records_path())  # document is taken from cache by id
                contracts = load_file(Path(c['path']))  # contract is loaded from path
                user_prompt = self.config.pm.compose_prompt(
                    "qa_dataset_generator_raw_v8.j2",
                    contracts=contracts,
                    raw_text=raw_text
                )
                messages = [
                    {"role": "user", "content": user_prompt}
                ]
                metrics = LLMMetrics()
                page_blocks = await call_llm(messages, self.config.model_config, coerce_to_json_list, metadata={}, metrics=metrics)
                print(metrics)
                valid_questions = [record for record in page_blocks if is_record_valid(record)]
                batch = []
                for i, q in enumerate(valid_questions, start=1):
                    q['id'] = f"{doc_id}-{i}"
                    question = form_question_sketch(q)
                    if question:
                        batch.append(dataclasses.asdict(question))
                logger.info(f"generated {len(batch)} questions")
                dataset_size += len(batch)
                res_path = self.save_locally(f"questions_{doc_id}.json", batch)
                processed_docs.append({"document_id": doc_id, "path": str(res_path)})
            except Exception as e:
                logger.error(e)
                error_count += 1

        logger.info(f"created {dataset_size} questions, errors: {error_count}")

        return {
            "qa_dataset.json": processed_docs
        }



