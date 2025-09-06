import asyncio
from pathlib import Path
from typing import Dict, Any

from agents.ContractWriterAgent import ContractWriterAgent
from agents.DocumentAnalyzerAgent import DocumentAnalyzerAgent
from agents.DownloadAgent import DownloadAgent
from agents.ExtractionAgent import ExtractionAgent
from agents.QAAnswerAgent import QAAnswerAgent
from agents.QADatasetGeneratorAgent import QADatasetGeneratorAgent
from agents.QAEvaluationAgent import QAEvaluationAgent
from agents.SemanticAnalyzerAgent import SemanticAnalyzerAgent
from core import DAG, DAGRunner
from middleware.ImageStorage import ImageStorage
from utils.common import ModelConfig
from utils.prompt_manager import PromptManager
from utils.settings import logger, ROOT_DIR, DATA_DIR


def get_document_pipeline_dag(paper_li: Path, dataset: Path = None) -> DAG:
    prompt_manager = PromptManager()
    image_store = ImageStorage()

    model_config = ModelConfig(
        name='o4-mini',
        model='o4-mini',
        temperature=1.0,
        retries=3
    )

    download_papers = DownloadAgent(
        name="download_papers",
        input_spec={
            "file_with_urls": f"str:{paper_li}"
        }
    )

    extract_pages = ExtractionAgent(
        name="extract_pages",
        input_spec={
            "file_paths": "agent:download_papers/file_paths.json"
        }
    )

    analyze_document = DocumentAnalyzerAgent(
        name="analyze_document",
        input_spec={
            "doc_ids": "agent:extract_pages/processed_document_ids.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config
    )

    semantic_analysis = SemanticAnalyzerAgent(
        name="semantic_analysis",
        input_spec={
            "doc_ids": "agent:analyze_document/processed_document_ids.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config
    )

    contract_generation = ContractWriterAgent(
        name="contract_generation",
        input_spec={
            "semantic_documents": "agent:semantic_analysis/semantic_documents.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config
    )

    question_generation = QADatasetGeneratorAgent(
        name="question_generation",
        input_spec={
            "contracts": "agent:contract_generation/contracts.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config
    )

    download_papers >> extract_pages >> analyze_document >> semantic_analysis >> contract_generation >> question_generation

    document_pipeline_dag = DAG(
        name="document_pipeline",
        tasks=[
            download_papers,
            extract_pages,
            analyze_document,
            semantic_analysis,
            contract_generation,
            question_generation
        ]
    )

    return document_pipeline_dag


def get_answer_pipeline_dag(dataset: Path = None, contracts: Path = None, options: Dict[str, Any] = None) -> DAG:
    prompt_manager = PromptManager()
    image_store = ImageStorage()

    model_config = ModelConfig(
        name='o4-mini',
        model='o4-mini',
        temperature=1.0,
        retries=3
    )

    question_answering = QAAnswerAgent(
        name="question_answering",
        input_spec={
            "qa_dataset": f"file:{str(dataset)}",
            "contracts": f"file:{str(contracts)}"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
        metadata=options
    )

    evaluation = QAEvaluationAgent(
        name="evaluation",
        input_spec={
            "answers": "agent:question_answering/answers.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
        metadata=options
    )

    question_answering >> evaluation

    test_pipeline_dag = DAG(
        name="test_pipeline",
        tasks=[
            question_answering,
            evaluation
        ]
    )

    return test_pipeline_dag


async def main():
    papers_li = ROOT_DIR / "test_data/papers_3.lst"
    runner = DAGRunner(
        dag=get_document_pipeline_dag(paper_li=papers_li),
        working_dir="runs")

    run_id = f"{runner.dag.name}-09022025-004"

    contracts = DATA_DIR / "contracts2.json"
    dataset = DATA_DIR / "full_dataset2.json"
    options = {'context_flags': 'CC'}
    runner = DAGRunner(
        dag=get_answer_pipeline_dag(contracts=contracts, dataset=dataset, options=options),
        working_dir="runs")

    run_id = f"{runner.dag.name}-test-2"

    await runner.run(experiment_id=run_id)


if __name__ == "__main__":
    asyncio.run(main())
