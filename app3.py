import asyncio
from pathlib import Path

from agents.ContractWriterAgent import ContractWriterAgent
from agents.DocumentAnalyzerAgent import DocumentAnalyzerAgent
from agents.DownloadAgent import DownloadAgent
from agents.ExtractionAgent import ExtractionAgent
from agents.SemanticAnalyzerAgent import SemanticAnalyzerAgent
from core import DAG, DAGRunner
from middleware.ImageStorage import ImageStorage
from utils.common import ModelConfig
from utils.prompt_manager import PromptManager
from utils.settings import logger, ROOT_DIR


def get_document_pipeline_dag(paper_li: Path) -> DAG:
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

    download_papers >> extract_pages >> analyze_document >> semantic_analysis >> contract_generation

    document_pipeline_dag = DAG(
        name="document_pipeline",
        tasks=[download_papers, extract_pages, analyze_document, semantic_analysis, contract_generation]
    )

    return document_pipeline_dag


async def main():
    papers_li = ROOT_DIR / "test_data/papers_material_science.lst"
    runner = DAGRunner(dag=get_document_pipeline_dag(paper_li=papers_li), working_dir="runs")

    logger.info("EXECUTING document pipeline")
    run_id = f"{runner.dag.name}-09022025-001"
    await runner.run(experiment_id=run_id)


if __name__ == "__main__":
    asyncio.run(main())
