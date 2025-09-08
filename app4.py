"""
CLI-enabled entrypoint for running the document or answer pipelines.

Usage examples:
  # Document pipeline (uses default test list if not provided)
  python app3.py document \
      --papers ./test_data/papers_3.lst \
      --working-dir runs

  # Answer pipeline (paths default to previous hardcoded files if omitted)
  python app3.py answer \
      --contracts ./data/contracts2.json \
      --dataset ./data/full_dataset2.json \
      --context-flags RAW_TEXT \
      --working-dir runs

  # Override model/runtime knobs
  python app3.py answer --contracts ./data/contracts2.json --dataset ./data/full_dataset2.json \
      --model o4-mini --temperature 0.7 --retries 3
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

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

DEFAULT_MODEL = "o4-mini"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_RETRIES = 3


# ------------------------------ DAG builders ------------------------------ #

def get_document_pipeline_dag(
    paper_li: Path,
    options: Optional[Dict[str, Any]] = None,
    model_config: Optional[ModelConfig] = None,
) -> DAG:
    """Build the document-processing pipeline DAG."""
    prompt_manager = PromptManager()
    image_store = ImageStorage()

    model_config = model_config or ModelConfig(
        name=DEFAULT_MODEL,
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        retries=DEFAULT_RETRIES,
    )

    download_papers = DownloadAgent(
        name="download_papers",
        input_spec={
            "file_with_urls": f"str:{paper_li}"
        },
    )

    extract_pages = ExtractionAgent(
        name="extract_pages",
        input_spec={
            "file_paths": "agent:download_papers/file_paths.json"
        },
    )

    analyze_document = DocumentAnalyzerAgent(
        name="analyze_document",
        input_spec={
            "doc_ids": "agent:extract_pages/processed_document_ids.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
    )

    semantic_analysis = SemanticAnalyzerAgent(
        name="semantic_analysis",
        input_spec={
            "doc_ids": "agent:analyze_document/processed_document_ids.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
    )

    contract_generation = ContractWriterAgent(
        name="contract_generation",
        input_spec={
            "semantic_documents": "agent:semantic_analysis/semantic_documents.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
    )

    question_generation = QADatasetGeneratorAgent(
        name="question_generation",
        input_spec={
            "contracts": "agent:contract_generation/contracts.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
        metadata=options
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
            question_generation,
        ],
    )

    return document_pipeline_dag


def get_answer_pipeline_dag(
    dataset: Path,
    contracts: Path,
    options: Optional[Dict[str, Any]] = None,
    model_config: Optional[ModelConfig] = None,
) -> DAG:
    """Build the QA+evaluation pipeline DAG."""
    prompt_manager = PromptManager()
    image_store = ImageStorage()

    model_config = model_config or ModelConfig(
        name=DEFAULT_MODEL,
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        retries=DEFAULT_RETRIES,
    )

    question_answering = QAAnswerAgent(
        name="question_answering",
        input_spec={
            "qa_dataset": f"file:{str(dataset)}",
            "contracts": f"file:{str(contracts)}",
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
        metadata=options,
    )

    evaluation = QAEvaluationAgent(
        name="evaluation",
        input_spec={
            "answers": "agent:question_answering/answers.json"
        },
        pm=prompt_manager,
        image_store=image_store,
        model_config=model_config,
        metadata=options,
    )

    question_answering >> evaluation

    test_pipeline_dag = DAG(
        name="test_pipeline",
        tasks=[
            question_answering,
            evaluation,
        ],
    )

    return test_pipeline_dag


# ------------------------------ runners ------------------------------ #

async def run_document(
    papers_list: Path,
    working_dir: str,
    run_id: Optional[str] = None,
    context_flags: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
) -> None:
    options = {"context_flags": context_flags} if context_flags else None
    dag = get_document_pipeline_dag(
        paper_li=papers_list,
        options=options,
        model_config=model_config
    )
    runner = DAGRunner(dag=dag, working_dir=working_dir)
    run_id = run_id or f"{dag.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    await runner.run(experiment_id=run_id)


async def run_answer(
    dataset: Path,
    contracts: Path,
    working_dir: str,
    context_flags: Optional[str] = None,
    run_id: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
) -> None:
    options = {"context_flags": context_flags} if context_flags else None
    dag = get_answer_pipeline_dag(
        dataset=dataset,
        contracts=contracts,
        options=options,
        model_config=model_config,
    )
    runner = DAGRunner(dag=dag, working_dir=working_dir)
    run_id = run_id or f"{dag.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    await runner.run(experiment_id=run_id)


# ------------------------------ CLI ------------------------------ #

@dataclass
class ModelArgs:
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    retries: int = DEFAULT_RETRIES

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            name=self.model,
            model=self.model,
            temperature=self.temperature,
            retries=self.retries,
        )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name/id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Number of retry attempts (default: {DEFAULT_RETRIES})",
    )


def _parse_model_args(ns: argparse.Namespace) -> ModelArgs:
    return ModelArgs(model=ns.model, temperature=ns.temperature, retries=ns.retries)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app4",
        description="Run document or answer pipelines via CLI.",
    )

    parser.add_argument(
        "--working-dir",
        default="runs",
        help="Directory to store run artifacts (default: runs)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # document subcommand
    p_doc = subparsers.add_parser("document", help="Run the document pipeline")
    p_doc.add_argument(
        "--papers",
        type=Path,
        default=(ROOT_DIR / "test_data/papers_3.lst"),
        help="Path to a file containing paper URLs (default: test_data/papers_3.lst)",
    )
    p_doc.add_argument(
        "--run-id",
        help="Optional experiment/run id (default: auto timestamp)",
    )
    _add_model_args(p_doc)

    # answer subcommand
    p_ans = subparsers.add_parser("answer", help="Run QA + evaluation pipeline")
    p_ans.add_argument(
        "--dataset",
        type=Path,
        default=(DATA_DIR / "full_dataset2.json"),
        help="Path to QA dataset JSON (default: data/full_dataset2.json)",
    )
    p_ans.add_argument(
        "--contracts",
        type=Path,
        default=(DATA_DIR / "contracts2.json"),
        help="Path to contracts JSON (default: data/contracts2.json)",
    )
    p_ans.add_argument(
        "--flags",
        default="RAW_TEXT",
        help="Optional context flags, e.g. 'CC' or 'RAW_TEXT' or 'CC|RAW_TEXT'",
    )
    p_ans.add_argument(
        "--run-id",
        help="Optional experiment/run id (default: auto timestamp)",
    )
    _add_model_args(p_ans)

    return parser


def _validate_path(path: Path, kind: str) -> None:
    if not path.exists():
        logger.error("%s not found: %s", kind, path)
        sys.exit(2)


async def _dispatch(ns: argparse.Namespace) -> None:
    model_cfg = _parse_model_args(ns).to_model_config()

    if ns.command == "document":
        _validate_path(ns.papers, "papers list")
        await run_document(
            papers_list=ns.papers,
            working_dir=ns.working_dir,
            run_id=ns.run_id,
            model_config=model_cfg,
        )
        return

    if ns.command == "answer":
        _validate_path(ns.dataset, "dataset file")
        _validate_path(ns.contracts, "contracts file")
        await run_answer(
            dataset=ns.dataset,
            contracts=ns.contracts,
            working_dir=ns.working_dir,
            context_flags=ns.flags,
            run_id=ns.run_id,
            model_config=model_cfg,
        )
        return

    # Should be unreachable because subparsers require a command
    logger.error("No command provided. Use --help for usage.")
    sys.exit(2)


def main() -> None:
    parser = make_parser()
    ns = parser.parse_args()
    try:
        asyncio.run(_dispatch(ns))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
