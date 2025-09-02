from pathlib import Path
from typing import Dict, Any, List

from core import Agent
from utils.common import DocumentBundle
from utils.extraction_utils import extract_pdf_pages
from utils.settings import logger


class ExtractionAgent(Agent):
    """An agent to fetch posts from a public API."""

    async def run(self, file_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"extracting data from {len(file_paths)} files")
        processed_files = []
        for r in file_paths:
            try:
                paper_id = r.get('paper_id', -1)
                path = r.get('file', "")
                if path and paper_id != -1:
                    bundle = DocumentBundle(str(paper_id))
                    file = Path(path)
                    out = bundle.pages_dir
                    extract_pdf_pages(pdf_path=file, out_dir=out)
                    processed_files.append(paper_id)
            except Exception as e:
                logger.error(e)

        return {
            "processed_document_ids.json": processed_files
        }


