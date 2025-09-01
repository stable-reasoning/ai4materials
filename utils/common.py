import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from utils.ioutils import get_document_bundle


@dataclass(frozen=True)
class SourceTxtBlock:
    """
    Minimal payload we keep after pruning.
    Note: `idx` is assigned AFTER filtering, so it does not reflect original position.
    """
    idx: int
    type: str
    text: str


class DocumentBundle:

    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.bundle_path = get_document_bundle(doc_id)
        self.pages_dir = self.bundle_path / "pages"
        self.assets_dir = self.bundle_path / "assets"
        self.out_dir = self.bundle_path

    def get_pages(self) -> List[Path]:
        page_files = sorted([p for p in self.pages_dir.iterdir() if p.suffix.lower() == ".png"])
        if not page_files:
            print(f"No .png files found in {self.pages_dir}", file=sys.stderr)
        return page_files or []

    def get_records_path(self) -> Path:
        return self.out_dir / f"rec_{self.doc_id}.json"

    def get_semantic_layer_path(self) -> Path:
        return self.out_dir / f"semantic_{self.doc_id}.json"

    def get_refs_path(self) -> Path:
        return self.out_dir / f"reference_{self.doc_id}.json"

    def get_figures_path(self) -> Path:
        return self.out_dir / f"figures_{self.doc_id}.json"

    def get_tables_path(self) -> Path:
        return self.out_dir / f"tables_{self.doc_id}.json"

    def get_qa_path(self) -> Path:
        return self.out_dir / f"qa_{self.doc_id}.json"
