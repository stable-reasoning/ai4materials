import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Dict, Any, Mapping, Iterable

from utils.settings import global_config


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model: str
    temperature: float
    retries: int = 1
    max_tokens: int = 65535


@dataclass(frozen=True)
class SourceTxtBlock:
    """
    Minimal payload we keep after pruning.
    Note: `idx` is assigned AFTER filtering, so it does not reflect original position.
    """
    idx: int
    type: str
    text: str


@dataclass(frozen=True)
class Question:
    question_id: str
    question_type: str
    question: str
    gold_answer: str
    gold_trace: str


@dataclass(frozen=False)
class Answer:
    question_id: str
    experiment_id: str
    config_name: str
    question_type: str
    gold_answer: str
    gold_trace: str
    pred_answer: str
    pred_trace: str
    eval_score: float = -1.0


def load_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    records: List[Dict[str, Any]]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        else:
            raise ValueError("Top-level JSON must be an array for non-JSONL mode.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    return records


# used to generate a lightweight version of records, dropping non-essential fields
def prune_and_validate(records: Sequence[Dict[str, Any]]) -> List[SourceTxtBlock]:
    blocks: List[SourceTxtBlock] = []
    for rec in records:
        if "idx" not in rec or "type" not in rec or "text" not in rec:
            continue

        block = SourceTxtBlock(
            idx=int(rec["idx"]),
            type=str(rec["type"]),
            text=str(rec["text"])
        )
        blocks.append(block)
    return blocks


def to_jsonl(blocks: Sequence[SourceTxtBlock]) -> str:
    lines = [json.dumps(asdict(b), ensure_ascii=False) for b in blocks]
    return "\n".join(lines)


def pick_keys(d: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def get_document_bundle(doc_id: str) -> Path:
    bundle_path = Path(global_config.docucache_path) / doc_id
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"docu bundle not found: {doc_id}")
    return bundle_path


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

    def get_refs_path(self) -> Path:
        return self.out_dir / f"reference_{self.doc_id}.json"

    def get_figures_path(self) -> Path:
        return self.out_dir / f"figures_{self.doc_id}.json"

    def get_tables_path(self) -> Path:
        return self.out_dir / f"tables_{self.doc_id}.json"
