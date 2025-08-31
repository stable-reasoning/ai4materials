from __future__ import annotations

import asyncio
import base64
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from openai import OpenAI

from utils.common import DocumentBundle
from utils.llm_backend import call_openai_parse
from utils.prompt_manager import PromptManager


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_numpy_gray(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("L"))


def encode_image_to_data_url(p: Path) -> str:
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    suffix = p.suffix.lower().lstrip(".") or "png"
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{b64}"


def get_boolean_flag(data: Dict[str, Any], key: str) -> bool:
    value = data.get(key)
    return str(value).lower().strip() == 'true'


def get_type(data: Dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    return str(value).lower().strip()


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


# TODO: add auto-injection of reference_header entry if not present - useful for multi-chapter books or conference vols
# TODO: accurate extraction of figures and tables using ViLM models
# TODO: link captions and figures

class DocumentProcessor:
    """
    Processes document pages from an LLM, enriches the data, merges split
    blocks, and collects reference entries.
    """

    def __init__(self, prompts: PromptManager, doc_id: str, llm_hook: Any):
        """Initializes the processor with empty lists for blocks and references."""
        self.all_blocks: List[Dict[str, Any]] = []
        self.references: Dict[str, Any] = {}
        self.figures: Dict[str, int] = {}
        self.tables: Dict[str, int] = {}
        self.file_map: Dict[int, Path] = {}
        self.prompts = prompts
        self.doc_bundle: DocumentBundle = DocumentBundle(doc_id)
        self.llm_hook = llm_hook

    async def process_page(self, page_number: int, image_path: Path):

        img64_url = encode_image_to_data_url(image_path)
        system_prompt = self.prompts.compose_prompt("analyse_page_sys_v1.j2")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img64_url}}
            ]},
        ]

        page_blocks = await call_openai_parse(self.llm_hook, messages, temperature=1.0)

        if not page_blocks:
            print(f"WARNING: No LLM response for page {page_number} ('{image_path}'). Skipping.")
            return

        try:
            # Enrich each block with the page number
            for block in page_blocks:
                block['page'] = page_number
                if get_type(block, 'type') == 'figure':
                    self.figures[block['text']] = page_number
                if get_type(block, 'type') == 'table':
                    self.tables[block['text']] = page_number

            self.all_blocks.extend(page_blocks)
            print(f"INFO: Successfully processed and added {len(page_blocks)} blocks from page {page_number}.")

        except (json.JSONDecodeError, TypeError) as e:
            print(f"ERROR: Failed to parse blocks for page {page_number}. Reason: {e}")

    async def process_document(self):
        for idx, page_path in enumerate(self.doc_bundle.get_pages(), start=1):
            try:
                print(f"--> {page_path.name}")
                self.file_map[idx] = page_path
                await self.process_page(idx, page_path)
            except Exception as e:
                msg = f"[error] {page_path.name}: {e}"
                print(msg, file=sys.stderr)

        print("\nINFO: Finalizing document...")
        self._merge_split_blocks()
        self._collect_references()
        self._collect_figures()
        self._collect_tables()
        with open(self.doc_bundle.get_records_path(), "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(self.all_blocks, ensure_ascii=False))
        with open(self.doc_bundle.get_refs_path(), "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(self.references, ensure_ascii=False))
        with open(self.doc_bundle.get_figures_path(), "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(self.figures, ensure_ascii=False))
        with open(self.doc_bundle.get_tables_path(), "w", encoding="utf-8") as f_out:
            f_out.write(json.dumps(self.tables, ensure_ascii=False))

    def _merge_split_blocks(self):
        """
        Merges consecutive blocks of the same type where the first block is
        marked with 'split: true'.
        """
        if not self.all_blocks:
            return

        merged_blocks: List[Dict[str, Any]] = []
        i = 0
        while i < len(self.all_blocks):
            current_block = self.all_blocks[i]

            # Check for a potential merge condition
            is_splittable = get_type(current_block, "type") in ("paragraph", "caption", "equation", "reference_entry")
            has_split_flag = get_boolean_flag(current_block, "split")
            has_next_block = i + 1 < len(self.all_blocks)

            if is_splittable and has_split_flag and has_next_block:
                next_block = self.all_blocks[i + 1]
                # Merge if the next block is of the same type
                if get_type(next_block, "type") == get_type(current_block, "type"):
                    new_block = current_block.copy()
                    new_block['text'] = current_block['text'] + ' ' + next_block['text']
                    new_block['split'] = get_boolean_flag(next_block, 'split')
                    merged_blocks.append(new_block)
                    i += 2  # Skip the next block since it's now merged
                    continue

            merged_blocks.append(current_block)
            i += 1

        self.all_blocks = merged_blocks

    def _collect_references(self):
        """
        Iterates through all blocks and collects 'reference_entry' types
        into a dictionary, keyed by their citation identifier (e.g., '1', '2').
        """
        self.references = {}
        for block in self.all_blocks:
            if get_type(block, "type") == "reference_entry":
                entry_key = block.get("ref", "")
                # Add a deep copy to prevent accidental mutation
                if entry_key:
                    self.references[entry_key] = block.get("text", "")

    def _copy_files(self, index: set):
        for src_idx in index:
            src_path = self.file_map[src_idx]
            if src_path.is_file():
                try:
                    shutil.copy2(src_path, self.doc_bundle.assets_dir / f"{src_idx}.png")
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
            else:
                print(f"Warning: Source '{src_path}' is not a file or does not exist. Skipping.")

    def _collect_figures(self):
        self._copy_files(set(self.figures.values()))

    def _collect_tables(self):
        self._copy_files(set(self.tables.values()))


async def main():
    client = OpenAI()
    prompts = PromptManager()
    doc_processor = DocumentProcessor(prompts=prompts, doc_id="0001", llm_hook=client)
    await doc_processor.process_document()


if __name__ == "__main__":
    asyncio.run(main())
