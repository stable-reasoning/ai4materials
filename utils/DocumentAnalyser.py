from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any

from middleware.ImageStorage import ImageStorage
from middleware.llm_middleware import coerce_to_json_list, GenericLLMCallable, call_llm
from utils.common import DocumentBundle, ModelConfig
from utils.prompt_manager import PromptManager
from utils.settings import logger


def get_boolean_flag(data: Dict[str, Any], key: str) -> bool:
    value = data.get(key)
    return str(value).lower().strip() == 'true'


def get_type(data: Dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    return str(value).lower().strip()


# TODO: add auto-injection of reference_header entry if not present - useful for multi-chapter books or conference vols
# TODO: accurate extraction of figures and tables using ViLM models
# TODO: link captions and figures

class DocumentProcessor:
    """
    Processes document pages from an LLM, enriches the data, merges split
    blocks, and collects reference entries.
    """

    def __init__(self, doc_bundle: DocumentBundle,
                 pm: PromptManager,
                 llm_hook: GenericLLMCallable,
                 model_config: ModelConfig,
                 im_store: ImageStorage):
        self.all_blocks: List[Dict[str, Any]] = []
        self.references: Dict[str, Any] = {}
        self.figures: Dict[str, int] = {}
        self.tables: Dict[str, int] = {}
        self.file_map: Dict[int, Path] = {}
        self.pm = pm
        self.doc_bundle = doc_bundle
        self.llm_hook = llm_hook
        self.im_store = im_store
        self.config = model_config

    async def process_page(self, page_number: int, image_path: Path):

        system_prompt = self.pm.compose_prompt("analyse_page_sys_v1.j2")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                self.im_store.get_image_entry(image_path)
            ]},
        ]

        page_blocks = await self.llm_hook(messages, self.config, coerce_to_json_list, metadata={})

        if not page_blocks:
            logger.warn(f"WARNING: No LLM response for page {page_number} ('{image_path}'). Skipping.")
            return

        # Enrich each block with the page number
        for block in page_blocks:
            block['page'] = page_number
            if get_type(block, 'type') == 'figure':
                self.figures[block['text']] = page_number
            if get_type(block, 'type') == 'table':
                self.tables[block['text']] = page_number

        self.all_blocks.extend(page_blocks)
        logger.info(f"INFO: Successfully processed and added {len(page_blocks)} blocks from page {page_number}.")

    async def process_document(self):
        for idx, page_path in enumerate(self.doc_bundle.get_pages(), start=1):
            try:
                logger.info(f"--> {page_path.name}")
                self.file_map[idx] = page_path
                await self.process_page(idx, page_path)
            except Exception as e:
                msg = f"[error] {page_path.name}: {e}"
                logger.error(msg, file=sys.stderr)

        logger.info("INFO: Finalizing document...")
        self._merge_split_blocks()
        self._enrich_with_idx()
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

    def _enrich_with_idx(self):
        if not self.all_blocks:
            return

        for i, rec in enumerate(self.all_blocks):
            rec["idx"] = i

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

    def _copy_files(self, indexed: Dict[str, int]):
        for label, src_idx in indexed.items():
            src_path = self.file_map[src_idx]
            if src_path.is_file():
                try:
                    shutil.copy2(src_path, self.doc_bundle.assets_dir / f"{label}.png")
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
            else:
                print(f"Warning: Source '{src_path}' is not a file or does not exist. Skipping.")

    def _collect_figures(self):
        self._copy_files(self.figures)

    def _collect_tables(self):
        self._copy_files(self.tables)


async def main():
    pm = PromptManager()
    image_store = ImageStorage()
    config = ModelConfig(
        name='o4-mini',
        model='o4-mini',
        temperature=1.0
    )
    doc_bundle = DocumentBundle("1")
    doc_processor = DocumentProcessor(doc_bundle=doc_bundle, pm=pm, model_config=config, im_store=image_store,
                                      llm_hook=call_llm)
    await doc_processor.process_document()


if __name__ == "__main__":
    asyncio.run(main())
