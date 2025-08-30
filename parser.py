#!/usr/bin/env python3
"""
LLM-based page parser → JSONL + clipped assets.

- Reads color .png pages from --pages_dir (default: /pages)
- Sends each page to an OpenAI Vision model with a precise parsing prompt
- Saves crops for all figures (incl. equations) to --assets_dir (default: ./assets)
- Writes one JSON object per page to --out_dir JSONL (default: ./out)

Examples:
  python parse_with_llm.py --pages_dir /pages --out out.jsonl --assets_dir ./assets
  python parse_with_llm.py --model gpt-4o-mini --use_strict
  python parse_with_llm.py --latex pix2tex
"""

from __future__ import annotations
import argparse, base64, json, os, re, sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import numpy as np
from dotenv import load_dotenv
from pix2tex.cli import LatexOCR  # type: ignore
# from latex_ocr import LatexOCR  # type: ignore

from PIL import Image

# OpenAI API (2024+ style)
from utils.prompt_manager import PromptManager

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore


# -------------------------- Data model --------------------------

@dataclass
class Block:
    page: int
    bbox: List[int]  # [x1,y1,x2,y2]
    type: str  # allowed: section_header, paragraph, equation_block, caption, figure, table, reference_header, reference_entry
    text: str  # text or unique name; for equations we will set eq_{QUAD}_...
    confidence: float
    latex: Optional[str] = None  # equations only (when --latex)

    def as_json(self) -> Dict[str, Any]:
        d = {
            "page": self.page,
            "bbox": [int(v) for v in self.bbox],
            "type": self.type,
            "text": self.text,
            "confidence": float(self.confidence),
        }
        if self.latex is not None:
            d["latex"] = self.latex
        return d


# -------------------------- Helpers --------------------------

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


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def inter_quadrant(box: List[int], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if cy < h / 2 and cx < w / 2:
        return "TL"
    if cy < h / 2 and cx >= w / 2:
        return "TR"
    if cy >= h / 2 and cx < w / 2:
        return "BL"
    return "BR"


def crop_save(pil: Image.Image, box: List[int], out_path: Path) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = pil.crop((x1, y1, x2, y2))
    ensure_dir(out_path.parent)
    crop.save(out_path)


def is_equation_like(gray: np.ndarray, box: List[int], page_w: int, page_h: int) -> bool:
    """Geometry/whitespace heuristic to tag equation-like areas (single baseline, modest area, high aspect, whitespace above/below)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    aspect = bw / bh
    area_ratio = (bw * bh) / float(page_w * page_h)
    # local bands
    pad = int(0.012 * page_h + bh * 0.3)
    top_y1, top_y2 = max(0, y1 - pad), y1
    bot_y1, bot_y2 = y2, min(page_h, y2 + pad)
    top_ws = gray[top_y1:top_y2, x1:x2].mean() / 255 if top_y2 > top_y1 else 1.0
    bot_ws = gray[bot_y1:bot_y2, x1:x2].mean() / 255 if bot_y2 > bot_y1 else 1.0
    centered = (x1 > 0.06 * page_w) and (x2 < 0.94 * page_w)
    return (aspect > 3.2 and 0.003 < area_ratio < 0.08 and centered and top_ws > 0.80 and bot_ws > 0.80)


def coerce_blocks(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    raise ValueError("LLM did not return a JSON list.")


# -------------------------- Optional eq → LaTeX --------------------------

def build_eq_to_latex(kind: Optional[str]) -> Optional[Callable[[Image.Image], str]]:
    if not kind:
        return None
    name = kind.strip().lower()
    if name == "pix2tex":
        try:
            model = LatexOCR()
            return lambda pil_img: model(pil_img)
        except Exception:
            print("[warn] pix2tex not available; continuing without LaTeX strings.", file=sys.stderr)
            return None
    #    if name == "latexocr":
    #        try:
    #            model = LatexOCR()
    #            return lambda pil_img: model(pil_img)
    #        except Exception:
    #            print("[warn] latexocr not available; continuing without LaTeX strings.", file=sys.stderr)
    #            return None
    print(f"[warn] Unknown --latex backend '{kind}'.", file=sys.stderr)
    return None


# -------------------------- LLM call --------------------------

def call_openai_parse(client, model: str,
                      prompts: PromptManager,
                      image_path: Path,
                      page_w: int, page_h: int,
                      temperature: float = 0.0,
                      max_retries: int = 3) -> List[Dict[str, Any]]:
    img_url = encode_image_to_data_url(image_path)
    last_err = None
    system_prompt = prompts.compose_prompt("analyse_page_sys_v1.j2")
    analyse_prompt = prompts.compose_prompt("analyse_page_v1.j2", page_w=page_w, page_h=page_h)
    print(analyse_prompt)
    print(f"image bytes: {len(img_url)}")
    for _ in range(max_retries):
        try:
            # Responses API (multi-modal)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": analyse_prompt},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]},
                ],
            )
            # Extracting and printing the response content
            print(f"prompt_tokens: {resp.usage.prompt_tokens}")
            print(f"completion_tokens: {resp.usage.completion_tokens}")

            content = resp.choices[0].message.content.strip()
            print(content)
            # Extract first [...] to avoid stray characters
            m = re.search(r"\[.*\]", content, flags=re.DOTALL)
            raw_json = m.group(0) if m else content
            data = json.loads(raw_json)
            return coerce_blocks(data)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"OpenAI parse failed for {image_path.name}: {last_err}")


# -------------------------- Main pipeline --------------------------

def process_page(
        client,
        model: str,
        prompts: PromptManager,
        page_path: Path,
        page_index: int,
        assets_dir: Path,
        eq2latex: Optional[Callable[[Image.Image], str]],
        rename_equations: bool,
        confidence_floor: float,
        temperature: float,
        max_retries: int,
) -> List[Dict[str, Any]]:
    pil = load_image(page_path)
    w, h = pil.size
    gray = pil_to_numpy_gray(pil)

    raw_blocks = call_openai_parse(client, model, prompts, page_path, w, h, temperature, max_retries)

    # Normalize + enforce deviations
    out_blocks: List[Block] = []
    eq_counter = 1
    fig_counter = 1

    for rb in raw_blocks:
        # Basic fields
        btype = rb.get("type", "").strip()
        bbox = [int(v) for v in rb.get("bbox", [0, 0, 0, 0])]
        text = str(rb.get("text", "") or "").strip()
        conf = float(rb.get("confidence", confidence_floor))

        # Decide if this figure is an equation (LLM might have already named it; we enforce our scheme)
        eq_like = (btype == "equation_block")
        is_fig = (btype == "figure")

        if eq_like:
            quad = inter_quadrant(bbox, w, h)
            # crop & save
            crop_save(pil, bbox, assets_dir / f"eq_{text}.png")
            # optional LaTeX
            latex_str = None
            if eq2latex is not None:
                try:
                    latex_str = eq2latex(pil.crop((bbox[0], bbox[1], bbox[2], bbox[3])))
                except Exception:
                    latex_str = None
            out_blocks.append(
                Block(page=page_index, bbox=bbox, type=btype, text=text, confidence=conf, latex=latex_str))
            eq_counter += 1
        elif is_fig:
            # regular figure
            if not text or text.lower().startswith("eq_"):
                text = f"figure_{page_index:02d}_{fig_counter:04d}"
            crop_save(pil, bbox, assets_dir / f"{text}.png")
            out_blocks.append(Block(page=page_index, bbox=bbox, type=btype, text=text, confidence=conf))
            fig_counter += 1
        else:
            # Other textual blocks unchanged
            out_blocks.append(Block(page=page_index, bbox=bbox, type=btype, text=text, confidence=conf))

    # Sort in reading order
    # out_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    # Emit page-level JSON (list of blocks)
    return [b.as_json() for b in out_blocks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages_dir", type=str, default="/pages")
    parser.add_argument("--out_dir", type=str, default="./out")
    parser.add_argument("--assets_dir", type=str, default="./assets")
    parser.add_argument("--model", type=str, default="gpt-4o")  # any OpenAI vision-capable chat model
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--latex", type=str, default="", help="Optional eq→LaTeX backend: 'pix2tex' or 'latexocr'")
    parser.add_argument("--use_strict", action="store_true", help="Fail if any page returns non-JSON")
    args = parser.parse_args()

    if OpenAI is None:
        print("Please install the OpenAI SDK: pip install openai", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    pages_dir = Path(args.pages_dir)
    assets_dir = Path(args.assets_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(assets_dir)
    ensure_dir(out_dir)

    eq2latex = build_eq_to_latex(args.latex)
    prompts = PromptManager()

    # Collect and sort page PNGs
    page_files = sorted([p for p in pages_dir.iterdir() if p.suffix.lower() == ".png"])
    if not page_files:
        print(f"No .png files found in {pages_dir}", file=sys.stderr)
        sys.exit(1)

    for idx, page_path in enumerate(page_files, start=1):
        with open(f"{out_dir}/p_{idx}.json", "w", encoding="utf-8") as f_out:
            try:
                print(f"--> {page_path.name}")
                page_json = process_page(
                    client=client,
                    model=args.model,
                    prompts=prompts,
                    page_path=page_path,
                    page_index=idx,
                    assets_dir=assets_dir,
                    eq2latex=eq2latex,
                    rename_equations=True,
                    confidence_floor=0.75,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
                f_out.write(json.dumps(page_json, ensure_ascii=False))
            except Exception as e:
                msg = f"[error] {page_path.name}: {e}"
                if args.use_strict:
                    raise
                print(msg, file=sys.stderr)
        if idx >= 1: break


load_dotenv()
warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == "__main__":
    main()
