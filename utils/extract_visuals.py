import os, re, json, argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np
import layoutparser as lp
import pytesseract

# ---------- Configurable patterns for caption labels ----------
FIG_PAT = re.compile(r"\b(?:fig(?:ure)?\.?\s*\d+[a-z]?)\b", flags=re.I)
TAB_PAT = re.compile(r"\b(?:table\.?\s*\d+[a-z]?)\b", flags=re.I)


@dataclass
class ExtractedItem:
    kind: str  # "figure" or "table"
    index: int  # sequential per kind by reading order
    bbox: Tuple[int, int, int, int]  # (x1,y1,x2,y2) on the page
    score: float  # detector confidence
    caption_label: Optional[str]  # e.g., "Fig. 1", "Table 2"
    caption_text: Optional[str]  # OCR’d line containing the label (if any)
    crop_path: str


def load_layout_model(conf_thresh: float = 0.6):
    """
    PubLayNet -> labels: Text, Title, List, Table, Figure
    Works very well on scientific PDFs/pages.
    """
    print(f"is_pytesseract_available:{lp.is_pytesseract_available()}")
    print(f"is_detectron2_available:{lp.is_detectron2_available()}")

    model = lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model_config",
        #model_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model",  # <- important
        extra_config=[
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", conf_thresh,
            "MODEL.WEIGHTS", "/home/alex/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth?dl=1"
        ],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
    )

    return model


def clamp(v, lo, hi): return max(lo, min(hi, v))


def expand_box(box: Tuple[int, int, int, int], pad: int, W: int, H: int):
    x1, y1, x2, y2 = box
    return (
        clamp(x1 - pad, 0, W - 1),
        clamp(y1 - pad, 0, H - 1),
        clamp(x2 + pad, 0, W - 1),
        clamp(y2 + pad, 0, H - 1)
    )


def ocr_lines(img: np.ndarray) -> List[str]:
    """
    OCR as individual lines; returns cleaned non-empty lines in reading order.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    # Light binarization helps for crisp OCR
    try:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    except:
        thr = gray
    cfg = "--oem 1 --psm 6 -l eng"
    data = pytesseract.image_to_data(thr, config=cfg, output_type=pytesseract.Output.DICT)
    lines = {}
    for i, txt in enumerate(data["text"]):
        if not txt or int(data["conf"][i]) < 30:
            continue
        ln = data["line_num"][i]
        lines.setdefault(ln, []).append(txt)
    ordered = [" ".join(lines[k]).strip() for k in sorted(lines.keys())]
    return [ln for ln in ordered if ln]


def find_caption_near(
        page_img: np.ndarray,
        item_box: Tuple[int, int, int, int],
        search_frac: float = 0.12
) -> Tuple[Optional[str], Optional[str]]:
    """
    Search for a caption label near the item:
    1) Prefer a strip BELOW the box
    2) Fallback to a strip ABOVE the box
    Return (caption_label, full_line_text_containing_label)
    """
    H, W = page_img.shape[:2]
    x1, y1, x2, y2 = item_box

    dy = int(H * search_frac)

    # 1) below
    yb1 = clamp(y2 + 1, 0, H - 1)
    yb2 = clamp(y2 + dy, 0, H - 1)
    below = page_img[yb1:yb2, max(0, x1 - 20):min(W, x2 + 20)]
    lines_below = ocr_lines(below) if below.size else []

    # 2) above
    ya1 = clamp(y1 - dy, 0, H - 1)
    ya2 = clamp(y1 - 1, 0, H - 1)
    above = page_img[ya1:ya2, max(0, x1 - 20):min(W, x2 + 20)]
    lines_above = ocr_lines(above) if above.size else []

    # Prefer below if label found; else check above.
    for ln in lines_below:
        m = FIG_PAT.search(ln) or TAB_PAT.search(ln)
        if m:
            return (m.group(0), ln.strip())
    for ln in lines_above:
        m = FIG_PAT.search(ln) or TAB_PAT.search(ln)
        if m:
            return (m.group(0), ln.strip())

    # Wider search: a full-width strip below (sometimes captions are centered)
    yb2w = clamp(y2 + int(H * 0.18), 0, H - 1)
    below_wide = page_img[yb1:yb2w, :]
    lines_wide = ocr_lines(below_wide) if below_wide.size else []
    for ln in lines_wide:
        m = FIG_PAT.search(ln) or TAB_PAT.search(ln)
        if m:
            return (m.group(0), ln.strip())

    return (None, None)


def detect_and_crop(
        image_path: str,
        out_dir: str,
        pad_px: Optional[int] = None,
        conf_thresh: float = 0.6
) -> List[ExtractedItem]:
    os.makedirs(out_dir, exist_ok=True)
    page = cv2.imread(image_path)
    if page is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = page.shape[:2]

    # tiny-small padding: default = max(8px, 0.5% of min dimension)
    if pad_px is None:
        pad_px = max(8, int(min(W, H) * 0.005))

    model = load_layout_model(conf_thresh)
    print("analysing...")
    layout = model.detect(page)
    print(layout)
    # Filter to figures & tables; sort by top-to-bottom, then left-to-right
    items = []
    for b in layout:
        if b.type not in ("Figure", "Table"):
            continue
        x1, y1, x2, y2 = map(int, b.block.points.flatten())
        items.append((b.type.lower(), (x1, y1, x2, y2), float(b.score)))

    items.sort(key=lambda it: (it[1][1], it[1][0]))  # by y1, then x1

    fig_count = 0
    tab_count = 0
    results: List[ExtractedItem] = []

    for kind, box, score in items:
        x1, y1, x2, y2 = expand_box(box, pad_px, W, H)
        crop = page[y1:y2, x1:x2]

        # Determine index per kind to keep deterministic names
        if kind == "figure":
            fig_count += 1
            idx = fig_count
            base = f"figure_{idx:02d}.png"
        else:
            tab_count += 1
            idx = tab_count
            base = f"table_{idx:02d}.png"

        crop_path = os.path.join(out_dir, base)
        cv2.imwrite(crop_path, crop)

        # Try to find the closest caption label
        label, label_line = find_caption_near(page, (x1, y1, x2, y2))

        results.append(ExtractedItem(
            kind=kind,
            index=idx,
            bbox=(x1, y1, x2, y2),
            score=score,
            caption_label=label,
            caption_text=label_line,
            crop_path=crop_path
        ))

    # Save a machine-readable manifest
    manifest_path = os.path.join(out_dir, "extractions.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # Optional: draw an overview image with boxes (for quick visual QA)
    overview = page.copy()
    for r in results:
        color = (0, 180, 0) if r.kind == "figure" else (180, 0, 0)
        x1, y1, x2, y2 = r.bbox
        cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
        tag = (r.caption_label or f"{r.kind[:3]}?").upper()
        cv2.putText(overview, tag, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "overview_boxes.png"), overview)

    return results


def main():
    ap = argparse.ArgumentParser(description="Extract figures and tables (with captions) from a paper page PNG.")
    ap.add_argument("image", help="Path to input PNG page")
    ap.add_argument("-o", "--out", default="extractions", help="Output directory")
    ap.add_argument("--pad", type=int, default=None,
                    help="Padding in pixels (default tiny: max(8px, 0.5% of min dimension))")
    ap.add_argument("--conf", type=float, default=0.6, help="Layout detector confidence threshold")
    args = ap.parse_args()

    results = detect_and_crop(args.image, args.out, pad_px=args.pad, conf_thresh=args.conf)
    print(f"Saved {len(results)} items to: {os.path.abspath(args.out)}")
    for r in results:
        print(f"- {r.kind}#{r.index:02d} | score={r.score:.2f} | label={r.caption_label} | crop={r.crop_path}")


if __name__ == "__main__":
    main()
