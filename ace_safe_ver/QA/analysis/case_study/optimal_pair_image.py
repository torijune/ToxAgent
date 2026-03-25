#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimal_pairs_top10.csv에 선정된 pair를 SAFE/SMILES 관점으로 시각화합니다.

출력:
1) pair별 카드 이미지: out_dir/pair_cards/pair_<rank>_row<row_index>.png
2) 전체 요약 이미지: out_dir/optimal_pairs_grid.png

카드 구성(2x2):
- Toxic SAFE (full SAFE string 기반 2D)
- Toxic SMILES
- Nontoxic SAFE (full SAFE string 기반 2D)
- Nontoxic SMILES
추가로 상단에 dataset/endpoint/score + SAFE/SMILES 문자열 일부를 표시합니다.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe.safe.viz import to_image as safe_to_image


DEFAULT_TOPK_CSV = SCRIPT_DIR / "optimal_pairs_top10.csv"
DEFAULT_MERGED_CSV = (
    PROJECT_ROOT
    / "ace_safe_ver"
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "optimal_pair_images"

MOL_SIZE = (320, 260)
PADDING = 20
TITLE_H = 98
TEXT_H = 110


def _safe_font(size: int):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _to_pil_from_safe_obj(img_obj, fallback_size=MOL_SIZE) -> Image.Image:
    """safe_to_image 반환 타입(str/bytes/PIL)을 PIL.Image로 통일."""
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, bytes):
        try:
            return Image.open(io.BytesIO(img_obj)).convert("RGB")
        except Exception:
            return Image.new("RGB", fallback_size, (245, 245, 245))
    if isinstance(img_obj, str):
        # SVG 문자열은 Pillow로 바로 열기 어려우므로 placeholder
        # (본 스크립트는 use_svg=False로 호출하므로 정상적이면 여기 거의 안 옴)
        return Image.new("RGB", fallback_size, (245, 245, 245))
    return Image.new("RGB", fallback_size, (245, 245, 245))


def _draw_smiles(smiles: str, size=MOL_SIZE) -> Image.Image:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        img = Image.new("RGB", size, (245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Invalid SMILES", fill="gray", font=_safe_font(14))
        return img
    return Draw.MolToImage(mol, size=size).convert("RGB")


def _draw_safe(safe_str: str, size=MOL_SIZE) -> Image.Image:
    if not safe_str or not str(safe_str).strip():
        img = Image.new("RGB", size, (245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "SAFE not available", fill="gray", font=_safe_font(14))
        return img
    try:
        # highlight_mode=None으로 plain molecule만 표시
        obj = safe_to_image(
            str(safe_str).strip(),
            mol_size=size,
            use_svg=False,
            highlight_mode=None,
        )
        return _to_pil_from_safe_obj(obj, fallback_size=size)
    except Exception:
        img = Image.new("RGB", size, (245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "SAFE render failed", fill="gray", font=_safe_font(14))
        return img


def _truncate(s: str, width: int = 95) -> str:
    s = (s or "").strip()
    if len(s) <= width:
        return s
    return s[: width - 3] + "..."


def _build_pair_card(
    row: pd.Series,
    toxic_safe: str,
    nontoxic_safe: str,
    out_path: Path,
) -> None:
    w, h = MOL_SIZE
    canvas_w = PADDING * 3 + w * 2
    canvas_h = TITLE_H + TEXT_H + PADDING * 3 + h * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)

    font_title = _safe_font(18)
    font_meta = _safe_font(13)
    font_caption = _safe_font(14)
    font_string = _safe_font(12)

    rank = int(row.get("rank", -1))
    row_idx = int(row.get("row_index", -1))
    ds = str(row.get("dataset_name", ""))
    endpoint = str(row.get("endpoint", ""))
    score = float(row.get("optimal_score", 0.0))
    sim = float(row.get("tanimoto_chiral_morgan", 0.0))
    tox_smiles = str(row.get("toxic_smiles", ""))
    non_smiles = str(row.get("nontoxic_smiles", ""))

    d.text((PADDING, 10), f"Rank {rank}  |  row_index={row_idx}", fill="black", font=font_title)
    d.text((PADDING, 34), f"dataset={ds}  |  endpoint={endpoint}", fill="black", font=font_meta)
    d.text((PADDING, 54), f"optimal_score={score:.6f}  |  tanimoto={sim:.6f}", fill="black", font=font_meta)

    safe_t_line = f"Toxic SAFE: {_truncate(toxic_safe)}"
    safe_n_line = f"Nontoxic SAFE: {_truncate(nontoxic_safe)}"
    smi_t_line = f"Toxic SMILES: {_truncate(tox_smiles)}"
    smi_n_line = f"Nontoxic SMILES: {_truncate(non_smiles)}"
    d.text((PADDING, 74), safe_t_line, fill=(20, 20, 20), font=font_string)
    d.text((PADDING, 90), safe_n_line, fill=(20, 20, 20), font=font_string)
    d.text((PADDING, 106), smi_t_line, fill=(20, 20, 20), font=font_string)
    d.text((PADDING, 122), smi_n_line, fill=(20, 20, 20), font=font_string)

    x1, x2 = PADDING, PADDING * 2 + w
    y1 = TITLE_H + TEXT_H + PADDING
    y2 = y1 + h + PADDING

    img_tox_safe = _draw_safe(toxic_safe)
    img_tox_smiles = _draw_smiles(tox_smiles)
    img_non_safe = _draw_safe(nontoxic_safe)
    img_non_smiles = _draw_smiles(non_smiles)

    d.text((x1, y1 - 18), "Toxic SAFE", fill="black", font=font_caption)
    d.text((x2, y1 - 18), "Toxic SMILES", fill="black", font=font_caption)
    d.text((x1, y2 - 18), "Nontoxic SAFE", fill="black", font=font_caption)
    d.text((x2, y2 - 18), "Nontoxic SMILES", fill="black", font=font_caption)

    canvas.paste(img_tox_safe, (x1, y1))
    canvas.paste(img_tox_smiles, (x2, y1))
    canvas.paste(img_non_safe, (x1, y2))
    canvas.paste(img_non_smiles, (x2, y2))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _make_combined_image(card_paths: list[Path], out_path: Path, cols: int = 2) -> None:
    if not card_paths:
        return
    cards = [Image.open(p).convert("RGB") for p in card_paths]
    cw, ch = cards[0].size
    cols = max(1, cols)
    rows = math.ceil(len(cards) / cols)
    pad = 18
    out_w = cols * cw + (cols + 1) * pad
    out_h = rows * ch + (rows + 1) * pad
    canvas = Image.new("RGB", (out_w, out_h), (250, 250, 250))

    for i, card in enumerate(cards):
        r = i // cols
        c = i % cols
        x = pad + c * (cw + pad)
        y = pad + r * (ch + pad)
        canvas.paste(card, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize optimal pairs with SAFE/SMILES images")
    ap.add_argument("--topk-csv", type=Path, default=DEFAULT_TOPK_CSV, help="optimal_pairs_top10.csv path")
    ap.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV, help="merged_test.csv path")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output image directory")
    ap.add_argument("--grid-cols", type=int, default=2, help="Combined grid column count")
    args = ap.parse_args()

    topk_df = pd.read_csv(args.topk_csv)
    merged_df = pd.read_csv(args.merged_csv)

    # SAFE 컬럼 선택
    tox_safe_col = "toxic_safe" if "toxic_safe" in merged_df.columns else None
    non_safe_col = "nontoxic_safe" if "nontoxic_safe" in merged_df.columns else None
    if tox_safe_col is None or non_safe_col is None:
        raise ValueError("merged_test.csv must include 'toxic_safe' and 'nontoxic_safe' columns")

    card_dir = args.out_dir / "pair_cards"
    card_paths: list[Path] = []

    for _, row in topk_df.iterrows():
        row_idx = int(row["row_index"])
        if row_idx < 0 or row_idx >= len(merged_df):
            continue
        mrow = merged_df.iloc[row_idx]
        tox_safe = str(mrow.get(tox_safe_col, "") or "")
        non_safe = str(mrow.get(non_safe_col, "") or "")

        rank = int(row.get("rank", 0))
        out_path = card_dir / f"pair_{rank:02d}_row{row_idx}.png"
        _build_pair_card(row=row, toxic_safe=tox_safe, nontoxic_safe=non_safe, out_path=out_path)
        card_paths.append(out_path)

    combined_path = args.out_dir / "optimal_pairs_grid.png"
    _make_combined_image(card_paths, combined_path, cols=args.grid_cols)

    print(f"Saved {len(card_paths)} pair card images to: {card_dir}")
    print(f"Saved combined grid image to: {combined_path}")


if __name__ == "__main__":
    main()
