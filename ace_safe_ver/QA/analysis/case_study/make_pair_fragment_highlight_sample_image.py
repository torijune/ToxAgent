#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
`optimal_pair_images/pair_fragment_highlight_sample.png`에
toxic/nontoxic의 SAFE/SMILES/fragments(SAFE) + dataset/endpoint를 같이 넣기.

기본값은 row_index=858, rank=1 (현재 존재하는 sample 이미지와 동일한 케이스)입니다.
기존 PNG 2장(only-fragment 하이라이트)을 그대로 사용해서 상단 텍스트만 확장합니다.
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent

DEFAULT_TOPK_CSV = SCRIPT_DIR / "optimal_pairs_top10.csv"
DEFAULT_MERGED_TEST_CSV = (
    PROJECT_ROOT
    / "ace_safe_ver"
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)

DEFAULT_TOPK_OUT_DIR = SCRIPT_DIR / "optimal_pair_images"
DEFAULT_SAMPLE_OUT = DEFAULT_TOPK_OUT_DIR / "pair_fragment_highlight_sample.png"
DEFAULT_FRAGMENT_HIGHLIGHT_DIR = (
    DEFAULT_TOPK_OUT_DIR / "pair_fragment_highlight_set"
)

DEFAULT_ROW_INDEX = 858
DEFAULT_RANK = 1


def _safe_font(size: int) -> ImageFont.ImageFont:
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


def _truncate(s: str, width_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= width_chars:
        return s
    if width_chars <= 3:
        return s[:width_chars]
    return s[: width_chars - 3] + "..."


def _read_topk_row_by_row_index(topk_csv: Path, row_index: int) -> dict:
    with topk_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("row_index", "-1") or "-1") == row_index:
                return row
    raise SystemExit(f"row_index={row_index} not found in {topk_csv}")


def _read_merged_row_by_index(merged_csv: Path, row_index: int) -> dict:
    with merged_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == row_index:
                return row
    raise SystemExit(f"row_index={row_index} out of range in {merged_csv} (len unknown)")


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise SystemExit(f"파일 없음: {path}")
    return Image.open(path).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate pair_fragment_highlight_sample.png with more metadata")
    ap.add_argument("--row-index", type=int, default=DEFAULT_ROW_INDEX)
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK)
    ap.add_argument("--topk-csv", type=Path, default=DEFAULT_TOPK_CSV)
    ap.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_TEST_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_SAMPLE_OUT)
    args = ap.parse_args()

    topk = _read_topk_row_by_row_index(args.topk_csv, args.row_index)
    merged = _read_merged_row_by_index(args.merged_csv, args.row_index)

    dataset_name = str(topk.get("dataset_name", "") or "").strip()
    endpoint = str(topk.get("endpoint", "") or "").strip()

    toxic_smiles = str(topk.get("toxic_smiles", "") or "").strip()
    nontoxic_smiles = str(topk.get("nontoxic_smiles", "") or "").strip()
    only_toxic_frag = str(topk.get("only_toxic_safe_fragment", "") or "").strip()
    only_nontoxic_frag = str(topk.get("only_nontoxic_safe_fragment", "") or "").strip()

    toxic_safe = str(merged.get("toxic_safe", "") or "").strip()
    nontoxic_safe = str(merged.get("nontoxic_safe", "") or "").strip()

    prefix = f"rank{args.rank:02d}_row{args.row_index}"
    row_dir = DEFAULT_FRAGMENT_HIGHLIGHT_DIR / f"row_{args.row_index}"
    toxic_img_path = row_dir / f"{prefix}_03_toxic_safe_highlight_only_toxic_frag_red.png"
    nontoxic_img_path = row_dir / f"{prefix}_04_nontoxic_safe_highlight_only_nontoxic_frag_green.png"

    img_tox = _load_image(toxic_img_path)
    img_non = _load_image(nontoxic_img_path)

    # ---- layout ----
    font_title = _safe_font(18)
    font_meta = _safe_font(13)
    font_line = _safe_font(12)
    font_label = _safe_font(14)

    gap = 20
    pad = 18
    line_w_chars = 120  # wrap/truncate target

    title = "Optimal Pair Fragment Highlight (SAFE)"

    lines: list[str] = []
    lines.append(f"row_index={args.row_index}  rank={args.rank}")
    lines.append(f"dataset={dataset_name}  endpoint={endpoint}")
    lines.append(f"Toxic SAFE: {_truncate(toxic_safe, line_w_chars)}")
    lines.append(f"Toxic SMILES: {_truncate(toxic_smiles, line_w_chars)}")
    lines.append(f"Toxic fragment(safe): {_truncate(only_toxic_frag, line_w_chars)}")
    lines.append(f"Nontoxic SAFE: {_truncate(nontoxic_safe, line_w_chars)}")
    lines.append(f"Nontoxic SMILES: {_truncate(nontoxic_smiles, line_w_chars)}")
    lines.append(f"Nontoxic fragment(safe): {_truncate(only_nontoxic_frag, line_w_chars)}")

    # Compute text height
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    def _font_for_wrapped_line(s: str) -> ImageFont.ImageFont:
        if s.startswith("dataset=") or s.startswith("row_index="):
            return font_meta
        return font_line

    y = pad
    bbox_title = dummy.textbbox((0, 0), title, font=font_title)
    y += (bbox_title[3] - bbox_title[1]) + 6
    for line in lines:
        wrapped = textwrap.wrap(line, width=110) if len(line) > 130 else [line]
        for wline in wrapped:
            font = _font_for_wrapped_line(wline)
            bb = dummy.textbbox((0, 0), wline, font=font)
            y += (bb[3] - bb[1]) + 4

    text_bottom = y + 6
    mol_h = max(img_tox.height, img_non.height)
    mol_h = mol_h if mol_h > 0 else 420

    canvas_w = img_tox.width + gap + img_non.width + pad * 2
    canvas_h = text_bottom + mol_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)

    # Title
    d.text((pad, pad), title, fill="black", font=font_title)

    ycur = pad + (dummy.textbbox((0, 0), title, font=font_title)[3]) + 6
    for line in lines:
        wrapped = textwrap.wrap(line, width=110) if len(line) > 130 else [line]
        for wline in wrapped:
            font = _font_for_wrapped_line(wline)
            d.text((pad, ycur), wline, fill="black", font=font)
            bb = dummy.textbbox((0, 0), wline, font=font)
            ycur += (bb[3] - bb[1]) + 4

    # Place images
    img_y = text_bottom
    canvas.paste(img_tox, (pad + (img_tox.width // 2 - img_tox.width // 2), img_y))
    canvas.paste(img_non, (pad + img_tox.width + gap, img_y))

    # Labels
    d.text((pad + 10, img_y - 24), "Toxic SAFE", fill="black", font=font_label)
    d.text((pad + img_tox.width + gap + 10, img_y - 24), "Nontoxic SAFE", fill="black", font=font_label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

