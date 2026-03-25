#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
row_858 등에서 생성된 4개 PNG를 2x2로 붙인다.

행 1: Toxic — 왼쪽 Full molecule | 오른쪽 Only fragment
행 2: Nontoxic — 동일

기본 입력: 이 스크립트 옆의 row_858/rank01_row858_*.png
출력 기본: row_858/pair_full_vs_fragment_grid.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _resize_to_height(im: Image.Image, target_h: int) -> Image.Image:
    w, h = im.size
    if h == target_h:
        return im
    new_w = max(1, round(w * target_h / h))
    return im.resize((new_w, target_h), Image.Resampling.LANCZOS)


def compose_grid(
    toxic_full: Path,
    toxic_only: Path,
    nontoxic_full: Path,
    nontoxic_only: Path,
    *,
    row_height: int = 420,
    gap: int = 16,
    pad: int = 20,
    label_h: int = 30,
    left_label_w: int = 108,
) -> Image.Image:
    t_full = Image.open(toxic_full).convert("RGBA")
    t_only = Image.open(toxic_only).convert("RGBA")
    nt_full = Image.open(nontoxic_full).convert("RGBA")
    nt_only = Image.open(nontoxic_only).convert("RGBA")

    t_full = _resize_to_height(t_full, row_height)
    t_only = _resize_to_height(t_only, row_height)
    nt_full = _resize_to_height(nt_full, row_height)
    nt_only = _resize_to_height(nt_only, row_height)

    col0_w = max(t_full.width, nt_full.width)
    col1_w = max(t_only.width, nt_only.width)

    body_h = row_height * 2 + gap
    total_w = pad * 2 + left_label_w + col0_w + gap + col1_w
    total_h = pad * 2 + label_h + body_h

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 17)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    y0 = pad + label_h
    x_cells = pad + left_label_w

    # 열 제목: Full molecule | Only fragment
    for text, cx_off, col_w in (
        ("Full molecule", 0, col0_w),
        ("Only fragment", col0_w + gap, col1_w),
    ):
        cx = x_cells + cx_off + col_w // 2
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, pad + 2), text, fill=(40, 40, 40), font=font)

    # Row 1: toxic
    y = y0
    canvas.paste(t_full, (x_cells + (col0_w - t_full.width) // 2, y))
    canvas.paste(t_only, (x_cells + col0_w + gap + (col1_w - t_only.width) // 2, y))
    tb = draw.textbbox((0, 0), "Toxic", font=font_small)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    draw.text(
        (pad + (left_label_w - tw) // 2, y + (row_height - th) // 2),
        "Toxic",
        fill=(180, 35, 35),
        font=font_small,
    )

    # Row 2: nontoxic
    y = y0 + row_height + gap
    canvas.paste(nt_full, (x_cells + (col0_w - nt_full.width) // 2, y))
    canvas.paste(nt_only, (x_cells + col0_w + gap + (col1_w - nt_only.width) // 2, y))
    tb = draw.textbbox((0, 0), "Nontoxic", font=font_small)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    draw.text(
        (pad + (left_label_w - tw) // 2, y + (row_height - th) // 2),
        "Nontoxic",
        fill=(28, 130, 42),
        font=font_small,
    )

    return canvas


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_row = script_dir / "row_858"
    ap = argparse.ArgumentParser(description="Full vs only-fragment 2x2 grid PNG")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=default_row,
        help="rank01_* PNG가 있는 디렉터리 (기본: row_858)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="출력 PNG (기본: input-dir/pair_full_vs_fragment_grid.png)",
    )
    ap.add_argument("--row-height", type=int, default=420)
    ap.add_argument("--gap", type=int, default=16)
    args = ap.parse_args()

    d = args.input_dir.resolve()
    toxic_full = d / "rank01_row858_01_toxic_safe_plain.png"
    nontoxic_full = d / "rank01_row858_02_nontoxic_safe_plain.png"
    toxic_only = d / "rank01_row858_06_only_toxic_fragment_safe_plain.png"
    nontoxic_only = d / "rank01_row858_07_only_nontoxic_fragment_safe_plain.png"

    for p in (toxic_full, nontoxic_full, toxic_only, nontoxic_only):
        if not p.is_file():
            raise SystemExit(f"파일 없음: {p}")

    out = args.out if args.out is not None else d / "pair_full_vs_fragment_grid.png"

    img = compose_grid(
        toxic_full,
        toxic_only,
        nontoxic_full,
        nontoxic_only,
        row_height=args.row_height,
        gap=args.gap,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
