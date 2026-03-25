#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimal_pairs_top10.csv에서 optimal pair 1개를 고른 뒤, 아래 8개 PNG를 저장합니다.

1) Toxic SAFE (plain)
2) Nontoxic SAFE (plain)
3) Toxic SAFE — only toxic fragment만 빨간색 하이라이트
4) Nontoxic SAFE — only nontoxic fragment만 초록색 하이라이트
5) Toxic 전체 SAFE에서 only toxic fragment를 제거한 나머지 (전체 맥락 유지)
6) Nontoxic 전체 SAFE에서 only nontoxic fragment를 제거한 나머지 (전체 맥락 유지)
7) Only toxic fragment SAFE만 — 전체 분자 이미지에서 해당 fragment 영역만 잘라 **각도·스케일 동일**
8) Only nontoxic fragment SAFE만 — 동일
"""

from __future__ import annotations

import argparse
import io
import itertools
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import datamol as dm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe.safe.converter import decode as safe_decode

DEFAULT_TOPK_CSV = SCRIPT_DIR / "optimal_pairs_top10.csv"
DEFAULT_MERGED_CSV = (
    PROJECT_ROOT
    / "ace_safe_ver"
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "optimal_pair_images" / "pair_fragment_highlight_set"
MOL_SIZE = (520, 420)

# RDKit highlight colors [0,1]
RED = (1.0, 0.15, 0.15)
GREEN = (0.15, 0.72, 0.18)


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


def _decode_safe_fragment_to_mol(safe_fragment: str) -> Optional[Chem.Mol]:
    decoded = safe_decode(
        str(safe_fragment).strip(),
        as_mol=False,
        remove_dummies=True,
        ignore_errors=True,
    )
    if not decoded:
        return None
    return dm.to_mol(str(decoded), remove_hs=False)


def _mol_from_full_safe(full_safe: str) -> Optional[Chem.Mol]:
    return dm.to_mol(str(full_safe).strip(), remove_hs=False)


def _placeholder(size: Tuple[int, int], msg: str) -> Image.Image:
    img = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.text((12, 12), msg, fill="gray", font=_safe_font(16))
    return img


def _draw_plain_mol(mol: Optional[Chem.Mol], size=MOL_SIZE) -> Image.Image:
    """전체 분자 plain — highlight·크롭 경로와 동일하게 rdMolDraw2D + Compute2DCoords 사용."""
    if mol is None:
        return _placeholder(size, "Molecule parse failed")
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    rdMolDraw2D.PrepareMolForDrawing(m)
    w, h = size
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d2d.DrawMolecule(m)
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")


def _substruct_match_parent_frag(parent: Chem.Mol, frag: Chem.Mol) -> Tuple[int, ...]:
    """부모에서 fragment에 대응하는 원자 인덱스 튜플 (없으면 빈 튜플)."""
    m = parent.GetSubstructMatch(frag, useChirality=True)
    if not m:
        m = parent.GetSubstructMatch(frag, useChirality=False)
    return tuple(int(x) for x in m)


def _draw_only_fragment_matching_full_molecule(
    parent: Optional[Chem.Mol],
    frag: Optional[Chem.Mol],
    size: Tuple[int, int] = MOL_SIZE,
    padding_px: float = 40.0,
) -> Image.Image:
    """
    Full molecule에 그려진 것과 동일한 2D 배치·상대 스케일로 fragment만 보이게 한다.

    부모를 동일 캔버스에 그린 뒤, substructure 원자들의 GetDrawCoords 로 bbox를 구해 크롭하고
    동일 해상도 캔버스에 맞게 비율 유지하여 가운데 배치한다.
    """
    if parent is None or frag is None:
        return _placeholder(size, "Molecule parse failed")
    pm = Chem.Mol(parent)
    fm = Chem.Mol(frag)
    try:
        Chem.SanitizeMol(pm)
        Chem.SanitizeMol(fm)
    except Exception:
        pass
    match = _substruct_match_parent_frag(pm, fm)
    if not match:
        try:
            atom_matches, _ = dm.substructure_matching_bonds(Chem.Mol(pm), fm)
            flat = list(dict.fromkeys(itertools.chain(*atom_matches)))
            match = tuple(int(x) for x in flat) if flat else ()
        except Exception:
            match = ()
    if not match:
        return _draw_plain_mol(fm, size=size)

    w, h = size
    rdDepictor.Compute2DCoords(pm)
    rdMolDraw2D.PrepareMolForDrawing(pm)
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d2d.DrawMolecule(pm)
    xs: List[float] = []
    ys: List[float] = []
    for idx in match:
        pt = d2d.GetDrawCoords(int(idx))
        xs.append(float(pt.x))
        ys.append(float(pt.y))
    d2d.FinishDrawing()
    full_img = Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")

    pad = float(padding_px)
    x0 = max(0.0, min(xs) - pad)
    y0 = max(0.0, min(ys) - pad)
    x1 = min(float(w), max(xs) + pad)
    y1 = min(float(h), max(ys) + pad)
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return _draw_plain_mol(fm, size=size)

    cropped = full_img.crop((int(x0), int(y0), int(x1), int(y1)))
    cropped = cropped.copy()
    cropped.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, (255, 255, 255))
    ox = (size[0] - cropped.width) // 2
    oy = (size[1] - cropped.height) // 2
    out.paste(cropped, (ox, oy))
    return out


def _highlight_one_color(
    mol: Optional[Chem.Mol],
    frag_mol: Optional[Chem.Mol],
    color: Tuple[float, float, float],
    size=MOL_SIZE,
) -> Image.Image:
    """
    하이라이트 색은 RDKit Draw.MolToImage가 무시하는 경우가 있어
    rdMolDraw2D(MolDraw2DCairo)로 그립니다.
    """
    if mol is None:
        return _placeholder(size, "Molecule parse failed")
    if frag_mol is None:
        return _draw_plain_mol(mol, size=size)
    mol_draw = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol_draw)
    atom_matches, bond_matches = dm.substructure_matching_bonds(mol_draw, frag_mol)
    atom_flat = list(dict.fromkeys(itertools.chain(*atom_matches)))
    bond_flat = list(dict.fromkeys(itertools.chain(*bond_matches)))
    ac = dict.fromkeys(atom_flat, color)
    bc = dict.fromkeys(bond_flat, color)
    rdMolDraw2D.PrepareMolForDrawing(mol_draw)
    w, h = size
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d2d.DrawMolecule(
        mol_draw,
        highlightAtoms=atom_flat,
        highlightBonds=bond_flat,
        highlightAtomColors=ac,
        highlightBondColors=bc,
    )
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")


def _mol_without_fragment(mol: Optional[Chem.Mol], frag: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    """전체 분자에서 only fragment에 해당하는 부분식을 삭제한 잔여 Mol (RDKit DeleteSubstructs)."""
    if mol is None or frag is None:
        return None
    try:
        parent = Chem.Mol(mol)
        query = Chem.Mol(frag)
        rem = AllChem.DeleteSubstructs(parent, query)
        if rem is None or rem.GetNumAtoms() == 0:
            return None
        Chem.SanitizeMol(rem)
        return rem
    except Exception:
        return None


def _draw_mol_cairo_plain(mol: Optional[Chem.Mol], size=MOL_SIZE) -> Image.Image:
    """2D 좌표를 새로 잡아 단일 분자 이미지 (잔여골격용)."""
    if mol is None:
        return _placeholder(size, "Molecule missing / delete failed")
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    rdMolDraw2D.PrepareMolForDrawing(m)
    w, h = size
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    d2d.DrawMolecule(m)
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")


def generate_pair_images(
    topk_row: "pd.Series",
    merged: "pd.DataFrame",
    *,
    out_dir: Optional[Path] = None,
    sample_index_hint: int = 0,
) -> Path:
    """
    topk CSV 한 행 + merged_test 전체 DataFrame으로 8장 PNG 저장.
    반환: 출력 디렉터리 경로.
    """
    row_index = int(topk_row["row_index"])
    if row_index < 0 or row_index >= len(merged):
        raise ValueError(f"row_index out of range in merged csv: {row_index}")

    mrow = merged.iloc[row_index]
    toxic_safe = str(mrow.get("toxic_safe", "") or "")
    nontoxic_safe = str(mrow.get("nontoxic_safe", "") or "")
    only_tox_frag = str(topk_row.get("only_toxic_safe_fragment", "") or "")
    only_non_frag = str(topk_row.get("only_nontoxic_safe_fragment", "") or "")

    mol_tox = _mol_from_full_safe(toxic_safe)
    mol_non = _mol_from_full_safe(nontoxic_safe)
    frag_tox = _decode_safe_fragment_to_mol(only_tox_frag)
    frag_non = _decode_safe_fragment_to_mol(only_non_frag)

    rem_tox = _mol_without_fragment(mol_tox, frag_tox)
    rem_non = _mol_without_fragment(mol_non, frag_non)

    resolved_out = out_dir
    if resolved_out is None:
        resolved_out = DEFAULT_OUT_DIR / f"row_{row_index}"
    resolved_out = Path(resolved_out)
    resolved_out.mkdir(parents=True, exist_ok=True)

    rank = int(topk_row.get("rank", sample_index_hint + 1))
    prefix = f"rank{rank:02d}_row{row_index}"

    paths = {
        f"{prefix}_01_toxic_safe_plain.png": _draw_plain_mol(mol_tox),
        f"{prefix}_02_nontoxic_safe_plain.png": _draw_plain_mol(mol_non),
        f"{prefix}_03_toxic_safe_highlight_only_toxic_frag_red.png": _highlight_one_color(
            mol_tox, frag_tox, RED
        ),
        f"{prefix}_04_nontoxic_safe_highlight_only_nontoxic_frag_green.png": _highlight_one_color(
            mol_non, frag_non, GREEN
        ),
        f"{prefix}_05_toxic_safe_without_only_toxic_fragment.png": _draw_mol_cairo_plain(rem_tox),
        f"{prefix}_06_nontoxic_safe_without_only_nontoxic_fragment.png": _draw_mol_cairo_plain(rem_non),
        f"{prefix}_07_only_toxic_fragment_safe_plain.png": _draw_only_fragment_matching_full_molecule(
            mol_tox, frag_tox
        ),
        f"{prefix}_08_only_nontoxic_fragment_safe_plain.png": _draw_only_fragment_matching_full_molecule(
            mol_non, frag_non
        ),
    }

    for name, img in paths.items():
        p = resolved_out / name
        img.save(p)
        print(f"Saved: {p}")

    print(f"Done. Directory: {resolved_out}")
    return resolved_out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Save 8 images for one optimal pair (SAFE + fragment highlights + remainder)"
    )
    ap.add_argument("--topk-csv", type=Path, default=DEFAULT_TOPK_CSV, help="optimal_pairs_top10.csv path")
    ap.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV, help="merged_test.csv path")
    ap.add_argument("--sample-index", type=int, default=0, help="0-based row index in topk csv (단일 모드)")
    ap.add_argument(
        "--batch",
        action="store_true",
        help="topk CSV의 모든 행에 대해 row_<row_index> 디렉터리에 각각 8장 생성",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="--batch 시 최대 N행만 처리 (앞에서부터)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="단일 모드: 출력 디렉터리 직접 지정 (기본: .../row_<row_index>). --batch 시 무시됨",
    )
    args = ap.parse_args()

    topk = pd.read_csv(args.topk_csv)
    merged = pd.read_csv(args.merged_csv)
    if topk.empty:
        raise ValueError(f"Empty topk csv: {args.topk_csv}")

    if args.batch:
        n = len(topk) if args.limit is None else min(len(topk), args.limit)
        for i in range(n):
            row = topk.iloc[i]
            generate_pair_images(row, merged, out_dir=None, sample_index_hint=i)
        print(f"Batch finished: {n} pair(s).")
        return

    if args.sample_index < 0 or args.sample_index >= len(topk):
        raise ValueError(f"sample-index must be in [0, {len(topk) - 1}]")

    row = topk.iloc[args.sample_index]
    row_index = int(row["row_index"])
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR / f"row_{row_index}"
    generate_pair_images(row, merged, out_dir=out_dir, sample_index_hint=args.sample_index)


if __name__ == "__main__":
    main()
