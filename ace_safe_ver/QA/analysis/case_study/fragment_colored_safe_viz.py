#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFE 문자열( '.' 로 분리되는 fragment들 )을 기준으로

1) 전체 분자 그림에서 fragment별로 서로 다른 색으로 하이라이트
2) 아래 SAFE 문자열에서도 동일 fragment를 동일 색으로 표시

를 한 이미지에 저장한다. toxic/nontoxic 모두 지원.

입력은 `merged_test.csv`의 toxic_safe / nontoxic_safe 를 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import datamol as dm
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


SCRIPT_DIR = Path(__file__).resolve().parent
_CASE_STUDY = SCRIPT_DIR
if str(_CASE_STUDY) not in sys.path:
    sys.path.insert(0, str(_CASE_STUDY))
from paths_bundle import setup_bundle_paths  # noqa: E402

_ACE = setup_bundle_paths(with_safe_pkg=True)
from safe.safe.converter import decode as safe_decode  # noqa: E402


DEFAULT_MERGED_CSV = (
    _ACE
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "optimal_pair_images" / "pair_fragment_highlight_set"

MOL_SIZE = (900, 520)
TOP_PAD = 18
BOTTOM_PAD = 16
TEXT_H = 120


# 색 팔레트(이미지 예시처럼 강한 색 위주)
PALETTE: Sequence[Tuple[float, float, float]] = (
    (0.00, 0.55, 0.55),  # teal
    (0.95, 0.20, 0.75),  # magenta
    (0.98, 0.80, 0.15),  # yellow
    (0.15, 0.60, 0.15),  # green
    (0.15, 0.35, 0.85),  # blue
    (0.90, 0.20, 0.20),  # red
    (0.55, 0.35, 0.85),  # purple
    (0.95, 0.55, 0.10),  # orange
)


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


def _tokenize_safe_fragments(s: str) -> List[str]:
    s = (s or "").strip().replace(" ", "")
    if not s:
        return []
    return [t for t in s.split(".") if t]


def _decode_safe_to_mol(full_safe: str) -> Optional[Chem.Mol]:
    # merged_test.csv의 toxic_safe/nontoxic_safe는 이미 full SAFE(전체) 문자열이다.
    # safe_decode는 SAFE -> SMILES. 여기서는 전체를 SMILES로 바꾼 뒤 Mol 생성.
    safe_str = (full_safe or "").strip()
    if not safe_str:
        return None
    decoded = safe_decode(
        safe_str,
        as_mol=False,
        remove_dummies=True,
        ignore_errors=True,
    )
    if not decoded:
        return None
    return dm.to_mol(str(decoded), remove_hs=False)


def _decode_fragment_to_mol(fragment_safe: str) -> Optional[Chem.Mol]:
    frag = (fragment_safe or "").strip()
    if not frag:
        return None
    decoded = safe_decode(
        frag,
        as_mol=False,
        remove_dummies=True,
        ignore_errors=True,
    )
    if not decoded:
        return None
    return dm.to_mol(str(decoded), remove_hs=False)


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    s = (smiles or "").strip()
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def _configure_draw_options(d2d: "rdMolDraw2D.MolDraw2D", bond_line_width: float = 4.5) -> None:
    opts = d2d.drawOptions()
    opts.bondLineWidth = float(bond_line_width)
    opts.scaleBondWidth = False
    # 요청: 하이라이트가 너무 두꺼움 → '얇게' 보이도록 설정
    # - 채워진 하이라이트(두껍게 보임) OFF
    # - 연속 하이라이트 OFF (blob처럼 보이는 것 방지)
    # - radius/굵기 multiplier 최소화
    opts.fillHighlights = False
    opts.continuousHighlight = False
    if hasattr(opts, "highlightRadius"):
        opts.highlightRadius = 0.12
    if hasattr(opts, "scaleHighlightBondWidth"):
        opts.scaleHighlightBondWidth = False
    if hasattr(opts, "highlightBondWidthMultiplier"):
        # 색 있는 bond 선을 더 굵게 보이게 (atom 점은 아예 안 그림)
        opts.highlightBondWidthMultiplier = 8


def _centroid_x(parent: Chem.Mol, atom_ids: Sequence[int]) -> float:
    """2D conformer 좌표 기준으로 atom 집합의 centroid x."""
    if not atom_ids:
        return 0.0
    try:
        conf = parent.GetConformer()
        xs = [float(conf.GetAtomPosition(int(a)).x) for a in atom_ids]
        return sum(xs) / max(len(xs), 1)
    except Exception:
        # conformer가 없으면 원자 인덱스 기반으로라도 안정적으로 정렬한다.
        xs = [float(int(a)) for a in atom_ids]
        return sum(xs) / max(len(xs), 1)


def _first_match_for_sort(parent: Chem.Mol, frag: Chem.Mol) -> Tuple[int, ...]:
    """
    정렬용 대표 매칭 선택:
    - 가능한 매칭들 중 (1) 원자 수 최대, (2) centroid x 최소
    """
    try:
        matches = parent.GetSubstructMatches(frag, uniquify=True, useChirality=False)
    except Exception:
        matches = ()
    if not matches:
        try:
            m = parent.GetSubstructMatch(frag, useChirality=False)
            matches = (m,) if m else ()
        except Exception:
            matches = ()
    if not matches:
        return ()
    best = None
    for m in matches:
        atom_ids = tuple(int(x) for x in m)
        score = (-len(atom_ids), _centroid_x(parent, atom_ids))
        if best is None or score < best[0]:
            best = (score, atom_ids)
    return best[1] if best is not None else ()


def _best_nonoverlap_match(
    parent: Chem.Mol,
    frag: Chem.Mol,
    used_atoms: Set[int],
) -> Tuple[int, ...]:
    """
    fragment가 parent에서 매칭되는 여러 경우 중, 이미 칠해진 원자(used_atoms)와
    겹침이 최소인 매칭을 선택한다.
    """
    try:
        matches = parent.GetSubstructMatches(frag, uniquify=True, useChirality=False)
    except Exception:
        matches = ()
    if not matches:
        try:
            m = parent.GetSubstructMatch(frag, useChirality=False)
            matches = (m,) if m else ()
        except Exception:
            matches = ()
    if not matches:
        return ()

    best = None
    for m in matches:
        mset = set(int(x) for x in m)
        overlap = len(mset & used_atoms)
        score = (overlap, -len(mset), _centroid_x(parent, m))  # overlap 최소 → 크기 최대 → x 왼쪽 우선
        if best is None or score < best[0]:
            best = (score, tuple(int(x) for x in m))
    return best[1] if best is not None else ()


def _bonds_within_atom_set(parent: Chem.Mol, atom_ids: Set[int]) -> List[int]:
    out: List[int] = []
    for b in parent.GetBonds():
        a1 = int(b.GetBeginAtomIdx())
        a2 = int(b.GetEndAtomIdx())
        if a1 in atom_ids and a2 in atom_ids:
            out.append(int(b.GetIdx()))
    return out


def _draw_molecule_with_fragment_colors(
    parent_mol: Chem.Mol,
    fragment_mols: Sequence[Chem.Mol],
    fragment_colors: Sequence[Tuple[float, float, float]],
    size: Tuple[int, int] = MOL_SIZE,
) -> Image.Image:
    """
    fragment별 substructure match를 찾아 원자/결합 하이라이트 색을 다르게 주고 그린다.
    match 실패 fragment는 건너뛴다.
    """
    m = Chem.Mol(parent_mol)
    rdDepictor.Compute2DCoords(m)
    rdMolDraw2D.PrepareMolForDrawing(m)
    w, h = size
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    _configure_draw_options(d2d)

    # 요청: atom 점(하이라이트 원) 없애기 → bond만 색칠
    highlight_bond_colors: Dict[int, Tuple[float, float, float]] = {}
    highlight_bonds: List[int] = []

    used_atoms: Set[int] = set()
    for frag_mol, col in zip(fragment_mols, fragment_colors):
        if frag_mol is None:
            continue
        match = _best_nonoverlap_match(m, frag_mol, used_atoms)
        if not match:
            continue
        aset = set(int(x) for x in match)
        bset = _bonds_within_atom_set(m, aset)

        for bi in bset:
            if bi not in highlight_bond_colors:
                highlight_bond_colors[bi] = col
            highlight_bonds.append(bi)

        used_atoms |= aset

    highlight_bonds = list(dict.fromkeys(highlight_bonds))

    # RDKit: bond 색만 줄 때도 (mol, highlightAtoms, highlightBonds, ...) 시그니처를 써야 함.
    d2d.DrawMolecule(
        m,
        [],
        highlight_bonds,
        {},
        highlight_bond_colors,
    )
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")


def _rgb255(c: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def _draw_colored_safe_string(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    fragments: Sequence[str],
    colors: Sequence[Tuple[float, float, float]],
    font: ImageFont.ImageFont,
    dot_color: Tuple[int, int, int] = (70, 70, 70),
) -> None:
    """
    SAFE 전체를 '.'로 split한 조각 단위로, 각 fragment를 지정 색으로 순서대로 렌더.
    """
    x, y = xy
    for i, (frag, col) in enumerate(zip(fragments, colors)):
        if i > 0:
            dot = "."
            draw.text((x, y), dot, fill=dot_color, font=font)
            x += draw.textbbox((0, 0), dot, font=font)[2]
        draw.text((x, y), frag, fill=_rgb255(col), font=font)
        x += draw.textbbox((0, 0), frag, font=font)[2]


def _fragment_atom_sets_for_parent(
    parent_mol: Chem.Mol,
    fragment_mols: Sequence[Optional[Chem.Mol]],
) -> List[Set[int]]:
    out: List[Set[int]] = []
    used_atoms: Set[int] = set()
    for fm in fragment_mols:
        if fm is None:
            out.append(set())
            continue
        m = _best_nonoverlap_match(parent_mol, fm, used_atoms)
        aset = set(int(x) for x in m) if m else set()
        out.append(aset)
        used_atoms |= aset
    return out


def _tokenize_original_smiles_with_atom_indices(smiles: str) -> List[Tuple[str, Optional[int]]]:
    """
    원본 SMILES 문자열을 가능한 그대로 유지한 채 토큰화한다.
    atom 토큰에만 atom index를 매긴다. (MolFromSmiles 파싱 순서 가정)
    """
    s = (smiles or "").strip()
    if not s:
        return []

    tokens: List[Tuple[str, Optional[int]]] = []
    atom_idx = 0
    i = 0
    two_char_elems = {
        "Cl",
        "Br",
        "Si",
        "Na",
        "Li",
        "Al",
        "Ca",
        "Fe",
        "Zn",
        "Cu",
        "Mg",
        "Mn",
        "Sn",
        "Se",
    }
    non_atom_single = set("-=#:/.()\\")

    while i < len(s):
        ch = s[i]

        if ch == "[":
            j = s.find("]", i + 1)
            if j == -1:
                tokens.append((s[i:], None))
                break
            tokens.append((s[i : j + 1], atom_idx))
            atom_idx += 1
            i = j + 1
            continue

        if ch == "%":
            if i + 2 < len(s) and s[i + 1].isdigit() and s[i + 2].isdigit():
                tokens.append((s[i : i + 3], None))
                i += 3
                continue
            tokens.append((ch, None))
            i += 1
            continue

        if ch.isdigit() or ch in non_atom_single:
            tokens.append((ch, None))
            i += 1
            continue

        if ch == "*":
            tokens.append((ch, atom_idx))
            atom_idx += 1
            i += 1
            continue

        if ch.isalpha():
            if i + 1 < len(s) and s[i : i + 2] in two_char_elems:
                tokens.append((s[i : i + 2], atom_idx))
                atom_idx += 1
                i += 2
                continue
            if ch.isupper():
                if i + 1 < len(s) and s[i + 1].islower():
                    tokens.append((s[i : i + 2], atom_idx))
                    atom_idx += 1
                    i += 2
                    continue
                tokens.append((ch, atom_idx))
                atom_idx += 1
                i += 1
                continue
            if ch in "bcnops":
                tokens.append((ch, atom_idx))
                atom_idx += 1
                i += 1
                continue

        tokens.append((ch, None))
        i += 1

    return tokens


def _draw_colored_smiles_tokens(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    smiles_tokens: Sequence[Tuple[str, Optional[int]]],
    atom_color_by_idx: Dict[int, Tuple[float, float, float]],
    font: ImageFont.ImageFont,
    default_color: Tuple[int, int, int] = (20, 20, 20),
) -> None:
    x, y = xy
    for text, atom_idx in smiles_tokens:
        if atom_idx is not None and atom_idx in atom_color_by_idx:
            fill = _rgb255(atom_color_by_idx[atom_idx])
        else:
            fill = default_color
        draw.text((x, y), text, fill=fill, font=font)
        x += draw.textbbox((0, 0), text, font=font)[2]


def _read_merged_row(merged_csv: Path, row_index: int) -> dict:
    with merged_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == row_index:
                return row
    raise SystemExit(f"row_index={row_index} out of range in {merged_csv}")


def build_image_for_safe(
    title: str,
    full_safe: str,
    original_smiles: str,
    out_path: Path,
    *,
    bond_line_width: float = 4.5,
) -> None:
    fragments = _tokenize_safe_fragments(full_safe)
    if not fragments:
        raise SystemExit("SAFE is empty; cannot build fragment-colored visualization.")

    parent_safe_mol = _decode_safe_to_mol(full_safe)
    if parent_safe_mol is None:
        raise SystemExit("SAFE decode failed for full molecule.")
    parent_smiles_mol = _mol_from_smiles(original_smiles)
    if parent_smiles_mol is None:
        # fallback: 최소한 결과는 나오도록 SAFE decode Mol 사용
        parent_smiles_mol = Chem.Mol(parent_safe_mol)
    try:
        rdDepictor.Compute2DCoords(parent_smiles_mol)
    except Exception:
        pass

    # 2D coords 먼저 잡아 fragment 위치 기반 정렬에 사용
    rdDepictor.Compute2DCoords(parent_safe_mol)

    frag_mols: List[Chem.Mol] = []
    tmp: List[Tuple[float, str, Optional[Chem.Mol]]] = []
    for frag in fragments:
        fm = _decode_fragment_to_mol(frag)
        if fm is None:
            tmp.append((1e9, frag, None))
            continue
        m = _first_match_for_sort(parent_safe_mol, fm)
        cx = _centroid_x(parent_safe_mol, m) if m else 1e9
        tmp.append((cx, frag, fm))

    # left->right 정렬 (매칭 실패는 뒤로)
    tmp.sort(key=lambda x: x[0])
    fragments_sorted = [t[1] for t in tmp]
    frag_mols = [t[2] for t in tmp]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(fragments_sorted))]

    # Molecule panel
    mol_img = _draw_molecule_with_fragment_colors(parent_safe_mol, frag_mols, colors, size=MOL_SIZE)

    # Original SMILES token color map (fragment가 문자열에서 분절되어도 atom 단위 색 유지)
    frag_atom_sets_smiles = _fragment_atom_sets_for_parent(parent_smiles_mol, frag_mols)
    atom_color_by_idx: Dict[int, Tuple[float, float, float]] = {}
    for aset, col in zip(frag_atom_sets_smiles, colors):
        for a in aset:
            atom_color_by_idx[a] = col
    smiles_tokens = _tokenize_original_smiles_with_atom_indices(original_smiles)
    if not smiles_tokens:
        smiles_tokens = [(original_smiles, None)]

    # Compose with title + colored SAFE text + colored SMILES text
    canvas_w = MOL_SIZE[0]
    canvas_h = TOP_PAD + 54 + MOL_SIZE[1] + 10 + (TEXT_H * 2)
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)
    font_title = _safe_font(44)
    font_safe = _safe_font(22)
    font_label = _safe_font(20)

    # Title centered
    tbb = d.textbbox((0, 0), title, font=font_title)
    tw = tbb[2] - tbb[0]
    d.text(((canvas_w - tw) // 2, TOP_PAD), title, fill=(0, 0, 0), font=font_title)

    # Molecule
    canvas.paste(mol_img, (0, TOP_PAD + 54))

    # SAFE string (colored fragments) at bottom
    y_text = TOP_PAD + 54 + MOL_SIZE[1] + 12
    _draw_colored_safe_string(
        d,
        (20, y_text),
        fragments_sorted,
        colors,
        font=font_safe,
    )
    y_smiles = y_text + 52
    d.text((20, y_smiles), "SMILES:", fill=(30, 30, 30), font=font_label)
    _draw_colored_smiles_tokens(
        d,
        (110, y_smiles),
        smiles_tokens,
        atom_color_by_idx=atom_color_by_idx,
        font=font_safe,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fragment-colored molecule + SAFE text visualization (toxic/nontoxic)")
    ap.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_CSV)
    ap.add_argument("--row-index", type=int, required=True, help="merged_test.csv의 0-based 행 인덱스")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="출력 디렉터리 (기본: pair_fragment_highlight_set/row_<row_index>/fragment_colored)",
    )
    ap.add_argument(
        "--bond-line-width",
        type=float,
        default=4.5,
        help="molecule bond line thickness (색 선이 잘 보이게 크게 권장: 4~7)",
    )
    args = ap.parse_args()

    row = _read_merged_row(args.merged_csv, args.row_index)
    tox_safe = str(row.get("toxic_safe", "") or "").strip()
    non_safe = str(row.get("nontoxic_safe", "") or "").strip()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = DEFAULT_OUT_DIR / f"row_{args.row_index}" / "fragment_colored"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_image_for_safe(
        "SAFE (toxic)",
        tox_safe,
        str(row.get("toxic_smiles", "") or ""),
        out_dir / "toxic_fragment_colored_safe.png",
        bond_line_width=args.bond_line_width,
    )
    build_image_for_safe(
        "SAFE (nontoxic)",
        non_safe,
        str(row.get("nontoxic_smiles", "") or ""),
        out_dir / "nontoxic_fragment_colored_safe.png",
        bond_line_width=args.bond_line_width,
    )

    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()

