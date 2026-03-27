#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
best_case/selected/test 아래의 best_cases_*.jsonl을 읽어
merged_test.csv 정보를 결합하고(case study용 정보 확장),
select_optimal_pair.py / optimal_pair_image.py / optimal_pair_fragment_highlight.py와 동일한 종류의
지표/시각화를 생성해 best_case/selected 아래에 저장한다.

입력(기본)
----------
ace_safe_ver/QA/analysis/case_study/best_case/selected/test/**/best_cases_*.jsonl

출력(기본)
----------
ace_safe_ver/QA/analysis/case_study/best_case/selected/augmented/test/<task>/both_repre/<step>/<model>/
  - augmented_best_cases.jsonl
  - augmented_best_cases.csv
  - images/
      - pair_cards/pair_<idx>_row<source_index>.png
      - pair_fragment_highlight_set/row_<source_index>/
            rankXX_rowYYY_01_...png  (총 8장, optimal_pair_fragment_highlight.py와 같은 네이밍)

주의
----
이 스크립트는 RDKit/Pillow/datamol/safe 패키지가 설치된 환경(venv)에서 실행해야 한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import datamol as dm
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CASE_STUDY_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CASE_STUDY_DIR.parent.parent.parent.parent

SELECTED_TEST_ROOT = SCRIPT_DIR / "selected" / "test"
MERGED_TEST_CSV = (
    PROJECT_ROOT
    / "ace_safe_ver"
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)

OUT_ROOT = SCRIPT_DIR / "selected" / "augmented"


# SAFE renderer (same as optimal_pair_image.py)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from safe.safe.viz import to_image as safe_to_image  # noqa: E402
from safe.safe.converter import decode as safe_decode  # noqa: E402


# ---------------------------------------------------------------------------
# Shared helpers (fonts, text)
# ---------------------------------------------------------------------------
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


def _truncate(s: str, width: int = 95) -> str:
    s = (s or "").strip()
    if len(s) <= width:
        return s
    return s[: width - 3] + "..."


def _normalize_answer(ans: Any) -> str:
    if isinstance(ans, dict):
        return str(ans.get("answer", "") or "").strip()
    return str(ans or "").strip()


def _to_pil_from_safe_obj(img_obj, fallback_size=(320, 260)) -> Image.Image:
    if isinstance(img_obj, Image.Image):
        return img_obj.convert("RGB")
    if isinstance(img_obj, bytes):
        try:
            return Image.open(io.BytesIO(img_obj)).convert("RGB")
        except Exception:
            return Image.new("RGB", fallback_size, (245, 245, 245))
    return Image.new("RGB", fallback_size, (245, 245, 245))


# ---------------------------------------------------------------------------
# Metrics (same family as select_optimal_pair.py)
# ---------------------------------------------------------------------------
def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    s = (smiles or "").strip()
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def _pair_metrics(
    toxic_smiles: str,
    nontoxic_smiles: str,
    radius: int = 2,
    fp_size: int = 2048,
    mcs_timeout_sec: int = 5,
) -> Optional[Dict[str, float]]:
    tox = _mol_from_smiles(toxic_smiles)
    non = _mol_from_smiles(nontoxic_smiles)
    if tox is None or non is None:
        return None

    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size, includeChirality=True)
    fp_tox = fpgen.GetFingerprint(tox)
    fp_non = fpgen.GetFingerprint(non)
    tanimoto = float(DataStructs.TanimotoSimilarity(fp_tox, fp_non))

    mcs = rdFMCS.FindMCS(
        [tox, non],
        timeout=mcs_timeout_sec,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchValences=False,
    )
    mcs_atoms = int(mcs.numAtoms or 0)
    mcs_bonds = int(mcs.numBonds or 0)

    tox_atoms = int(tox.GetNumHeavyAtoms())
    non_atoms = int(non.GetNumHeavyAtoms())
    tox_bonds = int(tox.GetNumBonds())
    non_bonds = int(non.GetNumBonds())
    tox_ring_count = int(tox.GetRingInfo().NumRings())
    non_ring_count = int(non.GetRingInfo().NumRings())
    tox_rotb = int(AllChem.CalcNumRotatableBonds(tox))
    non_rotb = int(AllChem.CalcNumRotatableBonds(non))

    max_atoms = max(tox_atoms, non_atoms, 1)
    max_bonds = max(tox_bonds, non_bonds, 1)

    mcs_atom_ratio = mcs_atoms / max_atoms
    mcs_bond_ratio = mcs_bonds / max_bonds

    atom_diff_abs = abs(tox_atoms - non_atoms)
    bond_diff_abs = abs(tox_bonds - non_bonds)
    atom_diff_ratio = atom_diff_abs / max_atoms
    bond_diff_ratio = bond_diff_abs / max_bonds
    ring_diff_abs = abs(tox_ring_count - non_ring_count)

    optimal_score = (
        0.50 * tanimoto
        + 0.35 * mcs_atom_ratio
        + 0.15 * mcs_bond_ratio
        - 0.10 * atom_diff_ratio
        - 0.05 * bond_diff_ratio
    )

    return {
        "tanimoto_chiral_morgan": tanimoto,
        "mcs_atoms": float(mcs_atoms),
        "mcs_bonds": float(mcs_bonds),
        "tox_heavy_atoms": float(tox_atoms),
        "non_heavy_atoms": float(non_atoms),
        "tox_bonds": float(tox_bonds),
        "non_bonds": float(non_bonds),
        "tox_ring_count": float(tox_ring_count),
        "non_ring_count": float(non_ring_count),
        "tox_rotb": float(tox_rotb),
        "non_rotb": float(non_rotb),
        "mcs_atom_ratio": float(mcs_atom_ratio),
        "mcs_bond_ratio": float(mcs_bond_ratio),
        "atom_diff_abs": float(atom_diff_abs),
        "bond_diff_abs": float(bond_diff_abs),
        "atom_diff_ratio": float(atom_diff_ratio),
        "bond_diff_ratio": float(bond_diff_ratio),
        "ring_diff_abs": float(ring_diff_abs),
        "optimal_score": float(optimal_score),
    }


def _fragment_size_from_safe_fragment(safe_fragment: str) -> Optional[int]:
    frag = str(safe_fragment or "").strip()
    if not frag:
        return None
    decoded = safe_decode(frag, as_mol=False, remove_dummies=False, ignore_errors=True)
    if not decoded:
        return None
    mol = Chem.MolFromSmiles(str(decoded))
    if mol is None:
        return None
    return int(mol.GetNumHeavyAtoms())


def _count_safe_fragments(safe_str: str) -> int:
    s = str(safe_str or "").strip()
    if not s:
        return 0
    return len([p for p in s.split(".") if p.strip()])


# ---------------------------------------------------------------------------
# Visualization: Pair card (like optimal_pair_image.py)
# ---------------------------------------------------------------------------
MOL_SIZE_CARD = (320, 260)
PADDING = 20
TITLE_H = 98
TEXT_H = 110


def _draw_smiles(smiles: str, size=MOL_SIZE_CARD) -> Image.Image:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        img = Image.new("RGB", size, (245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Invalid SMILES", fill="gray", font=_safe_font(14))
        return img
    return Draw.MolToImage(mol, size=size).convert("RGB")


def _draw_safe(safe_str: str, size=MOL_SIZE_CARD) -> Image.Image:
    if not safe_str or not str(safe_str).strip():
        img = Image.new("RGB", size, (245, 245, 245))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "SAFE not available", fill="gray", font=_safe_font(14))
        return img
    try:
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


def build_pair_card(
    *,
    rank: int,
    row_index: int,
    dataset_name: str,
    endpoint: str,
    toxic_safe: str,
    nontoxic_safe: str,
    toxic_smiles: str,
    nontoxic_smiles: str,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    w, h = MOL_SIZE_CARD
    canvas_w = PADDING * 3 + w * 2
    canvas_h = TITLE_H + TEXT_H + PADDING * 3 + h * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)

    font_title = _safe_font(18)
    font_meta = _safe_font(13)
    font_caption = _safe_font(14)
    font_string = _safe_font(12)

    score = float(metrics.get("optimal_score", 0.0))
    sim = float(metrics.get("tanimoto_chiral_morgan", 0.0))

    d.text((PADDING, 10), f"Rank {rank}  |  row_index={row_index}", fill="black", font=font_title)
    d.text((PADDING, 34), f"dataset={dataset_name}  |  endpoint={endpoint}", fill="black", font=font_meta)
    d.text((PADDING, 54), f"optimal_score={score:.6f}  |  tanimoto={sim:.6f}", fill="black", font=font_meta)

    d.text((PADDING, 74), f"Toxic SAFE: {_truncate(toxic_safe)}", fill=(20, 20, 20), font=font_string)
    d.text((PADDING, 90), f"Nontoxic SAFE: {_truncate(nontoxic_safe)}", fill=(20, 20, 20), font=font_string)
    d.text((PADDING, 106), f"Toxic SMILES: {_truncate(toxic_smiles)}", fill=(20, 20, 20), font=font_string)
    d.text((PADDING, 122), f"Nontoxic SMILES: {_truncate(nontoxic_smiles)}", fill=(20, 20, 20), font=font_string)

    x1, x2 = PADDING, PADDING * 2 + w
    y1 = TITLE_H + TEXT_H + PADDING
    y2 = y1 + h + PADDING

    img_tox_safe = _draw_safe(toxic_safe)
    img_tox_smiles = _draw_smiles(toxic_smiles)
    img_non_safe = _draw_safe(nontoxic_safe)
    img_non_smiles = _draw_smiles(nontoxic_smiles)

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


def _make_combined_image(card_paths: List[Path], out_path: Path, cols: int = 2) -> None:
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


# ---------------------------------------------------------------------------
# Visualization: Fragment highlight set (like optimal_pair_fragment_highlight.py)
# ---------------------------------------------------------------------------
MOL_SIZE = (520, 420)
RED = (1.0, 0.15, 0.15)
GREEN = (0.15, 0.72, 0.18)


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


def _configure_bond_line_width(d2d: "rdMolDraw2D.MolDraw2D", bond_line_width: float) -> None:
    opts = d2d.drawOptions()
    opts.bondLineWidth = float(bond_line_width)
    opts.scaleBondWidth = False
    if hasattr(opts, "scaleHighlightBondWidth"):
        opts.scaleHighlightBondWidth = False
    if hasattr(opts, "highlightBondWidthMultiplier"):
        opts.highlightBondWidthMultiplier = 1


def _draw_plain_mol(mol: Optional[Chem.Mol], size=MOL_SIZE, bond_line_width: float = 2.0) -> Image.Image:
    if mol is None:
        return _placeholder(size, "Molecule parse failed")
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    rdMolDraw2D.PrepareMolForDrawing(m)
    w, h = size
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    _configure_bond_line_width(d2d, bond_line_width)
    d2d.DrawMolecule(m)
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")


def _highlight_one_color(
    mol: Optional[Chem.Mol],
    frag_mol: Optional[Chem.Mol],
    color: Tuple[float, float, float],
    size=MOL_SIZE,
    bond_line_width: float = 2.0,
) -> Image.Image:
    if mol is None:
        return _placeholder(size, "Molecule parse failed")
    if frag_mol is None:
        return _draw_plain_mol(mol, size=size, bond_line_width=bond_line_width)
    mol_draw = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol_draw)
    atom_matches, bond_matches = dm.substructure_matching_bonds(mol_draw, frag_mol)
    atom_flat = list(dict.fromkeys([a for ms in atom_matches for a in ms]))
    bond_flat = list(dict.fromkeys([b for ms in bond_matches for b in ms]))
    ac = dict.fromkeys(atom_flat, color)
    bc = dict.fromkeys(bond_flat, color)
    rdMolDraw2D.PrepareMolForDrawing(mol_draw)
    w, h = size
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    _configure_bond_line_width(d2d, bond_line_width)
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


def _draw_mol_cairo_plain(mol: Optional[Chem.Mol], size=MOL_SIZE, bond_line_width: float = 2.0) -> Image.Image:
    if mol is None:
        return _placeholder(size, "Molecule missing / delete failed")
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    rdMolDraw2D.PrepareMolForDrawing(m)
    w, h = size
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    _configure_bond_line_width(d2d, bond_line_width)
    d2d.DrawMolecule(m)
    d2d.FinishDrawing()
    return Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")


def _substruct_match_parent_frag(parent: Chem.Mol, frag: Chem.Mol) -> Tuple[int, ...]:
    m = parent.GetSubstructMatch(frag, useChirality=True)
    if not m:
        m = parent.GetSubstructMatch(frag, useChirality=False)
    return tuple(int(x) for x in m)


def _draw_only_fragment_matching_full_molecule(
    parent: Optional[Chem.Mol],
    frag: Optional[Chem.Mol],
    size: Tuple[int, int] = MOL_SIZE,
    padding_px: float = 40.0,
    bond_line_width: float = 2.0,
) -> Image.Image:
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
            flat = list(dict.fromkeys([a for ms in atom_matches for a in ms]))
            match = tuple(int(x) for x in flat) if flat else ()
        except Exception:
            match = ()
    if not match:
        return _draw_plain_mol(fm, size=size, bond_line_width=bond_line_width)

    w, h = size
    rdDepictor.Compute2DCoords(pm)
    rdMolDraw2D.PrepareMolForDrawing(pm)
    d2d = rdMolDraw2D.MolDraw2DCairo(w, h)
    _configure_bond_line_width(d2d, bond_line_width)
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
        return _draw_plain_mol(fm, size=size, bond_line_width=bond_line_width)

    cropped = full_img.crop((int(x0), int(y0), int(x1), int(y1))).copy()
    cropped.thumbnail(size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, (255, 255, 255))
    ox = (size[0] - cropped.width) // 2
    oy = (size[1] - cropped.height) // 2
    out.paste(cropped, (ox, oy))
    return out


def generate_fragment_highlight_set(
    *,
    rank: int,
    row_index: int,
    toxic_safe: str,
    nontoxic_safe: str,
    only_toxic_safe_fragments: str,
    only_nontoxic_safe_fragments: str,
    out_dir: Path,
    bond_line_width: float = 2.0,
) -> None:
    mol_tox = _mol_from_full_safe(toxic_safe)
    mol_non = _mol_from_full_safe(nontoxic_safe)
    frag_tox = _decode_safe_fragment_to_mol(only_toxic_safe_fragments)
    frag_non = _decode_safe_fragment_to_mol(only_nontoxic_safe_fragments)
    rem_tox = _mol_without_fragment(mol_tox, frag_tox)
    rem_non = _mol_without_fragment(mol_non, frag_non)

    prefix = f"rank{rank:02d}_row{row_index}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        f"{prefix}_01_toxic_safe_plain.png": _draw_plain_mol(mol_tox, bond_line_width=bond_line_width),
        f"{prefix}_02_nontoxic_safe_plain.png": _draw_plain_mol(mol_non, bond_line_width=bond_line_width),
        f"{prefix}_03_toxic_safe_highlight_only_toxic_frag_red.png": _highlight_one_color(
            mol_tox, frag_tox, RED, bond_line_width=bond_line_width
        ),
        f"{prefix}_04_nontoxic_safe_highlight_only_nontoxic_frag_green.png": _highlight_one_color(
            mol_non, frag_non, GREEN, bond_line_width=bond_line_width
        ),
        f"{prefix}_05_toxic_safe_without_only_toxic_fragment.png": _draw_mol_cairo_plain(
            rem_tox, bond_line_width=bond_line_width
        ),
        f"{prefix}_06_nontoxic_safe_without_only_nontoxic_fragment.png": _draw_mol_cairo_plain(
            rem_non, bond_line_width=bond_line_width
        ),
        # 정석(1번): fragment 자체만 단독 2D로 렌더링 (crop/각도 맞춤(3번) 사용하지 않음)
        f"{prefix}_07_only_toxic_fragment_safe_plain.png": _draw_plain_mol(
            frag_tox, bond_line_width=bond_line_width
        ),
        f"{prefix}_08_only_nontoxic_fragment_safe_plain.png": _draw_plain_mol(
            frag_non, bond_line_width=bond_line_width
        ),
    }
    for name, img in paths.items():
        img.save(out_dir / name)


# ---------------------------------------------------------------------------
# IO: selected best_cases
# ---------------------------------------------------------------------------
def _iter_best_case_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/best_cases_*.jsonl"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_merged_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return [row for row in r]


@dataclass
class GroupKey:
    task: str
    step: str
    model: str


def _group_key_from_path(path: Path) -> GroupKey:
    # selected/test/<task>/both_repre/<step>/best_cases_<model>.jsonl
    parts = path.parts
    idx = parts.index("test")
    task = parts[idx + 1]
    step = parts[idx + 3]
    model = path.stem.replace("best_cases_", "", 1)
    return GroupKey(task=task, step=step, model=model)


def _write_augmented_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_augmented_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    # 간단 요약 CSV
    fields = [
        "rank",
        "source_index",
        "dataset_name",
        "endpoint",
        "tanimoto_chiral_morgan",
        "mcs_atom_ratio",
        "atom_diff_abs",
        "bond_diff_abs",
        "ring_diff_abs",
        "optimal_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            m = r.get("metrics", {}) or {}
            merged = r.get("merged", {}) or {}
            w.writerow(
                {
                    "rank": r.get("rank"),
                    "source_index": r.get("source_index"),
                    "dataset_name": merged.get("dataset_name", ""),
                    "endpoint": merged.get("endpoint", ""),
                    "tanimoto_chiral_morgan": m.get("tanimoto_chiral_morgan"),
                    "mcs_atom_ratio": m.get("mcs_atom_ratio"),
                    "atom_diff_abs": m.get("atom_diff_abs"),
                    "bond_diff_abs": m.get("bond_diff_abs"),
                    "ring_diff_abs": m.get("ring_diff_abs"),
                    "optimal_score": m.get("optimal_score"),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment selected best_case samples with merged_test info + visualizations")
    ap.add_argument("--selected-root", type=Path, default=SELECTED_TEST_ROOT)
    ap.add_argument("--merged-csv", type=Path, default=MERGED_TEST_CSV)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--grid-cols", type=int, default=2)
    ap.add_argument("--bond-line-width", type=float, default=2.0)
    args = ap.parse_args()

    if not args.selected_root.is_dir():
        raise SystemExit(f"selected root not found: {args.selected_root}")
    if not args.merged_csv.is_file():
        raise SystemExit(f"merged_test.csv not found: {args.merged_csv}")

    merged_rows = _load_merged_rows(args.merged_csv)
    best_files = _iter_best_case_files(args.selected_root)
    if not best_files:
        raise SystemExit(f"best_cases_*.jsonl not found under {args.selected_root}")

    for bf in best_files:
        key = _group_key_from_path(bf)
        rows = _load_jsonl(bf)
        if not rows:
            continue

        # augment + compute metrics
        augmented: List[Dict[str, Any]] = []
        card_paths: List[Path] = []

        out_group_dir = args.out_root / "test" / key.task / "both_repre" / key.step / key.model
        img_dir = out_group_dir / "images"
        cards_dir = img_dir / "pair_cards"
        frag_root = img_dir / "pair_fragment_highlight_set"

        for idx, r in enumerate(rows, 1):
            si = r.get("source_index")
            if si is None:
                continue
            si = int(si)
            if si < 0 or si >= len(merged_rows):
                continue
            mrow = merged_rows[si]

            tox_safe = str(mrow.get("toxic_safe", "") or "")
            non_safe = str(mrow.get("nontoxic_safe", "") or "")
            tox_smiles = str(mrow.get("toxic_safe_decoded_smiles", "") or mrow.get("toxic_smiles", "") or "")
            non_smiles = str(mrow.get("nontoxic_safe_decoded_smiles", "") or mrow.get("nontoxic_smiles", "") or "")
            only_tox = str(mrow.get("only_toxic_safe_fragments", "") or "")
            only_non = str(mrow.get("only_nontoxic_safe_fragments", "") or "")

            metrics = _pair_metrics(tox_smiles, non_smiles) or {}
            tox_frag_count = _count_safe_fragments(only_tox)
            non_frag_count = _count_safe_fragments(only_non)
            tox_frag_size = _fragment_size_from_safe_fragment(only_tox) if tox_frag_count == 1 else None
            non_frag_size = _fragment_size_from_safe_fragment(only_non) if non_frag_count == 1 else None

            aug = {
                **r,
                "rank": idx,
                "source_index": si,
                "pred_answer": _normalize_answer(r.get("pred")),
                "gold_answer": _normalize_answer(r.get("gold")),
                "merged": {
                    **(r.get("merged") or {}),
                    "dataset_name": mrow.get("dataset_name", ""),
                    "endpoint": mrow.get("endpoint", ""),
                    "toxic_safe": tox_safe,
                    "nontoxic_safe": non_safe,
                    "toxic_smiles": mrow.get("toxic_smiles", ""),
                    "nontoxic_smiles": mrow.get("nontoxic_smiles", ""),
                    "toxic_safe_decoded_smiles": tox_smiles,
                    "nontoxic_safe_decoded_smiles": non_smiles,
                    "only_toxic_safe_fragments": only_tox,
                    "only_nontoxic_safe_fragments": only_non,
                    "only_toxic_fragment_count": float(tox_frag_count),
                    "only_nontoxic_fragment_count": float(non_frag_count),
                    "only_toxic_fragment_size": float(tox_frag_size) if tox_frag_size is not None else None,
                    "only_nontoxic_fragment_size": float(non_frag_size) if non_frag_size is not None else None,
                },
                "metrics": metrics,
            }
            augmented.append(aug)

            # pair card
            card_path = cards_dir / f"pair_{idx:02d}_row{si}.png"
            build_pair_card(
                rank=idx,
                row_index=si,
                dataset_name=str(mrow.get("dataset_name", "") or ""),
                endpoint=str(mrow.get("endpoint", "") or ""),
                toxic_safe=tox_safe,
                nontoxic_safe=non_safe,
                toxic_smiles=tox_smiles,
                nontoxic_smiles=non_smiles,
                metrics=metrics,
                out_path=card_path,
            )
            card_paths.append(card_path)

            # fragment highlight set (8 images)
            out_row_dir = frag_root / f"row_{si}"
            generate_fragment_highlight_set(
                rank=idx,
                row_index=si,
                toxic_safe=tox_safe,
                nontoxic_safe=non_safe,
                only_toxic_safe_fragments=only_tox,
                only_nontoxic_safe_fragments=only_non,
                out_dir=out_row_dir,
                bond_line_width=args.bond_line_width,
            )

        # write augmented data
        out_jsonl = out_group_dir / "augmented_best_cases.jsonl"
        out_csv = out_group_dir / "augmented_best_cases.csv"
        _write_augmented_jsonl(out_jsonl, augmented)
        _write_augmented_csv(out_csv, augmented)

        # grid image
        grid_path = img_dir / "pair_cards_grid.png"
        _make_combined_image(card_paths, grid_path, cols=args.grid_cols)

        print(f"[OK] {key.task}/{key.step}/{key.model}: {len(augmented)} samples -> {out_group_dir}")


if __name__ == "__main__":
    main()

