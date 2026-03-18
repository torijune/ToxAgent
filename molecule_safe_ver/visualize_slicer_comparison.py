"""
smiles_to_safe_by_slicer.csv 를 사용해, 같은 분자에 대해 각 slicer가 어떻게 fragment를 만드는지 시각화.

- 기본: SAFE decode → SMARTS → 서브구조 매칭 (일부 원자 탈락 가능)
- --exact: SAFE와 동일한 slicer로 "자를 결합"만 구한 뒤, 그 결합 제거 시 연결 성분으로 배정 → 탈락 없음
"""
import argparse
import io
import re
from pathlib import Path
from collections import deque

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import datamol as dm
from tqdm import tqdm

import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from safe.safe.converter import decode as safe_decode
from safe.safe.converter import SAFEConverter
from safe.safe._exception import SAFEFragmentationError

INPUT_CSV = SCRIPT_DIR / "smiles_to_safe_by_slicer.csv"
OUTPUT_DIR = SCRIPT_DIR / "visualizations" / "slicer_comparison"

# slicer 컬럼 이름 (csv 컬럼과 동일)
SLICER_COLS = ["hr_safe", "rotatable_safe", "recap_safe", "mmpa_safe", "attach_safe", "brics_safe"]
SLICER_LABELS = ["hr", "rotatable", "recap", "mmpa", "attach", "brics"]

SEP_FRAG = "."
PANEL_SIZE = (280, 280)

# slicer 컬럼명 → converter에 넘길 slicer 이름
COL_TO_SLICER = {
    "hr_safe": "hr",
    "rotatable_safe": "rotatable",
    "recap_safe": "recap",
    "mmpa_safe": "mmpa",
    "attach_safe": "attach",
    "brics_safe": "brics",
}

COLORS_RGB = [
    (1.0, 0.25, 0.25), (0.2, 0.5, 1.0), (0.2, 0.8, 0.2), (1.0, 0.65, 0.0),
    (0.65, 0.0, 0.9), (0.0, 0.9, 0.9), (1.0, 0.4, 0.6), (0.85, 0.85, 0.0),
    (0.9, 0.45, 0.0), (0.3, 0.65, 0.5), (0.75, 0.0, 0.5), (0.0, 0.55, 0.55),
    (0.95, 0.5, 0.95), (0.5, 0.35, 0.85), (0.4, 0.85, 0.4), (0.95, 0.75, 0.35),
    (0.55, 0.55, 0.55), (0.95, 0.65, 0.45), (0.25, 0.65, 0.85), (0.85, 0.55, 0.25),
]


def _connected_components_after_cut(mol, bonds_to_cut: list[tuple[int, int]]) -> list[set[int]]:
    """자를 결합(bonds_to_cut = (a,b) 리스트)을 제거한 뒤 원자별 연결 성분 반환."""
    n = mol.GetNumAtoms()
    cut_set = {tuple(sorted((a, b))) for a, b in bonds_to_cut}
    # 인접 리스트 (자를 결합 제외)
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if tuple(sorted((a, b))) in cut_set:
            continue
        adj[a].append(b)
        adj[b].append(a)
    visited = set()
    components = []
    for start in range(n):
        if start in visited:
            continue
        comp = set()
        q = deque([start])
        while q:
            v = q.popleft()
            if v in visited:
                continue
            visited.add(v)
            comp.add(v)
            for w in adj[v]:
                if w not in visited:
                    q.append(w)
        components.append(comp)
    return components


def slicer_fragments_to_highlight_atoms_exact(
    mol,
    slicer_name: str,
) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]]]:
    """
    SAFE와 동일한 slicer로 자를 결합만 구한 뒤, 연결 성분으로 원자 배정.
    매칭 탈락 없음 (모든 원자가 정확히 한 fragment에 속함).
    """
    if mol is None or not slicer_name:
        return {}, {}
    # attach slicer는 SAFE 인코더에서 require_hs로 explicit H를 쓰지만, 여기서는 인덱스 일치를 위해 H 추가 없이 사용
    try:
        with dm.without_rdkit_log():
            conv = SAFEConverter(slicer=slicer_name, ignore_stereo=True)
            bonds_to_cut = conv._fragment(mol, allow_empty=True)
    except (SAFEFragmentationError, ValueError, Exception):
        return {}, {}
    if not bonds_to_cut:
        # 자를 결합 없음 → 분자 전체가 한 fragment
        n = mol.GetNumAtoms()
        if n == 0:
            return {}, {}
        color = COLORS_RGB[0]
        highlight_atom = {i: color for i in range(n)}
        highlight_bond = {b.GetIdx(): color for b in mol.GetBonds()}
        return highlight_atom, highlight_bond

    components = _connected_components_after_cut(mol, bonds_to_cut)
    highlight_atom: dict[int, tuple[float, float, float]] = {}
    for i, comp in enumerate(components):
        color = COLORS_RGB[i % len(COLORS_RGB)]
        for aid in comp:
            highlight_atom[aid] = color

    highlight_bond: dict[int, tuple[float, float, float]] = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in highlight_atom and a2 in highlight_atom and highlight_atom[a1] == highlight_atom[a2]:
            highlight_bond[bond.GetIdx()] = highlight_atom[a1]
    return highlight_atom, highlight_bond


def fragment_smiles_to_smarts(smiles_with_dummies: str) -> str:
    if not smiles_with_dummies or pd.isna(smiles_with_dummies):
        return ""
    return re.sub(r"\[\*[^\]]*\]", "*", str(smiles_with_dummies).strip())


def safe_fragments_to_highlight_atoms(
    mol, safe_str: str
) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]]]:
    if mol is None or not safe_str or pd.isna(safe_str) or not str(safe_str).strip():
        return {}, {}
    parts = [p.strip() for p in str(safe_str).split(SEP_FRAG) if p.strip()]
    if not parts:
        return {}, {}

    colors_rgb = [COLORS_RGB[i % len(COLORS_RGB)] for i in range(len(parts))]

    frag_matches: list[tuple[int, tuple[int, ...]]] = []
    for i, frag_safe in enumerate(parts):
        try:
            frag_smiles = safe_decode(frag_safe, remove_dummies=False, ignore_errors=True)
            if not frag_smiles:
                continue
            smarts = fragment_smiles_to_smarts(frag_smiles)
            if not smarts:
                continue
            q = dm.from_smarts(smarts)
            if q is None:
                continue
            matches = list(mol.GetSubstructMatches(q, uniquify=True))
            for m in matches:
                frag_matches.append((i, m))
        except Exception:
            continue

    frag_matches.sort(key=lambda x: len(x[1]))
    assigned = set()
    highlight_atom: dict[int, tuple[float, float, float]] = {}
    for frag_idx, match in frag_matches:
        color = colors_rgb[frag_idx]
        for aid in match:
            if aid not in assigned:
                highlight_atom[aid] = color
                assigned.add(aid)

    highlight_bond: dict[int, tuple[float, float, float]] = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in highlight_atom and a2 in highlight_atom and highlight_atom[a1] == highlight_atom[a2]:
            highlight_bond[bond.GetIdx()] = highlight_atom[a1]
    return highlight_atom, highlight_bond


def draw_mol_with_fragment_highlights(
    canonical_smiles: str,
    safe_str: str,
    size: tuple[int, int] = PANEL_SIZE,
) -> Image.Image | None:
    """동일 분자(canonical_smiles)에 safe_str 기준으로 fragment 하이라이트."""
    with dm.without_rdkit_log():
        mol = dm.to_mol(canonical_smiles)
    if mol is None:
        return None
    atom_colors, bond_colors = safe_fragments_to_highlight_atoms(mol, safe_str)
    if not atom_colors:
        return Draw.MolToImage(mol, size=size)
    w, h = size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightBonds=list(bond_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def draw_mol_with_fragment_highlights_exact(
    canonical_smiles: str,
    slicer_name: str,
    size: tuple[int, int] = PANEL_SIZE,
) -> Image.Image | None:
    """동일 slicer로 자를 결합만 구한 뒤 연결 성분으로 색칠 (탈락 없음)."""
    with dm.without_rdkit_log():
        mol = dm.to_mol(canonical_smiles)
    if mol is None:
        return None
    atom_colors, bond_colors = slicer_fragments_to_highlight_atoms_exact(mol, slicer_name)
    if not atom_colors:
        return Draw.MolToImage(mol, size=size)
    w, h = size
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightBonds=list(bond_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def draw_mol_rdkit(smiles: str, size: tuple[int, int] = PANEL_SIZE) -> Image.Image | None:
    with dm.without_rdkit_log():
        mol = dm.to_mol(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize how each slicer fragments the same molecule (from smiles_to_safe_by_slicer.csv)."
    )
    parser.add_argument("--limit", type=int, default=10, help="Max rows to visualize (default: 10)")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated row indices, e.g. 0,1,5")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use slicer bond-cut + connected components (no SMARTS matching); every atom assigned to a fragment.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"입력 파일이 없습니다: {INPUT_CSV}\n먼저 safe_converter_tuning.py 로 smiles_to_safe_by_slicer.csv 를 생성하세요."
        )

    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    required = ["canonical_smiles"] + SLICER_COLS
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if args.indices:
        idx_list = [int(x.strip()) for x in args.indices.split(",") if x.strip().isdigit()]
        df = df.iloc[idx_list].reset_index(drop=True)
    else:
        df = df.head(args.limit)

    n = len(df)
    mode = "exact (slicer bond-cut + connected components)" if args.exact else "SAFE→SMARTS matching"
    print(f"Visualizing {n} row(s) -> {out_dir} [mode: {mode}]")

    pad = 16
    label_h = 22
    w, h = PANEL_SIZE
    # 2행: 1행 = Original + hr + rotatable + recap (4열), 2행 = mmpa + attach + brics (3열)
    n_col1, n_col2 = 4, 3
    total_w = max(n_col1 * (w + pad) + pad, n_col2 * (w + pad) + pad)
    total_h = (h + label_h + pad) * 2 + pad

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        try:
            font = ImageFont.truetype("Arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

    for idx, row in tqdm(df.iterrows(), total=n, desc="Drawing"):
        canon = row["canonical_smiles"]
        if pd.isna(canon) or not str(canon).strip():
            continue

        img_orig = draw_mol_rdkit(canon)
        images = [img_orig]
        for col in SLICER_COLS:
            safe_str = row[col]
            if pd.isna(safe_str) or not str(safe_str).strip():
                images.append(draw_mol_rdkit(canon))  # 빈 SAFE면 하이라이트 없이
            else:
                images.append(draw_mol_with_fragment_highlights(canon, str(safe_str)))

        combined = Image.new("RGB", (total_w, total_h), "white")
        draw = ImageDraw.Draw(combined)

        # Row 1: Original, hr, rotatable, recap
        labels_row1 = ["Original", "hr", "rotatable", "recap"]
        for col_i, label in enumerate(labels_row1):
            x = pad + col_i * (w + pad)
            y0 = pad
            draw.text((x, y0), label, fill="black", font=font)
            if col_i < len(images) and images[col_i]:
                combined.paste(images[col_i], (x, y0 + label_h))

        # Row 2: mmpa, attach, brics
        labels_row2 = ["mmpa", "attach", "brics"]
        y1 = pad + label_h + h + pad
        for col_i, label in enumerate(labels_row2):
            x = pad + col_i * (w + pad)
            draw.text((x, y1), label, fill="black", font=font)
            img_idx = 4 + col_i
            if img_idx < len(images) and images[img_idx]:
                combined.paste(images[img_idx], (x, y1 + label_h))

        out_path = out_dir / f"row_{idx}.png"
        combined.save(out_path)

    print(f"Saved {n} image(s) under {out_dir}")


if __name__ == "__main__":
    main()
