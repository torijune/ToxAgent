"""
SAFE로 복원한 분자(SMILES)를 frag마다 하이라이트해서 시각화한 것과,
원본 SMILES를 RDKit으로 시각화한 것을 나란히 비교.

- 왼쪽: SAFE decode → 전체 분자 + SAFE fragment별 색상 하이라이트 (frag1, frag2, ... 각각 다른 색)
- 오른쪽: 원본 SMILES RDKit 시각화
"""
import argparse
import io
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image, ImageDraw, ImageFont
import datamol as dm
from tqdm import tqdm

# safe 패키지 (프로젝트 루트의 safe 폴더)
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from safe.safe.converter import decode as safe_decode

INPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles_matched.csv"
OUTPUT_DIR = SCRIPT_DIR / "visualizations" / "safe_vs_smiles"

SEP_FRAG = "."
IMG_SIZE = (450, 450)


def fragment_smiles_to_smarts(smiles_with_dummies: str) -> str:
    """[*], [*:2], [*:7] 등 부착점 더미를 *(RDKit SMARTS 임의 원자)로 치환"""
    if not smiles_with_dummies or pd.isna(smiles_with_dummies):
        return ""
    return re.sub(r"\[\*[^\]]*\]", "*", str(smiles_with_dummies).strip())


def safe_fragments_to_highlight_atoms(
    mol, safe_str: str
) -> tuple[dict[int, tuple[float, float, float]], dict[int, tuple[float, float, float]]]:
    """
    safe_str을 '.' 기준으로 split한 각 fragment를 decode → SMARTS → mol에서 매칭,
    fragment별로 서로 다른 색을 배정. 한 원자는 한 fragment에만 배정(겹치지 않게).
    반환: (atom_idx -> (R,G,B), bond_idx -> (R,G,B))
    """
    if mol is None or not safe_str or pd.isna(safe_str):
        return {}, {}
    parts = [p.strip() for p in str(safe_str).split(SEP_FRAG) if p.strip()]
    if not parts:
        return {}, {}

    # fragment별로 뚜렷한 색 (frag1, frag2, ... 구분용)
    colors_rgb = [
        (1.0, 0.25, 0.25), (0.2, 0.5, 1.0), (0.2, 0.8, 0.2), (1.0, 0.65, 0.0),
        (0.65, 0.0, 0.9), (0.0, 0.9, 0.9), (1.0, 0.4, 0.6), (0.85, 0.85, 0.0),
        (0.9, 0.45, 0.0), (0.3, 0.65, 0.5), (0.75, 0.0, 0.5), (0.0, 0.55, 0.55),
        (0.95, 0.5, 0.95), (0.5, 0.35, 0.85), (0.4, 0.85, 0.4), (0.95, 0.75, 0.35),
        (0.55, 0.55, 0.55), (0.95, 0.65, 0.45), (0.25, 0.65, 0.85), (0.85, 0.55, 0.25),
    ]
    colors_rgb = [colors_rgb[i % len(colors_rgb)] for i in range(len(parts))]

    # 각 fragment의 모든 매칭 수집 (frag_idx, match_atom_tuple)
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

    # 매칭 크기 순(작은 것 먼저) 정렬 → 작은 fragment가 먼저 원자 할당받음
    frag_matches.sort(key=lambda x: len(x[1]))

    assigned = set()
    highlight_atom: dict[int, tuple[float, float, float]] = {}
    for frag_idx, match in frag_matches:
        color = colors_rgb[frag_idx]
        for aid in match:
            if aid not in assigned:
                highlight_atom[aid] = color
                assigned.add(aid)

    # 같은 fragment 색의 원자들 사이 bond만 하이라이트
    highlight_bond: dict[int, tuple[float, float, float]] = {}
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in highlight_atom and a2 in highlight_atom and highlight_atom[a1] == highlight_atom[a2]:
            highlight_bond[bond.GetIdx()] = highlight_atom[a1]

    return highlight_atom, highlight_bond


def draw_mol_with_fragment_highlights(
    decoded_smiles: str,
    safe_str: str,
    size: tuple[int, int] = IMG_SIZE,
) -> Image.Image | None:
    """SAFE로 복원한 SMILES로 mol 그린 뒤, fragment마다 다른 색으로 하이라이트 (MolDraw2DCairo 사용)."""
    with dm.without_rdkit_log():
        mol = dm.to_mol(decoded_smiles)
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


def draw_mol_rdkit(smiles: str, size: tuple[int, int] = IMG_SIZE) -> Image.Image | None:
    """원본 SMILES를 RDKit으로 시각화 (하이라이트 없음)."""
    with dm.without_rdkit_log():
        mol = dm.to_mol(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def main():
    parser = argparse.ArgumentParser(
        description="Compare SAFE (frag-highlight) vs original SMILES visualization."
    )
    parser.add_argument("--limit", type=int, default=5, help="Max rows to process (default: 5)")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated row indices, e.g. 0,1,5")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    required = [
        "toxic_safe", "nontoxic_safe",
        "toxic_safe_decoded_smiles", "nontoxic_safe_decoded_smiles",
        "toxic_smiles", "nontoxic_smiles",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if args.indices:
        idx_list = [int(x.strip()) for x in args.indices.split(",") if x.strip().isdigit()]
        df = df.iloc[idx_list].reset_index(drop=True)
    else:
        df = df.head(args.limit)

    n = len(df)
    print(f"Visualizing {n} row(s) -> {out_dir}")

    pad = 30
    label_h = 44
    w, h = IMG_SIZE
    # 레이아웃: [ Toxic: SAFE view | Original ]  [ Nontoxic: SAFE view | Original ]
    total_w = w * 2 + pad * 3
    total_h = (h + label_h + pad) * 2 + pad

    for idx, row in tqdm(df.iterrows(), total=n, desc="Drawing"):
        toxic_safe = row["toxic_safe"]
        nontoxic_safe = row["nontoxic_safe"]
        toxic_decoded = row["toxic_safe_decoded_smiles"]
        nontoxic_decoded = row["nontoxic_safe_decoded_smiles"]
        toxic_smiles = row["toxic_smiles"]
        nontoxic_smiles = row["nontoxic_smiles"]

        img_toxic_safe = draw_mol_with_fragment_highlights(toxic_decoded, toxic_safe)
        img_toxic_orig = draw_mol_rdkit(toxic_smiles)
        img_nontoxic_safe = draw_mol_with_fragment_highlights(nontoxic_decoded, nontoxic_safe)
        img_nontoxic_orig = draw_mol_rdkit(nontoxic_smiles)

        combined = Image.new("RGB", (total_w, total_h), "white")
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except Exception:
            try:
                font = ImageFont.truetype("Arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()

        y0 = pad
        # Row 1: Toxic
        draw.text((pad, y0), "Toxic: SAFE (frag highlight)", fill="black", font=font)
        if img_toxic_safe:
            combined.paste(img_toxic_safe, (pad, y0 + label_h))
        draw.text((pad * 2 + w, y0), "Toxic: Original SMILES", fill="black", font=font)
        if img_toxic_orig:
            combined.paste(img_toxic_orig, (pad * 2 + w, y0 + label_h))

        y1 = y0 + label_h + h + pad
        # Row 2: Nontoxic
        draw.text((pad, y1), "Nontoxic: SAFE (frag highlight)", fill="black", font=font)
        if img_nontoxic_safe:
            combined.paste(img_nontoxic_safe, (pad, y1 + label_h))
        draw.text((pad * 2 + w, y1), "Nontoxic: Original SMILES", fill="black", font=font)
        if img_nontoxic_orig:
            combined.paste(img_nontoxic_orig, (pad * 2 + w, y1 + label_h))

        out_path = out_dir / f"row_{idx}.png"
        combined.save(out_path)

    print(f"Saved {n} image(s) under {out_dir}")


if __name__ == "__main__":
    main()
