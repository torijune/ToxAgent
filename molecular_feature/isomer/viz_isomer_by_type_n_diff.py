"""
isomer type(primary_isomer_type)별, n_diff별로 pair를 시각화하고,
이성질체를 유발하는 stereochemistry 차이(chiral center R/S, E/Z bond) 부위를 하이라이트한다.

입력: isomer_pairs_stereo_only_reclassified.csv (또는 _with_ftsim.csv)
출력: visualizations/by_isomer_type/{Enantiomer|Diastereomer|E/Z Isomer}/n_diff_{n}/...
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from PIL import Image, ImageDraw, ImageFont

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
ISOMER_DIR = SCRIPT_DIR
DEFAULT_CSV = ISOMER_DIR / "all_data_isomer" / "isomer_pairs_stereo_only_reclassified.csv"
DEFAULT_OUT_DIR = ISOMER_DIR / "visualizations" / "by_isomer_type"

MOL_SIZE = (420, 420)
TEXT_HEIGHT = 70
PADDING = 24

# 하이라이트 색: chiral 차이 = 노랑, E/Z 차이 = 빨강
CHIRAL_HIGHLIGHT_COLOR = (1.0, 0.85, 0.0)
EZ_HIGHLIGHT_COLOR = (1.0, 0.35, 0.35)


def _safe_eval_list(val: Any) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            out = ast.literal_eval(val)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    if pd.isna(val) or val == "" or val == "[]":
        return []
    return []


def get_stereo_diff_highlights_from_mol(
    toxic_smiles: str, nontoxic_smiles: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    RDKit으로 두 분자의 stereochemistry 차이 부위를 계산.
    same_skeleton 가정: atom index가 서로 대응한다고 봄.

    Returns:
        (diff_atom_indices, diff_bond_atom_pairs)
        - diff_atom_indices: config(R/S)가 다른 chiral center의 atom index (0-based, 둘 다 동일 적용)
        - diff_bond_atom_pairs: geometry(E/Z)가 다른 bond의 (a,b) 쌍 (sorted tuple, 0-based)
    """
    diff_atoms: list[int] = []
    diff_bonds: list[tuple[int, int]] = []

    mol_t = Chem.MolFromSmiles((toxic_smiles or "").strip()) if toxic_smiles else None
    mol_n = Chem.MolFromSmiles((nontoxic_smiles or "").strip()) if nontoxic_smiles else None
    if mol_t is None or mol_n is None:
        return diff_atoms, diff_bonds

    Chem.AssignStereochemistry(mol_t, force=True, cleanIt=True)
    Chem.AssignStereochemistry(mol_n, force=True, cleanIt=True)

    chiral_t: dict[int, str] = {}
    chiral_n: dict[int, str] = {}
    for mol, d in [(mol_t, chiral_t), (mol_n, chiral_n)]:
        for atom in mol.GetAtoms():
            tag = atom.GetChiralTag()
            if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                d[atom.GetIdx()] = "R"
            elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                d[atom.GetIdx()] = "S"

    for idx in set(chiral_t.keys()) | set(chiral_n.keys()):
        if chiral_t.get(idx) != chiral_n.get(idx):
            diff_atoms.append(idx)

    ez_t: dict[tuple[int, int], str] = {}
    ez_n: dict[tuple[int, int], str] = {}
    for mol, d in [(mol_t, ez_t), (mol_n, ez_n)]:
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            st = bond.GetStereo()
            if st == Chem.BondStereo.STEREOCIS:
                d[tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))] = "Z"
            elif st == Chem.BondStereo.STEREOTRANS:
                d[tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))] = "E"

    for k in set(ez_t.keys()) | set(ez_n.keys()):
        if ez_t.get(k) != ez_n.get(k):
            diff_bonds.append(k)

    return diff_atoms, diff_bonds


def _bond_pair_to_bond_indices(mol, bond_atom_pairs: list[tuple[int, int]]) -> list[int]:
    """(a,b) 쌍 리스트를 RDKit bond index 리스트로 변환."""
    indices = []
    for a, b in bond_atom_pairs:
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is not None:
            indices.append(bond.GetIdx())
    return indices


def _mol_image_with_highlights(
    smiles: str,
    highlight_atoms: list[int],
    highlight_bonds: list[tuple[int, int]],
    size: tuple[int, int] = MOL_SIZE,
) -> Image.Image | None:
    """SMILES를 그리되, chiral 차이 atom은 노랑, E/Z 차이 bond는 빨강으로 하이라이트."""
    if not RDKIT_AVAILABLE or not smiles or not str(smiles).strip():
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        bond_indices = _bond_pair_to_bond_indices(mol, highlight_bonds)
        if not highlight_atoms and not bond_indices:
            return Draw.MolToImage(mol, size=size)
        atom_colors = {i: CHIRAL_HIGHLIGHT_COLOR for i in highlight_atoms}
        bond_colors = {i: EZ_HIGHLIGHT_COLOR for i in bond_indices}
        return Draw.MolToImage(
            mol,
            size=size,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=atom_colors,
            highlightBonds=bond_indices,
            highlightBondColors=bond_colors,
        )
    except Exception:
        return None


def _default_font(size: int = 14):
    for name in ["Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"]:
        for base in ["/System/Library/Fonts", "/usr/share/fonts/truetype"]:
            p = Path(base) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    pass
    return ImageFont.load_default()


def _sanitize_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s)).strip("_") or "unknown"


def draw_pair_with_highlights(
    toxic_smiles: str,
    nontoxic_smiles: str,
    isomer_type: str,
    n_diff: int | float,
    pair_label: str,
    out_path: Path,
) -> None:
    """한 pair를 Toxic | Nontoxic 나란히 그리며, 스테레오 차이 부위 하이라이트."""
    diff_atoms, diff_bonds = get_stereo_diff_highlights_from_mol(toxic_smiles, nontoxic_smiles)

    img_tox = _mol_image_with_highlights(toxic_smiles, diff_atoms, diff_bonds)
    img_non = _mol_image_with_highlights(nontoxic_smiles, diff_atoms, diff_bonds)

    if img_tox is None:
        img_tox = Image.new("RGB", MOL_SIZE, (240, 240, 240))
        ImageDraw.Draw(img_tox).text(
            (MOL_SIZE[0] // 2 - 50, MOL_SIZE[1] // 2 - 10), "Invalid SMILES", fill="gray"
        )
    if img_non is None:
        img_non = Image.new("RGB", MOL_SIZE, (240, 240, 240))
        ImageDraw.Draw(img_non).text(
            (MOL_SIZE[0] // 2 - 50, MOL_SIZE[1] // 2 - 10), "Invalid SMILES", fill="gray"
        )

    total_width = MOL_SIZE[0] * 2 + PADDING
    total_height = MOL_SIZE[1] + TEXT_HEIGHT * 2 + PADDING
    combined = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(combined)
    font_title = _default_font(18)
    font_sub = _default_font(12)

    combined.paste(img_tox, (0, TEXT_HEIGHT))
    combined.paste(img_non, (MOL_SIZE[0] + PADDING, TEXT_HEIGHT))
    draw.text((MOL_SIZE[0] // 2 - 30, 8), "Toxic", fill="black", font=font_title)
    draw.text(
        (MOL_SIZE[0] + PADDING + MOL_SIZE[0] // 2 - 45, 8),
        "Nontoxic",
        fill="black",
        font=font_title,
    )
    info = f"{isomer_type}  |  n_diff = {int(n_diff)}  |  Yellow: chiral diff, Red: E/Z diff"
    draw.text((10, 8), pair_label, fill="gray", font=font_sub)
    draw.text((10, total_height - TEXT_HEIGHT + 8), info, fill="dimgray", font=font_sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"  Saved: {out_path}")


def run(
    csv_path: Path | None = None,
    out_dir: Path | None = None,
    max_per_group: int = 3,
) -> None:
    """
    CSV를 읽어 primary_isomer_type × n_diff별로 그룹하고,
    각 그룹에서 최대 max_per_group개 pair를 스테레오 차이 하이라이트와 함께 시각화한다.
    """
    csv_path = csv_path or DEFAULT_CSV
    out_dir = out_dir or DEFAULT_OUT_DIR

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit and PIL are required.")

    df = pd.read_csv(csv_path)
    for col in ["primary_isomer_type", "n_diff", "toxic_smiles", "nontoxic_smiles"]:
        if col not in df.columns:
            raise ValueError(f"CSV must have column: {col}")

    df = df.dropna(subset=["toxic_smiles", "nontoxic_smiles"]).copy()
    df["n_diff"] = df["n_diff"].fillna(0).astype(int)

    grouped = df.groupby(["primary_isomer_type", "n_diff"], dropna=False)
    total_saved = 0

    for (isomer_type, n_diff), grp in grouped:
        if pd.isna(isomer_type):
            isomer_type = "Unknown"
        isomer_safe = _sanitize_filename(str(isomer_type))
        n_diff_safe = _sanitize_filename(str(n_diff))
        group_dir = out_dir / isomer_safe / f"n_diff_{n_diff_safe}"
        samples = grp.head(max_per_group)

        for i, (_, row) in enumerate(samples.iterrows()):
            tox = str(row.get("toxic_smiles", "")).strip()
            non = str(row.get("nontoxic_smiles", "")).strip()
            if not tox or not non:
                continue
            out_path = group_dir / f"pair_{i + 1}.png"
            draw_pair_with_highlights(
                toxic_smiles=tox,
                nontoxic_smiles=non,
                isomer_type=str(isomer_type),
                n_diff=n_diff,
                pair_label=f"{isomer_safe}  n_diff={n_diff}  sample {i + 1}",
                out_path=out_path,
            )
            total_saved += 1

    print(f"Done. Saved {total_saved} images under {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Visualize isomer pairs by primary_isomer_type and n_diff with stereo-difference highlights."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input CSV (default: {DEFAULT_CSV})",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--max_per_group",
        type=int,
        default=3,
        help="Max number of pairs to draw per (isomer_type, n_diff) (default: 3)",
    )
    args = ap.parse_args()
    run(csv_path=args.input, out_dir=args.out_dir, max_per_group=args.max_per_group)


if __name__ == "__main__":
    main()
