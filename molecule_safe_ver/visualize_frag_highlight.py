"""
only_toxic_frag_smiles / only_nontoxic_frag_smiles에 해당하는 원자들을
toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles 이미지 위에 하이라이트하여 시각화.
atom index는 only_toxic_frag_atom_indices, only_nontoxic_frag_atom_indices 사용.
"""
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont
import datamol as dm
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles_matched.csv"
OUTPUT_DIR = SCRIPT_DIR / "visualizations" / "frag_highlight"

SEP_MATCH = "|"


def parse_atom_indices(indices_str: str) -> list[int]:
    """'0,1,2,3|4,5,6' -> [0,1,2,3,4,5,6] (모든 구간을 하나의 리스트로)"""
    if not indices_str or pd.isna(indices_str):
        return []
    out = []
    for part in str(indices_str).strip().split(SEP_MATCH):
        part = part.strip()
        if not part:
            continue
        for x in part.split(","):
            x = x.strip()
            if x.isdigit() or (x.lstrip("-").isdigit() and x.startswith("-")):
                out.append(int(x))
    return out


def draw_mol_with_highlight(
    smiles: str,
    highlight_atoms: list[int],
    size: tuple[int, int] = (500, 500),
    highlight_color: tuple[float, float, float] = (1.0, 0.75, 0.75),  # 연한 빨강
) -> Image.Image | None:
    """분자 이미지 생성. highlight_atoms가 있으면 해당 원자 하이라이트."""
    with dm.without_rdkit_log():
        mol = dm.to_mol(smiles)
    if mol is None:
        return None
    if highlight_atoms:
        img = Draw.MolToImage(
            mol,
            size=size,
            highlightAtoms=highlight_atoms,
            highlightColor=highlight_color,
        )
    else:
        img = Draw.MolToImage(mol, size=size)
    return img


def main():
    parser = argparse.ArgumentParser(description="Draw toxic/nontoxic molecules with only_toxic / only_nontoxic fragments highlighted.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (default: all)")
    parser.add_argument("--indices", type=str, default=None, help="Comma-separated row indices to visualize, e.g. 0,1,5")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: molecule_safe_ver/visualizations/frag_highlight)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = OUTPUT_DIR

    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required = [
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
        "only_toxic_frag_atom_indices",
        "only_nontoxic_frag_atom_indices",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    if args.indices:
        idx_list = [int(x.strip()) for x in args.indices.split(",") if x.strip().isdigit()]
        df = df.iloc[idx_list].reset_index(drop=True)
    elif args.limit is not None:
        df = df.head(args.limit)

    n = len(df)
    print(f"Visualizing {n} row(s) -> {out_dir}")

    for i, row in tqdm(df.iterrows(), total=n, desc="Drawing"):
        toxic_smiles = row["toxic_safe_decoded_smiles"]
        nontoxic_smiles = row["nontoxic_safe_decoded_smiles"]
        only_toxic_idx = parse_atom_indices(row["only_toxic_frag_atom_indices"])
        only_nontoxic_idx = parse_atom_indices(row["only_nontoxic_frag_atom_indices"])

        toxic_img = draw_mol_with_highlight(
            toxic_smiles,
            only_toxic_idx,
            highlight_color=(1.0, 0.7, 0.7),  # 연한 빨강 (only_toxic)
        )
        nontoxic_img = draw_mol_with_highlight(
            nontoxic_smiles,
            only_nontoxic_idx,
            highlight_color=(0.7, 0.7, 1.0),  # 연한 파랑 (only_nontoxic)
        )

        if toxic_img is None and nontoxic_img is None:
            continue
        w, h = 500, 500
        padding = 40
        label_h = 50
        total_w = w * 2 + padding * 3
        total_h = h + label_h + padding * 2

        combined = Image.new("RGB", (total_w, total_h), "white")
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            try:
                font = ImageFont.truetype("Arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()

        # Toxic (left)
        if toxic_img:
            combined.paste(toxic_img, (padding, label_h + padding))
        draw.text((padding, 12), "Toxic (only_toxic frag highlighted)", fill="black", font=font)

        # Nontoxic (right)
        if nontoxic_img:
            combined.paste(nontoxic_img, (padding * 2 + w, label_h + padding))
        draw.text((padding * 2 + w, 12), "Nontoxic (only_nontoxic frag highlighted)", fill="black", font=font)

        out_path = out_dir / f"row_{i}.png"
        combined.save(out_path)

    print(f"Saved {n} image(s) under {out_dir}")


if __name__ == "__main__":
    main()
