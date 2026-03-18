"""
isomer_FTSim.py 결과 CSV(tanimoto_sim 컬럼 포함)를 읽어,
유사도가 가장 낮은 5개 pair의 toxic_smiles / nontoxic_smiles를 시각화한다.

실행 전 isomer_FTSim.py를 먼저 실행해 *_with_ftsim.csv를 생성해야 한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
DEFAULT_FTSIM_CSV = ISOMER_DIR / "all_data_isomer" / "isomer_pairs_stereo_only_reclassified_with_ftsim.csv"
DEFAULT_OUT_DIR = ISOMER_DIR / "visualizations" / "low_ftsim_pairs"

# 시각화 크기
MOL_SIZE = (400, 400)
TEXT_HEIGHT = 80
PADDING = 24


def _mol_image(smiles: str, size: tuple[int, int] = MOL_SIZE) -> Image.Image | None:
    """SMILES를 분자 이미지로 그리기. 실패 시 None."""
    if not RDKIT_AVAILABLE or not smiles or not str(smiles).strip():
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


def _default_font(size: int = 14):
    """시스템 폰트 시도."""
    for name in ["Helvetica.ttc", "Arial.ttf", "DejaVuSans.ttf"]:
        for base in ["/System/Library/Fonts", "/usr/share/fonts/truetype"]:
            p = Path(base) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    pass
    return ImageFont.load_default()


def draw_one_pair(
    toxic_smiles: str,
    nontoxic_smiles: str,
    tanimoto_sim: float | None,
    pair_label: str,
    out_path: Path,
) -> None:
    """한 pair를 Toxic | Nontoxic 나란히 그려 저장."""
    img_tox = _mol_image(toxic_smiles)
    img_non = _mol_image(nontoxic_smiles)
    if img_tox is None:
        img_tox = Image.new("RGB", MOL_SIZE, (240, 240, 240))
        draw_placeholder = ImageDraw.Draw(img_tox)
        draw_placeholder.text((MOL_SIZE[0] // 2 - 40, MOL_SIZE[1] // 2 - 10), "Invalid SMILES", fill="gray")
    if img_non is None:
        img_non = Image.new("RGB", MOL_SIZE, (240, 240, 240))
        draw_placeholder = ImageDraw.Draw(img_non)
        draw_placeholder.text((MOL_SIZE[0] // 2 - 40, MOL_SIZE[1] // 2 - 10), "Invalid SMILES", fill="gray")

    sim_str = f"Tanimoto = {tanimoto_sim:.4f}" if tanimoto_sim is not None else "Tanimoto = N/A"
    total_width = MOL_SIZE[0] * 2 + PADDING
    total_height = MOL_SIZE[1] + TEXT_HEIGHT * 2 + PADDING

    combined = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(combined)
    font_title = _default_font(18)
    font_sim = _default_font(14)

    # Toxic (왼쪽)
    combined.paste(img_tox, (0, TEXT_HEIGHT))
    draw.text((MOL_SIZE[0] // 2 - 30, 8), "Toxic", fill="black", font=font_title)
    # Nontoxic (오른쪽)
    combined.paste(img_non, (MOL_SIZE[0] + PADDING, TEXT_HEIGHT))
    draw.text((MOL_SIZE[0] + PADDING + MOL_SIZE[0] // 2 - 45, 8), "Nontoxic", fill="black", font=font_title)
    # Tanimoto (중앙 하단)
    draw.text((total_width // 2 - 70, total_height - TEXT_HEIGHT + 10), sim_str, fill="black", font=font_sim)
    draw.text((10, 8), pair_label, fill="gray", font=_default_font(12))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"  Saved: {out_path}")


def draw_all_five(
    rows: list[dict],
    out_path: Path,
) -> None:
    """유사도 최하 5쌍을 한 이미지에 5행으로 그린다 (각 행: Toxic | Nontoxic)."""
    n = len(rows)
    row_height = MOL_SIZE[1] + TEXT_HEIGHT + PADDING
    total_width = MOL_SIZE[0] * 2 + PADDING
    total_height = n * row_height + PADDING

    combined = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(combined)
    font_title = _default_font(16)
    font_sim = _default_font(12)

    for i, r in enumerate(rows):
        tox_s = (r.get("toxic_smiles") or "").strip()
        non_s = (r.get("nontoxic_smiles") or "").strip()
        sim = r.get("tanimoto_sim")
        sim_str = f"Tanimoto = {sim:.4f}" if sim is not None else "N/A"
        y0 = i * row_height

        img_tox = _mol_image(tox_s)
        img_non = _mol_image(non_s)
        if img_tox is None:
            img_tox = Image.new("RGB", MOL_SIZE, (240, 240, 240))
        if img_non is None:
            img_non = Image.new("RGB", MOL_SIZE, (240, 240, 240))

        combined.paste(img_tox, (0, y0 + TEXT_HEIGHT))
        combined.paste(img_non, (MOL_SIZE[0] + PADDING, y0 + TEXT_HEIGHT))
        draw.text((10, y0 + 4), f"Pair {i + 1}", fill="gray", font=_default_font(12))
        draw.text((MOL_SIZE[0] // 2 - 25, y0 + 4), "Toxic", fill="black", font=font_title)
        draw.text((MOL_SIZE[0] + PADDING + MOL_SIZE[0] // 2 - 45, y0 + 4), "Nontoxic", fill="black", font=font_title)
        draw.text((total_width // 2 - 60, y0 + TEXT_HEIGHT + MOL_SIZE[1] + 4), sim_str, fill="black", font=font_sim)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"  Saved (combined): {out_path}")


def run(
    ftsim_csv: Path | None = None,
    out_dir: Path | None = None,
    top_k: int = 5,
    save_combined: bool = True,
) -> None:
    """
    FTSim 결과 CSV에서 tanimoto_sim 기준 최하 top_k개 pair를 골라 시각화한다.
    """
    ftsim_csv = ftsim_csv or DEFAULT_FTSIM_CSV
    out_dir = out_dir or DEFAULT_OUT_DIR

    if not ftsim_csv.exists():
        raise FileNotFoundError(
            f"FTSim CSV not found: {ftsim_csv}. Run isomer_FTSim.py first:\n"
            f"  python molecular_feature/isomer/src/isomer_FTSim.py"
        )
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit and PIL are required for visualization.")

    df = pd.read_csv(ftsim_csv)
    if "tanimoto_sim" not in df.columns:
        raise ValueError(f"CSV must contain column 'tanimoto_sim'. Run isomer_FTSim.py first.")
    if "toxic_smiles" not in df.columns or "nontoxic_smiles" not in df.columns:
        raise ValueError("CSV must contain columns: toxic_smiles, nontoxic_smiles")

    df_valid = df.dropna(subset=["tanimoto_sim"]).copy()
    df_valid = df_valid.sort_values("tanimoto_sim", ascending=True).reset_index(drop=True)
    n_available = len(df_valid)
    k = min(top_k, n_available)
    if k == 0:
        print("No rows with valid tanimoto_sim to visualize.")
        return

    low_rows = df_valid.head(k)
    print(f"Visualizing {k} pairs with lowest Tanimoto similarity (out of {n_available} valid).")

    for i, (_, row) in enumerate(low_rows.iterrows()):
        out_path = out_dir / f"low_ftsim_pair_{i + 1}.png"
        draw_one_pair(
            toxic_smiles=str(row.get("toxic_smiles", "")),
            nontoxic_smiles=str(row.get("nontoxic_smiles", "")),
            tanimoto_sim=row.get("tanimoto_sim"),
            pair_label=f"Pair {i + 1} (rank by lowest sim)",
            out_path=out_path,
        )

    if save_combined and k > 0:
        list_of_dicts = [
            {
                "toxic_smiles": str(row.get("toxic_smiles", "")),
                "nontoxic_smiles": str(row.get("nontoxic_smiles", "")),
                "tanimoto_sim": row.get("tanimoto_sim"),
            }
            for _, row in low_rows.iterrows()
        ]
        draw_all_five(list_of_dicts, out_dir / "low_ftsim_pairs_combined.png")

    print(f"Done. Output directory: {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Visualize toxic/nontoxic SMILES for the 5 pairs with lowest Tanimoto similarity."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_FTSIM_CSV,
        help=f"CSV with tanimoto_sim column (default: {DEFAULT_FTSIM_CSV})",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for images (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of lowest-similarity pairs to visualize (default: 5)",
    )
    ap.add_argument(
        "--no_combined",
        action="store_true",
        help="Do not save the combined image.",
    )
    args = ap.parse_args()
    run(
        ftsim_csv=args.input,
        out_dir=args.out_dir,
        top_k=args.top_k,
        save_combined=not args.no_combined,
    )


if __name__ == "__main__":
    main()
