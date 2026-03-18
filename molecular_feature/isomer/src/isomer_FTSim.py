"""
isomer_pairs_stereo_only_reclassified.csv에서 각 pair의 toxic_smiles, nontoxic_smiles를
canonicalize한 뒤 Morgan fingerprint Tanimoto similarity를 계산하고 통계를 출력한다.

출력:
- 통계 (개수, 평균, 표준편차, min, max, 25/50/75 백분위)
- 동일 데이터에 tanimoto_sim 컬럼을 추가한 CSV 저장 (viz_low_ftsim_pairs.py에서 사용)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
ISOMER_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_CSV = ISOMER_DIR / "all_data_isomer" / "isomer_pairs_stereo_only_reclassified.csv"
DEFAULT_OUTPUT_CSV = ISOMER_DIR / "all_data_isomer" / "isomer_pairs_stereo_only_reclassified_with_ftsim.csv"


def _canonical_smiles(smiles: str) -> str | None:
    """RDKit으로 canonical SMILES 반환. 실패 시 None."""
    if not RDKIT_AVAILABLE or pd.isna(smiles) or not str(smiles).strip():
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def _morgan_tanimoto(smiles1: str, smiles2: str, radius: int = 2, n_bits: int = 2048) -> float | None:
    """두 SMILES 간 Morgan fingerprint Tanimoto similarity. 실패 시 None."""
    if not RDKIT_AVAILABLE:
        return None
    mol1 = Chem.MolFromSmiles(smiles1) if isinstance(smiles1, str) and smiles1.strip() else None
    mol2 = Chem.MolFromSmiles(smiles2) if isinstance(smiles2, str) and smiles2.strip() else None
    if mol1 is None or mol2 is None:
        return None
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None


def run(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """
    CSV를 읽어 각 pair에 대해 canonical SMILES로 Tanimoto similarity를 계산하고
    통계를 출력한 뒤, tanimoto_sim 컬럼이 추가된 DataFrame을 반환하고 output_csv에 저장한다.
    """
    input_csv = input_csv or DEFAULT_INPUT_CSV
    output_csv = output_csv or DEFAULT_OUTPUT_CSV

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required. Install with: pip install rdkit")

    df = pd.read_csv(input_csv)
    if "toxic_smiles" not in df.columns or "nontoxic_smiles" not in df.columns:
        raise ValueError("CSV must have columns: toxic_smiles, nontoxic_smiles")

    canonical_toxic = []
    canonical_nontoxic = []
    tanimoto_list = []

    for _, row in df.iterrows():
        tox = _canonical_smiles(str(row["toxic_smiles"]) if pd.notna(row["toxic_smiles"]) else "")
        non = _canonical_smiles(
            str(row["nontoxic_smiles"]) if pd.notna(row["nontoxic_smiles"]) else ""
        )
        canonical_toxic.append(tox)
        canonical_nontoxic.append(non)
        sim = _morgan_tanimoto(tox or "", non or "") if (tox and non) else None
        tanimoto_list.append(sim)

    df = df.copy()
    df["canonical_toxic_smiles_ftsim"] = canonical_toxic
    df["canonical_nontoxic_smiles_ftsim"] = canonical_nontoxic
    df["tanimoto_sim"] = tanimoto_list

    valid = [s for s in tanimoto_list if s is not None]
    n_total = len(tanimoto_list)
    n_valid = len(valid)
    n_fail = n_total - n_valid

    if n_fail > 0:
        print(f"Warning: {n_fail} pairs had invalid/missing SMILES or Tanimoto computation failed.")

    if not valid:
        print("No valid Tanimoto similarities to report.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        return df

    n_valid = len(valid)
    mean_val = sum(valid) / n_valid
    variance = sum((x - mean_val) ** 2 for x in valid) / n_valid
    std_val = variance ** 0.5
    min_val = min(valid)
    max_val = max(valid)
    sorted_vals = sorted(valid)
    def _p(pct: float) -> float:
        idx = (pct / 100.0) * (n_valid - 1)
        i, f = int(idx), idx - int(idx)
        if i >= n_valid - 1:
            return sorted_vals[-1]
        return sorted_vals[i] * (1 - f) + sorted_vals[i + 1] * f
    p25, p50, p75 = _p(25), _p(50), _p(75)

    print("=== Tanimoto similarity (Morgan FP) statistics (toxic vs nontoxic, canonical SMILES) ===")
    print(f"  n_pairs (total):   {n_total}")
    print(f"  n_valid:           {n_valid}")
    print(f"  mean:              {mean_val:.4f}")
    print(f"  std:               {std_val:.4f}")
    print(f"  min:               {min_val:.4f}")
    print(f"  max:               {max_val:.4f}")
    print(f"  25% percentile:    {p25:.4f}")
    print(f"  50% percentile:    {p50:.4f}")
    print(f"  75% percentile:    {p75:.4f}")
    print(f"\nOutput saved to: {output_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Compute canonical SMILES and Morgan Tanimoto similarity for isomer pairs; print stats and save CSV."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path with tanimoto_sim column (default: {DEFAULT_OUTPUT_CSV})",
    )
    args = ap.parse_args()
    run(input_csv=args.input, output_csv=args.output)


if __name__ == "__main__":
    main()
