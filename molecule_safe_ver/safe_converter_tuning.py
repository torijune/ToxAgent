"""
converter.py SUPPORTED_SLICERS 기준으로 모든 slicer 타입에 대해
SMILES → canonical → SAFE 변환을 수행하고, 결과를 한 CSV에 저장.

출력 컬럼: smiles, canonical_smiles, hr_safe, rotatable_safe, recap_safe, mmpa_safe, attach_safe, brics_safe
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import datamol as dm
from tqdm import tqdm

# 프로젝트 루트(ToxAgent)에 safe 패키지가 있음
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe.safe.converter import encode as safe_encode
from safe.safe._exception import SAFEEncodeError, SAFEFragmentationError

# converter.py 38행과 동일한 순서
SUPPORTED_SLICERS = ["hr", "rotatable", "recap", "mmpa", "attach", "brics"]

INPUT_CSV = SCRIPT_DIR / "unique_smiles.csv"
OUTPUT_CSV = SCRIPT_DIR / "smiles_to_safe_by_slicer.csv"


def canonical_smiles(smiles: str):
    """SMILES를 canonical form으로 변환. 실패 시 None."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    try:
        with dm.without_rdkit_log():
            mol = dm.to_mol(str(smiles).strip())
            if mol is None:
                return None
            return dm.standardize_smiles(dm.to_smiles(mol, canonical=True))
    except Exception:
        return None


def encode_with_slicer(smiles: str, slicer: str) -> str | None:
    """주어진 slicer로 SAFE 인코딩. 실패 시 None."""
    if not smiles or pd.isna(smiles):
        return None
    try:
        with dm.without_rdkit_log():
            return safe_encode(
                str(smiles).strip(),
                canonical=True,
                slicer=slicer,
                ignore_stereo=True,
            )
    except (SAFEEncodeError, SAFEFragmentationError, ValueError, Exception):
        return None


def main():
    parser = argparse.ArgumentParser(description="Run SAFE encoding with all supported slicers.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows (for testing).")
    args = parser.parse_args()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"입력 파일이 없습니다. 먼저 extract_unique_smiles.py 로 {INPUT_CSV.name} 을 생성하세요."
        )

    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if "smiles" not in df.columns:
        raise ValueError("입력 CSV에 'smiles' 컬럼이 없습니다.")
    smiles_list = df["smiles"].astype(str).str.strip().tolist()
    if args.limit is not None:
        smiles_list = smiles_list[: args.limit]
        print(f"Limiting to first {args.limit} rows.")
    n = len(smiles_list)

    # canonical_smiles 계산
    print("Computing canonical SMILES...")
    canonical_list = []
    for smi in tqdm(smiles_list, desc="canonical"):
        canonical_list.append(canonical_smiles(smi))

    # 각 slicer별 SAFE 컬럼
    result_cols = {f"{slicer}_safe": [] for slicer in SUPPORTED_SLICERS}
    for slicer in SUPPORTED_SLICERS:
        print(f"Encoding with slicer='{slicer}'...")
        for i, canon in enumerate(tqdm(canonical_list, desc=slicer, leave=False)):
            if canon is None:
                result_cols[f"{slicer}_safe"].append("")
            else:
                safe_str = encode_with_slicer(canon, slicer)
                result_cols[f"{slicer}_safe"].append(safe_str if safe_str else "")

    out_df = pd.DataFrame({
        "smiles": smiles_list,
        "canonical_smiles": [c if c is not None else "" for c in canonical_list],
        **result_cols,
    })
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} (rows={len(out_df)}, columns={list(out_df.columns)})")


if __name__ == "__main__":
    main()
