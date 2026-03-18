"""
molecule_safe_ver/unique_smiles.csv 에 있는 SMILES 목록을 불러와
각 SMILES에 대해 canonical 적용 후 SAFE로 변환하여 1:1 매칭 CSV로 저장.
실행 전에 extract_unique_smiles.py 로 unique_smiles.csv 를 먼저 생성해야 함.
"""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 프로젝트 루트의 safe 패키지 import를 위해 path 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "safe"))

import datamol as dm
from safe import encode as safe_encode
from safe import SAFEEncodeError, SAFEFragmentationError

# 경로
SCRIPT_DIR = Path(__file__).resolve().parent
UNIQUE_SMILES_CSV = SCRIPT_DIR / "unique_smiles.csv"
OUTPUT_CSV = SCRIPT_DIR / "smiles_to_safe.csv"

def canonical_smiles(smiles: str):
    """SMILES를 canonical form으로 변환. 실패 시 None 반환."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    with dm.without_rdkit_log():
        try:
            mol = dm.to_mol(str(smiles))
            if mol is None:
                return None
            return dm.standardize_smiles(dm.to_smiles(mol, canonical=True))
        except Exception:
            return None


def smiles_to_safe(smiles: str):
    """Canonical SMILES를 SAFE 문자열로 변환. 실패 시 None 반환."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    with dm.without_rdkit_log():
        try:
            return safe_encode(str(smiles), canonical=True)
        except (SAFEEncodeError, SAFEFragmentationError, Exception):
            return None


def main():
    if not UNIQUE_SMILES_CSV.exists():
        raise FileNotFoundError(
            f"unique_smiles.csv 가 없습니다. 먼저 extract_unique_smiles.py 를 실행하세요.\n경로: {UNIQUE_SMILES_CSV}"
        )

    print(f"Loading: {UNIQUE_SMILES_CSV}")
    df = pd.read_csv(UNIQUE_SMILES_CSV)
    if "smiles" not in df.columns:
        raise ValueError("unique_smiles.csv 에 'smiles' 컬럼이 없습니다.")
    smiles_list = df["smiles"].astype(str).str.strip().tolist()
    n_total = len(smiles_list)
    print(f"Total unique SMILES: {n_total}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("smiles,canonical_smiles,safe\n")
        for smiles in tqdm(smiles_list, desc="SMILES → canonical → SAFE"):
            canon = canonical_smiles(smiles)
            safe_str = smiles_to_safe(canon) if canon is not None else None
            # CSV 이스케이프: 필드에 쉼표/따옴표/개행 있으면 따옴표로 감싼다
            def escape(s):
                if s is None:
                    return ""
                s = str(s)
                if "," in s or '"' in s or "\n" in s:
                    return '"' + s.replace('"', '""') + '"'
                return s

            line = f"{escape(smiles)},{escape(canon)},{escape(safe_str)}\n"
            f.write(line)
            f.flush()  # 실시간으로 디스크에 반영

    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
