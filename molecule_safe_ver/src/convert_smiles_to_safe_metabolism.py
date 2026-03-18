from __future__ import annotations

"""
metabolism_ver/pairs.csv 에 등장하는 모든 SMILES를 대상으로
canonical SMILES → SAFE 로 변환한 매핑 CSV를 생성한다.

출력:
  molecule_safe_ver/smiles_to_safe_metabolism.csv

그 후 ace_safe_ver/src/build_metabolism.py 실행 시
  --mapping molecule_safe_ver/smiles_to_safe_metabolism.csv
를 넘겨주면 metabolism pairs에도 SAFE를 제대로 붙일 수 있다.
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ToxAgent 루트를 sys.path에 넣고 local safe 패키지에서 converter import
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../ToxAgent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datamol as dm
from safe.safe.converter import encode as safe_encode, SAFEEncodeError, SAFEFragmentationError


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
METABOLISM_PAIRS = (
    REPO_ROOT / "molecularACE_ver" / "metabolism_ver" / "pairs.csv"
)
OUTPUT_CSV = SCRIPT_DIR.parent / "smiles_to_safe_metabolism.csv"


def canonical_smiles(smiles: str):
    """SMILES를 canonical form으로 변환. 실패 시 None."""
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
    """Canonical SMILES를 SAFE 문자열로 변환. 실패 시 None."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    with dm.without_rdkit_log():
        try:
            return safe_encode(str(smiles), canonical=True)
        except (SAFEEncodeError, SAFEFragmentationError, Exception):
            return None


def main() -> None:
    if not METABOLISM_PAIRS.exists():
        raise FileNotFoundError(f"Metabolism pairs.csv not found: {METABOLISM_PAIRS}")

    print(f"Loading metabolism pairs: {METABOLISM_PAIRS}")
    df = pd.read_csv(METABOLISM_PAIRS)
    for col in ["toxic_smiles", "nontoxic_smiles"]:
        if col not in df.columns:
            raise ValueError(f"Pairs CSV must have column '{col}'. Found: {list(df.columns)}")

    smiles_set = set(
        str(s).strip()
        for col in ["toxic_smiles", "nontoxic_smiles"]
        for s in df[col].dropna().astype(str).tolist()
        if str(s).strip()
    )
    smiles_list = sorted(smiles_set)
    n_total = len(smiles_list)
    print(f"Total unique metabolism SMILES: {n_total}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        f.write("smiles,canonical_smiles,safe\n")

        def escape(s: str | None) -> str:
            if s is None:
                return ""
            s = str(s)
            if "," in s or '"' in s or "\n" in s:
                return '"' + s.replace('"', '""') + '"'
            return s

        for smiles in tqdm(smiles_list, desc="metabolism SMILES → canonical → SAFE"):
            canon = canonical_smiles(smiles)
            safe_str = smiles_to_safe(canon) if canon is not None else None
            line = f"{escape(smiles)},{escape(canon)},{escape(safe_str)}\n"
            f.write(line)

    print(f"Saved metabolism mapping: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

