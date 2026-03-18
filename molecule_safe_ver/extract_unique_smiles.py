"""
pairs_fg_stereo_merged_nodot.csv에서 toxic_smiles, nontoxic_smiles의
unique한 값들을 모아서 molecule_safe_ver/unique_smiles.csv 로 저장.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "molecular_feature" / "pairs_fg_stereo_merged_nodot.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "unique_smiles.csv"


def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, usecols=["toxic_smiles", "nontoxic_smiles"])

    toxic = df["toxic_smiles"].dropna().astype(str).str.strip()
    nontoxic = df["nontoxic_smiles"].dropna().astype(str).str.strip()
    toxic = toxic[toxic != ""]
    nontoxic = nontoxic[nontoxic != ""]

    unique_smiles = pd.Series(
        list(set(toxic.tolist()) | set(nontoxic.tolist())),
        name="smiles",
    ).sort_values(ignore_index=True)

    unique_smiles.to_frame().to_csv(OUTPUT_CSV, index=False)
    print(f"Unique SMILES count: {len(unique_smiles)}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
