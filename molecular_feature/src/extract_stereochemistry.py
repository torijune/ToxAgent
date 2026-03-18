"""
Filtered pairs CSV의 toxic_smiles, nontoxic_smiles 각각에 대해 Stereochemistry 정보를 추출하여
toxic_smiles_stereo, nontoxic_smiles_stereo 관련 컬럼으로 저장합니다.

- Mol_stereo.stereo_evaluation_metrics.extract_stereochemistry_info 사용 (RDKit 기반)
- 입력: scaffold_sim/results/tanimoto_512_0.7_pairs_final_canonical.csv (canonical된 최종 pair)
- 추출 시 canonical=True: FG와 통일되도록 SMILES를 canonicalize한 뒤 stereo 추출
- 출력: molecular_feature/stereochemistry/ pairs_with_stereochemistry.csv
  컬럼: 기존 + toxic_chiral_centers, toxic_ez_bonds, toxic_has_chirality, toxic_has_ez_bonds,
        nontoxic_chiral_centers, nontoxic_ez_bonds, nontoxic_has_chirality, nontoxic_has_ez_bonds
"""
from pathlib import Path
import sys

# 프로젝트 루트(detoxicity_model)를 path에 추가하여 Mol_stereo 모듈 import
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from Mol_stereo.stereo_evaluation_metrics import extract_stereochemistry_info

# 경로 설정
BASE = Path(__file__).resolve().parent.parent  # molecular_feature
INPUT_CSV = ROOT / "scaffold_sim" / "results" / "tanimoto_512_0.7_pairs_final_canonical.csv"
OUT_DIR = BASE / "stereochemistry"
OUT_CSV = OUT_DIR / "pairs_with_stereochemistry.csv"


def _canonical_smiles(smiles: str):
    """RDKit canonical SMILES. stereo/chiral atom index를 통일하기 위해 추출 전에 적용."""
    if not smiles or pd.isna(smiles):
        return ""
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return Chem.MolToSmiles(mol) if mol else str(smiles).strip()
    except Exception:
        return str(smiles).strip()


def main(
    input_csv: Path = INPUT_CSV,
    out_csv: Path = OUT_CSV,
    verbose: bool = True,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    for col in ["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    toxic_chiral_centers = []
    toxic_ez_bonds = []
    toxic_has_chirality = []
    toxic_has_ez_bonds = []
    nontoxic_chiral_centers = []
    nontoxic_ez_bonds = []
    nontoxic_has_chirality = []
    nontoxic_has_ez_bonds = []

    n = len(df)
    it = tqdm(df.itertuples(index=False), total=n, desc="Stereochemistry extraction") if verbose else df.itertuples(index=False)

    for row in it:
        toxic_smi = str(row.toxic_smiles)
        nontoxic_smi = str(row.nontoxic_smiles)
        # canonical=True: 추출 시에도 canonicalize하여 FG와 동일한 mol 기준 유지
        cano_t = _canonical_smiles(toxic_smi) or toxic_smi
        cano_n = _canonical_smiles(nontoxic_smi) or nontoxic_smi

        t_stereo = extract_stereochemistry_info(cano_t) or {}
        n_stereo = extract_stereochemistry_info(cano_n) or {}

        toxic_chiral_centers.append(t_stereo.get("chiral_centers", []))
        toxic_ez_bonds.append(t_stereo.get("ez_bonds", []))
        toxic_has_chirality.append(t_stereo.get("has_chirality", False))
        toxic_has_ez_bonds.append(t_stereo.get("has_ez_bonds", False))

        nontoxic_chiral_centers.append(n_stereo.get("chiral_centers", []))
        nontoxic_ez_bonds.append(n_stereo.get("ez_bonds", []))
        nontoxic_has_chirality.append(n_stereo.get("has_chirality", False))
        nontoxic_has_ez_bonds.append(n_stereo.get("has_ez_bonds", False))

    df["toxic_chiral_centers"] = toxic_chiral_centers
    df["toxic_ez_bonds"] = toxic_ez_bonds
    df["toxic_has_chirality"] = toxic_has_chirality
    df["toxic_has_ez_bonds"] = toxic_has_ez_bonds
    df["nontoxic_chiral_centers"] = nontoxic_chiral_centers
    df["nontoxic_ez_bonds"] = nontoxic_ez_bonds
    df["nontoxic_has_chirality"] = nontoxic_has_chirality
    df["nontoxic_has_ez_bonds"] = nontoxic_has_ez_bonds

    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"Saved: {out_csv}")
        print(f"  Rows: {len(df):,}")
        print(f"  toxic_smiles_stereo / nontoxic_smiles_stereo columns: toxic_*, nontoxic_* (chiral_centers, ez_bonds, has_chirality, has_ez_bonds)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract stereochemistry for filtered pairs (toxic/nontoxic_smiles).")
    p.add_argument("--input", type=Path, default=INPUT_CSV, help="Input pairs CSV path")
    p.add_argument("--output", type=Path, default=OUT_CSV, help="Output CSV path")
    p.add_argument("--quiet", action="store_true", help="Less output")
    args = p.parse_args()
    main(input_csv=args.input, out_csv=args.output, verbose=not args.quiet)
