"""
Filtered pairs CSV의 toxic_smiles, nontoxic_smiles 각각에 대해 Functional Group 정보를 추출하여
toxic_smiles_fg, nontoxic_smiles_fg 관련 컬럼으로 저장합니다.

- FunctionalGroup_Tox_Preprocessor.FunctionalGroupExtractor(AccFG) 사용
- AccFG는 canonical=True 로 동작 → fg_full 의 atom index는 **canonical SMILES**로 만든 mol 기준.
- RDKit 2D 시각화 시 인덱스가 맞으려면, **canonical SMILES**로 mol을 그려야 함.
  → 출력에 toxic_canonical_smiles, nontoxic_canonical_smiles 를 저장해 두었으므로
    시각화 시 이 컬럼으로 mol을 만들고 fg_full 원자 인덱스를 사용하면 됨.

- 입력: scaffold_sim/results/tanimoto_512_0.7_pairs_final_canonical.csv (canonical된 최종 pair)
- 추출 시 AccFG run(..., canonical=True) 유지 (이미 canonical 입력이어도 동일 mol 기준으로 통일)
- 출력: molecular_feature/functional_group/ pairs_with_fg.csv
  컬럼: 기존 + toxic_canonical_smiles, nontoxic_canonical_smiles (시각화용),
        toxic_fg_*, nontoxic_fg_* (fg_names, fg_counts, total_fg_count, fg_full)
"""
from pathlib import Path
import sys

# 프로젝트 루트(detoxicity_model)를 path에 추가하여 AccFG, FunctionalGroupExtractor import
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from rdkit import Chem

from FunctionalGroup_Tox_Preprocessor import FunctionalGroupExtractor


def _canonical_smiles(smiles: str):
    """AccFG와 동일: RDKit canonical SMILES. FG atom index는 이 mol 기준."""
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None

# 경로 설정
BASE = Path(__file__).resolve().parent.parent  # molecular_feature
INPUT_CSV = ROOT / "scaffold_sim" / "results" / "tanimoto_512_0.7_pairs_final_canonical.csv"
OUT_DIR = BASE / "functional_group"
OUT_CSV = OUT_DIR / "pairs_with_fg.csv"


def main(
    input_csv: Path = INPUT_CSV,
    out_csv: Path = OUT_CSV,
    lite_mode: bool = True,
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

    extractor = FunctionalGroupExtractor(lite_mode=lite_mode, verbose=verbose)

    toxic_canonical_smiles = []
    nontoxic_canonical_smiles = []
    toxic_fg_names = []
    toxic_fg_counts = []
    toxic_total_fg_count = []
    toxic_fg_full = []
    nontoxic_fg_names = []
    nontoxic_fg_counts = []
    nontoxic_total_fg_count = []
    nontoxic_fg_full = []

    n = len(df)
    it = tqdm(df.itertuples(index=False), total=n, desc="FG extraction") if verbose else df.itertuples(index=False)

    for row in it:
        toxic_smi = row.toxic_smiles
        nontoxic_smi = row.nontoxic_smiles

        # FG atom index는 canonical mol 기준 → canonical SMILES로 추출하고 저장 (시각화 시 이걸로 그리면 인덱스 일치)
        cano_t = _canonical_smiles(toxic_smi)
        cano_n = _canonical_smiles(nontoxic_smi)
        toxic_canonical_smiles.append(cano_t if cano_t else toxic_smi)
        nontoxic_canonical_smiles.append(cano_n if cano_n else nontoxic_smi)

        t_fg = extractor.extract_single_fg_properties(str(toxic_smi))
        n_fg = extractor.extract_single_fg_properties(str(nontoxic_smi))

        toxic_fg_names.append(t_fg["fg_names"])
        toxic_fg_counts.append(t_fg["fg_counts"])
        toxic_total_fg_count.append(t_fg["total_fg_count"])
        toxic_fg_full.append(t_fg["fg_full"])

        nontoxic_fg_names.append(n_fg["fg_names"])
        nontoxic_fg_counts.append(n_fg["fg_counts"])
        nontoxic_total_fg_count.append(n_fg["total_fg_count"])
        nontoxic_fg_full.append(n_fg["fg_full"])

    df["toxic_canonical_smiles"] = toxic_canonical_smiles
    df["nontoxic_canonical_smiles"] = nontoxic_canonical_smiles
    df["toxic_fg_names"] = toxic_fg_names
    df["toxic_fg_counts"] = toxic_fg_counts
    df["toxic_total_fg_count"] = toxic_total_fg_count
    df["toxic_fg_full"] = toxic_fg_full
    df["nontoxic_fg_names"] = nontoxic_fg_names
    df["nontoxic_fg_counts"] = nontoxic_fg_counts
    df["nontoxic_total_fg_count"] = nontoxic_total_fg_count
    df["nontoxic_fg_full"] = nontoxic_fg_full

    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"Saved: {out_csv}")
        print(f"  Rows: {len(df):,}")
        print(f"  FG는 canonical=True 기준. 시각화 시 toxic_canonical_smiles / nontoxic_canonical_smiles 로 mol 그리면 fg_full atom index와 일치.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract FG for filtered pairs (toxic/nontoxic_smiles).")
    p.add_argument("--input", type=Path, default=INPUT_CSV, help="Input pairs CSV path")
    p.add_argument("--output", type=Path, default=OUT_CSV, help="Output CSV path")
    p.add_argument("--no-lite", action="store_true", help="Use AccFG full mode (slower)")
    p.add_argument("--quiet", action="store_true", help="Less output")
    args = p.parse_args()
    main(input_csv=args.input, out_csv=args.output, lite_mode=not args.no_lite, verbose=not args.quiet)
