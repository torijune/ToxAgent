"""
isomer_pairs_all_data.csv에 대해:
- 각 toxic/nontoxic SMILES를 canonicalize
- RDKit으로 stereochemistry 정보를 제거한 SMILES 생성 (isomericSmiles=False)
- 두 no-stereo SMILES가 동일하면 → ONLY stereochemistry 차이 (same skeleton)
- same_skeleton=True 인 pair만 남기고, 해당 pair들에 대해 isomer를 새로 분류해 저장
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
ISOMER_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_CSV = ISOMER_DIR / "all_data_isomer" / "isomer_pairs_all_data.csv"
DEFAULT_OUTPUT_CSV = ISOMER_DIR / "all_data_isomer" / "isomer_pairs_stereo_only_reclassified.csv"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from find_isomer import classify_isomer_type
    FIND_ISOMER_AVAILABLE = True
except ImportError:
    FIND_ISOMER_AVAILABLE = False


def _canonical_smiles(smiles: str) -> Optional[str]:
    """RDKit canonical SMILES. 실패 시 None."""
    if not RDKIT_AVAILABLE or pd.isna(smiles) or not str(smiles).strip():
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
    except Exception:
        return None


def _no_stereo_smiles(smiles: str) -> Optional[str]:
    """Canonical SMILES에서 stereochemistry 제거. 골격만 비교용."""
    if not RDKIT_AVAILABLE or pd.isna(smiles) or not str(smiles).strip():
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None


def _safe_eval_list(value: Any) -> List:
    """문자열/리스트를 안전하게 list로."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            result = ast.literal_eval(value)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    if pd.isna(value) or value == "" or value == "[]":
        return []
    return []


def run(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    1. CSV 로드
    2. 각 행에 대해 canonical SMILES, no-stereo SMILES 계산 → same_skeleton 여부
    3. same_skeleton=True 인 행만 유지
    4. 해당 행들에 대해 find_isomer.classify_isomer_type으로 isomer 재분류
    5. 결과 저장
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv or DEFAULT_OUTPUT_CSV)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit required.")
    if not FIND_ISOMER_AVAILABLE:
        raise RuntimeError("find_isomer.classify_isomer_type required.")

    df = pd.read_csv(input_csv)
    if "toxic_smiles" not in df.columns or "nontoxic_smiles" not in df.columns:
        raise ValueError("CSV must have toxic_smiles, nontoxic_smiles")

    if verbose:
        print("=" * 80)
        print("Compare canonical no-stereo → filter same_skeleton → reclassify isomer")
        print("=" * 80)
        print(f"Input: {input_csv} (rows: {len(df):,})")

    # 1) Canonical + no-stereo
    toxic_canonical = []
    nontoxic_canonical = []
    toxic_no_stereo = []
    nontoxic_no_stereo = []
    same_skeleton = []

    it = tqdm(df.itertuples(index=False), total=len(df), desc="Canonical + no-stereo") if verbose else df.itertuples(index=False)
    for row in it:
        t_smi = getattr(row, "toxic_smiles", None)
        n_smi = getattr(row, "nontoxic_smiles", None)
        t_can = _canonical_smiles(t_smi) if t_smi else None
        n_can = _canonical_smiles(n_smi) if n_smi else None
        t_ns = _no_stereo_smiles(t_smi) if t_smi else None
        n_ns = _no_stereo_smiles(n_smi) if n_smi else None
        toxic_canonical.append(t_can)
        nontoxic_canonical.append(n_can)
        toxic_no_stereo.append(t_ns)
        nontoxic_no_stereo.append(n_ns)
        same_skeleton.append(
            (t_ns is not None and n_ns is not None and t_ns == n_ns)
        )

    df["toxic_canonical_smiles"] = toxic_canonical
    df["nontoxic_canonical_smiles"] = nontoxic_canonical
    df["toxic_no_stereo_smiles"] = toxic_no_stereo
    df["nontoxic_no_stereo_smiles"] = nontoxic_no_stereo
    df["same_skeleton"] = same_skeleton

    # 2) Filter: same_skeleton only
    df_stereo = df.loc[df["same_skeleton"]].copy().reset_index(drop=True)
    if verbose:
        print(f"Same skeleton (ONLY stereochemistry difference): {len(df_stereo):,} / {len(df):,}")

    if len(df_stereo) == 0:
        df_stereo.to_csv(output_csv, index=False)
        if verbose:
            print(f"Saved (empty): {output_csv}")
        return df_stereo

    # 3) Re-classify isomer using canonical SMILES and existing chiral/ez columns
    classify_cols = [
        "is_position_isomer", "position_different_fgs", "is_fg_isomer", "fg_isomer_diff",
        "is_enantiomer", "is_diastereomer", "is_ez_isomer", "isomer_types",
        "primary_isomer_type", "n_diff",
    ]
    # Drop old classification so we can replace with new
    for c in classify_cols:
        if c in df_stereo.columns:
            df_stereo = df_stereo.drop(columns=[c])

    classifications = []
    it2 = tqdm(df_stereo.iterrows(), total=len(df_stereo), desc="Reclassify isomer") if verbose else df_stereo.iterrows()
    for idx, row in it2:
        # Build row for classify_isomer_type: canonical SMILES + chiral/ez from CSV
        t_can = row.get("toxic_canonical_smiles") or row.get("toxic_smiles")
        n_can = row.get("nontoxic_canonical_smiles") or row.get("nontoxic_smiles")
        t_chiral = row.get("toxic_chiral_centers")
        n_chiral = row.get("nontoxic_chiral_centers")
        t_ez = row.get("toxic_ez_bonds")
        n_ez = row.get("nontoxic_ez_bonds")
        # Parse if string (CSV stores as string)
        if isinstance(t_chiral, str):
            t_chiral = _safe_eval_list(t_chiral)
        if isinstance(n_chiral, str):
            n_chiral = _safe_eval_list(n_chiral)
        if isinstance(t_ez, str):
            t_ez = _safe_eval_list(t_ez)
        if isinstance(n_ez, str):
            n_ez = _safe_eval_list(n_ez)

        row_dict = {
            "toxic_canonical_smiles": t_can,
            "nontoxic_canonical_smiles": n_can,
            "toxic_smiles": t_can,
            "nontoxic_smiles": n_can,
            "toxic_fg_full": {},
            "nontoxic_fg_full": {},
            "toxic_chiral_centers": t_chiral,
            "nontoxic_chiral_centers": n_chiral,
            "toxic_ez_bonds": t_ez,
            "nontoxic_ez_bonds": n_ez,
        }
        try:
            classification = classify_isomer_type(pd.Series(row_dict))
            classifications.append(classification)
        except Exception as e:
            if verbose:
                tqdm.write(f"Row {idx} classify error: {e}")
            classifications.append({
                "is_position_isomer": False,
                "position_different_fgs": [],
                "is_fg_isomer": False,
                "fg_isomer_diff": {},
                "is_enantiomer": False,
                "is_diastereomer": False,
                "is_ez_isomer": False,
                "isomer_types": [],
                "primary_isomer_type": "Unknown",
                "n_diff": 0,
            })

    class_df = pd.DataFrame(classifications)
    result_df = pd.concat([df_stereo.reset_index(drop=True), class_df], axis=1)

    # CSV 저장 시 list/dict 컬럼 문자열로
    result_df_csv = result_df.copy()
    for col in ["position_different_fgs", "fg_isomer_diff", "isomer_types",
                "toxic_chiral_centers", "nontoxic_chiral_centers", "toxic_ez_bonds", "nontoxic_ez_bonds"]:
        if col in result_df_csv.columns:
            result_df_csv[col] = result_df_csv[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
    result_df_csv.to_csv(output_csv, index=False)

    if verbose:
        print(f"\nSaved: {output_csv}")
        print(f"Rows (stereo-only, reclassified): {len(result_df):,}")
        if "primary_isomer_type" in result_df.columns:
            print("Primary isomer type counts:")
            for ptype, count in result_df["primary_isomer_type"].value_counts().items():
                print(f"  - {ptype}: {count:,}")
        print("=" * 80)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Canonical + no-stereo 비교 후 same_skeleton만 남기고 isomer 재분류"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_CSV, help="입력 CSV")
    parser.add_argument("--output", type=Path, default=None, help="출력 CSV")
    parser.add_argument("--quiet", action="store_true", help="출력 최소화")
    args = parser.parse_args()
    run(
        input_csv=args.input,
        output_csv=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
