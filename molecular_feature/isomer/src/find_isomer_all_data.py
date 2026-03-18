"""
Raw_data 내 모든 데이터셋·엔드포인트에서
독성-비독성 pair 중 molecular formula가 같은 isomer를 찾아 분류합니다.

파이프라인:
1. 같은 데이터셋·같은 endpoint에서 독성-비독성 SMILES가 동일한 샘플은 drop
2. SMILES에 "."(염)이 포함된 행은 drop
3. 남은 독성/비독성으로 동일 데이터셋·엔드포인트 내 전체 독성-비독성 pair 생성 후,
   RDKit으로 molecular formula 계산 → 동일 formula pair만 유지
4. find_isomer.py 로직으로 Enantiomer / Diastereomer / E/Z Isomer 등 분류
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    rdMolDescriptors = None

# 프로젝트 루트(detoxicity_model) 및 isomer/src
SCRIPT_DIR = Path(__file__).resolve().parent
ISOMER_DIR = SCRIPT_DIR.parent
MOLECULAR_FEATURE_DIR = ISOMER_DIR.parent
PROJECT_ROOT = MOLECULAR_FEATURE_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "Raw_data"
OUT_DIR = ISOMER_DIR / "all_data_isomer"

for _path in [str(PROJECT_ROOT), str(SCRIPT_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# find_isomer 분류 함수 및 Mol_stereo 스테레오 추출
try:
    from find_isomer import classify_isomer_type
    FIND_ISOMER_AVAILABLE = True
except ImportError:
    FIND_ISOMER_AVAILABLE = False

try:
    from Mol_stereo.stereo_evaluation_metrics import extract_stereochemistry_info
    MOL_STEREO_AVAILABLE = True
except ImportError:
    extract_stereochemistry_info = None
    MOL_STEREO_AVAILABLE = False


def _smiles_has_dot(smiles: str) -> bool:
    """SMILES에 '.'(염/연결)이 포함되어 있으면 True."""
    if pd.isna(smiles) or not isinstance(smiles, str):
        return True
    return "." in str(smiles).strip()


def _get_molecular_formula(smiles: str) -> Optional[str]:
    """RDKit으로 molecular formula 반환. 실패 시 None."""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def _normalize_label(value: Any) -> Optional[int]:
    """Y 값을 0(비독성) 또는 1(독성)으로 정규화. 그 외는 None."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        if value == 0 or value == 0.0:
            return 0
        if value == 1 or value == 1.0:
            return 1
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("0", "false", "no", "n"):
            return 0
        if v in ("1", "true", "yes", "y"):
            return 1
    return None


def _infer_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(smiles_col, label_col, task_col) 추론. Task 없으면 task_col=None."""
    smiles_col = None
    label_col = None
    task_col = None
    for c in df.columns:
        if c.strip() == "X":
            smiles_col = c
        elif c.strip() == "Y":
            label_col = c
        elif c.strip() == "Task":
            task_col = c
    return smiles_col, label_col, task_col


def load_raw_dataset(
    csv_path: Path,
    dataset_name: str,
    drop_dot_smiles: bool = True,
) -> List[Dict[str, Any]]:
    """
    Raw_data CSV 하나를 읽어 (dataset, endpoint, smiles, label) 리스트로 반환.
    - drop_dot_smiles=True 이면 '.' 포함 SMILES 행 제거.
    - label 0/1만 유지.
    """
    df = pd.read_csv(csv_path)
    smiles_col, label_col, task_col = _infer_columns(df)
    if not smiles_col or not label_col:
        return []

    rows = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        if pd.isna(smi):
            continue
        if drop_dot_smiles and _smiles_has_dot(smi):
            continue
        lab = _normalize_label(row[label_col])
        if lab is None:
            continue
        endpoint = str(row[task_col]).strip() if task_col and task_col in row.index else dataset_name
        rows.append({
            "dataset": dataset_name,
            "endpoint": endpoint,
            "smiles": str(smi).strip(),
            "label": lab,
        })
    return rows


def build_same_formula_pairs(
    rows: List[Dict[str, Any]],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    (dataset, endpoint) 단위로 이미 필터된 rows에서
    동일 molecular formula를 가진 독성-비독성 pair만 생성.
    - 같은 dataset, endpoint 내에서만 pair 형성.
    - toxic_smiles != nontoxic_smiles 인 pair만 (같으면 제외).
    """
    # (dataset, endpoint) -> toxic set, nontoxic set (unique SMILES)
    by_key: Dict[Tuple[str, str], Tuple[Set[str], Set[str]]] = defaultdict(lambda: (set(), set()))
    for r in rows:
        k = (r["dataset"], r["endpoint"])
        s = r["smiles"]
        if r["label"] == 1:
            by_key[k][0].add(s)
        else:
            by_key[k][1].add(s)

    # (dataset, endpoint) -> toxic SMILES -> formula; nontoxic SMILES -> formula
    # formula -> (toxic_smiles_list, nontoxic_smiles_list)
    pairs_out: List[Dict[str, Any]] = []

    for (dataset, endpoint), (toxic_smiles_set, nontoxic_smiles_set) in by_key.items():
        # 같은 SMILES는 pair 제외 (독성/비독성 동일 분자 제거)
        toxic_smiles_set = toxic_smiles_set - nontoxic_smiles_set
        nontoxic_smiles_set = nontoxic_smiles_set - toxic_smiles_set
        if not toxic_smiles_set or not nontoxic_smiles_set:
            continue

        toxic_formula: Dict[str, Set[str]] = defaultdict(set)
        nontoxic_formula: Dict[str, Set[str]] = defaultdict(set)
        for s in toxic_smiles_set:
            f = _get_molecular_formula(s)
            if f:
                toxic_formula[f].add(s)
        for s in nontoxic_smiles_set:
            f = _get_molecular_formula(s)
            if f:
                nontoxic_formula[f].add(s)

        common_formulas = set(toxic_formula.keys()) & set(nontoxic_formula.keys())
        for formula in common_formulas:
            for t_smi in toxic_formula[formula]:
                for n_smi in nontoxic_formula[formula]:
                    if t_smi == n_smi:
                        continue
                    pairs_out.append({
                        "dataset": dataset,
                        "endpoint": endpoint,
                        "toxic_smiles": t_smi,
                        "nontoxic_smiles": n_smi,
                        "molecular_formula": formula,
                    })

    return pairs_out


def classify_pairs_with_isomer(
    pairs: List[Dict[str, Any]],
    use_stereo_only: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    각 pair에 대해 stereo 정보 추출 후 find_isomer.classify_isomer_type 호출.
    Raw_data에는 fg_full이 없으므로 toxic_fg_full, nontoxic_fg_full는 빈 dict로 두고
    Enantiomer / Diastereomer / E/Z Isomer만 분류.
    """
    if not MOL_STEREO_AVAILABLE or not FIND_ISOMER_AVAILABLE:
        raise RuntimeError("Mol_stereo.stereo_evaluation_metrics and find_isomer are required.")

    rows = []
    it = tqdm(pairs, desc="Classify isomer") if verbose else pairs
    for p in it:
        t_smi = p["toxic_smiles"]
        n_smi = p["nontoxic_smiles"]
        t_stereo = extract_stereochemistry_info(t_smi) or {}
        n_stereo = extract_stereochemistry_info(n_smi) or {}

        row_dict = {
            "dataset": p["dataset"],
            "endpoint": p["endpoint"],
            "toxic_smiles": t_smi,
            "nontoxic_smiles": n_smi,
            "molecular_formula": p["molecular_formula"],
            "toxic_fg_full": {},
            "nontoxic_fg_full": {},
            "toxic_chiral_centers": t_stereo.get("chiral_centers", []),
            "nontoxic_chiral_centers": n_stereo.get("chiral_centers", []),
            "toxic_ez_bonds": t_stereo.get("ez_bonds", []),
            "nontoxic_ez_bonds": n_stereo.get("ez_bonds", []),
        }
        try:
            classification = classify_isomer_type(pd.Series(row_dict))
            row_dict.update(classification)
        except Exception as e:
            if verbose:
                tqdm.write(f"Classify error for pair: {e}")
            row_dict.update({
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
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if use_stereo_only and len(df) > 0:
        stereo_mask = (
            df["is_enantiomer"] | df["is_diastereomer"] | df["is_ez_isomer"]
        )
        df = df.loc[stereo_mask].reset_index(drop=True)
    return df


def run(
    raw_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    stereo_only: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Raw_data 전체에 대해 isomer 파이프라인 실행.
    - stereo_only: True면 Enantiomer/Diastereomer/E-Z isomer에 해당하는 pair만 저장.
    """
    raw_data_dir = Path(raw_data_dir or RAW_DATA_DIR)
    output_dir = Path(output_dir or OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_csv or output_dir / "isomer_pairs_all_data.csv")

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required.")
    if not MOL_STEREO_AVAILABLE or not FIND_ISOMER_AVAILABLE:
        raise RuntimeError("Mol_stereo.stereo_evaluation_metrics and find_isomer are required.")

    csv_files = sorted(raw_data_dir.glob("*.csv"))
    if not csv_files:
        if verbose:
            print(f"No CSV files in {raw_data_dir}")
        return pd.DataFrame()

    if verbose:
        print("=" * 80)
        print("Isomer pipeline: Raw_data → same-formula toxic/nontoxic pairs → isomer classification")
        print("=" * 80)
        print(f"Raw data dir: {raw_data_dir}")
        print(f"Output CSV: {output_csv}")
        print(f"Datasets: {[f.stem for f in csv_files]}")
        print()

    all_rows: List[Dict[str, Any]] = []
    for csv_path in csv_files:
        dataset_name = csv_path.stem
        rows = load_raw_dataset(csv_path, dataset_name, drop_dot_smiles=True)
        if verbose:
            print(f"  {dataset_name}: {len(rows):,} rows (after drop dot & valid label)")
        all_rows.extend(rows)

    if not all_rows:
        if verbose:
            print("No rows after loading. Exit.")
        return pd.DataFrame()

    if verbose:
        print(f"\nTotal rows: {len(all_rows):,}")
        print("Building same-formula pairs (dataset × endpoint)...")

    pairs = build_same_formula_pairs(all_rows, verbose=verbose)
    if verbose:
        print(f"Same-formula pairs: {len(pairs):,}")

    if not pairs:
        if verbose:
            print("No same-formula pairs. Exit.")
        return pd.DataFrame()

    if verbose:
        print("Classifying isomer types (Enantiomer / Diastereomer / E-Z)...")

    result_df = classify_pairs_with_isomer(pairs, use_stereo_only=stereo_only, verbose=verbose)

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
        print(f"Pairs (stereo-only): {len(result_df):,}")
        if len(result_df) > 0:
            print("Primary isomer type counts:")
            for ptype, count in result_df["primary_isomer_type"].value_counts().items():
                print(f"  - {ptype}: {count:,}")
        print("=" * 80)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Raw_data 전체에서 molecular formula 같은 독성-비독성 isomer pair 찾기"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Raw_data 디렉터리 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="출력 디렉터리",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="출력 CSV 경로 (지정 시 --output-dir 무시)",
    )
    parser.add_argument(
        "--all-isomer",
        action="store_true",
        help="스테레오 이성질체만이 아니라 모든 isomer pair 저장 (Position/FG 포함)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="출력 최소화",
    )
    args = parser.parse_args()
    run(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        stereo_only=not args.all_isomer,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
