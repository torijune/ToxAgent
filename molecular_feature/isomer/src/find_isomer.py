"""
pairs_fg_stereo_merged_nodot.csv를 raw data로 사용하여
Enantiomer, Diastereomer, E/Z Isomer만 찾아 분류합니다.

- Enantiomer: 모든 chiral center가 R↔S 완전 반대, 동일 골격
- Diastereomer: chiral center 일부만 상이 (Enantiomer 아님). n_diff = 입체 화학이 다른 atom(bond) 개수
- E/Z Isomer: E/Z bond geometry 상이 (양쪽 모두 E/Z bond 있을 때)

출력에는 위 세 가지 스테레오 이성질체에 해당하는 pair만 포함됩니다.
"""
from __future__ import annotations

import ast
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# 기본 경로: molecular_feature/isomer/src/ 기준
SCRIPT_DIR = Path(__file__).resolve().parent
MOLECULAR_FEATURE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_CSV = MOLECULAR_FEATURE_DIR / "pairs_fg_stereo_merged_nodot.csv"
DEFAULT_OUTPUT_CSV = MOLECULAR_FEATURE_DIR / "isomer" / "pairs_stereo_isomer_only.csv"
DEFAULT_OUTPUT_PKL = MOLECULAR_FEATURE_DIR / "isomer" / "pairs_stereo_isomer_only.pkl"


def safe_eval_dict(value: Any) -> Dict:
    """문자열/딕셔너리를 안전하게 dict로 변환."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = ast.literal_eval(value)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}
    return {}


def safe_eval_list(value: Any) -> List:
    """문자열/리스트를 안전하게 list로 변환."""
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


def chiral_centers_to_info(centers: Any) -> Dict[str, Any]:
    """
    CSV 형식의 chiral_centers ([{atom_idx, config: 'R'|'S'}, ...])를
    Mol_isomer classify 형식으로 변환.
    """
    centers = safe_eval_list(centers)
    if not centers:
        return {"chiral_center_count": 0, "chirality_types": {"R": 0, "S": 0}}
    r_count = sum(1 for c in centers if isinstance(c, dict) and c.get("config") == "R")
    s_count = sum(1 for c in centers if isinstance(c, dict) and c.get("config") == "S")
    return {
        "chiral_center_count": len(centers),
        "chirality_types": {"R": r_count, "S": s_count},
    }


def ez_bonds_to_info(ez_bonds: Any) -> Dict[str, Any]:
    """
    CSV 형식의 ez_bonds ([{bond, geometry: 'E'|'Z'}, ...] 또는 [{geometry: 'E'|'Z'}, ...])를
    Mol_isomer classify 형식으로 변환.
    """
    bonds = safe_eval_list(ez_bonds)
    if not bonds:
        return {"ez_bond_count": 0, "geometry_types": {"E": 0, "Z": 0}}
    e_count = sum(1 for b in bonds if isinstance(b, dict) and b.get("geometry") == "E")
    z_count = sum(1 for b in bonds if isinstance(b, dict) and b.get("geometry") == "Z")
    return {
        "ez_bond_count": len(bonds),
        "geometry_types": {"E": e_count, "Z": z_count},
    }


# ---------- Mol_isomer와 동일한 검출 로직 ----------


def detect_position_isomer(
    toxic_fg_full: Dict, nontoxic_fg_full: Dict
) -> Tuple[bool, List[str]]:
    """Position Isomer: 같은 FG type, 다른 atom indices."""
    toxic_fg_full = safe_eval_dict(toxic_fg_full)
    nontoxic_fg_full = safe_eval_dict(nontoxic_fg_full)
    toxic_fg_types = set(toxic_fg_full.keys())
    nontoxic_fg_types = set(nontoxic_fg_full.keys())
    if toxic_fg_types != nontoxic_fg_types:
        return False, []
    position_different_fgs = []
    for fg_name in toxic_fg_types:
        toxic_positions = {
            tuple(sorted(idx)) if isinstance(idx, (list, tuple)) else (idx,)
            for idx in toxic_fg_full.get(fg_name, [])
        }
        nontoxic_positions = {
            tuple(sorted(idx)) if isinstance(idx, (list, tuple)) else (idx,)
            for idx in nontoxic_fg_full.get(fg_name, [])
        }
        if toxic_positions != nontoxic_positions:
            position_different_fgs.append(fg_name)
    return len(position_different_fgs) > 0, position_different_fgs


def detect_fg_isomer(
    toxic_fg_full: Dict, nontoxic_fg_full: Dict
) -> Tuple[bool, Dict]:
    """Functional Group Isomer: FG type이 다른 경우."""
    toxic_fg_full = safe_eval_dict(toxic_fg_full)
    nontoxic_fg_full = safe_eval_dict(nontoxic_fg_full)
    toxic_fg_types = set(toxic_fg_full.keys())
    nontoxic_fg_types = set(nontoxic_fg_full.keys())
    if toxic_fg_types == nontoxic_fg_types:
        return False, {}
    return True, {
        "only_in_toxic": list(toxic_fg_types - nontoxic_fg_types),
        "only_in_nontoxic": list(nontoxic_fg_types - toxic_fg_types),
        "common": list(toxic_fg_types & nontoxic_fg_types),
    }


def detect_enantiomer(
    toxic_smiles: str,
    nontoxic_smiles: str,
    toxic_chiral_info: Dict,
    nontoxic_chiral_info: Dict,
) -> bool:
    """Enantiomer: 모든 chiral center가 R↔S 완전 반대, 동일 골격."""
    toxic_chiral_info = safe_eval_dict(toxic_chiral_info)
    nontoxic_chiral_info = safe_eval_dict(nontoxic_chiral_info)
    toxic_count = toxic_chiral_info.get("chiral_center_count", 0)
    nontoxic_count = nontoxic_chiral_info.get("chiral_center_count", 0)
    if toxic_count != nontoxic_count or toxic_count == 0:
        return False
    toxic_chirality = toxic_chiral_info.get("chirality_types", {})
    nontoxic_chirality = nontoxic_chiral_info.get("chirality_types", {})
    toxic_R = toxic_chirality.get("R", 0)
    toxic_S = toxic_chirality.get("S", 0)
    nontoxic_R = nontoxic_chirality.get("R", 0)
    nontoxic_S = nontoxic_chirality.get("S", 0)
    is_enantiomer_basic = (
        toxic_R == nontoxic_S
        and toxic_S == nontoxic_R
        and toxic_R + toxic_S == toxic_count
    )
    if not is_enantiomer_basic:
        return False
    if not RDKIT_AVAILABLE:
        return is_enantiomer_basic
    try:
        mol1 = Chem.MolFromSmiles(toxic_smiles)
        mol2 = Chem.MolFromSmiles(nontoxic_smiles)
        if mol1 is None or mol2 is None:
            return is_enantiomer_basic
        smiles1_no_stereo = Chem.MolToSmiles(mol1, isomericSmiles=False)
        smiles2_no_stereo = Chem.MolToSmiles(mol2, isomericSmiles=False)
        if smiles1_no_stereo != smiles2_no_stereo:
            return False
        return True
    except Exception:
        return is_enantiomer_basic


def detect_diastereomer(
    toxic_smiles: str,
    nontoxic_smiles: str,
    toxic_chiral_info: Dict,
    nontoxic_chiral_info: Dict,
    is_enantiomer: bool,
) -> bool:
    """Diastereomer: 양쪽 모두 chiral center 있고, Enantiomer가 아닌 경우."""
    if is_enantiomer:
        return False
    toxic_chiral_info = safe_eval_dict(toxic_chiral_info)
    nontoxic_chiral_info = safe_eval_dict(nontoxic_chiral_info)
    toxic_count = toxic_chiral_info.get("chiral_center_count", 0)
    nontoxic_count = nontoxic_chiral_info.get("chiral_center_count", 0)
    if toxic_count == 0 or nontoxic_count == 0:
        return False
    try:
        mol1 = Chem.MolFromSmiles(toxic_smiles)
        mol2 = Chem.MolFromSmiles(nontoxic_smiles)
        if mol1 is None or mol2 is None:
            return True
        if toxic_count == nontoxic_count:
            return True
        return True
    except Exception:
        return toxic_count > 0 and nontoxic_count > 0


def detect_ez_isomer(
    toxic_ez_info: Dict,
    nontoxic_ez_info: Dict,
) -> bool:
    """E/Z Isomer: 양쪽 모두 E/Z bond 있고, geometry 비율이 다른 경우."""
    toxic_ez_info = safe_eval_dict(toxic_ez_info)
    nontoxic_ez_info = safe_eval_dict(nontoxic_ez_info)
    toxic_count = toxic_ez_info.get("ez_bond_count", 0)
    nontoxic_count = nontoxic_ez_info.get("ez_bond_count", 0)
    if toxic_count == 0 and nontoxic_count == 0:
        return False
    if toxic_count == 0 or nontoxic_count == 0:
        return False
    toxic_geometry = toxic_ez_info.get("geometry_types", {})
    nontoxic_geometry = nontoxic_ez_info.get("geometry_types", {})
    toxic_E = toxic_geometry.get("E", 0)
    toxic_Z = toxic_geometry.get("Z", 0)
    nontoxic_E = nontoxic_geometry.get("E", 0)
    nontoxic_Z = nontoxic_geometry.get("Z", 0)
    if (toxic_E, toxic_Z) != (nontoxic_E, nontoxic_Z):
        return True
    return False


def _bond_to_key(bond: Any) -> Optional[Tuple[int, ...]]:
    """bond (list/tuple of 2 ints)를 정규화된 키로 변환."""
    if bond is None:
        return None
    if isinstance(bond, (list, tuple)) and len(bond) >= 2:
        return tuple(sorted([int(bond[0]), int(bond[1])]))
    return None


def compute_n_stereo_diff(
    toxic_chiral_centers: Any,
    nontoxic_chiral_centers: Any,
    toxic_ez_bonds: Any,
    nontoxic_ez_bonds: Any,
) -> int:
    """
    두 분자 간 **입체 화학이 다른 atom(chiral center) 또는 bond(E/Z)의 개수**.

    - **Chiral**: 같은 atom_idx에서 config(R/S)가 다르면 1개씩. 한쪽에만 있는 center도 1.
    - **E/Z**: 같은 bond에서 geometry(E/Z)가 다르면 1개씩. 한쪽에만 있는 bond도 1.

    예: toxic 2R·1S, nontoxic 1R·2S (동일 3개 center, 전부 반대) → 3.
    """
    toxic_chiral = safe_eval_list(toxic_chiral_centers)
    nontoxic_chiral = safe_eval_list(nontoxic_chiral_centers)
    toxic_ez = safe_eval_list(toxic_ez_bonds)
    nontoxic_ez = safe_eval_list(nontoxic_ez_bonds)

    # Chiral: atom_idx -> config (R/S)
    t_c: Dict[int, str] = {}
    for c in toxic_chiral:
        if isinstance(c, dict) and "atom_idx" in c and "config" in c:
            t_c[int(c["atom_idx"])] = str(c["config"]).strip().upper()
    n_c: Dict[int, str] = {}
    for c in nontoxic_chiral:
        if isinstance(c, dict) and "atom_idx" in c and "config" in c:
            n_c[int(c["atom_idx"])] = str(c["config"]).strip().upper()

    chiral_diff = 0
    for idx in set(t_c.keys()) | set(n_c.keys()):
        if t_c.get(idx) != n_c.get(idx):
            chiral_diff += 1

    # E/Z: bond (sorted tuple) -> geometry (E/Z)
    t_ez: Dict[Tuple[int, ...], str] = {}
    for b in toxic_ez:
        if not isinstance(b, dict):
            continue
        bond = b.get("bond")
        k = _bond_to_key(bond)
        if k is not None and "geometry" in b:
            t_ez[k] = str(b["geometry"]).strip().upper()
    n_ez: Dict[Tuple[int, ...], str] = {}
    for b in nontoxic_ez:
        if not isinstance(b, dict):
            continue
        bond = b.get("bond")
        k = _bond_to_key(bond)
        if k is not None and "geometry" in b:
            n_ez[k] = str(b["geometry"]).strip().upper()

    ez_diff = 0
    for k in set(t_ez.keys()) | set(n_ez.keys()):
        if t_ez.get(k) != n_ez.get(k):
            ez_diff += 1

    return chiral_diff + ez_diff


def determine_primary_isomer_type(
    isomer_types: List[str],
    is_fg: bool,
    is_position: bool,
    is_enantiomer: bool,
    is_diastereomer: bool,
    is_ez: bool,
) -> str:
    """Primary isomer type 문자열 결정 (복합 타입 포함)."""
    structural_types = ["Position Isomer", "Functional Group Isomer"]
    stereo_types = ["Enantiomer", "Diastereomer", "E/Z Isomer"]
    has_structural = any(t in isomer_types for t in structural_types)
    has_stereo = any(t in isomer_types for t in stereo_types)
    if has_structural and has_stereo:
        if "Functional Group Isomer" in isomer_types:
            if is_enantiomer:
                return "Functional Group Isomer + Enantiomer"
            if is_diastereomer:
                return "Functional Group Isomer + Diastereomer"
            if is_ez:
                return "Functional Group Isomer + E/Z Isomer"
            return "Functional Group Isomer + Stereoisomer"
        if "Position Isomer" in isomer_types:
            if is_enantiomer:
                return "Position Isomer + Enantiomer"
            if is_diastereomer:
                return "Position Isomer + Diastereomer"
            if is_ez:
                return "Position Isomer + E/Z Isomer"
            return "Position Isomer + Stereoisomer"
    if is_enantiomer:
        return "Enantiomer"
    if is_diastereomer:
        return "Diastereomer"
    if is_ez:
        return "E/Z Isomer"
    if is_position:
        return "Position Isomer"
    if is_fg:
        return "Functional Group Isomer"
    return "Unknown"


def classify_isomer_type(row: pd.Series) -> Dict[str, Any]:
    """
    CSV 한 행(row)에 대해 이성질체 타입 분류.
    pairs_fg_stereo_merged_nodot.csv 컬럼명에 맞춤.
    """
    result = {
        "is_position_isomer": False,
        "position_different_fgs": [],
        "is_fg_isomer": False,
        "fg_isomer_diff": {},
        "is_enantiomer": False,
        "is_diastereomer": False,
        "is_ez_isomer": False,
        "isomer_types": [],
        "primary_isomer_type": "Unknown",
        "n_diff": 0,  # stereochemistry 차이 개수 (Diastereomer 등에서 의미 있음)
    }

    # 컬럼명 유연 처리 (소문자 등)
    def get_col(row: pd.Series, *candidates: str) -> Any:
        for c in candidates:
            if c in row.index:
                return row[c]
            low = c.lower()
            for col in row.index:
                if col.lower() == low:
                    return row[col]
        return None

    toxic_fg_full = get_col(row, "toxic_fg_full") or {}
    nontoxic_fg_full = get_col(row, "nontoxic_fg_full") or {}

    # 1. Position Isomer
    is_position, position_fgs = detect_position_isomer(toxic_fg_full, nontoxic_fg_full)
    result["is_position_isomer"] = is_position
    result["position_different_fgs"] = position_fgs

    # 2. Functional Group Isomer
    is_fg, fg_diff = detect_fg_isomer(toxic_fg_full, nontoxic_fg_full)
    result["is_fg_isomer"] = is_fg
    result["fg_isomer_diff"] = fg_diff

    # SMILES (canonical 우선)
    toxic_smiles = get_col(row, "toxic_canonical_smiles") or get_col(row, "toxic_smiles") or ""
    nontoxic_smiles = get_col(row, "nontoxic_canonical_smiles") or get_col(row, "nontoxic_smiles") or ""

    # 3. Chiral info (CSV: toxic_chiral_centers → chirality_types 유도)
    toxic_chiral_raw = get_col(row, "toxic_chiral_centers")
    nontoxic_chiral_raw = get_col(row, "nontoxic_chiral_centers")
    toxic_chiral_info = chiral_centers_to_info(toxic_chiral_raw)
    nontoxic_chiral_info = chiral_centers_to_info(nontoxic_chiral_raw)

    is_enantiomer = detect_enantiomer(
        toxic_smiles, nontoxic_smiles, toxic_chiral_info, nontoxic_chiral_info
    )
    result["is_enantiomer"] = is_enantiomer

    is_diastereomer = detect_diastereomer(
        toxic_smiles,
        nontoxic_smiles,
        toxic_chiral_info,
        nontoxic_chiral_info,
        is_enantiomer,
    )
    result["is_diastereomer"] = is_diastereomer

    # 4. E/Z Isomer (CSV: toxic_ez_bonds → geometry_types 유도)
    toxic_ez_raw = get_col(row, "toxic_ez_bonds")
    nontoxic_ez_raw = get_col(row, "nontoxic_ez_bonds")
    toxic_ez_info = ez_bonds_to_info(toxic_ez_raw)
    nontoxic_ez_info = ez_bonds_to_info(nontoxic_ez_raw)

    is_ez = detect_ez_isomer(toxic_ez_info, nontoxic_ez_info)
    result["is_ez_isomer"] = is_ez

    # 4-1. n_diff: 입체 화학이 다른 atom(chiral center) / bond(E/Z) 개수
    result["n_diff"] = compute_n_stereo_diff(
        toxic_chiral_raw, nontoxic_chiral_raw, toxic_ez_raw, nontoxic_ez_raw
    )

    # 5. isomer_types 리스트 및 primary_isomer_type
    isomer_types = []
    if is_position:
        isomer_types.append("Position Isomer")
    if is_fg:
        isomer_types.append("Functional Group Isomer")
    if is_enantiomer:
        isomer_types.append("Enantiomer")
    elif is_diastereomer:
        isomer_types.append("Diastereomer")
    if is_ez:
        isomer_types.append("E/Z Isomer")
    result["isomer_types"] = isomer_types
    result["primary_isomer_type"] = determine_primary_isomer_type(
        isomer_types, is_fg, is_position, is_enantiomer, is_diastereomer, is_ez
    )

    return result


def run(
    input_csv: Path = DEFAULT_INPUT_CSV,
    output_csv: Optional[Path] = None,
    output_pkl: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    CSV를 읽어 Enantiomer, Diastereomer, E/Z Isomer만 분류하고,
    해당하는 pair만 저장합니다. Diastereomer 등에는 스테레오 차이 개수(n_diff) 컬럼을 추가합니다.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if output_csv is None:
        output_csv = DEFAULT_OUTPUT_CSV
    if output_pkl is None:
        output_pkl = DEFAULT_OUTPUT_PKL
    output_csv = Path(output_csv)
    output_pkl = Path(output_pkl)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("🔬 Enantiomer / Diastereomer / E/Z Isomer만 분류 (스테레오 이성질체만 출력)")
        print("=" * 80)
        print(f"입력: {input_csv}")
        print(f"출력 CSV: {output_csv}")
        print(f"출력 PKL: {output_pkl}")
        print()

    df = pd.read_csv(input_csv)
    if verbose:
        print(f"총 pairs: {len(df):,}")

    classifications = []
    for idx, row in df.iterrows():
        if verbose and (idx + 1) % 2000 == 0:
            print(f"  처리 중: {idx + 1:,}/{len(df):,}")
        try:
            classification = classify_isomer_type(row)
            classifications.append(classification)
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Row {idx} 오류: {e}")
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
    result_df = pd.concat([df.reset_index(drop=True), class_df], axis=1)

    # Enantiomer, Diastereomer, E/Z Isomer에 해당하는 pair만 유지
    stereo_mask = (
        result_df["is_enantiomer"] | result_df["is_diastereomer"] | result_df["is_ez_isomer"]
    )
    result_df = result_df.loc[stereo_mask].reset_index(drop=True)
    if verbose:
        print(f"스테레오 이성질체(Enantiomer/Diastereomer/E/Z Isomer) pair만 필터: {len(result_df):,}개")

    # CSV 저장 시 list/dict 컬럼은 문자열로
    result_df_csv = result_df.copy()
    for col in ["position_different_fgs", "fg_isomer_diff", "isomer_types"]:
        if col in result_df_csv.columns:
            result_df_csv[col] = result_df_csv[col].apply(
                lambda x: str(x) if isinstance(x, (list, dict)) else x
            )
    result_df_csv.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n💾 CSV 저장: {output_csv}")

    try:
        import pickle
        with open(output_pkl, "wb") as f:
            pickle.dump(result_df, f)
        if verbose:
            print(f"💾 PKL 저장: {output_pkl}")
    except Exception as e:
        if verbose:
            print(f"⚠️ PKL 저장 실패: {e}")

    if verbose:
        print("\n📊 스테레오 이성질체 타입별 개수:")
        print(f"  - Enantiomer:   {result_df['is_enantiomer'].sum():,}")
        print(f"  - Diastereomer: {result_df['is_diastereomer'].sum():,}")
        print(f"  - E/Z Isomer:   {result_df['is_ez_isomer'].sum():,}")
        print("\n📋 Primary isomer type 분포:")
        for ptype, count in result_df["primary_isomer_type"].value_counts().items():
            print(f"  - {ptype}: {count:,}")
        diastereomer_df = result_df[result_df["is_diastereomer"]]
        if len(diastereomer_df) > 0:
            print("\n📐 Diastereomer n_diff (스테레오 차이 개수):")
            print(f"  - min: {diastereomer_df['n_diff'].min()}, max: {diastereomer_df['n_diff'].max()}, mean: {diastereomer_df['n_diff'].mean():.2f}")
        print("\n" + "=" * 80)
        print("✅ 완료")
        print("=" * 80)

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="pairs_fg_stereo_merged_nodot.csv에서 Enantiomer/Diastereomer/E/Z Isomer만 찾아 분류 (n_diff 포함)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="입력 CSV 경로",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="출력 CSV 경로",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=None,
        help="출력 PKL 경로",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="출력 최소화",
    )
    args = parser.parse_args()
    run(
        input_csv=args.input,
        output_csv=args.output_csv,
        output_pkl=args.output_pkl,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
