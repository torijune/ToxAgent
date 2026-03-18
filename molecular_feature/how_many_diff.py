"""
Merged CSV에 FG 차이 개수, stereo 차이 개수, 합계(n_diff_features) 컬럼을 추가합니다.

- n_fg_diff: unique_fg(JSON 리스트)의 길이 = 서로 다른 FG 피처 개수
- n_stereo_diff: 실제 차이가 있는 chiral center 개수 + E/Z bond 개수
- n_diff_features: n_fg_diff + n_stereo_diff
"""
from pathlib import Path
import json
import ast
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT_CSV = BASE / "pairs_fg_stereo_merged.csv"
OUT_CSV = BASE / "pairs_fg_stereo_merged.csv"


def _parse_unique_fg(s):
    """unique_fg 컬럼(JSON 문자열)을 파싱해 리스트 길이 반환."""
    if pd.isna(s) or str(s).strip() in ("", "[]"):
        return 0
    try:
        out = json.loads(s) if isinstance(s, str) else s
        return len(out) if isinstance(out, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0


def _parse_bool(s):
    """CSV에서 bool/문자열 → True/False."""
    if pd.isna(s):
        return False
    if isinstance(s, bool):
        return bool(s)
    return str(s).strip().lower() in ("true", "1", "yes")


def _parse_chiral_centers(s):
    """chiral_centers 컬럼을 파싱해 리스트 반환."""
    if pd.isna(s) or str(s).strip() in ("", "[]"):
        return []
    try:
        if isinstance(s, str):
            return ast.literal_eval(s)
        return s if isinstance(s, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def _parse_ez_bonds(s):
    """ez_bonds 컬럼을 파싱해 리스트 반환."""
    if pd.isna(s) or str(s).strip() in ("", "[]"):
        return []
    try:
        if isinstance(s, str):
            return ast.literal_eval(s)
        return s if isinstance(s, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def _count_chiral_diff(tx_chiral, nt_chiral):
    """차이가 있는 chiral center 개수를 계산.
    
    Args:
        tx_chiral: toxic_chiral_centers 리스트 [{'atom_idx': int, 'config': 'R'/'S'}, ...]
        nt_chiral: nontoxic_chiral_centers 리스트 [{'atom_idx': int, 'config': 'R'/'S'}, ...]
        
    Returns:
        차이가 있는 chiral center 개수
    """
    if not tx_chiral and not nt_chiral:
        return 0
    
    # atom_idx를 키로 하는 딕셔너리 생성
    tx_dict = {}
    for c in tx_chiral:
        if isinstance(c, dict):
            atom_idx = c.get('atom_idx')
            config = c.get('config', '')
            if atom_idx is not None:
                # 'R' 또는 'S' 추출 (혹시 'R/@@' 같은 형식일 수도 있음)
                if 'R' in str(config):
                    tx_dict[atom_idx] = 'R'
                elif 'S' in str(config):
                    tx_dict[atom_idx] = 'S'
    
    nt_dict = {}
    for c in nt_chiral:
        if isinstance(c, dict):
            atom_idx = c.get('atom_idx')
            config = c.get('config', '')
            if atom_idx is not None:
                if 'R' in str(config):
                    nt_dict[atom_idx] = 'R'
                elif 'S' in str(config):
                    nt_dict[atom_idx] = 'S'
    
    # 모든 chiral center atom_idx 수집
    all_atoms = set(tx_dict.keys()) | set(nt_dict.keys())
    
    # 차이가 있는 것 카운트
    diff_count = 0
    for atom_idx in all_atoms:
        tx_config = tx_dict.get(atom_idx)
        nt_config = nt_dict.get(atom_idx)
        if tx_config != nt_config:
            diff_count += 1
    
    return diff_count


def _count_ez_diff(tx_ez, nt_ez):
    """차이가 있는 E/Z bond 개수를 계산.
    
    Args:
        tx_ez: toxic_ez_bonds 리스트 [{'bond': (int, int), 'geometry': 'E'/'Z'}, ...]
        nt_ez: nontoxic_ez_bonds 리스트 [{'bond': (int, int), 'geometry': 'E'/'Z'}, ...]
        
    Returns:
        차이가 있는 E/Z bond 개수
    """
    if not tx_ez and not nt_ez:
        return 0
    
    # bond를 키로 하는 딕셔너리 생성 (정렬된 튜플로)
    tx_dict = {}
    for b in tx_ez:
        if isinstance(b, dict):
            bond = b.get('bond')
            geometry = b.get('geometry', '')
            if bond is not None:
                # bond를 정렬된 튜플로 변환
                if isinstance(bond, (list, tuple)) and len(bond) >= 2:
                    bond_tuple = tuple(sorted([bond[0], bond[1]]))
                    tx_dict[bond_tuple] = str(geometry).upper()
    
    nt_dict = {}
    for b in nt_ez:
        if isinstance(b, dict):
            bond = b.get('bond')
            geometry = b.get('geometry', '')
            if bond is not None:
                if isinstance(bond, (list, tuple)) and len(bond) >= 2:
                    bond_tuple = tuple(sorted([bond[0], bond[1]]))
                    nt_dict[bond_tuple] = str(geometry).upper()
    
    # 모든 bond 수집
    all_bonds = set(tx_dict.keys()) | set(nt_dict.keys())
    
    # 차이가 있는 것 카운트
    diff_count = 0
    for bond in all_bonds:
        tx_geom = tx_dict.get(bond)
        nt_geom = nt_dict.get(bond)
        if tx_geom != nt_geom:
            diff_count += 1
    
    return diff_count


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)

    n_fg_diff = df["unique_fg"].apply(_parse_unique_fg)

    # 실제 차이가 있는 chiral center와 E/Z bond 개수 계산
    print("Calculating n_stereo_diff from actual differences...")
    n_stereo_diff_list = []
    
    for idx, row in df.iterrows():
        tx_chiral = _parse_chiral_centers(row.get("toxic_chiral_centers", []))
        nt_chiral = _parse_chiral_centers(row.get("nontoxic_chiral_centers", []))
        tx_ez = _parse_ez_bonds(row.get("toxic_ez_bonds", []))
        nt_ez = _parse_ez_bonds(row.get("nontoxic_ez_bonds", []))
        
        chiral_diff_count = _count_chiral_diff(tx_chiral, nt_chiral)
        ez_diff_count = _count_ez_diff(tx_ez, nt_ez)
        
        n_stereo_diff = chiral_diff_count + ez_diff_count
        n_stereo_diff_list.append(n_stereo_diff)
    
    n_stereo_diff = pd.Series(n_stereo_diff_list)

    df["n_fg_diff"] = n_fg_diff
    df["n_stereo_diff"] = n_stereo_diff
    df["n_diff_features"] = n_fg_diff + n_stereo_diff

    df.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}")
    print(f"  Updated columns: n_fg_diff, n_stereo_diff, n_diff_features")
    print(f"  n_fg_diff:    min={df['n_fg_diff'].min()}, max={df['n_fg_diff'].max()}, mean={df['n_fg_diff'].mean():.2f}")
    print(f"  n_stereo_diff: min={df['n_stereo_diff'].min()}, max={df['n_stereo_diff'].max()}, mean={df['n_stereo_diff'].mean():.2f}")
    print(f"  n_diff_features: min={df['n_diff_features'].min()}, max={df['n_diff_features'].max()}, mean={df['n_diff_features'].mean():.2f}")
    
    # 분포 확인
    print(f"\n  n_stereo_diff distribution:")
    print(df['n_stereo_diff'].value_counts().sort_index().head(10))


if __name__ == "__main__":
    main()
