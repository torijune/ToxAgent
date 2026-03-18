"""
pairs_fg_stereo_merged.csv의 diff 컬럼들을 재계산하여 덮어씌우기.

atom_index_diff는 제외하고, FG 이름 차이만 diff로 간주.
"""
from pathlib import Path
import ast
import json
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE / "pairs_fg_stereo_merged_nodot.csv"
OUTPUT_CSV = BASE / "pairs_fg_stereo_merged_nodot.csv"  # 같은 파일에 덮어씌우기


def _parse_list(s):
    if pd.isna(s) or str(s).strip() in ("", "[]"):
        return []
    try:
        out = ast.literal_eval(s) if isinstance(s, str) else s
        return list(out) if isinstance(out, list) else []
    except Exception:
        return []


def _parse_fg_full(s):
    if pd.isna(s) or str(s).strip() in ("", "{}"):
        return {}
    try:
        out = ast.literal_eval(s) if isinstance(s, str) else s
        if not isinstance(out, dict):
            return {}
        return {
            k: [tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in (v if isinstance(v, list) else [v])]
            for k, v in out.items()
        }
    except Exception:
        return {}


def _normalize_indices(indices_list):
    """리스트 내 각 원소를 tuple로, 정렬 가능하게."""
    return sorted(tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in indices_list)


def compute_fg_diff_strict(tx_names: list, nt_names: list, tx_full: dict, nt_full: dict):
    """
    FG diff 계산 (atom_index_diff 제외).
    
    Returns:
        (has_fg_diff: bool, unique_fg: list of dict, n_fg_diff: int)
    """
    if not tx_names and not nt_names:
        return False, [], 0
    # 둘 다 FG가 있어야 strict
    if not tx_names or not nt_names:
        return False, [], 0  # 한쪽만 FG 있으면 diff 아님

    tx_set = set(tx_names)
    nt_set = set(nt_names)
    only_in_tx = tx_set - nt_set
    only_in_nt = nt_set - tx_set

    unique_fg = []

    # FG 이름이 다른 것들만 unique_fg에 추가 (atom_index_diff 제외)
    for fg in only_in_tx:
        unique_fg.append({
            "fg_name": fg,
            "reason": "name_only_in_toxic",
            "toxic_atom_indices": _normalize_indices(tx_full.get(fg, [])),
            "nontoxic_atom_indices": [],
        })
    for fg in only_in_nt:
        unique_fg.append({
            "fg_name": fg,
            "reason": "name_only_in_nontoxic",
            "toxic_atom_indices": [],
            "nontoxic_atom_indices": _normalize_indices(nt_full.get(fg, [])),
        })

    has_fg_diff = len(unique_fg) > 0
    n_fg_diff = len(unique_fg)
    return has_fg_diff, unique_fg, n_fg_diff


def _parse_chiral_centers(s):
    if pd.isna(s) or s == "[]" or str(s).strip() == "":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _parse_ez_bonds(s):
    if pd.isna(s) or s == "[]" or str(s).strip() == "":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _parse_bool(s):
    if pd.isna(s):
        return False
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("true", "1", "yes")


def compute_stereo_diff_loose(
    t_has_chirality: bool,
    n_has_chirality: bool,
    t_centers: list,
    n_centers: list,
    t_has_ez: bool,
    n_has_ez: bool,
    t_ez: list,
    n_ez: list,
) -> tuple[bool, bool, str, int]:
    """
    Stereo diff 계산 (loose 기준).
    
    Returns:
        (chiral_diff_loose, ez_diff_loose, stereo_diff_type_loose, n_stereo_diff)
    """
    # Chiral: unique (한쪽만 있음) or count change or R/S type change
    t_R = sum(1 for c in t_centers if isinstance(c, dict) and c.get("config") == "R")
    t_S = sum(1 for c in t_centers if isinstance(c, dict) and c.get("config") == "S")
    n_R = sum(1 for c in n_centers if isinstance(c, dict) and c.get("config") == "R")
    n_S = sum(1 for c in n_centers if isinstance(c, dict) and c.get("config") == "S")

    chiral_count_diff = len(t_centers) != len(n_centers)
    chiral_R_change = (n_R - t_R) != 0
    chiral_S_change = (n_S - t_S) != 0
    one_has_chiral_other_not = t_has_chirality != n_has_chirality

    chiral_diff_loose = (
        one_has_chiral_other_not
        or chiral_count_diff
        or chiral_R_change
        or chiral_S_change
    )

    # E/Z: unique (한쪽만 있음) or count change or E/Z geometry change
    t_E = sum(1 for b in t_ez if isinstance(b, dict) and b.get("geometry") == "E")
    t_Z = sum(1 for b in t_ez if isinstance(b, dict) and b.get("geometry") == "Z")
    n_E = sum(1 for b in n_ez if isinstance(b, dict) and b.get("geometry") == "E")
    n_Z = sum(1 for b in n_ez if isinstance(b, dict) and b.get("geometry") == "Z")

    ez_count_diff = len(t_ez) != len(n_ez)
    ez_E_change = (n_E - t_E) != 0
    ez_Z_change = (n_Z - t_Z) != 0
    one_has_ez_other_not = t_has_ez != n_has_ez

    ez_diff_loose = (
        one_has_ez_other_not
        or ez_count_diff
        or ez_E_change
        or ez_Z_change
    )

    if chiral_diff_loose and ez_diff_loose:
        label = "both"
        n_stereo_diff = 2
    elif chiral_diff_loose:
        label = "chiral_only"
        n_stereo_diff = 1
    elif ez_diff_loose:
        label = "ez_only"
        n_stereo_diff = 1
    else:
        label = ""
        n_stereo_diff = 0

    return chiral_diff_loose, ez_diff_loose, label, n_stereo_diff


def determine_diff_type(has_fg_diff: bool, stereo_diff_type: str) -> str:
    """diff_type 결정: 'only_fg', 'only_stereo', 'both', ''"""
    if has_fg_diff and stereo_diff_type:
        return "both"
    elif has_fg_diff:
        return "only_fg"
    elif stereo_diff_type:
        return "only_stereo"
    else:
        return ""


def determine_step_num(n_fg_diff: int, n_stereo_diff: int) -> str:
    """step_num 결정: 'one_step' (총 diff가 1개) or 'multi_step' (2개 이상)"""
    total_diff = n_fg_diff + n_stereo_diff
    if total_diff == 0:
        return ""
    elif total_diff == 1:
        return "one_step"
    else:
        return "multi_step"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")

    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Total rows: {len(df):,}")

    # 필요한 컬럼 확인
    required_cols = [
        "toxic_fg_names", "nontoxic_fg_names", "toxic_fg_full", "nontoxic_fg_full",
        "toxic_has_chirality", "toxic_chiral_centers", "toxic_has_ez_bonds", "toxic_ez_bonds",
        "nontoxic_has_chirality", "nontoxic_chiral_centers", "nontoxic_has_ez_bonds", "nontoxic_ez_bonds",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 재계산
    has_fg_diff_list = []
    unique_fg_list = []
    n_fg_diff_list = []
    chiral_diff_loose_list = []
    ez_diff_loose_list = []
    stereo_diff_type_loose_list = []
    n_stereo_diff_list = []
    n_diff_features_list = []
    diff_type_list = []
    step_num_list = []

    print("Recomputing diffs...")
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  Processed: {idx + 1:,} / {len(df):,}")

        # FG diff
        tx_names = _parse_list(row["toxic_fg_names"])
        nt_names = _parse_list(row["nontoxic_fg_names"])
        tx_full = _parse_fg_full(row["toxic_fg_full"])
        nt_full = _parse_fg_full(row["nontoxic_fg_full"])
        has_fg_diff, unique_fg, n_fg_diff = compute_fg_diff_strict(tx_names, nt_names, tx_full, nt_full)

        # Stereo diff
        t_has_c = _parse_bool(row["toxic_has_chirality"])
        n_has_c = _parse_bool(row["nontoxic_has_chirality"])
        t_centers = _parse_chiral_centers(row["toxic_chiral_centers"])
        n_centers = _parse_chiral_centers(row["nontoxic_chiral_centers"])
        t_has_ez = _parse_bool(row["toxic_has_ez_bonds"])
        n_has_ez = _parse_bool(row["nontoxic_has_ez_bonds"])
        t_ez = _parse_ez_bonds(row["toxic_ez_bonds"])
        n_ez = _parse_ez_bonds(row["nontoxic_ez_bonds"])

        chiral_diff, ez_diff, stereo_type, n_stereo_diff = compute_stereo_diff_loose(
            t_has_c, n_has_c, t_centers, n_centers,
            t_has_ez, n_has_ez, t_ez, n_ez,
        )

        # Derived columns
        n_diff_features = n_fg_diff + n_stereo_diff
        diff_type = determine_diff_type(has_fg_diff, stereo_type)
        step_num = determine_step_num(n_fg_diff, n_stereo_diff)

        # Append
        has_fg_diff_list.append(has_fg_diff)
        unique_fg_list.append(json.dumps(unique_fg, ensure_ascii=False) if unique_fg else "[]")
        n_fg_diff_list.append(n_fg_diff)
        chiral_diff_loose_list.append(chiral_diff)
        ez_diff_loose_list.append(ez_diff)
        stereo_diff_type_loose_list.append(stereo_type)
        n_stereo_diff_list.append(n_stereo_diff)
        n_diff_features_list.append(n_diff_features)
        diff_type_list.append(diff_type)
        step_num_list.append(step_num)

    # Update DataFrame
    df = df.copy()
    df["has_fg_diff"] = has_fg_diff_list
    df["unique_fg"] = unique_fg_list
    df["n_fg_diff"] = n_fg_diff_list
    df["chiral_diff_loose"] = chiral_diff_loose_list
    df["ez_diff_loose"] = ez_diff_loose_list
    df["stereo_diff_type_loose"] = stereo_diff_type_loose_list
    df["n_stereo_diff"] = n_stereo_diff_list
    df["n_diff_features"] = n_diff_features_list
    df["diff_type"] = diff_type_list
    df["step_num"] = step_num_list

    # Save
    print(f"\nSaving to: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    # Statistics
    print("\n=== Statistics ===")
    print(f"Total rows: {len(df):,}")
    print(f"  has_fg_diff=True: {df['has_fg_diff'].sum():,}")
    print(f"  stereo_diff_type != '': {(df['stereo_diff_type_loose'] != '').sum():,}")
    print(f"  diff_type='only_fg': {(df['diff_type'] == 'only_fg').sum():,}")
    print(f"  diff_type='only_stereo': {(df['diff_type'] == 'only_stereo').sum():,}")
    print(f"  diff_type='both': {(df['diff_type'] == 'both').sum():,}")
    print(f"  step_num='one_step': {(df['step_num'] == 'one_step').sum():,}")
    print(f"  step_num='multi_step': {(df['step_num'] == 'multi_step').sum():,}")


if __name__ == "__main__":
    main()
