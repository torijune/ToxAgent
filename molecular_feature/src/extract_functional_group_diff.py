"""
FG diff 추출 (strict, 1번만): 두 SMILES **모두** functional group이 있을 때만 고려.

1. FG 이름 set이 다르면 → 다른 FG를 unique_fg로, 공통 FG는 atom index 비교
   1-2-1-1: 공통 FG 중 atom index가 다르면 unique_fg에 추가
   1-2-1-2: atom index까지 같으면 해당 FG는 패스
2. FG 이름 set이 같으면 → 동일 FG끼리 atom index 비교
   1-1-1: atom index가 하나라도 다르면 차이 있음, 해당 FG 이름 + atom index를 unique_fg로
   1-1-2: 모두 같으면 diff 없음 (패스)

출력: pairs_fg_diff_only.csv (둘 다 FG 있는 pair 중 has_fg_diff=True 만)
"""
from pathlib import Path
import ast
import json
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE / "functional_group" / "pairs_with_fg.csv"
OUT_DIR = BASE / "functional_group"
OUT_CSV = OUT_DIR / "pairs_fg_diff_only.csv"


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
        # value: list of (atom indices). CSV에서 tuple이 list로 올 수 있음 → tuple로 정규화
        return {
            k: [tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in (v if isinstance(v, list) else [v])]
            for k, v in out.items()
        }
    except Exception:
        return {}


def _normalize_indices(indices_list):
    """리스트 내 각 원소를 tuple로, 정렬 가능하게."""
    return sorted(tuple(x) if isinstance(x, (list, tuple)) else (x,) for x in indices_list)


def _same_atom_indices(tx_indices, nt_indices):
    """동일 FG의 atom index 리스트가 같은지 (길이·구성 동일)."""
    a = _normalize_indices(tx_indices or [])
    b = _normalize_indices(nt_indices or [])
    return a == b


def compute_fg_diff_strict(tx_names: list, nt_names: list, tx_full: dict, nt_full: dict):
    """
    둘 다 FG가 있을 때만 호출. (한쪽이라도 FG 없으면 None 반환)

    Returns:
        (has_fg_diff: bool, unique_fg: list of dict)

    unique_fg 각 항목:
        {"fg_name", "reason", "toxic_atom_indices", "nontoxic_atom_indices"}

    현재 버전에서는 **atom index 차이만 있는 경우는 diff로 보지 않는다.**
    즉, FG 이름이 한쪽에만 존재하는 경우만 diff로 간주한다.

    reason:
        - "name_only_in_toxic"
        - "name_only_in_nontoxic"
        - (과거에는 "atom_index_diff"도 있었지만 SMILES 표현 차이만으로
          인덱스가 달라지는 경우를 diff로 잘못 잡을 수 있어 제거했다.)
    """
    if not tx_names and not nt_names:
        return False, []
    # 둘 다 FG가 있어야 strict
    if not tx_names or not nt_names:
        return None, []  # skip (strict에서는 제외)

    tx_set = set(tx_names)
    nt_set = set(nt_names)
    only_in_tx = tx_set - nt_set
    only_in_nt = nt_set - tx_set
    common = tx_set & nt_set

    unique_fg = []

    # 1-2. FG 이름이 다른 것들 → unique_fg
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

    # 공통 FG에 대해서는 이제 atom index 차이만 있는 경우는 diff로 보지 않는다.
    # (SMILES 길이나 부분구조 표현 방식 때문에 인덱스만 달라지는 경우가 많기 때문)
    # 예전 로직:
    # for fg in common:
    #     ti = tx_full.get(fg, [])
    #     ni = nt_full.get(fg, [])
    #     if not _same_atom_indices(ti, ni):
    #         unique_fg.append({
    #             "fg_name": fg,
    #             "reason": "atom_index_diff",
    #             "toxic_atom_indices": _normalize_indices(ti),
    #             "nontoxic_atom_indices": _normalize_indices(ni),
    #         })

    has_fg_diff = len(unique_fg) > 0
    return has_fg_diff, unique_fg


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    for col in ["toxic_fg_names", "nontoxic_fg_names", "toxic_fg_full", "nontoxic_fg_full"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    has_fg_diff_list = []
    unique_fg_list = []

    for _, row in df.iterrows():
        tx_names = _parse_list(row["toxic_fg_names"])
        nt_names = _parse_list(row["nontoxic_fg_names"])
        tx_full = _parse_fg_full(row["toxic_fg_full"])
        nt_full = _parse_fg_full(row["nontoxic_fg_full"])

        has_diff, unique_fg = compute_fg_diff_strict(tx_names, nt_names, tx_full, nt_full)

        if has_diff is None:
            # strict: 둘 다 FG 없거나 한쪽만 FG 있음 → diff 아님 (제외)
            has_fg_diff_list.append(False)
            unique_fg_list.append([])
        else:
            has_fg_diff_list.append(has_diff)
            unique_fg_list.append(unique_fg)

    df = df.copy()
    df["has_fg_diff"] = has_fg_diff_list
    df["unique_fg"] = [json.dumps(u, ensure_ascii=False) for u in unique_fg_list]

    out_df = df.loc[df["has_fg_diff"]].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    n_both_fg = sum(1 for _, r in df.iterrows() if _parse_list(r["toxic_fg_names"]) and _parse_list(r["nontoxic_fg_names"]))
    print(f"Saved: {OUT_CSV}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Rows with both having FG (strict pool): {n_both_fg:,}")
    print(f"  Rows with FG diff (strict): {len(out_df):,}")


if __name__ == "__main__":
    main()
