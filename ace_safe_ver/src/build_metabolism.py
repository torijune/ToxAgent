from __future__ import annotations

"""
metabolism_ver에서 생성한 MolecularACE 스타일 pairs를
ACE SAFE 파이프라인(pairs_to_safe → compare_safe → filter_safe_pairs)과
동일한 방식으로 SAFE pairing 및 필터링하는 스크립트.

입력:
  - data/metabolism_ver/pairs.csv (또는 --pairs)
  - data/smiles_to_safe.csv (또는 --mapping)

출력 (기본, ace_safe_ver 루트):
  - pairs_safe_metabolism.csv
  - pairs_safe_metabolism_compared.csv
  - pairs_safe_metabolism_filtered.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ACE_SAFE_VER_DIR = SCRIPT_DIR.parent
if str(ACE_SAFE_VER_DIR) not in sys.path:
    sys.path.insert(0, str(ACE_SAFE_VER_DIR))
import ace_local  # noqa: E402

DEFAULT_METABOLISM_PAIRS = ace_local.DEFAULT_METABOLISM_PAIRS_CSV
DEFAULT_SMILES_TO_SAFE_CSV = ace_local.DEFAULT_SMILES_TO_SAFE_CSV

# 출력 경로 기본값
DEFAULT_OUT_PAIRS_SAFE = ACE_SAFE_VER_DIR / "pairs_safe_metabolism.csv"
DEFAULT_OUT_COMPARED = ACE_SAFE_VER_DIR / "pairs_safe_metabolism_compared.csv"
DEFAULT_OUT_FILTERED = ACE_SAFE_VER_DIR / "pairs_safe_metabolism_filtered.csv"

SEP = "."


def load_safe_mapping(mapping_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    smiles_to_safe.csv 를 읽어 (원본 SMILES → SAFE), (canonical_smiles → SAFE) 딕셔너리 반환.
    pairs_to_safe.py 의 로직과 동일.
    """
    map_df = pd.read_csv(mapping_path)
    if "smiles" not in map_df.columns or "safe" not in map_df.columns:
        raise ValueError(
            f"Mapping CSV must have columns 'smiles' and 'safe'. Found: {list(map_df.columns)}"
        )
    smiles_to_safe = dict(
        zip(map_df["smiles"].astype(str).str.strip(), map_df["safe"].fillna("").astype(str))
    )
    canon_to_safe: dict[str, str] = {}
    if "canonical_smiles" in map_df.columns:
        canon_to_safe = dict(
            zip(
                map_df["canonical_smiles"].astype(str).str.strip(),
                map_df["safe"].fillna("").astype(str),
            )
        )
        canon_to_safe = {k: v for k, v in canon_to_safe.items() if k and str(k) != "nan"}
    return smiles_to_safe, canon_to_safe


def lookup_safe(
    smiles_series: pd.Series,
    smiles_to_safe: dict[str, str],
    canon_to_safe: dict[str, str],
) -> list[str]:
    """SMILES 시리즈에 대해 SAFE 문자열 리스트 반환 (없으면 빈 문자열)."""
    out: list[str] = []
    for s in smiles_series:
        s = str(s).strip() if pd.notna(s) else ""
        safe_str = smiles_to_safe.get(s)
        if safe_str is None and s:
            safe_str = canon_to_safe.get(s, "")
        out.append(safe_str if safe_str is not None else "")
    return out


def safe_to_fragments(safe_str: Any) -> set[str]:
    """SAFE 문자열을 dot으로 split한 fragment set (빈 토큰 제거)."""
    if pd.isna(safe_str) or not str(safe_str).strip():
        return set()
    return {s.strip() for s in str(safe_str).split(SEP) if s.strip()}


def compare_fragments(toxic_safe: Any, nontoxic_safe: Any) -> tuple[set[str], set[str], set[str]]:
    """
    toxic_safe, nontoxic_safe 문자열에 대해
    공통 fragment set, toxic 전용, nontoxic 전용을 계산.
    """
    t_set = safe_to_fragments(toxic_safe)
    n_set = safe_to_fragments(nontoxic_safe)
    common = t_set & n_set
    only_toxic = t_set - n_set
    only_nontoxic = n_set - t_set
    return common, only_toxic, only_nontoxic


def _has_any_fragment_ge(s: Any, min_length: int) -> bool:
    """SAFE fragment 문자열에 길이가 min_length 이상인 fragment가 하나라도 있으면 True."""
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return False
    parts = [p.strip() for p in t.split(SEP) if p.strip()]
    return any(len(p) >= min_length for p in parts)


def run(
    pairs_path: Path,
    mapping_path: Path,
    out_pairs_safe: Path,
    out_compared: Path,
    out_filtered: Path,
    max_frag_len: int,
    max_frag_num: int,
) -> None:
    # 1) 매핑 로드
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found: {mapping_path}\n"
            "Place smiles_to_safe.csv under ace_safe_ver/data/ or set --mapping."
        )
    if not pairs_path.exists():
        raise FileNotFoundError(f"Metabolism pairs CSV not found: {pairs_path}")

    print(f"[metabolism] Loading mapping: {mapping_path}")
    smiles_to_safe, canon_to_safe = load_safe_mapping(mapping_path)

    # 2) metabolism pairs 로드 및 SAFE 부착 (pairs_safe_metabolism.csv)
    print(f"[metabolism] Loading pairs: {pairs_path}")
    df = pd.read_csv(pairs_path)
    for col in ["toxic_smiles", "nontoxic_smiles"]:
        if col not in df.columns:
            raise ValueError(f"Pairs CSV must have column '{col}'. Found: {list(df.columns)}")

    n_rows = len(df)
    df = df.assign(
        toxic_safe=lookup_safe(df["toxic_smiles"], smiles_to_safe, canon_to_safe),
        nontoxic_safe=lookup_safe(df["nontoxic_smiles"], smiles_to_safe, canon_to_safe),
    )

    toxic_miss = (df["toxic_safe"] == "").sum()
    nontoxic_miss = (df["nontoxic_safe"] == "").sum()
    print(f"[metabolism] Rows with missing toxic_safe:   {toxic_miss} / {n_rows}")
    print(f"[metabolism] Rows with missing nontoxic_safe: {nontoxic_miss} / {n_rows}")

    out_pairs_safe.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_pairs_safe, index=False)
    print(f"[metabolism] Saved pairs_safe_metabolism -> {out_pairs_safe}")

    # 3) SAFE fragment 비교 (compare_safe 로직)
    print("[metabolism] Comparing SAFE fragments (common / only_toxic / only_nontoxic)")
    n = len(df)
    common_list: list[str] = []
    only_toxic_list: list[str] = []
    only_nontoxic_list: list[str] = []
    n_common_list: list[int] = []
    n_only_toxic_list: list[int] = []
    n_only_nontoxic_list: list[int] = []
    toxic_fragments_str_list: list[str] = []
    nontoxic_fragments_str_list: list[str] = []

    for _, row in df.iterrows():
        toxic_safe = row.get("toxic_safe", "")
        nontoxic_safe = row.get("nontoxic_safe", "")
        common, only_toxic, only_nontoxic = compare_fragments(toxic_safe, nontoxic_safe)

        toxic_fragments_str_list.append(SEP.join(sorted(safe_to_fragments(toxic_safe))))
        nontoxic_fragments_str_list.append(SEP.join(sorted(safe_to_fragments(nontoxic_safe))))
        common_list.append(SEP.join(sorted(common)))
        only_toxic_list.append(SEP.join(sorted(only_toxic)))
        only_nontoxic_list.append(SEP.join(sorted(only_nontoxic)))
        n_common_list.append(len(common))
        n_only_toxic_list.append(len(only_toxic))
        n_only_nontoxic_list.append(len(only_nontoxic))

    df_comp = df.assign(
        toxic_safe_fragments=toxic_fragments_str_list,
        nontoxic_safe_fragments=nontoxic_fragments_str_list,
        common_safe_fragments=common_list,
        only_toxic_safe_fragments=only_toxic_list,
        only_nontoxic_safe_fragments=only_nontoxic_list,
        n_common_safe=n_common_list,
        n_only_toxic_safe=n_only_toxic_list,
        n_only_nontoxic_safe=n_only_nontoxic_list,
    )

    df_comp.to_csv(out_compared, index=False)
    print(f"[metabolism] Saved pairs_safe_metabolism_compared -> {out_compared} ({n} rows)")

    # 4) 필터 적용 (filter_safe_pairs.py 와 동일)
    print("[metabolism] Applying SAFE fragment filters (common!=0, diff!=0, length/num thresholds)")
    n_start = len(df_comp)

    # 4-1) n_common_safe != 0
    df_f = df_comp[df_comp["n_common_safe"].ne(0)].copy()
    n_after_common = len(df_f)
    print(
        f"  Filter 1 (n_common_safe != 0): {n_start} -> {n_after_common} "
        f"(-{n_start - n_after_common})"
    )

    # 4-2) not (both only == 0)
    mask_both_zero = (df_f["n_only_nontoxic_safe"] == 0) & (df_f["n_only_toxic_safe"] == 0)
    df_f = df_f[~mask_both_zero].copy()
    n_after_diff = len(df_f)
    print(
        f"  Filter 2 (has only_toxic or only_nontoxic): "
        f"{n_after_common} -> {n_after_diff} (-{n_after_common - n_after_diff})"
    )

    # 4-3) no fragment length >= max_frag_len
    mask_long = df_f.apply(
        lambda r: _has_any_fragment_ge(r.get("only_toxic_safe_fragments"), max_frag_len)
        or _has_any_fragment_ge(r.get("only_nontoxic_safe_fragments"), max_frag_len),
        axis=1,
    )
    df_f = df_f[~mask_long].copy()
    n_after_len = len(df_f)
    print(
        f"  Filter 3 (no fragment length >= {max_frag_len}): "
        f"{n_after_diff} -> {n_after_len} (-{n_after_diff - n_after_len})"
    )

    # 4-4) n_only_* <= max_frag_num
    mask_too_many = (df_f["n_only_toxic_safe"] > max_frag_num) | (
        df_f["n_only_nontoxic_safe"] > max_frag_num
    )
    df_f = df_f[~mask_too_many].copy()
    n_final = len(df_f)
    print(
        f"  Filter 4 (n_only_* <= {max_frag_num}): "
        f"{n_after_len} -> {n_final} (-{n_after_len - n_final})"
    )

    out_filtered.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(out_filtered, index=False)
    print(
        f"[metabolism] Saved pairs_safe_metabolism_filtered -> {out_filtered} "
        f"({n_final} rows, total removed {n_start - n_final})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build SAFE-based metabolism pairs with the same pipeline as ACE SAFE:\n"
            "pairs_to_safe -> compare_safe -> filter_safe_pairs"
        )
    )
    ap.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_METABOLISM_PAIRS,
        help=f"Metabolism pairs CSV (default: {DEFAULT_METABOLISM_PAIRS})",
    )
    ap.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_SMILES_TO_SAFE_CSV,
        help="smiles→SAFE mapping CSV (default: ace_safe_ver/data/smiles_to_safe.csv)",
    )
    ap.add_argument(
        "--out_pairs_safe",
        type=Path,
        default=DEFAULT_OUT_PAIRS_SAFE,
        help=f"Output SAFE-attached pairs CSV (default: {DEFAULT_OUT_PAIRS_SAFE})",
    )
    ap.add_argument(
        "--out_compared",
        type=Path,
        default=DEFAULT_OUT_COMPARED,
        help=f"Output compared CSV (default: {DEFAULT_OUT_COMPARED})",
    )
    ap.add_argument(
        "--out_filtered",
        type=Path,
        default=DEFAULT_OUT_FILTERED,
        help=f"Output filtered CSV (default: {DEFAULT_OUT_FILTERED})",
    )
    ap.add_argument(
        "--max-frag-len",
        type=int,
        default=28,
        help="Drop pair if any fragment length >= this (default 28).",
    )
    ap.add_argument(
        "--max-frag-num",
        type=int,
        default=4,
        help=(
            "Keep only rows with n_only_toxic_safe <= this and "
            "n_only_nontoxic_safe <= this (default 4)."
        ),
    )
    args = ap.parse_args()

    run(
        pairs_path=args.pairs,
        mapping_path=args.mapping,
        out_pairs_safe=args.out_pairs_safe,
        out_compared=args.out_compared,
        out_filtered=args.out_filtered,
        max_frag_len=args.max_frag_len,
        max_frag_num=args.max_frag_num,
    )


if __name__ == "__main__":
    main()

