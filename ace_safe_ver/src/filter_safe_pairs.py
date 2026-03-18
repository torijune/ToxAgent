"""
pairs_safe_compared.csv에 molecule_safe_ver와 동일한 전처리(필터)를 적용합니다.

적용 필터 (molecule_safe_ver의 노트북 + remove_frag_len_outlier + remove_frag_num_outlier):
  1. n_common_safe != 0                    (공통 fragment 최소 1개)
  2. NOT (n_only_nontoxic_safe==0 AND n_only_toxic_safe==0)  (차이가 있는 pair만)
  3. only_toxic / only_nontoxic 에 length >= max_frag_len 인 fragment 없음 (기본 28)
  4. n_only_toxic_safe <= max_frag_num, n_only_nontoxic_safe <= max_frag_num (기본 4)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ACE_SAFE_VER_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT = ACE_SAFE_VER_DIR / "pairs_safe_compared.csv"
DEFAULT_OUTPUT = ACE_SAFE_VER_DIR / "pairs_safe_filtered.csv"

COL_TOXIC = "only_toxic_safe_fragments"
COL_NONTOXIC = "only_nontoxic_safe_fragments"
SEP = "."


def _has_any_fragment_ge(s: Any, min_length: int) -> bool:
    """SAFE fragment 문자열에 길이가 min_length 이상인 fragment가 하나라도 있으면 True."""
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return False
    parts = [p.strip() for p in t.split(SEP) if p.strip()]
    return any(len(p) >= min_length for p in parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply same filters as molecule_safe_ver to ACE pairs_safe_compared."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV (pairs_safe_compared.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output filtered CSV",
    )
    parser.add_argument(
        "--max-frag-len",
        type=int,
        default=28,
        help="Drop pair if any fragment length >= this (default 28).",
    )
    parser.add_argument(
        "--max-frag-num",
        type=int,
        default=4,
        help="Keep only rows with n_only_toxic_safe <= this and n_only_nontoxic_safe <= this (default 4).",
    )
    args = parser.parse_args()

    inp = args.input if args.input.is_absolute() else (ACE_SAFE_VER_DIR / args.input).resolve()
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_csv(inp)
    for c in [
        "n_common_safe",
        "n_only_toxic_safe",
        "n_only_nontoxic_safe",
        COL_TOXIC,
        COL_NONTOXIC,
    ]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}. Run compare_safe.py first.")

    n_start = len(df)

    # 1) n_common_safe != 0
    df = df[df["n_common_safe"].ne(0)].copy()
    n_after_common = len(df)
    print(f"Filter 1 (n_common_safe != 0): {n_start} -> {n_after_common} (-{n_start - n_after_common})")

    # 2) not (both only == 0)
    mask_both_zero = (df["n_only_nontoxic_safe"] == 0) & (df["n_only_toxic_safe"] == 0)
    df = df[~mask_both_zero].copy()
    n_after_diff = len(df)
    print(f"Filter 2 (has only_toxic or only_nontoxic): {n_after_common} -> {n_after_diff} (-{n_after_common - n_after_diff})")

    # 3) no fragment length >= max_frag_len
    mask_long = df.apply(
        lambda r: _has_any_fragment_ge(r.get(COL_TOXIC), args.max_frag_len)
        or _has_any_fragment_ge(r.get(COL_NONTOXIC), args.max_frag_len),
        axis=1,
    )
    df = df[~mask_long].copy()
    n_after_len = len(df)
    print(f"Filter 3 (no fragment length >= {args.max_frag_len}): {n_after_diff} -> {n_after_len} (-{n_after_diff - n_after_len})")

    # 4) n_only_* <= max_frag_num
    mask_too_many = (df["n_only_toxic_safe"] > args.max_frag_num) | (
        df["n_only_nontoxic_safe"] > args.max_frag_num
    )
    df = df[~mask_too_many].copy()
    n_final = len(df)
    print(f"Filter 4 (n_only_* <= {args.max_frag_num}): {n_after_len} -> {n_final} (-{n_after_len - n_final})")

    out = args.output if args.output.is_absolute() else (ACE_SAFE_VER_DIR / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out} ({n_final} rows, total removed {n_start - n_final})")


if __name__ == "__main__":
    main()
