#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw data(commom_frage_pairs_with_smiles_no_long_frag.csv)에서
only_toxic_safe_fragments, only_nontoxic_safe_fragments를 dot(".")으로 분리했을 때
fragment 개수가 5개 이상인 pair는 제거하고 새 CSV로 저장.

- 기본: 두 컬럼 모두 fragment 개수 < 5 인 행만 유지.
- 옵션: --max_frag_num (기본 4, 즉 5개 이상이면 제거), --output 경로.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COL_TOXIC = "only_toxic_safe_fragments"
COL_NONTOXIC = "only_nontoxic_safe_fragments"
SEP = "."


def count_fragments(s: str) -> int:
    """Dot으로 구분된 SAFE 문자열의 fragment 개수 (빈 토큰 제외)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return 0
    s = str(s).strip()
    if not s:
        return 0
    parts = [p.strip() for p in s.replace(" ", "").split(SEP) if p.strip()]
    return len(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop pairs where only_toxic or only_nontoxic has >= max_frag_num fragments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "commom_frage_pairs_with_smiles_no_long_frag.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--max_frag_num",
        type=int,
        default=4,
        help="Keep only rows where both columns have fragment count <= this (default 4 → drop 5+).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: input stem + _max{N}frag.csv",
    )
    args = parser.parse_args()

    inp = args.input
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    df = pd.read_csv(inp)
    for col in [COL_TOXIC, COL_NONTOXIC]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    n_toxic = df[COL_TOXIC].apply(count_fragments)
    n_nontoxic = df[COL_NONTOXIC].apply(count_fragments)

    # 5개 이상이면 제거 → max_frag_num 초과이면 제거 (기본 max_frag_num=4 이므로 5개 이상 제거)
    mask_drop = (n_toxic > args.max_frag_num) | (n_nontoxic > args.max_frag_num)
    df_kept = df[~mask_drop].copy()

    out = args.output
    if out is None:
        out = inp.parent / f"{inp.stem}_max{args.max_frag_num}frag.csv"

    df_kept.to_csv(out, index=False)
    n_removed = mask_drop.sum()
    print(f"Removed {n_removed} pairs (fragment count > {args.max_frag_num} in either column).")
    print(f"Kept {len(df_kept)} pairs -> {out}")


if __name__ == "__main__":
    main()
