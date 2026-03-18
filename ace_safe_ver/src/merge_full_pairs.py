from __future__ import annotations

"""
여러 SAFE pair CSV들을 하나의 full dataset으로 병합하는 스크립트.

기본 입력:
  - pairs_safe_filtered_herg_merged.csv
  - pairs_safe_metabolism_filtered_valid.csv
  - pairs_safe_sider_filtered_valid.csv

또는 --inputs 로 임의의 CSV 리스트를 지정할 수 있습니다.

병합 규칙:
  - 입력 CSV들 간 컬럼이 다를 수 있으므로, 기본은 "공통 컬럼(intersection)"만 사용하여 concat 합니다.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ACE_SAFE_DIR = SCRIPT_DIR.parent

HERG_CSV = ACE_SAFE_DIR / "pairs_safe_filtered_herg_merged.csv"
METAB_CSV = ACE_SAFE_DIR / "pairs_safe_metabolism_filtered_valid.csv"
SIDER_CSV = ACE_SAFE_DIR / "pairs_safe_sider_filtered_valid.csv"

OUT_FULL = ACE_SAFE_DIR / "pairs_safe_filtered_full.csv"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    df = pd.read_csv(path)
    print(f"[merge_full] Loaded {path.name}: {len(df):,} rows, {len(df.columns)} columns")
    return df


def merge_full(
    input_paths: list[Path],
    out_path: Path,
) -> None:
    if not input_paths:
        raise ValueError("No input CSVs provided.")

    dfs: List[pd.DataFrame] = [_load_csv(p) for p in input_paths]

    # 공통 컬럼(intersection)만 사용해서 병합
    common_cols = set(dfs[0].columns)
    for df in dfs[1:]:
        common_cols &= set(df.columns)
    common_cols_list = sorted(common_cols)
    if not common_cols_list:
        raise ValueError("No common columns across input CSVs; cannot merge.")

    trimmed = [df[common_cols_list].copy() for df in dfs]
    df_full = pd.concat(trimmed, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(out_path, index=False)
    print(f"[merge_full] Saved full dataset -> {out_path} (rows={len(df_full):,})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge multiple SAFE pair CSVs into one full CSV."
    )
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Input CSV paths. If provided, overrides --herg/--metab/--sider.",
    )
    ap.add_argument(
        "--herg",
        type=Path,
        default=HERG_CSV,
        help=f"herg merged CSV (default: {HERG_CSV})",
    )
    ap.add_argument(
        "--metab",
        type=Path,
        default=METAB_CSV,
        help=f"metabolism filtered+valid CSV (default: {METAB_CSV})",
    )
    ap.add_argument(
        "--sider",
        type=Path,
        default=SIDER_CSV,
        help=f"sider filtered+valid CSV (default: {SIDER_CSV})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_FULL,
        help=f"output full CSV (default: {OUT_FULL})",
    )
    args = ap.parse_args()

    merge_full(
        input_paths=(
            [Path(p).expanduser().resolve() for p in args.inputs]
            if args.inputs
            else [args.herg, args.metab, args.sider]
        ),
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

