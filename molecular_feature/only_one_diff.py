"""
Merged CSV에서 n_diff_features == 1 인 pair만 추려서 저장합니다.
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE / "pairs_fg_stereo_merged.csv"
OUT_CSV = BASE / "pairs_one_diff_only.csv"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    if "n_diff_features" not in df.columns:
        raise ValueError("Column n_diff_features not found. Run how_many_diff.py first.")

    one_diff = df.loc[df["n_diff_features"] == 1].copy()
    one_diff.to_csv(OUT_CSV, index=False)

    n_fg_only = ((one_diff["n_fg_diff"] == 1) & (one_diff["n_stereo_diff"] == 0)).sum()
    n_stereo_only = ((one_diff["n_fg_diff"] == 0) & (one_diff["n_stereo_diff"] == 1)).sum()

    print(f"Saved: {OUT_CSV}")
    print(f"  Total rows (n_diff_features == 1): {len(one_diff):,}")
    print(f"  FG diff only (1):  {n_fg_only:,}")
    print(f"  Stereo diff only (1): {n_stereo_only:,}")


if __name__ == "__main__":
    main()
