from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "smiles_to_safe_ace.csv"
DEFAULT_TRAIN_OUT = PROJECT_ROOT / "smiles_to_safe_ace_train.csv"
DEFAULT_TEST_OUT = PROJECT_ROOT / "smiles_to_safe_ace_test.csv"


def split_smiles_to_safe(
    input_csv: Path,
    train_out: Path,
    test_out: Path,
    frac_train: float = 0.9,
    seed: int = 0,
) -> None:
    input_csv = input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_csv}")

    # 필수 컬럼 체크
    required = {"smiles", "canonical_smiles", "safe"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    n = len(df)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = int(round(frac_train * n))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    train_out = train_out.expanduser().resolve()
    test_out = test_out.expanduser().resolve()
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)

    print(f"[OK] Input : {input_csv}")
    print(f"[OK] Train : {train_out} ({len(df_train):,} rows)")
    print(f"[OK] Test  : {test_out} ({len(df_test):,} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Randomly split smiles_to_safe_ace.csv into train/test (default 9:1)."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    ap.add_argument(
        "--train_out",
        type=Path,
        default=DEFAULT_TRAIN_OUT,
        help=f"Output train CSV (default: {DEFAULT_TRAIN_OUT})",
    )
    ap.add_argument(
        "--test_out",
        type=Path,
        default=DEFAULT_TEST_OUT,
        help=f"Output test CSV (default: {DEFAULT_TEST_OUT})",
    )
    ap.add_argument(
        "--frac_train",
        type=float,
        default=0.9,
        help="Fraction for train split (default: 0.9 → 9:1 train:test).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    args = ap.parse_args()

    split_smiles_to_safe(
        input_csv=args.input,
        train_out=args.train_out,
        test_out=args.test_out,
        frac_train=args.frac_train,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

