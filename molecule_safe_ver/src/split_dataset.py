"""
Endpoint-wise scaffold split for SAFE pair dataset.

Input
-----
- commom_frage_pairs_with_smiles_no_long_frag_max4frag.csv

This script groups rows by (dataset_name, endpoint) and performs a Bemis–Murcko
scaffold split (ScaffoldSplitter) within each group to produce train/valid/test
CSV files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
# molecule_safe_ver 루트 디렉토리
PROJECT_ROOT = SCRIPT_DIR.parent

# Local import (avoid package install requirement)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from spliter import ScaffoldSplitter
except Exception as e:  # pragma: no cover
    raise ImportError(
        f"ScaffoldSplitter import 실패: {e}. 경로를 확인하세요: {SCRIPT_DIR / 'spliter.py'}"
    )

DEFAULT_INPUT = PROJECT_ROOT / "commom_frage_pairs_with_smiles_no_long_frag_max4frag.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "splits" / "scaffold_by_endpoint"


def _sanitize_for_path(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "unknown"
    # Replace path-hostile characters
    for ch in ["/", "\\", ":", ";", "|", "?", "*", "<", ">", "\"", "\n", "\t"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._ ") or "unknown"


def _choose_smiles_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--smiles_col={requested!r} 컬럼이 CSV에 없습니다.")
        return requested
    # Prefer decoded/canonical-ish smiles if present
    for c in ["toxic_safe_decoded_smiles", "toxic_smiles"]:
        if c in df.columns:
            return c
    raise ValueError("SMILES 컬럼을 찾을 수 없습니다. (--smiles_col로 지정하세요)")


def _scaffold_split_indices(
    smiles_list: list[str],
    frac_train: float,
    frac_valid: float,
    frac_test: float,
    seed: Optional[int],
) -> Tuple[list[int], list[int], list[int]]:
    class _SimpleDataset:
        def __init__(self, ids: list[str]):
            self.ids = ids

        def __len__(self) -> int:
            return len(self.ids)

    ds = _SimpleDataset(smiles_list)
    splitter = ScaffoldSplitter()
    train_inds, valid_inds, test_inds = splitter.split(
        ds,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed,
    )
    return list(train_inds), list(valid_inds), list(test_inds)


def run(
    input_csv: Path,
    out_dir: Path,
    smiles_col: Optional[str],
    frac_train: float,
    frac_valid: float,
    frac_test: float,
    seed: Optional[int],
) -> None:
    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "dataset_name" not in df.columns or "endpoint" not in df.columns:
        raise ValueError("CSV must have columns: dataset_name, endpoint")

    smiles_col = _choose_smiles_col(df, smiles_col)
    required = ["dataset_name", "endpoint", smiles_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Ensure string
    df[smiles_col] = df[smiles_col].fillna("").astype(str)

    groups = df.groupby(["dataset_name", "endpoint"], dropna=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for (dataset_name, endpoint), g in groups:
        g = g.copy().reset_index(drop=True)
        # Filter empty SMILES rows
        mask = g[smiles_col].str.strip().astype(bool)
        g_valid = g.loc[mask].copy().reset_index(drop=True)
        n_total = len(g)
        n_valid = len(g_valid)

        dataset_dir = out_dir / _sanitize_for_path(dataset_name)
        endpoint_dir = dataset_dir / _sanitize_for_path(endpoint)
        endpoint_dir.mkdir(parents=True, exist_ok=True)

        if n_valid == 0:
            # Save empty splits for completeness
            (endpoint_dir / "train.csv").write_text("", encoding="utf-8")
            (endpoint_dir / "valid.csv").write_text("", encoding="utf-8")
            (endpoint_dir / "test.csv").write_text("", encoding="utf-8")
            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "endpoint": endpoint,
                    "smiles_col": smiles_col,
                    "n_total": n_total,
                    "n_valid_smiles": n_valid,
                    "n_train": 0,
                    "n_valid": 0,
                    "n_test": 0,
                    "note": "no valid smiles",
                }
            )
            continue

        smiles_list = g_valid[smiles_col].astype(str).tolist()
        train_inds, valid_inds, test_inds = _scaffold_split_indices(
            smiles_list=smiles_list,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=seed,
        )

        df_train = g_valid.iloc[train_inds].copy()
        df_valid = g_valid.iloc[valid_inds].copy()
        df_test = g_valid.iloc[test_inds].copy()

        df_train.to_csv(endpoint_dir / "train.csv", index=False)
        df_valid.to_csv(endpoint_dir / "valid.csv", index=False)
        df_test.to_csv(endpoint_dir / "test.csv", index=False)

        # Also save a single file with split labels
        g_out = g_valid.copy()
        g_out["split"] = ""
        g_out.loc[train_inds, "split"] = "train"
        g_out.loc[valid_inds, "split"] = "valid"
        g_out.loc[test_inds, "split"] = "test"
        g_out.to_csv(endpoint_dir / "all_with_split.csv", index=False)

        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "endpoint": endpoint,
                "smiles_col": smiles_col,
                "n_total": n_total,
                "n_valid_smiles": n_valid,
                "n_train": len(df_train),
                "n_valid": len(df_valid),
                "n_test": len(df_test),
                "note": "",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset_name", "endpoint"], kind="stable"
    )
    summary_path = out_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved split summary -> {summary_path}")
    print(f"Output dir -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Endpoint-wise ScaffoldSplitter train/valid/test split for SAFE pair CSV."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--smiles_col",
        type=str,
        default=None,
        help="Which SMILES column to use for scaffold split (default: toxic_safe_decoded_smiles if exists else toxic_smiles).",
    )
    ap.add_argument("--frac_train", type=float, default=0.8)
    ap.add_argument("--frac_valid", type=float, default=0.1)
    ap.add_argument("--frac_test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run(
        input_csv=args.input,
        out_dir=args.out_dir,
        smiles_col=args.smiles_col,
        frac_train=args.frac_train,
        frac_valid=args.frac_valid,
        frac_test=args.frac_test,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

