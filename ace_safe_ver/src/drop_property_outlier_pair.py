#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property delta CSV에서 outlier pair를 drop합니다.

- 대상 컬럼: delta_abs_<descriptor>
- outlier 기준: Tukey boxplot IQR rule (1.5*IQR), 전체 데이터 기준으로 fence 계산
  Q1 = 25% quantile, Q3 = 75% quantile, IQR = Q3 - Q1
  lower_fence = Q1 - 1.5*IQR
  upper_fence = Q3 + 1.5*IQR
  outlier: x < lower_fence or x > upper_fence

여기서는 delta_abs_* (절댓값 delta)라서 실질적으로 x > upper_fence가 outlier가 됩니다.

출력:
- kept(=outlier_any drop 후 남은 pair) CSV
- dropped(=outlier_any인 pair) CSV
- thresholds CSV (컬럼별 Q1/Q3/IQR/fence)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DESCRIPTORS = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]


def _iqr_fences(x: pd.Series) -> Tuple[float, float, float, float, float]:
    """(q1, q3, iqr, lower, upper). NaN 제외."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def compute_thresholds(df: pd.DataFrame, delta_abs_cols: List[str]) -> Dict[str, Dict[str, float]]:
    thr: Dict[str, Dict[str, float]] = {}
    for c in delta_abs_cols:
        q1, q3, iqr, lower, upper = _iqr_fences(df[c])
        thr[c] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
        }
    return thr


def outlier_any_mask(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.Series:
    masks = []
    for c, t in thresholds.items():
        x = pd.to_numeric(df[c], errors="coerce")
        lower = t["lower_fence"]
        upper = t["upper_fence"]
        if np.isnan(lower) or np.isnan(upper):
            masks.append(pd.Series(False, index=df.index))
        else:
            masks.append((x < lower) | (x > upper))
    if not masks:
        return pd.Series(False, index=df.index)
    m = masks[0].copy()
    for mm in masks[1:]:
        m |= mm
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Drop pairs that are outliers in ANY selected property deltas.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("/ssd1/jueon/wj/detoxicity_model/ace_safe_ver/pairs_safe_filtered_full_herg_metabolism_sider_property_delta.csv"),
        help="Input CSV containing delta_abs_* columns",
    )
    ap.add_argument(
        "--descriptors",
        type=str,
        default=",".join(DEFAULT_DESCRIPTORS),
        help="Comma-separated descriptor names (default: MW,logP,TPSA,HBD,HBA,RotB)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input_dir>/dropped_property_outliers",
    )
    ap.add_argument(
        "--kept_csv",
        type=Path,
        default=None,
        help="Kept output CSV path (after drop). Default: <out_dir>/pairs_property_outlier_dropped.csv",
    )
    ap.add_argument(
        "--dropped_csv",
        type=Path,
        default=None,
        help="Dropped output CSV path (outliers only). Default: <out_dir>/pairs_property_outliers_only.csv",
    )
    ap.add_argument(
        "--thresholds_csv",
        type=Path,
        default=None,
        help="Thresholds output CSV path. Default: <out_dir>/outlier_thresholds_iqr_1p5.csv",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    descriptors = [d.strip() for d in str(args.descriptors).split(",") if d.strip()]
    df = pd.read_csv(in_path)

    delta_abs_cols = []
    missing = []
    for d in descriptors:
        c = f"delta_abs_{d}"
        if c in df.columns:
            delta_abs_cols.append(c)
        else:
            missing.append(c)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else (in_path.parent / "dropped_property_outliers")
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_csv = Path(args.kept_csv) if args.kept_csv is not None else (out_dir / "pairs_property_outlier_dropped.csv")
    dropped_csv = Path(args.dropped_csv) if args.dropped_csv is not None else (out_dir / "pairs_property_outliers_only.csv")
    thresholds_csv = (
        Path(args.thresholds_csv) if args.thresholds_csv is not None else (out_dir / "outlier_thresholds_iqr_1p5.csv")
    )

    # thresholds (global)
    thresholds = compute_thresholds(df, delta_abs_cols)
    thr_rows = []
    for c, t in thresholds.items():
        thr_rows.append(
            {
                "descriptor": c.replace("delta_abs_", ""),
                "column": c,
                **t,
            }
        )
    thr_df = pd.DataFrame(thr_rows)[["descriptor", "column", "q1", "q3", "iqr", "lower_fence", "upper_fence"]]
    thr_df.to_csv(thresholds_csv, index=False)

    # outlier_any mask
    out_any = outlier_any_mask(df, thresholds)

    dropped = df.loc[out_any].copy()
    kept = df.loc[~out_any].copy()

    kept.to_csv(kept_csv, index=False)
    dropped.to_csv(dropped_csv, index=False)

    total = len(df)
    out_n = int(out_any.sum())
    print(f"Input: {in_path} (rows: {total:,})")
    print(f"Outlier-any dropped: {out_n:,} ({out_n / max(total, 1):.4f})")
    print(f"Remaining kept: {len(kept):,} ({len(kept) / max(total, 1):.4f})")
    print(f"Saved kept: {kept_csv}")
    print(f"Saved dropped: {dropped_csv}")
    print(f"Saved thresholds: {thresholds_csv}")


if __name__ == "__main__":
    main()

