#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Property delta CSV에서 outlier pair 카운트.

대상 descriptor (default):
  DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]

Outlier 기준: Tukey boxplot IQR rule (1.5*IQR)
  Q1 = 25% quantile, Q3 = 75% quantile, IQR = Q3 - Q1
  lower_fence = Q1 - 1.5 * IQR
  upper_fence = Q3 + 1.5 * IQR
  outlier: x < lower_fence or x > upper_fence

여기서는 delta_abs_* 컬럼을 사용하므로 주로 x > upper_fence가 outlier입니다.

출력:
- 전체 기준(outlier_any) pair 수, drop 후 남는 수
- endpoint 기준으로 동일한 통계 테이블 생성 및 CSV 저장
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]


def iqr_fences(x: pd.Series) -> Tuple[float, float, float, float, float]:
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


def compute_global_thresholds(df: pd.DataFrame, delta_abs_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """컬럼별 IQR fence 계산."""
    thr: Dict[str, Dict[str, float]] = {}
    for c in delta_abs_cols:
        q1, q3, iqr, lower, upper = iqr_fences(df[c])
        thr[c] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
        }
    return thr


def build_outlier_mask(df: pd.DataFrame, thresholds: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """각 컬럼별 outlier mask 및 any mask 생성."""
    masks = {}
    for c, t in thresholds.items():
        x = pd.to_numeric(df[c], errors="coerce")
        lower = t["lower_fence"]
        upper = t["upper_fence"]
        if np.isnan(lower) or np.isnan(upper):
            masks[c] = pd.Series(False, index=df.index)
        else:
            masks[c] = (x < lower) | (x > upper)
    mask_df = pd.DataFrame(masks)
    mask_df["outlier_any"] = mask_df.any(axis=1)
    return mask_df


def summarize_counts(df: pd.DataFrame, outlier_any: pd.Series) -> Dict[str, float]:
    total = int(len(df))
    out_n = int(outlier_any.sum())
    remain = total - out_n
    return {
        "total_pairs": total,
        "outlier_pairs_any": out_n,
        "outlier_ratio_any": (out_n / total) if total else 0.0,
        "remaining_after_drop": remain,
        "remaining_ratio": (remain / total) if total else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Count outlier pairs (any of selected descriptors) overall and by endpoint.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("/ssd1/jueon/wj/detoxicity_model/ace_safe_ver/pairs_safe_filtered_full_herg_metabolism_sider_property_delta.csv"),
        help="Input CSV with delta_abs_* columns",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for summaries. Default: <input_dir>/outlier_counts",
    )
    ap.add_argument(
        "--endpoint_col",
        type=str,
        default="endpoint",
        help="Endpoint column name (default: endpoint)",
    )
    ap.add_argument(
        "--descriptors",
        type=str,
        default=",".join(DESCRIPTOR_NAMES),
        help="Comma-separated descriptor names (default: MW,logP,TPSA,HBD,HBA,RotB)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else (in_path.parent / "outlier_counts")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    endpoint_col = args.endpoint_col
    if endpoint_col not in df.columns:
        raise ValueError(f"Missing endpoint column: {endpoint_col}")

    descriptors = [d.strip() for d in str(args.descriptors).split(",") if d.strip()]
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

    # 1) global thresholds (전체 기준)
    thresholds = compute_global_thresholds(df, delta_abs_cols)

    # 2) outlier masks
    mask_df = build_outlier_mask(df, thresholds)
    outlier_any = mask_df["outlier_any"]

    # 3) overall summary
    overall = summarize_counts(df, outlier_any)
    overall_df = pd.DataFrame([overall])
    overall_csv = out_dir / "overall_outlier_counts.csv"
    overall_df.to_csv(overall_csv, index=False)

    # 4) endpoint summary
    rows = []
    for endpoint, g in df.groupby(endpoint_col, dropna=False):
        idx = g.index
        s = summarize_counts(g, outlier_any.loc[idx])
        s["endpoint"] = endpoint
        rows.append(s)
    ep_df = pd.DataFrame(rows).sort_values(["outlier_pairs_any", "total_pairs"], ascending=[False, False])
    ep_csv = out_dir / "endpoint_outlier_counts.csv"
    ep_df.to_csv(ep_csv, index=False)

    # 5) thresholds 저장
    thr_rows = []
    for c, t in thresholds.items():
        thr_rows.append({
            "column": c,
            "descriptor": c.replace("delta_abs_", ""),
            **t,
        })
    thr_df = pd.DataFrame(thr_rows)[["descriptor", "column", "q1", "q3", "iqr", "lower_fence", "upper_fence"]]
    thr_csv = out_dir / "outlier_thresholds_iqr_1p5.csv"
    thr_df.to_csv(thr_csv, index=False)

    # 6) drop 후 남는 데이터 저장(원하면 사용)
    kept_df = df.loc[~outlier_any].copy()
    kept_csv = out_dir / "pairs_dropped_outliers_any.csv"
    kept_df.to_csv(kept_csv, index=False)

    print(f"Saved: {overall_csv}")
    print(f"Saved: {ep_csv}")
    print(f"Saved: {thr_csv}")
    print(f"Saved (kept after drop): {kept_csv}")
    print(f"Overall: total={overall['total_pairs']:,}, outlier_any={overall['outlier_pairs_any']:,}, remaining={overall['remaining_after_drop']:,}")


if __name__ == "__main__":
    main()

