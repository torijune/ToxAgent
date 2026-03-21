#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
property delta CSV에 대해
- 각 property별 delta_abs_* boxplot 시각화 저장
- boxplot outlier 기준(IQR rule)과 outlier 개수/비율 요약 저장

Outlier 기준(일반적인 Tukey boxplot):
  Q1 = 25% quantile, Q3 = 75% quantile, IQR = Q3 - Q1
  lower_fence = Q1 - 1.5 * IQR
  upper_fence = Q3 + 1.5 * IQR
  outlier: x < lower_fence or x > upper_fence

이 스크립트는 delta_abs_* (절댓값 delta)이므로 실질적으로는 x > upper_fence 인 점들이 outlier가 됩니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB"]


def _iqr_outlier_thresholds(x: pd.Series) -> Tuple[float, float, float, float, float]:
    """(q1, q3, iqr, lower_fence, upper_fence) 반환. NaN은 제외."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def summarize_outliers(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    """value_col에 대해 outlier 요약 dict 반환."""
    x = pd.to_numeric(df[value_col], errors="coerce")
    q1, q3, iqr, lower, upper = _iqr_outlier_thresholds(x)
    valid = x.notna().sum()
    if valid == 0 or np.isnan(upper):
        return {
            "valid_n": int(valid),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "outlier_n": 0,
            "outlier_ratio": 0.0,
        }
    outlier_mask = (x < lower) | (x > upper)
    out_n = int(outlier_mask.sum())
    return {
        "valid_n": int(valid),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": float(lower),
        "upper_fence": float(upper),
        "outlier_n": out_n,
        "outlier_ratio": out_n / float(valid),
    }


def save_boxplots(
    df: pd.DataFrame,
    cols: List[str],
    out_dir: Path,
    title_prefix: str = "delta_abs",
    showfliers: bool = True,
    dpi: int = 200,
) -> None:
    """
    각 컬럼별 boxplot PNG 저장.
    matplotlib/seaborn은 함수 내부에서 import해서(의존성 없을 때 에러 메시지 명확히) 사용.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as e:
        raise RuntimeError(
            "matplotlib/seaborn이 필요합니다. "
            "예: pip install matplotlib seaborn"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) == 0:
            continue
        plt.figure(figsize=(6, 4))
        ax = sns.boxplot(x=x, showfliers=showfliers, color="#4C72B0")
        ax.set_title(f"{title_prefix}: {c}")
        ax.set_xlabel(c)
        plt.tight_layout()
        out_path = out_dir / f"boxplot_{c}.png"
        plt.savefig(out_path, dpi=dpi)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Boxplot visualization + IQR outlier summary for property deltas.")
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV path (e.g. pairs_safe_filtered_full_herg_metabolism_sider_property_delta.csv)",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default: <input_dir>/viz_property_delta",
    )
    ap.add_argument(
        "--no_fliers",
        action="store_true",
        help="Boxplot에서 outlier 점(fliers) 표시 안함.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG DPI (default: 200)",
    )
    args = ap.parse_args()

    in_path = args.input
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    out_dir = args.out_dir or (in_path.parent / "viz_property_delta")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # delta_abs_* 컬럼 자동 수집 (기본 8개 우선)
    cols = []
    for name in DESCRIPTOR_NAMES:
        c = f"delta_abs_{name}"
        if c in df.columns:
            cols.append(c)

    if not cols:
        # fallback: delta_abs_ prefix로 수집
        cols = [c for c in df.columns if c.startswith("delta_abs_")]

    if not cols:
        raise ValueError("No delta_abs_* columns found in CSV.")

    # 1) outlier summary
    summary_rows = []
    for c in cols:
        s = summarize_outliers(df, c)
        s["property"] = c.replace("delta_abs_", "")
        s["column"] = c
        summary_rows.append(s)
    summary_df = pd.DataFrame(summary_rows)[
        ["property", "column", "valid_n", "q1", "q3", "iqr", "lower_fence", "upper_fence", "outlier_n", "outlier_ratio"]
    ]
    summary_csv = out_dir / "outlier_summary_iqr_1p5.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 2) boxplots
    save_boxplots(
        df=df,
        cols=cols,
        out_dir=out_dir / "boxplots",
        title_prefix="delta_abs",
        showfliers=not args.no_fliers,
        dpi=args.dpi,
    )

    print(f"Saved outlier summary: {summary_csv}")
    print(f"Saved boxplots: {(out_dir / 'boxplots')}")
    print("Outlier rule: Q1 - 1.5*IQR, Q3 + 1.5*IQR (Tukey boxplot).")


if __name__ == "__main__":
    main()

