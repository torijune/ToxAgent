#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw data(commom_frage_pairs_with_smiles.csv)에서
only_toxic_safe_fragments / only_nontoxic_safe_fragments 중
하나라도 fragment 길이가 너무 긴(length >= 28) 경우 해당 pair를 제거하고 새 CSV로 저장.

- 기본: length >= 28 인 fragment가 하나라도 있으면 그 pair 제거 → commom_frage_pairs_with_smiles_no_long_frag.csv
- 옵션: --max_frag_len, --out_no_long_frag

추가로 Tukey(사분위수) 기반 outlier 통계를 frag_len_outlier_thresholds.json 에 저장한다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COL_TOXIC = "only_toxic_safe_fragments"
COL_NONTOXIC = "only_nontoxic_safe_fragments"
SEP = "."


def _collect_fragment_lengths(values) -> List[int]:
    lengths: List[int] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        parts = [p.strip() for p in s.split(SEP) if p.strip()]
        lengths.extend([len(p) for p in parts])
    return lengths


def _quantiles(arr: List[int]) -> Tuple[float, float, float]:
    """Q1, Q2(median), Q3."""
    import numpy as np

    x = np.array(arr, dtype=float)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    try:
        q1, q2, q3 = np.percentile(x, [25, 50, 75], method="linear")
    except TypeError:  # numpy 구버전 호환
        q1, q2, q3 = np.percentile(x, [25, 50, 75], interpolation="linear")
    return float(q1), float(q2), float(q3)


def compute_outlier_threshold(lengths: List[int], k: float = 1.5) -> Dict[str, Any]:
    """Tukey 기준으로 outlier 임계값과 요약 통계를 계산."""
    if not lengths:
        return {
            "N": 0,
            "Q1": None,
            "median": None,
            "Q3": None,
            "IQR": None,
            "k": k,
            "upper_fence": None,
            "outlier_min_length": None,
            "n_outliers": 0,
            "outlier_rate": 0.0,
            "min": None,
            "max": None,
            "mean": None,
        }

    import numpy as np

    q1, q2, q3 = _quantiles(lengths)
    iqr = q3 - q1
    upper_fence = q3 + k * iqr
    outlier_min_length = int(math.floor(upper_fence) + 1)

    n = len(lengths)
    n_out = int(sum(1 for v in lengths if v >= outlier_min_length))
    return {
        "N": n,
        "Q1": q1,
        "median": q2,
        "Q3": q3,
        "IQR": iqr,
        "k": k,
        "upper_fence": upper_fence,
        "outlier_min_length": outlier_min_length,
        "n_outliers": n_out,
        "outlier_rate": float(n_out / max(n, 1)),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "mean": float(np.mean(np.array(lengths, dtype=float))),
    }


def _has_any_fragment_ge(s: Any, min_length: int) -> bool:
    """SAFE fragment 문자열에 길이가 min_length 이상인 fragment가 하나라도 있으면 True."""
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return False
    parts = [p.strip() for p in t.split(SEP) if p.strip()]
    return any(len(p) >= min_length for p in parts)


def _row_has_long_fragment(row, col_toxic: str, col_nontoxic: str, min_length: int) -> bool:
    """해당 row의 only_toxic / only_nontoxic 중 하나라도 길이 >= min_length 인 fragment가 있으면 True (제거 대상)."""
    if _has_any_fragment_ge(row.get(col_toxic), min_length):
        return True
    if _has_any_fragment_ge(row.get(col_nontoxic), min_length):
        return True
    return False


def _filter_frag_string(s: Any, max_len_inclusive: int) -> str:
    """SAFE fragment 문자열에서 길이가 max_len_inclusive 초과인 fragment를 제거."""
    if s is None:
        return ""
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return ""
    parts = [p.strip() for p in t.split(SEP) if p.strip()]
    kept = [p for p in parts if len(p) <= max_len_inclusive]
    return SEP.join(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "commom_frage_pairs_with_smiles.csv"),
        help="Input CSV (default: molecule_safe_ver/commom_frage_pairs_with_smiles.csv)",
    )
    ap.add_argument("--k", type=float, default=1.5, help="Tukey fence multiplier k (default 1.5)")
    ap.add_argument(
        "--out_json",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "frag_len_outlier_thresholds.json"),
        help="Output JSON path",
    )
    ap.add_argument("--write_filtered", action="store_true", help="Write filtered CSV with outlier fragments removed.")
    ap.add_argument(
        "--out_csv",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "commom_frage_pairs_with_smiles_filtered_by_len.csv"),
        help="Output CSV path (only used with --write_filtered)",
    )
    ap.add_argument(
        "--max_frag_len",
        type=int,
        default=28,
        help="Fragment length >= this → pair 제거 (default: 28)",
    )
    ap.add_argument(
        "--out_no_long_frag",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "commom_frage_pairs_with_smiles_no_long_frag.csv"),
        help="제거 후 pair만 저장할 새 CSV 경로 (length >= max_frag_len 인 fragment가 하나라도 있으면 해당 pair 제외)",
    )
    args = ap.parse_args()

    import pandas as pd

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    for c in [COL_TOXIC, COL_NONTOXIC]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # pair 제거: only_toxic / only_nontoxic 에 length >= max_frag_len 인 fragment가 하나라도 있으면 제거
    mask_remove = df.apply(
        lambda r: _row_has_long_fragment(r, COL_TOXIC, COL_NONTOXIC, args.max_frag_len),
        axis=1,
    )
    df_kept = df[~mask_remove].copy()
    n_removed = mask_remove.sum()
    n_kept = len(df_kept)
    out_no_long = Path(args.out_no_long_frag)
    out_no_long.parent.mkdir(parents=True, exist_ok=True)
    df_kept.to_csv(out_no_long, index=False)
    print("\n" + "=" * 60)
    print("Pair 제거 (fragment length >= {} 인 경우 해당 pair 제외)".format(args.max_frag_len))
    print("=" * 60)
    print(f"  제거된 pair 수: {n_removed}")
    print(f"  남은 pair 수:   {n_kept}")
    print(f"  저장: {out_no_long}")

    toxic_lengths = _collect_fragment_lengths(df[COL_TOXIC].values)
    nontoxic_lengths = _collect_fragment_lengths(df[COL_NONTOXIC].values)
    combined_lengths = toxic_lengths + nontoxic_lengths

    toxic_thr = compute_outlier_threshold(toxic_lengths, k=args.k)
    nontoxic_thr = compute_outlier_threshold(nontoxic_lengths, k=args.k)
    combined_thr = compute_outlier_threshold(combined_lengths, k=args.k)

    print("\n" + "=" * 90)
    print("Fragment length outlier 기준 (Tukey: Q3 + k*IQR)")
    print("=" * 90)
    print(f"k = {args.k}")
    print("\n[only_toxic_safe_fragments]")
    print(f"  Q1={toxic_thr['Q1']:.2f}, Q3={toxic_thr['Q3']:.2f}, IQR={toxic_thr['IQR']:.2f}, upper_fence={toxic_thr['upper_fence']:.2f}")
    print(f"  ==> outlier 기준: length >= {toxic_thr['outlier_min_length']} (문자 수)")
    print(f"  outliers: {toxic_thr['n_outliers']}/{toxic_thr['N']} ({toxic_thr['outlier_rate']*100:.2f}%)")

    print("\n[only_nontoxic_safe_fragments]")
    print(f"  Q1={nontoxic_thr['Q1']:.2f}, Q3={nontoxic_thr['Q3']:.2f}, IQR={nontoxic_thr['IQR']:.2f}, upper_fence={nontoxic_thr['upper_fence']:.2f}")
    print(f"  ==> outlier 기준: length >= {nontoxic_thr['outlier_min_length']} (문자 수)")
    print(f"  outliers: {nontoxic_thr['n_outliers']}/{nontoxic_thr['N']} ({nontoxic_thr['outlier_rate']*100:.2f}%)")

    print("\n[combined (toxic+nontoxic)]")
    print(f"  Q1={combined_thr['Q1']:.2f}, Q3={combined_thr['Q3']:.2f}, IQR={combined_thr['IQR']:.2f}, upper_fence={combined_thr['upper_fence']:.2f}")
    print(f"  ==> outlier 기준: length >= {combined_thr['outlier_min_length']} (문자 수)")
    print(f"  outliers: {combined_thr['n_outliers']}/{combined_thr['N']} ({combined_thr['outlier_rate']*100:.2f}%)")

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "k": args.k,
                "only_toxic_safe_fragments": toxic_thr,
                "only_nontoxic_safe_fragments": nontoxic_thr,
                "combined": combined_thr,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved: {out_json_path}")

    if args.write_filtered:
        # 기준은 각 컬럼별 outlier_min_length를 사용 (length >= outlier_min_length 는 제거)
        tox_keep_max = int(toxic_thr["outlier_min_length"] - 1) if toxic_thr["outlier_min_length"] else 10**9
        non_keep_max = int(nontoxic_thr["outlier_min_length"] - 1) if nontoxic_thr["outlier_min_length"] else 10**9

        df[f"{COL_TOXIC}_len_filtered"] = df[COL_TOXIC].apply(lambda x: _filter_frag_string(x, tox_keep_max))
        df[f"{COL_NONTOXIC}_len_filtered"] = df[COL_NONTOXIC].apply(lambda x: _filter_frag_string(x, non_keep_max))

        out_csv_path = Path(args.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv_path, index=False)
        print(f"Saved filtered CSV: {out_csv_path}")


if __name__ == "__main__":
    main()

