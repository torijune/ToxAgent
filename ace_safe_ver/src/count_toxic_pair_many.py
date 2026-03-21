#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merged_test.csv(또는 임의 pair CSV)에서
toxic_smiles 기준으로 여러 개의 nontoxic_smiles와 pairing된 toxic_smiles 통계를 계산합니다.

요약 (endpoint 내부 기준):
- 같은 endpoint 내부에서 n_unique_nontoxic >= K 인 (endpoint, toxic_smiles) 개수
- 그 (endpoint, toxic_smiles)들이 차지하는 row 개수 (전체 row 중 몇 %)
- endpoint별 동일 통계(기본 저장)
- (옵션) (endpoint, toxic) 상세 테이블 저장 (unique nontoxic 개수, row 수)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def main() -> None:
    ap = argparse.ArgumentParser(description="Count toxic_smiles paired with multiple nontoxic_smiles WITHIN each endpoint.")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("/ssd1/jueon/wj/detoxicity_model/ace_safe_ver/splits/scaffold_by_endpoint_property_outlier_dropped/merged_test.csv"),
        help="Input CSV path",
    )
    ap.add_argument("--toxic_col", type=str, default="toxic_smiles", help="toxic smiles column name")
    ap.add_argument("--nontoxic_col", type=str, default="nontoxic_smiles", help="nontoxic smiles column name")
    ap.add_argument("--endpoint_col", type=str, default="endpoint", help="endpoint column name (optional)")
    ap.add_argument(
        "--min_unique",
        type=int,
        default=2,
        help="Minimum number of unique nontoxic_smiles to count as 'many' (default: 2)",
    )
    ap.add_argument(
        "--no_canonical",
        action="store_true",
        help="RDKit canonical SMILES 변환을 하지 않고 원본 문자열로 계산.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory for CSV summaries. Default: <input_dir>/toxic_pair_many_stats",
    )
    ap.add_argument(
        "--no_endpoint",
        action="store_true",
        help="Skip endpoint-level summary even if endpoint column exists.",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")

    df = pd.read_csv(in_path)
    for c in (args.toxic_col, args.nontoxic_col):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if args.endpoint_col not in df.columns:
        raise ValueError(
            f"Missing endpoint column: {args.endpoint_col}. "
            "이 스크립트는 '같은 endpoint 내부' 기준 통계를 계산합니다."
        )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (in_path.parent / "toxic_pair_many_stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonicalize (optional)
    toxic_key_col = args.toxic_col
    nontoxic_key_col = args.nontoxic_col
    if not args.no_canonical:
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit이 없어 canonical SMILES 변환을 할 수 없습니다. --no_canonical을 사용하거나 RDKit을 설치하세요.")

        cache: dict[str, str | None] = {}

        def _cano(smiles: object) -> str | None:
            if smiles is None:
                return None
            s = str(smiles).strip()
            if not s:
                return None
            if s in cache:
                return cache[s]
            try:
                mol = Chem.MolFromSmiles(s)
                cano = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
            except Exception:
                cano = None
            cache[s] = cano
            return cano

        df["toxic_canonical_smiles"] = df[args.toxic_col].map(_cano)
        df["nontoxic_canonical_smiles"] = df[args.nontoxic_col].map(_cano)
        toxic_key_col = "toxic_canonical_smiles"
        nontoxic_key_col = "nontoxic_canonical_smiles"

        # canonicalization 실패(None)인 row는 그룹 통계에서 제외(의미 없는 키)
        before = len(df)
        df = df.dropna(subset=[toxic_key_col, nontoxic_key_col]).reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            print(f"[canonical] dropped rows with invalid SMILES: {dropped:,}")

    # (endpoint, toxic)별: unique nontoxic 개수 + row 수 (endpoint 내부 기준)
    g = df.groupby([args.endpoint_col, toxic_key_col], dropna=False)
    toxic_stats = g.agg(
        n_rows=(nontoxic_key_col, "size"),
        n_unique_nontoxic=(nontoxic_key_col, pd.Series.nunique),
    ).reset_index()
    toxic_stats = toxic_stats.sort_values(["n_unique_nontoxic", "n_rows"], ascending=[False, False])

    many_mask = toxic_stats["n_unique_nontoxic"] >= int(args.min_unique)
    many_pairs = toxic_stats.loc[many_mask, [args.endpoint_col, toxic_key_col]]

    # 전체 요약 (endpoint 내부 기준을 합산)
    total_rows = int(len(df))
    total_unique_toxic_global = int(df[toxic_key_col].nunique(dropna=False))
    total_unique_endpoint_toxic = int(toxic_stats.shape[0])  # (endpoint, toxic) 고유 개수

    # outlier many (endpoint,toxic) 수
    many_endpoint_toxic_n = int(many_pairs.shape[0])

    # 해당 (endpoint,toxic)에 속하는 row 수
    many_key = set(zip(many_pairs[args.endpoint_col], many_pairs[toxic_key_col]))
    many_rows_n = int(
        df[[args.endpoint_col, toxic_key_col]]
        .apply(lambda r: (r[args.endpoint_col], r[toxic_key_col]) in many_key, axis=1)
        .sum()
    )

    overall = pd.DataFrame(
        [
            {
                "input": str(in_path),
                "min_unique_nontoxic": int(args.min_unique),
                "total_rows": total_rows,
                "total_unique_toxic_global": total_unique_toxic_global,
                "total_unique_endpoint_toxic": total_unique_endpoint_toxic,
                "endpoint_toxic_with_many_nontoxic": many_endpoint_toxic_n,
                "rows_from_many_endpoint_toxic": many_rows_n,
                "rows_from_many_endpoint_toxic_ratio": (many_rows_n / total_rows) if total_rows else 0.0,
            }
        ]
    )
    overall_csv = out_dir / "overall_summary.csv"
    overall.to_csv(overall_csv, index=False)

    # toxic 상세 저장
    toxic_csv = out_dir / "toxic_level_stats.csv"
    toxic_stats.to_csv(toxic_csv, index=False)

    # endpoint별 요약 (옵션) - endpoint 내부 기준 그대로
    endpoint_csv = None
    if (not args.no_endpoint) and (args.endpoint_col in df.columns):
        rows = []
        for endpoint, sub in df.groupby(args.endpoint_col, dropna=False):
            sub_g = sub.groupby(toxic_key_col, dropna=False).agg(
                n_rows=(nontoxic_key_col, "size"),
                n_unique_nontoxic=(nontoxic_key_col, pd.Series.nunique),
            )
            many_toxic_ep = int((sub_g["n_unique_nontoxic"] >= int(args.min_unique)).sum())
            many_rows_ep = int(
                sub[sub[toxic_key_col].isin(sub_g[sub_g["n_unique_nontoxic"] >= int(args.min_unique)].index)].shape[0]
            )
            total_rows_ep = int(len(sub))
            total_toxic_ep = int(sub[toxic_key_col].nunique(dropna=False))
            rows.append(
                {
                    "endpoint": endpoint,
                    "min_unique_nontoxic": int(args.min_unique),
                    "total_rows": total_rows_ep,
                    "total_unique_toxic": total_toxic_ep,
                    "toxic_with_many_nontoxic": many_toxic_ep,
                    "rows_from_many_toxic": many_rows_ep,
                    "rows_from_many_toxic_ratio": (many_rows_ep / total_rows_ep) if total_rows_ep else 0.0,
                }
            )
        endpoint_df = pd.DataFrame(rows).sort_values(
            ["toxic_with_many_nontoxic", "rows_from_many_toxic"],
            ascending=[False, False],
        )
        endpoint_csv = out_dir / "endpoint_summary.csv"
        endpoint_df.to_csv(endpoint_csv, index=False)

    # 콘솔 출력
    print("=" * 80)
    print("Toxic paired with multiple nontoxic stats (WITHIN endpoint)")
    print("=" * 80)
    print(f"Input: {in_path}")
    print(f"Rule (within endpoint): n_unique_nontoxic >= {int(args.min_unique)}")
    print(f"Key columns: toxic={toxic_key_col}, nontoxic={nontoxic_key_col}")
    print(f"Total unique toxic_smiles (global, ignoring endpoint): {total_unique_toxic_global:,}")
    print(f"Total unique (endpoint, toxic_smiles): {total_unique_endpoint_toxic:,}")
    print(f"(endpoint, toxic_smiles) with many nontoxic: {many_endpoint_toxic_n:,}")
    print(f"Rows from those (endpoint,toxic): {many_rows_n:,} / {total_rows:,} ({(many_rows_n / total_rows) if total_rows else 0.0:.4f})")
    print(f"Saved: {overall_csv}")
    print(f"Saved: {toxic_csv}")
    if endpoint_csv is not None:
        print(f"Saved: {endpoint_csv}")


if __name__ == "__main__":
    main()

