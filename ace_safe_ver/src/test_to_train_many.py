#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merged_test.csv에서 "같은 endpoint 내부에서 toxic_smiles가 여러 nontoxic_smiles와 pairing" 되는 row들을
전부 train으로 이동시키고, test에는 1:1 매칭만 남깁니다.

정의 (within endpoint):
- (endpoint, toxic_smiles) 그룹에서 nontoxic_smiles의 unique 개수 >= K 이면
  해당 (endpoint, toxic_smiles)에 속한 test row 전부를 "many-pair"로 보고 train으로 이동.

출력:
- merged_train.csv (기존 train + moved rows)
- merged_test.csv (remaining rows)
- moved_from_test.csv (train으로 이동된 row만)
- summary.csv (전체 및 endpoint별 이동 통계)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Move many-pair toxic rows from test to train (within endpoint).")
    ap.add_argument(
        "--train",
        type=Path,
        default=Path("/ssd1/jueon/wj/detoxicity_model/ace_safe_ver/splits/scaffold_by_endpoint_property_outlier_dropped/merged_train.csv"),
        help="Input train CSV path",
    )
    ap.add_argument(
        "--test",
        type=Path,
        default=Path("/ssd1/jueon/wj/detoxicity_model/ace_safe_ver/splits/scaffold_by_endpoint_property_outlier_dropped/merged_test.csv"),
        help="Input test CSV path",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default: sibling dir named scaffold_by_endpoint_property_outlier_dropped_moved_many",
    )
    ap.add_argument("--endpoint_col", type=str, default="endpoint", help="Endpoint column name")
    ap.add_argument("--toxic_col", type=str, default="toxic_smiles", help="Toxic smiles column name")
    ap.add_argument("--nontoxic_col", type=str, default="nontoxic_smiles", help="Non-toxic smiles column name")
    ap.add_argument(
        "--min_unique",
        type=int,
        default=2,
        help="Minimum unique nontoxic_smiles within (endpoint,toxic) to be moved (default: 2)",
    )
    ap.add_argument(
        "--no_canonical",
        action="store_true",
        help="RDKit canonical SMILES 변환을 하지 않고 원본 문자열로 (endpoint,toxic) 그룹을 계산.",
    )
    args = ap.parse_args()

    train_path = Path(args.train)
    test_path = Path(args.test)
    if not train_path.exists():
        raise FileNotFoundError(f"Train not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test not found: {test_path}")

    # default out_dir: 같은 splits 폴더 아래 새 디렉터리
    if args.out_dir is None:
        base = test_path.parent
        out_dir = base.parent / (base.name + "_moved_many_to_train")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    key_cols = [args.endpoint_col, args.toxic_col, args.nontoxic_col]
    _ensure_cols(train_df, key_cols)
    _ensure_cols(test_df, key_cols)

    # Canonicalize (default) so we treat identical molecules consistently
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

        train_df["toxic_canonical_smiles"] = train_df[args.toxic_col].map(_cano)
        train_df["nontoxic_canonical_smiles"] = train_df[args.nontoxic_col].map(_cano)
        test_df["toxic_canonical_smiles"] = test_df[args.toxic_col].map(_cano)
        test_df["nontoxic_canonical_smiles"] = test_df[args.nontoxic_col].map(_cano)

        toxic_key_col = "toxic_canonical_smiles"
        nontoxic_key_col = "nontoxic_canonical_smiles"

    # 1) test에서 (endpoint,toxic)별 unique nontoxic 개수 계산 (same endpoint 기준)
    # canonical 사용 시 invalid(None) row는 many-key 계산에서 제외 (원치 않는 대규모 이동 방지)
    valid_for_key = test_df[toxic_key_col].notna() & test_df[nontoxic_key_col].notna()
    grp = (
        test_df.loc[valid_for_key]
        .groupby([args.endpoint_col, toxic_key_col], dropna=False)[nontoxic_key_col]
        .nunique()
        .reset_index(name="n_unique_nontoxic")
    )
    many_keys = grp[grp["n_unique_nontoxic"] >= int(args.min_unique)][[args.endpoint_col, toxic_key_col]]

    # 빠른 membership 체크를 위해 merge 사용
    many_keys["_many"] = True
    test_flagged = test_df.merge(many_keys, on=[args.endpoint_col, toxic_key_col], how="left")
    # key 계산에 사용 가능한(valid) row만 이동 대상으로 포함
    move_mask = test_flagged["_many"].fillna(False).astype(bool) & valid_for_key.reindex(test_flagged.index, fill_value=False)

    moved_df = test_flagged.loc[move_mask].drop(columns=["_many"])
    kept_test_df = test_flagged.loc[~move_mask].drop(columns=["_many"])

    # 2) train에 append
    # 컬럼 순서/집합: train 기준 유지, train에 없는 컬럼이 test에 있을 수 있으니 outer concat 후 train 컬럼 우선
    all_cols = list(dict.fromkeys(list(train_df.columns) + list(moved_df.columns)))
    new_train_df = pd.concat(
        [train_df.reindex(columns=all_cols), moved_df.reindex(columns=all_cols)],
        ignore_index=True,
    )
    new_test_df = kept_test_df.reindex(columns=all_cols)

    # 3) 저장
    out_train = out_dir / "merged_train.csv"
    out_test = out_dir / "merged_test.csv"
    out_moved = out_dir / "moved_from_test.csv"
    out_summary = out_dir / "summary.csv"

    new_train_df.to_csv(out_train, index=False)
    new_test_df.to_csv(out_test, index=False)
    moved_df.to_csv(out_moved, index=False)

    # 4) summary (전체 + endpoint별)
    total_test_before = int(len(test_df))
    moved_n = int(len(moved_df))
    kept_n = int(len(kept_test_df))
    total_train_before = int(len(train_df))
    total_train_after = int(len(new_train_df))

    overall_row = {
        "min_unique": int(args.min_unique),
        "canonical": int(not args.no_canonical),
        "toxic_key_col": toxic_key_col,
        "nontoxic_key_col": nontoxic_key_col,
        "train_before_rows": total_train_before,
        "test_before_rows": total_test_before,
        "moved_rows": moved_n,
        "test_after_rows": kept_n,
        "train_after_rows": total_train_after,
        "moved_ratio_of_test": (moved_n / total_test_before) if total_test_before else 0.0,
    }

    ep_rows: List[Dict] = []
    for ep, sub in test_df.groupby(args.endpoint_col, dropna=False):
        sub_valid = sub[sub[toxic_key_col].notna() & sub[nontoxic_key_col].notna()]
        sub_grp = sub_valid.groupby(toxic_key_col, dropna=False)[nontoxic_key_col].nunique()
        many_toxic = set(sub_grp[sub_grp >= int(args.min_unique)].index.tolist())
        moved_rows_ep = int(sub_valid[sub_valid[toxic_key_col].isin(many_toxic)].shape[0])
        total_rows_ep = int(len(sub))
        ep_rows.append(
            {
                "endpoint": ep,
                "min_unique": int(args.min_unique),
                "test_rows": total_rows_ep,
                "moved_rows": moved_rows_ep,
                "kept_rows": total_rows_ep - moved_rows_ep,
                "moved_ratio": (moved_rows_ep / total_rows_ep) if total_rows_ep else 0.0,
                "many_toxic_smiles": int(len(many_toxic)),
            }
        )

    summary_df = pd.DataFrame([overall_row] + ep_rows)
    summary_df.to_csv(out_summary, index=False)

    # 콘솔 출력
    print("=" * 80)
    print("Move many-pair rows from test -> train (within endpoint)")
    print("=" * 80)
    print(f"Train: {train_path} ({total_train_before:,} rows)")
    print(f"Test : {test_path} ({total_test_before:,} rows)")
    print(f"Rule: unique nontoxic within (endpoint,toxic) >= {int(args.min_unique)}")
    print(f"Key columns: toxic={toxic_key_col}, nontoxic={nontoxic_key_col}")
    print(f"Moved: {moved_n:,} rows  | Kept in test: {kept_n:,} rows")
    print(f"Saved train: {out_train}")
    print(f"Saved test : {out_test}")
    print(f"Saved moved: {out_moved}")
    print(f"Saved summary: {out_summary}")


if __name__ == "__main__":
    main()

