"""
herg_central, herg, herg_karim 통합:

1. pairs_safe_filtered.csv에서 해당 데이터셋들의 toxic_smiles / nontoxic_smiles로
   per-SMILES 레이블 수집 (canonical 기준).
2. 같은 canonical SMILES가 독성/비독성으로 서로 다르게 나오면 → 해당 SMILES drop (conflict).
3. 같으면 통합. 같은 (toxic_canon, nontoxic_canon) pair는 한 번만 유지(deduplication).
4. 최종: conflict 없는 pair만 남기고, (toxic_canon, nontoxic_canon) 기준 중복 제거 후 dataset_name/endpoint를 herg_unified로 통일해 저장.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

try:
    from rdkit import Chem
except ImportError:
    Chem = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "pairs_safe_filtered_valid.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "pairs_safe_filtered_herg_merged.csv"

HERG_DATASETS = {"herg_central", "herg", "herg_karim"}
MERGED_DATASET = "herg_unified"
MERGED_ENDPOINT = "herg_unified"


def _canon_smiles(s: str) -> str:
    if Chem is None:
        return (s or "").strip()
    s = (s or "").strip()
    if not s:
        return ""
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def merge_herg(
    input_csv: Path,
    output_csv: Path,
    herg_datasets: set[str] | None = None,
    merged_name: str = MERGED_DATASET,
    merged_endpoint: str = MERGED_ENDPOINT,
    deduplicate_pairs: bool = True,
) -> pd.DataFrame:
    herg_datasets = herg_datasets or HERG_DATASETS
    if Chem is None:
        raise ImportError("RDKit required for canonical SMILES. pip install rdkit")

    df = pd.read_csv(input_csv)
    if "dataset_name" not in df.columns or "toxic_smiles" not in df.columns or "nontoxic_smiles" not in df.columns:
        raise ValueError("CSV must have dataset_name, toxic_smiles, nontoxic_smiles")

    mask_herg = df["dataset_name"].isin(herg_datasets)
    df_herg = df.loc[mask_herg].copy()
    df_other = df.loc[~mask_herg].copy()

    if df_herg.empty:
        print("[WARN] No rows for herg_central/herg/herg_karim; output = rest only.")
        df_out = df_other
        df_out.to_csv(output_csv, index=False)
        return df_out

    # 1) 수집: canonical_smiles -> set of labels (1=toxic, 0=nontoxic)
    canon_to_labels: dict[str, set[int]] = defaultdict(set)
    for _, row in df_herg.iterrows():
        t = _canon_smiles(str(row["toxic_smiles"]))
        n = _canon_smiles(str(row["nontoxic_smiles"]))
        if t:
            canon_to_labels[t].add(1)
        if n:
            canon_to_labels[n].add(0)

    # 2) conflict: 두 레이블 다 있으면 drop
    conflicted = {c for c, labels in canon_to_labels.items() if len(labels) > 1}

    # 3) pair 유지: toxic_canon, nontoxic_canon 둘 다 conflicted가 아닌 행만
    def both_ok(row) -> bool:
        tc = _canon_smiles(str(row["toxic_smiles"]))
        nc = _canon_smiles(str(row["nontoxic_smiles"]))
        return (tc not in conflicted and nc not in conflicted) and tc and nc

    df_herg["_toxic_canon"] = df_herg["toxic_smiles"].astype(str).map(_canon_smiles)
    df_herg["_nontoxic_canon"] = df_herg["nontoxic_smiles"].astype(str).map(_canon_smiles)
    keep = ~df_herg["_toxic_canon"].isin(conflicted) & ~df_herg["_nontoxic_canon"].isin(conflicted)
    keep &= df_herg["_toxic_canon"].astype(bool) & df_herg["_nontoxic_canon"].astype(bool)

    df_herg_kept = df_herg.loc[keep].copy()
    n_before_dedup = len(df_herg_kept)
    if deduplicate_pairs:
        # 레이블이 같고 동일한 (toxic_canon, nontoxic_canon) pair → 한 행만 유지
        df_herg_kept = df_herg_kept.drop_duplicates(
            subset=["_toxic_canon", "_nontoxic_canon"], keep="first"
        )
    n_keep_pair = len(df_herg_kept)
    n_dup_dropped = n_before_dedup - n_keep_pair
    df_herg_kept = df_herg_kept.drop(columns=["_toxic_canon", "_nontoxic_canon"])
    df_herg_kept["dataset_name"] = merged_name
    df_herg_kept["endpoint"] = merged_endpoint

    n_drop_pair = int((~keep).sum())
    n_conflict_smiles = len(conflicted)

    df_out = pd.concat([df_other, df_herg_kept], ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print(f"[OK] Input  : {input_csv}")
    print(f"[OK] Output : {output_csv}")
    print(f"[HERG] datasets merged: {sorted(herg_datasets)} -> {merged_name}/{merged_endpoint}")
    print(f"[HERG] conflicted SMILES (dropped): {n_conflict_smiles:,}")
    print(f"[HERG] pairs dropped (contain conflicted SMILES): {n_drop_pair:,}")
    if deduplicate_pairs and n_dup_dropped:
        print(f"[HERG] duplicate pairs removed (same toxic_canon, nontoxic_canon): {n_dup_dropped:,}")
    print(f"[HERG] pairs kept (unified): {n_keep_pair:,}")
    print(f"[OK] Total rows: {len(df_out):,} (other: {len(df_other):,}, herg unified: {n_keep_pair:,})")
    return df_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge herg_central, herg, herg_karim with conflict drop.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input pairs_safe_filtered CSV")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    ap.add_argument("--merged_name", type=str, default=MERGED_DATASET, help="dataset_name for merged")
    ap.add_argument("--merged_endpoint", type=str, default=MERGED_ENDPOINT, help="endpoint for merged")
    ap.add_argument(
        "--no_dedup",
        action="store_true",
        help="Do not deduplicate by (toxic_canon, nontoxic_canon); keep all rows (default: deduplicate).",
    )
    args = ap.parse_args()
    merge_herg(
        input_csv=args.input,
        output_csv=args.output,
        merged_name=args.merged_name,
        merged_endpoint=args.merged_endpoint,
        deduplicate_pairs=not args.no_dedup,
    )


if __name__ == "__main__":
    main()
