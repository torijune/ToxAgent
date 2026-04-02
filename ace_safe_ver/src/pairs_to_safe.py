"""
data/pairs.csv 등 (toxic_smiles, nontoxic_smiles) pair에
data/smiles_to_safe.csv 매핑을 사용해 toxic_safe, nontoxic_safe 를 붙여
pairs_safe.csv 를 생성합니다.

로직은 기존 molecule_safe_ver/src/matching_pairs_safe.py 와 동일합니다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ACE_SAFE_VER_DIR = SCRIPT_DIR.parent
if str(ACE_SAFE_VER_DIR) not in sys.path:
    sys.path.insert(0, str(ACE_SAFE_VER_DIR))
import ace_local  # noqa: E402

DEFAULT_ACE_PAIRS_CSV = ace_local.DEFAULT_MOLECULARACE_PAIRS_CSV
DEFAULT_SMILES_TO_SAFE_CSV = ace_local.DEFAULT_SMILES_TO_SAFE_CSV
DEFAULT_OUTPUT_CSV = ACE_SAFE_VER_DIR / "pairs_safe.csv"


def load_safe_mapping(mapping_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    smiles_to_safe.csv 를 읽어 (원본 SMILES → SAFE), (canonical_smiles → SAFE) 딕셔너리 반환.
    smiles, canonical_smiles(optional), safe 컬럼 형식.
    """
    map_df = pd.read_csv(mapping_path)
    if "smiles" not in map_df.columns or "safe" not in map_df.columns:
        raise ValueError(
            f"Mapping CSV must have columns 'smiles' and 'safe'. Found: {list(map_df.columns)}"
        )
    smiles_to_safe = dict(
        zip(map_df["smiles"].astype(str).str.strip(), map_df["safe"].fillna("").astype(str))
    )
    canon_to_safe = {}
    if "canonical_smiles" in map_df.columns:
        canon_to_safe = dict(
            zip(
                map_df["canonical_smiles"].astype(str).str.strip(),
                map_df["safe"].fillna("").astype(str),
            )
        )
        canon_to_safe = {k: v for k, v in canon_to_safe.items() if k and str(k) != "nan"}
    return smiles_to_safe, canon_to_safe


def lookup_safe(
    smiles_series: pd.Series,
    smiles_to_safe: dict[str, str],
    canon_to_safe: dict[str, str],
) -> list[str]:
    """SMILES 시리즈에 대해 SAFE 문자열 리스트 반환 (없으면 빈 문자열)."""
    out = []
    for s in smiles_series:
        s = str(s).strip() if pd.notna(s) else ""
        safe_str = smiles_to_safe.get(s)
        if safe_str is None and s:
            safe_str = canon_to_safe.get(s, "")
        out.append(safe_str if safe_str is not None else "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach toxic_safe, nontoxic_safe to ACE pairs using smiles→SAFE mapping CSV."
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_ACE_PAIRS_CSV,
        help=f"ACE pairs CSV (default: {DEFAULT_ACE_PAIRS_CSV})",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_SMILES_TO_SAFE_CSV,
        help=f"smiles→SAFE mapping CSV (default: ace_safe_ver/data/smiles_to_safe.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output pairs_safe CSV (default: ace_safe_ver/pairs_safe.csv)",
    )
    args = parser.parse_args()

    if not args.mapping.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found: {args.mapping}\n"
            "Place smiles_to_safe.csv under ace_safe_ver/data/ or set --mapping."
        )
    if not args.pairs.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {args.pairs}")

    print(f"Loading mapping: {args.mapping}")
    smiles_to_safe, canon_to_safe = load_safe_mapping(args.mapping)

    print(f"Loading pairs: {args.pairs}")
    df = pd.read_csv(args.pairs)
    for col in ["toxic_smiles", "nontoxic_smiles"]:
        if col not in df.columns:
            raise ValueError(f"Pairs CSV must have '{col}'. Found: {list(df.columns)}")

    n = len(df)
    df = df.assign(
        toxic_safe=lookup_safe(df["toxic_smiles"], smiles_to_safe, canon_to_safe),
        nontoxic_safe=lookup_safe(df["nontoxic_smiles"], smiles_to_safe, canon_to_safe),
    )

    toxic_miss = (df["toxic_safe"] == "").sum()
    nontoxic_miss = (df["nontoxic_safe"] == "").sum()
    print(f"Rows with missing toxic_safe:   {toxic_miss} / {n}")
    print(f"Rows with missing nontoxic_safe: {nontoxic_miss} / {n}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
