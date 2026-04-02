"""
ACE pair에 등장하는 모든 SMILES를 수집한 뒤,
아직 매핑에 없는 것만 canonical → SAFE 변환하여
ace_safe_ver용 smiles_to_safe CSV를 만듭니다.
기존 data/smiles_to_safe.csv 가 있으면 병합해 재사용합니다.

사용 순서:
  1. python build_ace_smiles_to_safe.py   # ACE 전용 매핑 생성 (없는 것만 인코딩)
  2. python pairs_to_safe.py --mapping ../smiles_to_safe_ace.csv  # pair에 SAFE 붙이기

Merged train/test에서 task raw 데이터 생성 (--merged_train / --merged_test):
  merged_train.csv, merged_test.csv에서 toxic/nontoxic decoded_smiles + safe 쌍을 수집하여
  endpoint·독성 여부 무관 unique (smiles, safe) raw 데이터를 만듦.
  출력: smiles, canonical_smiles, safe, split (train/test) → smiles↔SAFE task용.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ACE_SAFE_VER_DIR = SCRIPT_DIR.parent
if str(ACE_SAFE_VER_DIR) not in sys.path:
    sys.path.insert(0, str(ACE_SAFE_VER_DIR))
import ace_local  # noqa: E402

ace_local.ensure_safe_pkg_path()

import datamol as dm
from safe.safe.converter import SAFEEncodeError, SAFEFragmentationError, encode as safe_encode

DEFAULT_PAIRS_CSV = ace_local.DEFAULT_MOLECULARACE_PAIRS_CSV
DEFAULT_EXISTING_MAPPING = ace_local.DEFAULT_SMILES_TO_SAFE_CSV
DEFAULT_OUTPUT_CSV = ACE_SAFE_VER_DIR / "smiles_to_safe_ace.csv"

# merged train/test 기반 task raw 출력 기본 경로
DEFAULT_MERGED_SPLIT_DIR = ACE_SAFE_VER_DIR / "splits" / "scaffold_by_endpoint_unseen_ver"
DEFAULT_TASK_RAW_OUTPUT = ACE_SAFE_VER_DIR / "smiles_safe_task_raw.csv"

REQUIRED_MERGED_COLS = [
    "toxic_safe_decoded_smiles",
    "nontoxic_safe_decoded_smiles",
    "toxic_safe",
    "nontoxic_safe",
]


def _escape_csv(s: str | None) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def canonical_smiles(smiles: str) -> str | None:
    """SMILES를 canonical form으로. 실패 시 None."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    with dm.without_rdkit_log():
        try:
            mol = dm.to_mol(str(smiles).strip())
            if mol is None:
                return None
            return dm.standardize_smiles(dm.to_smiles(mol, canonical=True))
        except Exception:
            return None


def encode_safe(smiles: str) -> str | None:
    """Canonical SMILES를 SAFE 문자열로. 실패 시 None."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    with dm.without_rdkit_log():
        try:
            return safe_encode(str(smiles).strip(), canonical=True)
        except (SAFEEncodeError, SAFEFragmentationError, Exception):
            return None


def _valid_pair(smiles: str, safe: str) -> bool:
    if pd.isna(smiles) or pd.isna(safe):
        return False
    s = str(smiles).strip()
    t = str(safe).strip()
    if not s or not t or s.lower() == "nan" or t.lower() == "nan":
        return False
    return True


def build_task_raw_from_merged(
    merged_train_path: Path,
    merged_test_path: Path,
    output_path: Path,
    canonicalize: bool = True,
) -> None:
    """
    merged_train.csv, merged_test.csv에서 (decoded_smiles, safe) 쌍을 수집하여
    unique (smiles, safe) raw 데이터를 저장. split 컬럼으로 train/test 유지.
    동일 (canonical_smiles, safe)가 train/test 둘 다 있으면 test로 표시(데이터 누수 방지).
    """
    for p in (merged_train_path, merged_test_path):
        if not p.exists():
            raise FileNotFoundError(f"Merged CSV not found: {p}")

    def load_and_collect(path: Path, split_label: str) -> list[tuple[str, str, str]]:
        df = pd.read_csv(path)
        for c in REQUIRED_MERGED_COLS:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' in {path}. Found: {list(df.columns)}")
        rows: list[tuple[str, str, str]] = []
        for _, row in df.iterrows():
            for smi_col, safe_col in [
                ("toxic_safe_decoded_smiles", "toxic_safe"),
                ("nontoxic_safe_decoded_smiles", "nontoxic_safe"),
            ]:
                smi = row[smi_col]
                safe = row[safe_col]
                if not _valid_pair(smi, safe):
                    continue
                rows.append((str(smi).strip(), str(safe).strip(), split_label))
        return rows

    train_pairs = load_and_collect(merged_train_path, "train")
    test_pairs = load_and_collect(merged_test_path, "test")
    print(f"Collected: {len(train_pairs)} (train) + {len(test_pairs)} (test) raw pairs")

    # Deduplicate by (canonical_smiles, safe); if same pair in train and test, mark as test
    key_to_row: dict[tuple[str, str], tuple[str, str, str]] = {}
    for smiles, safe, split in train_pairs + test_pairs:
        if canonicalize:
            canon = canonical_smiles(smiles)
            key = (canon if canon else smiles, safe)
            use_smiles = canon if canon else smiles
        else:
            key = (smiles, safe)
            use_smiles = smiles
        if not key[0] or not key[1]:
            continue
        existing = key_to_row.get(key)
        if existing is None:
            key_to_row[key] = (use_smiles, safe, split)
        else:
            # already seen: if current is test, upgrade to test
            if split == "test":
                key_to_row[key] = (existing[0], existing[1], "test")

    rows_out = list(key_to_row.values())
    rows_out.sort(key=lambda x: (x[2], x[0], x[1]))  # train first, then by smiles, safe

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows_out, columns=["smiles", "safe", "split"])
    out_df.insert(1, "canonical_smiles", out_df["smiles"])  # smiles가 이미 canonical (canonicalize=True 시)
    out_df.to_csv(output_path, index=False)
    n_train = sum(1 for _, _, s in rows_out if s == "train")
    n_test = sum(1 for _, _, s in rows_out if s == "test")
    print(f"Saved: {output_path} (unique rows={len(rows_out)}, train={n_train}, test={n_test})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build smiles→SAFE mapping for all SMILES appearing in ACE pairs, or task raw from merged train/test."
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS_CSV,
        help="ACE pairs CSV (used when not --merged_train/--merged_test).",
    )
    parser.add_argument(
        "--existing",
        type=Path,
        default=DEFAULT_EXISTING_MAPPING,
        help="Existing mapping CSV to merge (skip encoding for these SMILES). None to disable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output smiles_to_safe CSV for ACE",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Ignore --existing; encode all ACE SMILES from scratch.",
    )
    parser.add_argument(
        "--merged_train",
        type=Path,
        default=None,
        help="Merged train CSV (with toxic/nontoxic decoded_smiles and safe). With --merged_test, build task raw only.",
    )
    parser.add_argument(
        "--merged_test",
        type=Path,
        default=None,
        help="Merged test CSV. With --merged_train, build task raw only.",
    )
    parser.add_argument(
        "--merged_dir",
        type=Path,
        default=None,
        help="Directory containing merged_train.csv and merged_test.csv (overrides --merged_train/--merged_test paths).",
    )
    parser.add_argument(
        "--from_merged",
        action="store_true",
        help=f"Use merged train/test from default dir: {DEFAULT_MERGED_SPLIT_DIR} (merged_train.csv, merged_test.csv).",
    )
    parser.add_argument(
        "--output_raw",
        type=Path,
        default=DEFAULT_TASK_RAW_OUTPUT,
        help="Output path for smiles/safe task raw CSV (default: ace_safe_ver/smiles_safe_task_raw.csv).",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        dest="no_canonicalize",
        help="In merged mode, do not canonicalize SMILES for dedup (use as-is).",
    )
    args = parser.parse_args()

    # Merged train/test 모드: task raw만 생성
    if args.from_merged:
        args.merged_dir = DEFAULT_MERGED_SPLIT_DIR
    if args.merged_dir is not None:
        args.merged_train = args.merged_dir / "merged_train.csv"
        args.merged_test = args.merged_dir / "merged_test.csv"
    if args.merged_train is not None and args.merged_test is not None:
        build_task_raw_from_merged(
            merged_train_path=args.merged_train,
            merged_test_path=args.merged_test,
            output_path=args.output_raw,
            canonicalize=not args.no_canonicalize,
        )
        return

    if not args.pairs.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {args.pairs}")

    # 1) ACE pair에서 고유 SMILES 수집
    print(f"Loading pairs: {args.pairs}")
    df = pd.read_csv(args.pairs)
    for c in ["toxic_smiles", "nontoxic_smiles"]:
        if c not in df.columns:
            raise ValueError(f"Need column '{c}'. Found: {list(df.columns)}")
    unique_smiles = set()
    for s in df["toxic_smiles"].astype(str).str.strip():
        if s and s != "nan":
            unique_smiles.add(s)
    for s in df["nontoxic_smiles"].astype(str).str.strip():
        if s and s != "nan":
            unique_smiles.add(s)
    unique_smiles = sorted(unique_smiles)
    print(f"Unique SMILES in ACE pairs: {len(unique_smiles)}")

    # 2) 기존 매핑 로드 (병합 시)
    existing_by_smiles: dict[str, tuple[str, str]] = {}  # smiles or canonical -> (canonical, safe)
    if not args.no_merge and args.existing and args.existing.exists():
        print(f"Loading existing mapping: {args.existing}")
        exist_df = pd.read_csv(args.existing)
        if "smiles" in exist_df.columns and "safe" in exist_df.columns:
            canon_col = "canonical_smiles" if "canonical_smiles" in exist_df.columns else "smiles"
            for _, row in exist_df.iterrows():
                orig = str(row["smiles"]).strip()
                if not orig or orig == "nan":
                    continue
                canon = str(row.get(canon_col, orig)).strip() if canon_col in row else orig
                safe = str(row["safe"]).strip() if pd.notna(row["safe"]) else ""
                existing_by_smiles[orig] = (canon, safe)
                if canon and canon != "nan" and canon != orig:
                    existing_by_smiles[canon] = (canon, safe)
            print(f"  Loaded {len(existing_by_smiles)} entries")
    else:
        if args.no_merge:
            print("--no-merge: encoding all ACE SMILES from scratch.")
        else:
            print("No existing mapping found; encoding all ACE SMILES.")

    # 3) 없는 것만 canonical + SAFE 인코딩
    to_encode = [s for s in unique_smiles if s not in existing_by_smiles]
    print(f"To encode (not in existing): {len(to_encode)}")

    # smi -> (canon, safe) 통합
    merged: dict[str, tuple[str, str]] = dict(existing_by_smiles)
    for s in tqdm(to_encode, desc="Canonical + SAFE"):
        canon = canonical_smiles(s)
        if canon is None:
            merged[s] = ("", "")
            continue
        safe_str = encode_safe(canon)
        merged[s] = (canon, safe_str if safe_str else "")

    # 4) unique_smiles 순서로 출력용 리스트 생성 (한 번씩만)
    ordered: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for s in unique_smiles:
        if s in seen:
            continue
        seen.add(s)
        canon, safe = merged.get(s, ("", ""))
        ordered.append((s, canon, safe))

    # 5) 저장
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        f.write("smiles,canonical_smiles,safe\n")
        for smiles, canon, safe in ordered:
            line = f"{_escape_csv(smiles)},{_escape_csv(canon)},{_escape_csv(safe)}\n"
            f.write(line)
    print(f"Saved: {args.output} ({len(ordered)} rows)")
    n_with_safe = sum(1 for _, _, s in ordered if s)
    print(f"Rows with non-empty SAFE: {n_with_safe} / {len(ordered)}")


if __name__ == "__main__":
    main()
