from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SPLIT_ROOT = PROJECT_ROOT / "splits" / "scaffold_by_endpoint_unseen_ver"


# use_unseen_split 시 split_dataset.py가 쓰는 서브디렉터리
TRAIN_TEST_SUBDIR = "train_test"
UNSEEN_TEST_SUBDIR = "unseen_endpoint_test"


def _collect_split_paths(split_root: Path, split_name: str) -> list[Path]:
    """split_root 아래의 모든 dataset/endpoint/<split_name>.csv 경로를 수집."""
    paths: list[Path] = []
    for dataset_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        for endpoint_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            csv_path = endpoint_dir / f"{split_name}.csv"
            if csv_path.exists():
                paths.append(csv_path)
    return paths


def _collect_paths_unseen_structure(split_root: Path, split_name: str) -> list[Path]:
    """
    use_unseen_split 구조에서 경로 수집.
    - train: train_test/<dataset>/<endpoint>/train.csv 만
    - test:  train_test/<dataset>/<endpoint>/test.csv + unseen_endpoint_test/<dataset>/<endpoint>/test.csv
    """
    paths: list[Path] = []
    train_test_dir = split_root / TRAIN_TEST_SUBDIR
    unseen_dir = split_root / UNSEEN_TEST_SUBDIR

    if split_name == "train":
        if train_test_dir.is_dir():
            paths = _collect_split_paths(train_test_dir, "train")
        return paths

    if split_name == "test":
        if train_test_dir.is_dir():
            paths = _collect_split_paths(train_test_dir, "test")
        if unseen_dir.is_dir():
            paths.extend(_collect_split_paths(unseen_dir, "test"))
        return paths

    # valid: train_test 쪽만 (unseen에는 valid 없음)
    if split_name == "valid" and train_test_dir.is_dir():
        return _collect_split_paths(train_test_dir, "valid")
    return paths


def merge_splits(
    split_root: Path,
    out_dir: Path,
    include_valid: bool = True,
    use_unseen_structure: Optional[bool] = None,
) -> None:
    split_root = split_root.resolve()
    out_dir = out_dir.resolve()
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    # use_unseen_structure 미지정 시: train_test/ 존재하면 unseen 구조로 간주
    if use_unseen_structure is None:
        use_unseen_structure = (split_root / TRAIN_TEST_SUBDIR).is_dir()
    if use_unseen_structure:
        print(f"[INFO] Using unseen split structure: {TRAIN_TEST_SUBDIR}/ + {UNSEEN_TEST_SUBDIR}/ (test only)")

    out_dir.mkdir(parents=True, exist_ok=True)

    def _collect(split_name: str) -> list[Path]:
        if use_unseen_structure:
            return _collect_paths_unseen_structure(split_root, split_name)
        return _collect_split_paths(split_root, split_name)

    def _merge(split_name: str) -> Path:
        paths = _collect(split_name)
        if not paths:
            print(f"[WARN] No '{split_name}.csv' found under {split_root}")
            return out_dir / f"merged_{split_name}.csv"

        dfs = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                if df.empty:
                    continue
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Skip {p}: {e}")
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
        else:
            merged = pd.DataFrame()
        out_path = out_dir / f"merged_{split_name}.csv"
        merged.to_csv(out_path, index=False)
        print(f"[OK] Saved merged {split_name}: {out_path} (rows={len(merged):,})")
        return out_path

    train_path = _merge("train")
    test_path = _merge("test")
    valid_path = None
    if include_valid:
        valid_path = _merge("valid")

    print("[DONE] Merged splits:")
    print(f"  train -> {train_path}")
    print(f"  test  -> {test_path}")
    if include_valid:
        print(f"  valid -> {valid_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Merge per-endpoint scaffold splits (train/valid/test) into single CSVs.\n"
            "Expects structure: <split_root>/<dataset>/<endpoint>/{train,valid,test}.csv"
        )
    )
    ap.add_argument(
        "--split_root",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
        help=f"Root directory containing per-endpoint splits (default: {DEFAULT_SPLIT_ROOT})",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_SPLIT_ROOT,
        help="Output directory for merged CSVs (default: same as split_root)",
    )
    ap.add_argument(
        "--no_valid",
        action="store_true",
        help="Do not merge valid.csv (only train/test).",
    )
    ap.add_argument(
        "--use_unseen_structure",
        action="store_true",
        dest="use_unseen_structure",
        help="split_root 아래 train_test/ + unseen_endpoint_test/ 구조로 merge (train은 train_test만, test는 둘 다).",
    )
    ap.add_argument(
        "--no_unseen_structure",
        action="store_false",
        dest="use_unseen_structure",
        help="평면 구조만 사용 (split_root/<dataset>/<endpoint>/). 기본은 train_test/ 있으면 자동으로 unseen 구조 사용.",
    )
    ap.set_defaults(use_unseen_structure=None)
    args = ap.parse_args()

    merge_splits(
        split_root=args.split_root,
        out_dir=args.out_dir,
        include_valid=not args.no_valid,
        use_unseen_structure=args.use_unseen_structure,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

