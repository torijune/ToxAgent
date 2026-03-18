#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shuffle merged_train.csv and merged_test.csv in-place.

This is useful when you want randomized row order *at the dataset level*,
and then build QA without per-task shuffling.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def _read_csv_rows(path: Path) -> Tuple[List[str], List[List[str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]
    return header, rows


def _write_csv_rows(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def shuffle_csv_inplace(path: Path, seed: int, make_backup: bool = True) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    backup_path = path.with_suffix(path.suffix + ".bak")
    if make_backup:
        shutil.copy2(path, backup_path)

    header, rows = _read_csv_rows(path)
    rng = random.Random(seed)
    rng.shuffle(rows)
    _write_csv_rows(path, header, rows)
    return backup_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Shuffle merged_train.csv and merged_test.csv in-place.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    ap.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create .bak backups before overwriting.",
    )
    ap.add_argument(
        "--dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing merged_train.csv and merged_test.csv.",
    )
    args = ap.parse_args()

    d = Path(args.dir)
    train_path = d / "merged_train.csv"
    test_path = d / "merged_test.csv"

    for p in [train_path, test_path]:
        backup = shuffle_csv_inplace(p, seed=args.seed, make_backup=not args.no_backup)
        if args.no_backup:
            print(f"[shuffled] {p}")
        else:
            print(f"[shuffled] {p} (backup: {backup})")


if __name__ == "__main__":
    main()

