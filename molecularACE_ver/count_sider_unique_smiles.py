#!/usr/bin/env python3
"""molecularACE_ver/sider.csv — X 열(SMILES) 고유 개수."""

from __future__ import annotations

import csv
from pathlib import Path


def main():
    p = Path(__file__).resolve().parent / "sider.csv"
    if not p.is_file():
        print("sider.csv not found")
        return
    seen: set[str] = set()
    with open(p, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        key = None
        if r.fieldnames:
            for h in r.fieldnames:
                if (h or "").strip().lstrip("\ufeff") == "X":
                    key = h
                    break
        if not key:
            print("no X column", r.fieldnames)
            return
        for row in r:
            s = (row.get(key) or "").strip()
            if s:
                seen.add(s)
    print(f"sider\t{len(seen)}")


if __name__ == "__main__":
    main()
