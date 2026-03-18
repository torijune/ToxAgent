#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
commom_frage_pairs_with_smiles.csv의
only_toxic_safe_fragments / only_nontoxic_safe_fragments에서
fragment 문자열 길이(len) 분포를 boxplot으로 시각화한다.

추가:
- 이상치 제거 후 데이터셋(예: commom_frage_pairs_with_smiles_no_long_frag.csv)도 함께 읽어서
  raw vs filtered boxplot을 한 번에 그린다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "commom_frage_pairs_with_smiles.csv"),
    )
    ap.add_argument(
        "--filtered_input",
        type=str,
        default=str(Path(__file__).resolve().parent / "commom_frage_pairs_with_smiles_no_long_frag.csv"),
        help="Outlier 제거 후 CSV (default: commom_frage_pairs_with_smiles_no_long_frag.csv)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "fragment_length_boxplot_raw_vs_filtered.png"),
    )
    ap.add_argument("--no_outliers", action="store_true")
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    import matplotlib
    matplotlib.use("Agg", force=True)

    # ✅ Python 3.14 + matplotlib에서 Path.__deepcopy__ 재귀 버그 회피용 몽키패치
    import matplotlib.path as mpath

    def _safe_path_deepcopy(self, memo):
        verts = self.vertices.copy()
        codes = None if self.codes is None else self.codes.copy()
        # Path 생성자에 vertices/codes만 넣어도 렌더링/저장엔 충분
        return mpath.Path(verts, codes)

    mpath.Path.__deepcopy__ = _safe_path_deepcopy

    import matplotlib.pyplot as plt

    input_path = Path(args.input)
    filtered_path = Path(args.filtered_input) if args.filtered_input else None
    out_path = Path(args.out)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    for col in [COL_TOXIC, COL_NONTOXIC]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    toxic_lengths = _collect_fragment_lengths(df[COL_TOXIC].values)
    nontoxic_lengths = _collect_fragment_lengths(df[COL_NONTOXIC].values)

    # filtered (없으면 raw만 그림)
    toxic_lengths_f = []
    nontoxic_lengths_f = []
    if filtered_path is not None and filtered_path.exists():
        df_f = pd.read_csv(filtered_path)
        for col in [COL_TOXIC, COL_NONTOXIC]:
            if col not in df_f.columns:
                raise ValueError(f"Missing column in filtered CSV: {col}")
        toxic_lengths_f = _collect_fragment_lengths(df_f[COL_TOXIC].values)
        nontoxic_lengths_f = _collect_fragment_lengths(df_f[COL_NONTOXIC].values)

    if len(toxic_lengths) == 0 and len(nontoxic_lengths) == 0:
        raise RuntimeError("수집된 fragment 길이가 없습니다 (컬럼이 비어있을 수 있음).")

    # global y range for 비교용 (logy면 하한 1로 고정)
    all_lengths = toxic_lengths + nontoxic_lengths + toxic_lengths_f + nontoxic_lengths_f
    ymax = max(all_lengths) if all_lengths else 1
    ymin = 1 if args.logy else 0

    if toxic_lengths_f or nontoxic_lengths_f:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        ax_raw, ax_f = axes[0], axes[1]
        panels = [
            ("Raw", ax_raw, toxic_lengths, nontoxic_lengths),
            ("Filtered", ax_f, toxic_lengths_f, nontoxic_lengths_f),
        ]
    else:
        fig, ax_only = plt.subplots(figsize=(8, 5))
        panels = [("Raw", ax_only, toxic_lengths, nontoxic_lengths)]

    for title, ax, tox, non in panels:
        ax.boxplot(
            [tox, non],
            labels=["only_toxic", "only_nontoxic"],
            showfliers=(not args.no_outliers),
        )
        ax.set_title(title)
        ax.set_xlabel("Fragments")
        if args.logy:
            ax.set_yscale("log")
        ax.set_ylim(bottom=ymin, top=ymax * (1.08 if not args.logy else 1.2))

        # N 표시
        y_text = ymax * (1.03 if not args.logy else 1.1)
        ax.text(1, y_text, f"N={len(tox)}", ha="center", va="bottom", fontsize=9)
        ax.text(2, y_text, f"N={len(non)}", ha="center", va="bottom", fontsize=9)

    # 공통 y-label
    fig.suptitle("Fragment string length (characters) boxplot")
    fig.text(0.04, 0.5, "Fragment length (chars)", va="center", rotation="vertical")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ Python 3.14 + matplotlib에서 bbox_inches="tight"가 deepcopy 재귀를 유발하는 케이스 회피
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.15, wspace=0.25)
    fig.savefig(out_path, dpi=200)  # bbox_inches="tight" 제거
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()