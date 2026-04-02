#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# X축: 라벨(MW, MolLogP, …) 눈금 숫자
XLABEL_FONTSIZE = 15
XTICK_FONTSIZE = 13
# 좌·하단 축 테두리(스파인) 및 눈금선 굵기
SPINE_LINEWIDTH = 0.55
TICK_LINEWIDTH = 0.55

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MERGED_TRAIN = SCRIPT_DIR / "merged_train.csv"
DEFAULT_MERGED_TEST = SCRIPT_DIR / "merged_test.csv"


def _resolve_csv_path(csv_path: Path) -> Path:
    """
    cwd가 어디든 스크립트와 같은 디렉터리의 merged_*.csv를 찾을 수 있게 한다.
    (예: repo 루트 기준 상대경로를 split 폴더에서 실행했을 때 FileNotFound 방지)
    """
    p = Path(csv_path)
    if p.exists():
        return p.resolve()
    same_dir = SCRIPT_DIR / p.name
    if same_dir.exists():
        return same_dir.resolve()
    if not p.is_absolute():
        under_script = SCRIPT_DIR / p
        if under_script.exists():
            return under_script.resolve()
    raise FileNotFoundError(str(p))


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _read_csv(csv_path: Path) -> pd.DataFrame:
    p = _resolve_csv_path(csv_path)
    return pd.read_csv(p)


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return s


def _try_import_seaborn():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:
        return None


def _set_plot_style() -> None:
    sns = _try_import_seaborn()
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook")
        return
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.color": "#D0D0D0",
            "axes.edgecolor": "#B0B0B0",
            "axes.linewidth": SPINE_LINEWIDTH,
            "font.size": 11,
        }
    )


def _kde_or_hist(ax, values: pd.Series, *, color: str, label: str) -> None:
    sns = _try_import_seaborn()
    vals = values.dropna().astype(float)
    if len(vals) < 3:
        return

    if sns is not None:
        try:
            sns.kdeplot(
                x=vals,
                ax=ax,
                fill=True,
                alpha=0.45,
                linewidth=2.25,
                color=color,
                label=label,
            )
            return
        except Exception:
            pass

    ax.hist(vals, bins=40, density=True, alpha=0.25, color=color, label=label)


def _plot_one_property(
    ax,
    df: pd.DataFrame,
    prop: str,
    *,
    tox_color: str,
    non_color: str,
) -> None:
    tox = _series(df, f"toxic_{prop}")
    non = _series(df, f"nontoxic_{prop}")

    _kde_or_hist(ax, tox, color=tox_color, label="toxic")
    _kde_or_hist(ax, non, color=non_color, label="nontoxic")

    if len(tox) > 0:
        ax.axvline(
            float(tox.mean()),
            color=tox_color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label="mean toxic",
        )
    if len(non) > 0:
        ax.axvline(
            float(non.mean()),
            color=non_color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.95,
            label="mean nontoxic",
        )

    # 서브플롯 타이틀은 예시(캡처)처럼 없음. x-label로만 구분.
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(SPINE_LINEWIDTH)
    ax.tick_params(axis="both", width=TICK_LINEWIDTH, length=4)


def plot_property_grid(
    df: pd.DataFrame,
    *,
    out_path: Path,
    properties: List[Tuple[str, str]],
) -> None:
    _set_plot_style()

    # toxic / nontoxic 색 대비 (색약 친화 Wong 팔레트 계열)
    tox_color = "#E41A1C"  # 선명한 빨강
    non_color = "#377EB8"  # 선명한 파랑

    n = len(properties)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 7.5), constrained_layout=False)
    if nrows == 1:
        axes = [axes]  # type: ignore

    flat_axes = [ax for row in axes for ax in (row if isinstance(row, Iterable) else [row])]  # type: ignore

    for i, (prop, label) in enumerate(properties):
        ax = flat_axes[i]
        _plot_one_property(ax, df, prop, tox_color=tox_color, non_color=non_color)
        ax.set_xlabel(label, fontsize=XLABEL_FONTSIZE)
        ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE)

    # hide unused axes
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.subplots_adjust(left=0.07, bottom=0.11, right=0.98, top=0.96, hspace=0.32, wspace=0.28)

    # MW 패널 내부 오른쪽 빈 여백에 범례 (밀도가 왼쪽에 몰릴 때 활용)
    ax_mw = flat_axes[0]
    handles, leg_labels = ax_mw.get_legend_handles_labels()
    ax_mw.legend(
        handles,
        leg_labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax_mw.transAxes,
        ncol=1,
        fontsize=9,
        frameon=True,
        fancybox=True,
        edgecolor="#CCCCCC",
        facecolor="white",
        framealpha=0.95,
        borderaxespad=0.35,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot toxic vs nontoxic molecular property density grids for train/test splits."
    )
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_MERGED_TRAIN,
        help=f"merged_train.csv path (default: {DEFAULT_MERGED_TRAIN})",
    )
    ap.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_MERGED_TEST,
        help=f"merged_test.csv path (default: {DEFAULT_MERGED_TEST})",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="output directory (default: this folder)",
    )
    ap.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="output format",
    )
    args = ap.parse_args()

    out_dir = _ensure_out_dir(args.out_dir)

    # (column suffix, plot label)
    properties: List[Tuple[str, str]] = [
        ("MW", "MW"),
        ("logP", "MolLogP"),
        ("TPSA", "TPSA"),
        ("HBD", "NumHDonors"),
        ("HBA", "NumHAcceptors"),
        ("RotB", "RotB"),
    ]

    train_df = _read_csv(args.train_csv)
    test_df = _read_csv(args.test_csv)
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    plot_property_grid(
        train_df,
        out_path=out_dir / f"property_density_train.{args.format}",
        properties=properties,
    )
    plot_property_grid(
        test_df,
        out_path=out_dir / f"property_density_test.{args.format}",
        properties=properties,
    )
    plot_property_grid(
        all_df,
        out_path=out_dir / f"property_density_all.{args.format}",
        properties=properties,
    )

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

