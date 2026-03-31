#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _read_csv(csv_path: Path) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
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
            "axes.linewidth": 1.0,
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
                alpha=0.35,
                linewidth=2.0,
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
    title: str,
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


def plot_property_grid(
    df: pd.DataFrame,
    *,
    split_name: str,
    out_path: Path,
    properties: List[Tuple[str, str]],
    fig_title: str,
) -> None:
    _set_plot_style()

    # 예시 이미지 팔레트 느낌(보라/파랑)
    tox_color = "#8B5A73"  # muted purple (Source)
    non_color = "#2E6FBA"  # blue (Recover)

    n = len(properties)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 7.5), constrained_layout=True)
    if nrows == 1:
        axes = [axes]  # type: ignore

    flat_axes = [ax for row in axes for ax in (row if isinstance(row, Iterable) else [row])]  # type: ignore

    for i, (prop, label) in enumerate(properties):
        ax = flat_axes[i]
        _plot_one_property(ax, df, prop, title=label, tox_color=tox_color, non_color=non_color)
        ax.set_xlabel(label)
        if i == 0:
            ax.legend(frameon=True, fontsize=9, loc="upper left")
        else:
            ax.legend().remove()

    # hide unused axes
    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle(f"{fig_title} ({split_name})", fontsize=16, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot toxic vs nontoxic molecular property density grids for train/test splits."
    )
    ap.add_argument("--train-csv", type=Path, required=True, help="merged_train.csv path")
    ap.add_argument("--test-csv", type=Path, required=True, help="merged_test.csv path")
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
        split_name="train",
        out_path=out_dir / f"property_density_train.{args.format}",
        properties=properties,
        fig_title="Toxic vs Nontoxic property distributions",
    )
    plot_property_grid(
        test_df,
        split_name="test",
        out_path=out_dir / f"property_density_test.{args.format}",
        properties=properties,
        fig_title="Toxic vs Nontoxic property distributions",
    )
    plot_property_grid(
        all_df,
        split_name="all",
        out_path=out_dir / f"property_density_all.{args.format}",
        properties=properties,
        fig_title="Toxic vs Nontoxic property distributions",
    )

    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

