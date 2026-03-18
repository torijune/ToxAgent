"""
저장된 sim_matrices에서 endpoint별 FT sim (w/ chirality, w/o chirality)을 읽어,
임계값 0.5 ~ 0.9 (0.05 단위)별 샘플 수 분포를 dataset·전체로 집계합니다.

출력 CSV 컬럼: dataset, 0.5_w_chiar, 0.5_wo_chiar, 0.55_w_chiar, 0.55_wo_chiar, ..., 0.9_w_chiar, 0.9_wo_chiar
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

OUT_DIR = Path(__file__).resolve().parent
SIM_MATRICES_DIR = OUT_DIR / "sim_matrices"

# 0.5, 0.55, 0.6, ..., 0.9
THRESHOLDS = [round(0.5 + i * 0.05, 2) for i in range(9)]


def main() -> None:
    if not SIM_MATRICES_DIR.exists():
        print(f"Not found: {SIM_MATRICES_DIR}")
        return

    exclude = {"toxcast_df"}

    # dataset -> { thresh: (count_wo, count_w) }  (sum over endpoints in dataset)
    by_dataset: dict[str, dict[float, tuple[int, int]]] = {}
    total_counts: dict[float, tuple[int, int]] = {t: (0, 0) for t in THRESHOLDS}

    for dataset_dir in sorted(SIM_MATRICES_DIR.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in exclude:
            continue
        dataset = dataset_dir.name
        if dataset not in by_dataset:
            by_dataset[dataset] = {t: (0, 0) for t in THRESHOLDS}

        for npz_path in tqdm(sorted(dataset_dir.glob("*.npz")), desc=dataset, leave=False):
            try:
                data = np.load(npz_path)
                sub_wo = data["substructure_sim"]
                sub_w = data["substructure_sim_w_chiral"]
            except Exception:
                continue

            for t in THRESHOLDS:
                n_wo = int(np.sum(np.isfinite(sub_wo) & (sub_wo >= t)))
                n_w = int(np.sum(np.isfinite(sub_w) & (sub_w >= t)))
                prev_wo, prev_w = by_dataset[dataset][t]
                by_dataset[dataset][t] = (prev_wo + n_wo, prev_w + n_w)
                tot_wo, tot_w = total_counts[t]
                total_counts[t] = (tot_wo + n_wo, tot_w + n_w)

    # Build table: dataset, 0.5_w_chiar, 0.5_wo_chiar, 0.55_w_chiar, ...
    col_order = []
    for t in THRESHOLDS:
        col_order.append(f"{t}_w_chiar")
        col_order.append(f"{t}_wo_chiar")

    rows = []
    for dataset in sorted(by_dataset.keys()):
        row = {"dataset": dataset}
        for t in THRESHOLDS:
            c_wo, c_w = by_dataset[dataset][t]
            row[f"{t}_w_chiar"] = c_w
            row[f"{t}_wo_chiar"] = c_wo
        rows.append(row)

    # total row
    total_row = {"dataset": "total"}
    for t in THRESHOLDS:
        total_row[f"{t}_w_chiar"] = total_counts[t][1]
        total_row[f"{t}_wo_chiar"] = total_counts[t][0]
    rows.append(total_row)

    df = pd.DataFrame(rows)
    df = df[["dataset"] + col_order]

    out_csv = OUT_DIR / "ft_sim_threshold_distribution.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print()
    print("FT sim (w/ chirality, w/o chirality) 임계값별 샘플 수")
    print("(각 셀: 해당 threshold 이상인 (toxic, nontoxic) 쌍 수)")
    print()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 20)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
