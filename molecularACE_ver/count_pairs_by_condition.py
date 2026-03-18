"""
저장된 sim_matrices/*.npz를 읽어, 각 조건별로 유사도 ≥ 0.9 인 (toxic, nontoxic) 쌍 수를 집계합니다.

OR 조건이므로 "해당 조건 하나만 만족하는 수"와 "전체 조건 중 하나라도 만족(OR) 수"를 함께 출력합니다.
ToxCast(toxcast_df)는 제외합니다.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SIM_MATRICES_DIR = OUT_DIR / "sim_matrices"
SIM_THRESHOLD = 0.9


def main() -> None:
    if not SIM_MATRICES_DIR.exists():
        print(f"Not found: {SIM_MATRICES_DIR}")
        return

    exclude = {"toxcast_df"}
    rows = []
    totals = {
        "sub_wo": 0,
        "sub_w_chiral": 0,
        "scaffold": 0,
        "smiles": 0,
        "or_any": 0,
    }

    for dataset_dir in sorted(SIM_MATRICES_DIR.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in exclude:
            continue
        dataset = dataset_dir.name
        for npz_path in sorted(dataset_dir.glob("*.npz")):
            endpoint = npz_path.stem
            try:
                data = np.load(npz_path)
                sub_wo = data["substructure_sim"]
                sub_w = data["substructure_sim_w_chiral"]
                scaffold = data["scaffold_sim"]
                smiles = data["smiles_sim"]
            except Exception as e:
                print(f"Skip {dataset}/{endpoint}: {e}")
                continue

            n_sub_wo = int(np.sum(np.isfinite(sub_wo) & (sub_wo >= SIM_THRESHOLD)))
            n_sub_w = int(np.sum(np.isfinite(sub_w) & (sub_w >= SIM_THRESHOLD)))
            n_scaffold = int(np.sum(np.isfinite(scaffold) & (scaffold >= SIM_THRESHOLD)))
            n_smiles = int(np.sum(np.isfinite(smiles) & (smiles >= SIM_THRESHOLD)))
            or_mask = (
                (np.isfinite(sub_wo) & (sub_wo >= SIM_THRESHOLD))
                | (np.isfinite(sub_w) & (sub_w >= SIM_THRESHOLD))
                | (np.isfinite(scaffold) & (scaffold >= SIM_THRESHOLD))
                | (np.isfinite(smiles) & (smiles >= SIM_THRESHOLD))
            )
            n_or = int(np.sum(or_mask))

            rows.append({
                "dataset": dataset,
                "endpoint": endpoint,
                "sub_wo_ge09": n_sub_wo,
                "sub_w_chiral_ge09": n_sub_w,
                "scaffold_ge09": n_scaffold,
                "smiles_ge09": n_smiles,
                "or_any_ge09": n_or,
            })
            totals["sub_wo"] += n_sub_wo
            totals["sub_w_chiral"] += n_sub_w
            totals["scaffold"] += n_scaffold
            totals["smiles"] += n_smiles
            totals["or_any"] += n_or

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "pair_counts_by_condition.csv"
    if len(df) > 0:
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv} ({len(df)} endpoints)")
    else:
        print("No endpoints processed.")
        return

    print()
    print("=" * 60)
    print("각 조건별 유사도 ≥ 0.9 인 (toxic, nontoxic) 쌍 수 (ToxCast 제외)")
    print("=" * 60)
    print(f"  Substructure (w/o chirality) ≥ 0.9  : {totals['sub_wo']:>12,}  (FG용)")
    print(f"  Substructure (w/ chirality)  ≥ 0.9  : {totals['sub_w_chiral']:>12,}  (Stereo/Isomer용)")
    print(f"  Scaffold                         ≥ 0.9  : {totals['scaffold']:>12,}")
    print(f"  SMILES similarity                 ≥ 0.9  : {totals['smiles']:>12,}")
    print("-" * 60)
    print(f"  OR (위 조건 중 하나라도 만족)        : {totals['or_any']:>12,}")
    print()
    print("※ 동일 (i,j)가 여러 조건을 동시에 만족할 수 있어서,")
    print("  위 네 개 수의 합과 OR 수는 일치하지 않습니다.")


if __name__ == "__main__":
    main()
