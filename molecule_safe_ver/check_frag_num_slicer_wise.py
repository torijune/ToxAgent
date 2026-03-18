"""
smiles_to_safe_by_slicer.csv 에서 각 slicer별로:
1) 분자당 fragment 개수 통계 (dot 개수 → fragment 수 = dot 수 + 1, 빈 칸 제외)
2) fragment 크기(복잡도) 통계: 각 fragment 문자열 길이 기준
"""
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "smiles_to_safe_by_slicer.csv"
OUTPUT_CSV = SCRIPT_DIR / "slicer_frag_stats.csv"

SLICER_COLS = ["hr_safe", "rotatable_safe", "recap_safe", "mmpa_safe", "attach_safe", "brics_safe"]
SEP = "."


def get_n_fragments_and_sizes(safe_str) -> tuple[int, list[int]]:
    """
    SAFE 문자열에서 fragment 개수와 각 fragment의 크기(문자 수) 반환.
    빈 문자열/NaN이면 (0, []).
    """
    if safe_str is None or (isinstance(safe_str, float) and np.isnan(safe_str)):
        return 0, []
    s = str(safe_str).strip()
    if not s:
        return 0, []
    parts = [p.strip() for p in s.split(SEP) if p.strip()]
    n = len(parts)
    sizes = [len(p) for p in parts]
    return n, sizes


def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    for c in SLICER_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # slicer별로 분자당 fragment 수, 전체 fragment 크기 수집
    stats = {}
    for col in SLICER_COLS:
        n_frags_per_mol = []
        all_frag_sizes = []
        for safe_str in tqdm(df[col], desc=col.replace("_safe", ""), leave=False):
            n, sizes = get_n_fragments_and_sizes(safe_str)
            if n > 0:
                n_frags_per_mol.append(n)
                all_frag_sizes.extend(sizes)
        stats[col] = {
            "n_molecules_with_frags": len(n_frags_per_mol),
            "n_molecules_empty": len(df) - len(n_frags_per_mol),
            "n_fragments_mean": np.mean(n_frags_per_mol) if n_frags_per_mol else 0,
            "n_fragments_median": np.median(n_frags_per_mol) if n_frags_per_mol else 0,
            "n_fragments_std": np.std(n_frags_per_mol) if n_frags_per_mol else 0,
            "n_fragments_min": min(n_frags_per_mol) if n_frags_per_mol else 0,
            "n_fragments_max": max(n_frags_per_mol) if n_frags_per_mol else 0,
            "frag_size_mean": np.mean(all_frag_sizes) if all_frag_sizes else 0,
            "frag_size_median": np.median(all_frag_sizes) if all_frag_sizes else 0,
            "frag_size_std": np.std(all_frag_sizes) if all_frag_sizes else 0,
            "frag_size_min": min(all_frag_sizes) if all_frag_sizes else 0,
            "frag_size_max": max(all_frag_sizes) if all_frag_sizes else 0,
            "total_fragments": len(all_frag_sizes),
        }

    # 요약 테이블 출력
    slicer_short = [c.replace("_safe", "") for c in SLICER_COLS]
    print("\n" + "=" * 80)
    print("분자당 fragment 개수 (분자당 dot+1, 빈 SAFE 제외)")
    print("=" * 80)
    print(f"{'slicer':<12} {'n_mol':>8} {'mean':>8} {'median':>8} {'std':>8} {'min':>6} {'max':>6}")
    print("-" * 60)
    for col in SLICER_COLS:
        s = stats[col]
        name = col.replace("_safe", "")
        print(
            f"{name:<12} {s['n_molecules_with_frags']:>8} "
            f"{s['n_fragments_mean']:>8.2f} {s['n_fragments_median']:>8.1f} "
            f"{s['n_fragments_std']:>8.2f} {s['n_fragments_min']:>6} {s['n_fragments_max']:>6}"
        )

    print("\n" + "=" * 80)
    print("Fragment 크기(복잡도) — 각 fragment 문자열 길이 통계")
    print("=" * 80)
    print(f"{'slicer':<12} {'N_frag':>8} {'mean':>8} {'median':>8} {'std':>8} {'min':>6} {'max':>6}")
    print("-" * 60)
    for col in SLICER_COLS:
        s = stats[col]
        name = col.replace("_safe", "")
        print(
            f"{name:<12} {s['total_fragments']:>8} "
            f"{s['frag_size_mean']:>8.2f} {s['frag_size_median']:>8.1f} "
            f"{s['frag_size_std']:>8.2f} {s['frag_size_min']:>6} {s['frag_size_max']:>6}"
        )

    # CSV로 저장 (행: slicer, 열: 지표)
    out_rows = []
    for col in SLICER_COLS:
        s = stats[col]
        out_rows.append({
            "slicer": col.replace("_safe", ""),
            "n_molecules_with_frags": s["n_molecules_with_frags"],
            "n_molecules_empty": s["n_molecules_empty"],
            "n_fragments_mean": round(s["n_fragments_mean"], 4),
            "n_fragments_median": round(s["n_fragments_median"], 4),
            "n_fragments_std": round(s["n_fragments_std"], 4),
            "n_fragments_min": s["n_fragments_min"],
            "n_fragments_max": s["n_fragments_max"],
            "total_fragments": s["total_fragments"],
            "frag_size_mean": round(s["frag_size_mean"], 4),
            "frag_size_median": round(s["frag_size_median"], 4),
            "frag_size_std": round(s["frag_size_std"], 4),
            "frag_size_min": s["frag_size_min"],
            "frag_size_max": s["frag_size_max"],
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
