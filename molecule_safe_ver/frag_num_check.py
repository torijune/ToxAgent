"""
commom_frage_pairs_with_smiles.csv 의 only_toxic_safe_fragments, only_nontoxic_safe_fragments에 대해
- 각 row의 fragment 개수 (1개 또는 여러 개, dot 구분)
- 각 fragment의 길이(문자 수) 및 크기 관련 통계
를 계산하여 출력·저장한다.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles.csv"
OUTPUT_CSV = SCRIPT_DIR / "frag_stats_only_toxic_nontoxic.csv"
OUTPUT_JSON = SCRIPT_DIR / "frag_stats_only_toxic_nontoxic.json"

COL_TOXIC = "only_toxic_safe_fragments"
COL_NONTOXIC = "only_nontoxic_safe_fragments"
SEP = "."


def parse_fragments(s: str) -> tuple[int, list[int]]:
    """
    SAFE fragment 문자열(dot 구분)에서 fragment 개수와 각 fragment의 문자 길이 리스트 반환.
    빈 문자열/NaN이면 (0, []).
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 0, []
    s = str(s).strip()
    if not s:
        return 0, []
    parts = [p.strip() for p in s.split(SEP) if p.strip()]
    n = len(parts)
    lengths = [len(p) for p in parts]
    return n, lengths


def collect_stats(series: pd.Series) -> dict:
    """한 컬럼(only_toxic 또는 only_nontoxic)에 대해 통계 수집."""
    n_frags_per_row: list[int] = []
    all_lengths: list[int] = []
    for val in series:
        n, lengths = parse_fragments(val)
        if n > 0:
            n_frags_per_row.append(n)
            all_lengths.extend(lengths)

    if not n_frags_per_row:
        return {
            "n_rows_with_frags": 0,
            "n_rows_empty": len(series),
            "n_fragments_per_row_mean": 0.0,
            "n_fragments_per_row_median": 0.0,
            "n_fragments_per_row_std": 0.0,
            "n_fragments_per_row_min": 0,
            "n_fragments_per_row_max": 0,
            "total_fragments": 0,
            "frag_length_mean": 0.0,
            "frag_length_median": 0.0,
            "frag_length_std": 0.0,
            "frag_length_min": 0,
            "frag_length_max": 0,
            "dist_n_fragments": {},
            "dist_length_bins": {},
        }

    n_frags_arr = np.array(n_frags_per_row)
    len_arr = np.array(all_lengths)
    dist_n = Counter(n_frags_per_row)
    # 길이 구간별 분포 (1–5, 6–10, 11–20, 21–50, 51+)
    bins_edges = [0, 5, 10, 20, 50, 10_000]
    hist, _ = np.histogram(len_arr, bins=bins_edges)
    bin_labels = ["1-5", "6-10", "11-20", "21-50", "51+"]
    dist_bins = {label: int(hist[i]) for i, label in enumerate(bin_labels)}

    return {
        "n_rows_with_frags": len(n_frags_per_row),
        "n_rows_empty": int(len(series) - len(n_frags_per_row)),
        "n_fragments_per_row_mean": float(np.mean(n_frags_arr)),
        "n_fragments_per_row_median": float(np.median(n_frags_arr)),
        "n_fragments_per_row_std": float(np.std(n_frags_arr)),
        "n_fragments_per_row_min": int(np.min(n_frags_arr)),
        "n_fragments_per_row_max": int(np.max(n_frags_arr)),
        "total_fragments": len(all_lengths),
        "frag_length_mean": float(np.mean(len_arr)),
        "frag_length_median": float(np.median(len_arr)),
        "frag_length_std": float(np.std(len_arr)),
        "frag_length_min": int(np.min(len_arr)),
        "frag_length_max": int(np.max(len_arr)),
        "dist_n_fragments": dict(sorted(dist_n.items())),
        "dist_length_bins": dist_bins,
    }


def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    for col in [COL_TOXIC, COL_NONTOXIC]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    stats_toxic = collect_stats(df[COL_TOXIC])
    stats_nontoxic = collect_stats(df[COL_NONTOXIC])

    # 요약 테이블: row당 fragment 개수
    print("\n" + "=" * 80)
    print("Row당 fragment 개수 (only_toxic / only_nontoxic)")
    print("=" * 80)
    print(f"{'column':<28} {'n_rows':>8} {'mean':>8} {'median':>8} {'std':>8} {'min':>6} {'max':>6}")
    print("-" * 80)
    for name, s in [("only_toxic_safe_fragments", stats_toxic), ("only_nontoxic_safe_fragments", stats_nontoxic)]:
        print(
            f"{name:<28} {s['n_rows_with_frags']:>8} "
            f"{s['n_fragments_per_row_mean']:>8.2f} {s['n_fragments_per_row_median']:>8.1f} "
            f"{s['n_fragments_per_row_std']:>8.2f} {s['n_fragments_per_row_min']:>6} {s['n_fragments_per_row_max']:>6}"
        )

    # 요약 테이블: fragment 길이(문자 수)
    print("\n" + "=" * 80)
    print("Fragment 길이(문자 수) 통계")
    print("=" * 80)
    print(f"{'column':<28} {'N_frag':>8} {'mean':>8} {'median':>8} {'std':>8} {'min':>6} {'max':>6}")
    print("-" * 80)
    for name, s in [("only_toxic_safe_fragments", stats_toxic), ("only_nontoxic_safe_fragments", stats_nontoxic)]:
        print(
            f"{name:<28} {s['total_fragments']:>8} "
            f"{s['frag_length_mean']:>8.2f} {s['frag_length_median']:>8.1f} "
            f"{s['frag_length_std']:>8.2f} {s['frag_length_min']:>6} {s['frag_length_max']:>6}"
        )

    # Row당 fragment 개수 분포
    print("\n" + "=" * 80)
    print("Row당 fragment 개수 분포 (n_fragments → row 수)")
    print("=" * 80)
    for name, s in [("only_toxic_safe_fragments", stats_toxic), ("only_nontoxic_safe_fragments", stats_nontoxic)]:
        print(f"\n{name}:")
        for k, v in s["dist_n_fragments"].items():
            print(f"  {k}개: {v} rows")

    # Fragment 길이 구간별 분포
    print("\n" + "=" * 80)
    print("Fragment 길이 구간별 개수 (문자 수)")
    print("=" * 80)
    for name, s in [("only_toxic_safe_fragments", stats_toxic), ("only_nontoxic_safe_fragments", stats_nontoxic)]:
        print(f"\n{name}: {s['dist_length_bins']}")

    # CSV 저장 (flat 요약)
    out_rows = []
    for col_name, s in [("only_toxic_safe_fragments", stats_toxic), ("only_nontoxic_safe_fragments", stats_nontoxic)]:
        out_rows.append({
            "column": col_name,
            "n_rows_with_frags": s["n_rows_with_frags"],
            "n_rows_empty": s["n_rows_empty"],
            "n_fragments_per_row_mean": round(s["n_fragments_per_row_mean"], 4),
            "n_fragments_per_row_median": round(s["n_fragments_per_row_median"], 4),
            "n_fragments_per_row_std": round(s["n_fragments_per_row_std"], 4),
            "n_fragments_per_row_min": s["n_fragments_per_row_min"],
            "n_fragments_per_row_max": s["n_fragments_per_row_max"],
            "total_fragments": s["total_fragments"],
            "frag_length_mean": round(s["frag_length_mean"], 4),
            "frag_length_median": round(s["frag_length_median"], 4),
            "frag_length_std": round(s["frag_length_std"], 4),
            "frag_length_min": s["frag_length_min"],
            "frag_length_max": s["frag_length_max"],
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # JSON 저장 (분포 포함)
    import json
    export = {
        "only_toxic_safe_fragments": {k: v for k, v in stats_toxic.items()},
        "only_nontoxic_safe_fragments": {k: v for k, v in stats_nontoxic.items()},
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
