"""
FG diff 전용 CSV와 stereo diff 전용(loose) CSV를 하나로 합칩니다.

- Merge key: (dataset_name, endpoint, toxic_smiles, nontoxic_smiles)
- Outer join으로 합쳐서, FG에만 있는 pair / stereo에만 있는 pair / 둘 다 있는 pair 모두 포함
- FG 관련 컬럼, stereo 관련 컬럼을 모두 유지해 두 diff가 모두 있는 pair를 식별 가능하게 함

출력: molecular_feature/pairs_fg_stereo_merged.csv
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
FG_CSV = BASE / "functional_group" / "pairs_fg_diff_only.csv"
STEREo_CSV = BASE / "stereochemistry" / "pairs_stereo_diff_only_loose.csv"
OUT_CSV = BASE / "pairs_fg_stereo_merged.csv"

KEY_COLS = ["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"]


def main():
    if not FG_CSV.exists():
        raise FileNotFoundError(f"Not found: {FG_CSV}")
    if not STEREo_CSV.exists():
        raise FileNotFoundError(f"Not found: {STEREo_CSV}")

    fg_df = pd.read_csv(FG_CSV)
    stereo_df = pd.read_csv(STEREo_CSV)

    for c in KEY_COLS:
        if c not in fg_df.columns or c not in stereo_df.columns:
            raise ValueError(f"Missing key column: {c}")

    # Outer merge: 한쪽에만 있어도 행 유지
    merged = pd.merge(
        fg_df,
        stereo_df,
        on=KEY_COLS,
        how="outer",
        suffixes=("_fg", "_stereo"),
    )

    # 공통 컬럼(KEY 제외)이면 _fg, _stereo 붙어 있음 → 하나로 합침
    all_cols = list(merged.columns)
    drop_cols = []
    for c in all_cols:
        if c in KEY_COLS:
            continue
        if c.endswith("_fg"):
            base = c[:-3]  # remove "_fg"
            if f"{base}_stereo" in merged.columns:
                # 동일 컬럼이 양쪽에 있음 → 한쪽 값으로 채우기 (있으면 그대로, 없으면 다른 쪽)
                merged[base] = merged[c].fillna(merged[f"{base}_stereo"])
                drop_cols.extend([c, f"{base}_stereo"])
        elif c.endswith("_stereo") and c not in drop_cols:
            base = c[:-7]
            if f"{base}_fg" in merged.columns:
                continue  # 이미 위에서 처리
            merged[base] = merged[c]
            drop_cols.append(c)

    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # 컬럼 순서 정리: key → 공통(스캐폴드, tanimoto, delta 등) → FG 전용 → stereo 전용
    key_first = [c for c in KEY_COLS if c in merged.columns]
    common = [
        "toxic_scaffold_smiles", "nontoxic_scaffold_smiles",
        "tanimoto_sim", "delta_MW", "delta_logP", "delta_TPSA",
        "delta_HBD", "delta_HBA", "delta_RotB", "delta_RingCount", "delta_AromRingCount",
    ]
    fg_cols = [
        "toxic_canonical_smiles", "nontoxic_canonical_smiles",
        "toxic_fg_names", "toxic_fg_counts", "toxic_total_fg_count", "toxic_fg_full",
        "nontoxic_fg_names", "nontoxic_fg_counts", "nontoxic_total_fg_count", "nontoxic_fg_full",
        "has_fg_diff", "unique_fg",
    ]
    stereo_cols = [
        "toxic_chiral_centers", "toxic_ez_bonds", "toxic_has_chirality", "toxic_has_ez_bonds",
        "nontoxic_chiral_centers", "nontoxic_ez_bonds", "nontoxic_has_chirality", "nontoxic_has_ez_bonds",
        "chiral_diff_loose", "ez_diff_loose", "stereo_diff_type_loose",
    ]
    order = key_first + [c for c in common if c in merged.columns]
    order += [c for c in fg_cols if c in merged.columns]
    order += [c for c in stereo_cols if c in merged.columns]
    order += [c for c in merged.columns if c not in order]
    merged = merged[order]

    merged.to_csv(OUT_CSV, index=False)

    n_both = ((merged["has_fg_diff"] == True) & (merged["stereo_diff_type_loose"].notna()) & (merged["stereo_diff_type_loose"] != "")).sum()
    n_fg_only = ((merged["has_fg_diff"] == True) & ((merged["stereo_diff_type_loose"].isna()) | (merged["stereo_diff_type_loose"] == ""))).sum()
    n_stereo_only = (((merged["has_fg_diff"] != True) | merged["has_fg_diff"].isna()) & (merged["stereo_diff_type_loose"].notna()) & (merged["stereo_diff_type_loose"] != "")).sum()

    print(f"Saved: {OUT_CSV}")
    print(f"  Total rows: {len(merged):,}")
    print(f"  FG diff only: {n_fg_only:,}")
    print(f"  Stereo diff only (loose): {n_stereo_only:,}")
    print(f"  Both FG diff and stereo diff: {n_both:,}")


if __name__ == "__main__":
    main()
