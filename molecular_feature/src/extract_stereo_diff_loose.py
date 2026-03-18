"""
Mol_stereo 방식(loose)으로 stereo 차이가 있는 pair만 필터링합니다.

차이 조건 (Mol_stereo.has_stereochemistry_difference와 동일한 관대한 기준):
- 한쪽만 chiral center 있음 / 한쪽만 E·Z bond 있음 → 차이
- chiral center 개수 차이 (count_change != 0) → 차이
- R/S 개수 차이 (chirality_type_changes) → 차이
- E/Z bond 개수 차이 (count_change != 0) → 차이
- E/Z 기하 개수 차이 (geometry_changes) → 차이

입력: pairs_with_stereochemistry.csv
출력: pairs_stereo_diff_only_loose.csv (이전 파일명 + _loose.csv)
"""
from pathlib import Path
import ast
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE / "stereochemistry" / "pairs_with_stereochemistry.csv"
OUT_DIR = BASE / "stereochemistry"
OUT_CSV = OUT_DIR / "pairs_stereo_diff_only_loose.csv"


def _parse_chiral_centers(s):
    if pd.isna(s) or s == "[]" or str(s).strip() == "":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _parse_ez_bonds(s):
    if pd.isna(s) or s == "[]" or str(s).strip() == "":
        return []
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _parse_bool(s):
    if pd.isna(s):
        return False
    if isinstance(s, bool):
        return s
    return str(s).strip().lower() in ("true", "1", "yes")


def _has_stereochemistry_difference_loose(
    t_has_chirality: bool,
    n_has_chirality: bool,
    t_centers: list,
    n_centers: list,
    t_has_ez: bool,
    n_has_ez: bool,
    t_ez: list,
    n_ez: list,
) -> tuple[bool, bool, str]:
    """
    Mol_stereo 스타일로 stereo 차이 여부 판단 (관대한 기준).

    Returns:
        (chiral_diff_loose, ez_diff_loose, stereo_diff_type_loose)
        stereo_diff_type_loose: "both" | "chiral_only" | "ez_only" | ""
    """
    # Chiral: unique (한쪽만 있음) or count change or R/S type change
    t_R = sum(1 for c in t_centers if isinstance(c, dict) and c.get("config") == "R")
    t_S = sum(1 for c in t_centers if isinstance(c, dict) and c.get("config") == "S")
    n_R = sum(1 for c in n_centers if isinstance(c, dict) and c.get("config") == "R")
    n_S = sum(1 for c in n_centers if isinstance(c, dict) and c.get("config") == "S")

    chiral_count_diff = len(t_centers) != len(n_centers)
    chiral_R_change = (n_R - t_R) != 0
    chiral_S_change = (n_S - t_S) != 0
    one_has_chiral_other_not = t_has_chirality != n_has_chirality

    chiral_diff_loose = (
        one_has_chiral_other_not
        or chiral_count_diff
        or chiral_R_change
        or chiral_S_change
    )

    # E/Z: unique (한쪽만 있음) or count change or E/Z geometry change
    t_E = sum(1 for b in t_ez if isinstance(b, dict) and b.get("geometry") == "E")
    t_Z = sum(1 for b in t_ez if isinstance(b, dict) and b.get("geometry") == "Z")
    n_E = sum(1 for b in n_ez if isinstance(b, dict) and b.get("geometry") == "E")
    n_Z = sum(1 for b in n_ez if isinstance(b, dict) and b.get("geometry") == "Z")

    ez_count_diff = len(t_ez) != len(n_ez)
    ez_E_change = (n_E - t_E) != 0
    ez_Z_change = (n_Z - t_Z) != 0
    one_has_ez_other_not = t_has_ez != n_has_ez

    ez_diff_loose = (
        one_has_ez_other_not
        or ez_count_diff
        or ez_E_change
        or ez_Z_change
    )

    if chiral_diff_loose and ez_diff_loose:
        label = "both"
    elif chiral_diff_loose:
        label = "chiral_only"
    elif ez_diff_loose:
        label = "ez_only"
    else:
        label = ""

    return chiral_diff_loose, ez_diff_loose, label


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = [
        "toxic_has_chirality",
        "toxic_chiral_centers",
        "toxic_has_ez_bonds",
        "toxic_ez_bonds",
        "nontoxic_has_chirality",
        "nontoxic_chiral_centers",
        "nontoxic_has_ez_bonds",
        "nontoxic_ez_bonds",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    chiral_diff_loose = []
    ez_diff_loose = []
    stereo_diff_type_loose = []

    for _, row in df.iterrows():
        t_has_c = _parse_bool(row["toxic_has_chirality"])
        n_has_c = _parse_bool(row["nontoxic_has_chirality"])
        t_centers = _parse_chiral_centers(row["toxic_chiral_centers"])
        n_centers = _parse_chiral_centers(row["nontoxic_chiral_centers"])

        t_has_ez = _parse_bool(row["toxic_has_ez_bonds"])
        n_has_ez = _parse_bool(row["nontoxic_has_ez_bonds"])
        t_ez = _parse_ez_bonds(row["toxic_ez_bonds"])
        n_ez = _parse_ez_bonds(row["nontoxic_ez_bonds"])

        c_diff, e_diff, label = _has_stereochemistry_difference_loose(
            t_has_c, n_has_c, t_centers, n_centers,
            t_has_ez, n_has_ez, t_ez, n_ez,
        )
        chiral_diff_loose.append(c_diff)
        ez_diff_loose.append(e_diff)
        stereo_diff_type_loose.append(label)

    df = df.copy()
    df["chiral_diff_loose"] = chiral_diff_loose
    df["ez_diff_loose"] = ez_diff_loose
    df["stereo_diff_type_loose"] = stereo_diff_type_loose

    mask = df["stereo_diff_type_loose"] != ""
    out_df = df.loc[mask].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    n_total = len(df)
    n_out = len(out_df)
    print(f"Saved: {OUT_CSV}")
    print(f"  Total pairs (input): {n_total:,}")
    print(f"  Pairs with stereo diff (loose): {n_out:,}")
    print(f"  chiral_only: {(df['stereo_diff_type_loose'] == 'chiral_only').sum():,}")
    print(f"  ez_only:     {(df['stereo_diff_type_loose'] == 'ez_only').sum():,}")
    print(f"  both:        {(df['stereo_diff_type_loose'] == 'both').sum():,}")


if __name__ == "__main__":
    main()
