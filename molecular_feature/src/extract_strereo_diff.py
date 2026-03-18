"""
유사도로 필터링된 pair들(pairs_with_stereochemistry.csv) 중에서
toxic–nontoxic 간 stereochemistry 차이가 있는 pair만 추출합니다.

차이 조건:
1) 둘 다 chiral center 있음 + chiral center 정보(개수·R/S 구성) 다름  → chiral_diff
2) 둘 다 E/Z bond 있음 + E/Z bond 정보(개수·E/Z 구성) 다름         → ez_diff
3) 위 두 조건 모두 성립                                        → both

위 1~3 중 하나라도 만족하면 stereo_diff 있는 pair로 필터링해 저장합니다.
"""
from pathlib import Path
import ast
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE / "stereochemistry" / "pairs_with_stereochemistry.csv"
OUT_DIR = BASE / "stereochemistry"
OUT_CSV = OUT_DIR / "pairs_stereo_diff_only.csv"


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


def _chiral_info_different(t_centers: list, n_centers: list) -> bool:
    """둘 다 chiral center 있을 때, 정보(개수·R/S 구성)가 다른지."""
    if not t_centers or not n_centers:
        return False
    t_configs = tuple(sorted(c.get("config", "") for c in t_centers if isinstance(c, dict)))
    n_configs = tuple(sorted(c.get("config", "") for c in n_centers if isinstance(c, dict)))
    return (len(t_configs) != len(n_configs)) or (t_configs != n_configs)


def _ez_info_different(t_ez: list, n_ez: list) -> bool:
    """둘 다 E/Z bond 있을 때, 정보(개수·E/Z 구성)가 다른지."""
    if not t_ez or not n_ez:
        return False
    t_geoms = tuple(sorted(b.get("geometry", "") for b in t_ez if isinstance(b, dict)))
    n_geoms = tuple(sorted(b.get("geometry", "") for b in n_ez if isinstance(b, dict)))
    return (len(t_geoms) != len(n_geoms)) or (t_geoms != n_geoms)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = [
        "toxic_has_chirality", "toxic_chiral_centers", "toxic_has_ez_bonds", "toxic_ez_bonds",
        "nontoxic_has_chirality", "nontoxic_chiral_centers", "nontoxic_has_ez_bonds", "nontoxic_ez_bonds",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    chiral_diff = []
    ez_diff = []
    stereo_diff_type = []  # "chiral_only" | "ez_only" | "both"

    for _, row in df.iterrows():
        t_has_c = _parse_bool(row["toxic_has_chirality"])
        n_has_c = _parse_bool(row["nontoxic_has_chirality"])
        t_centers = _parse_chiral_centers(row["toxic_chiral_centers"])
        n_centers = _parse_chiral_centers(row["nontoxic_chiral_centers"])

        t_has_ez = _parse_bool(row["toxic_has_ez_bonds"])
        n_has_ez = _parse_bool(row["nontoxic_has_ez_bonds"])
        t_ez = _parse_ez_bonds(row["toxic_ez_bonds"])
        n_ez = _parse_ez_bonds(row["nontoxic_ez_bonds"])

        c_diff = t_has_c and n_has_c and _chiral_info_different(t_centers, n_centers)
        e_diff = t_has_ez and n_has_ez and _ez_info_different(t_ez, n_ez)

        chiral_diff.append(c_diff)
        ez_diff.append(e_diff)
        if c_diff and e_diff:
            stereo_diff_type.append("both")
        elif c_diff:
            stereo_diff_type.append("chiral_only")
        elif e_diff:
            stereo_diff_type.append("ez_only")
        else:
            stereo_diff_type.append("")

    df = df.copy()
    df["chiral_diff"] = chiral_diff
    df["ez_diff"] = ez_diff
    df["stereo_diff_type"] = stereo_diff_type

    # 필터: stereo 차이가 있는 pair만 (1~3 중 하나라도 만족)
    mask = (df["stereo_diff_type"] != "")
    out_df = df.loc[mask].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    n_total = len(df)
    n_out = len(out_df)
    print(f"Saved: {OUT_CSV}")
    print(f"  Total pairs (input): {n_total:,}")
    print(f"  Pairs with stereo diff: {n_out:,}")
    print(f"  chiral_only: {(df['stereo_diff_type'] == 'chiral_only').sum():,}")
    print(f"  ez_only:     {(df['stereo_diff_type'] == 'ez_only').sum():,}")
    print(f"  both:       {(df['stereo_diff_type'] == 'both').sum():,}")


if __name__ == "__main__":
    main()
