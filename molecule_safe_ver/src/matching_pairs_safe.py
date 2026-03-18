"""
smiles_to_safe.csv 매칭 테이블을 사용해
pairs_fg_stereo_merged_nodot.csv의 toxic_smiles, nontoxic_smiles에 대응하는
toxic_safe, nontoxic_safe를 붙여 pairs_safe.csv를 생성.
"""
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

SMILES_TO_SAFE_CSV = SCRIPT_DIR / "smiles_to_safe.csv"
PAIRS_CSV = PROJECT_ROOT / "molecular_feature" / "pairs_fg_stereo_merged_nodot.csv"
OUTPUT_CSV = SCRIPT_DIR / "pairs_safe.csv"


def main():
    print(f"Loading mapping: {SMILES_TO_SAFE_CSV}")
    map_df = pd.read_csv(SMILES_TO_SAFE_CSV)
    # smiles -> safe (원본 SMILES 기준 매칭; canonical_smiles로도 보조 매칭)
    smiles_to_safe = dict(zip(map_df["smiles"].astype(str).str.strip(), map_df["safe"].fillna("")))
    canon_to_safe = dict(
        zip(
            map_df["canonical_smiles"].astype(str).str.strip(),
            map_df["safe"].fillna(""),
        )
    )
    # 빈 문자열이 아닌 것만 사용해 canonical으로 보조 조회
    canon_to_safe = {k: v for k, v in canon_to_safe.items() if k and str(k) != "nan"}

    print(f"Loading pairs: {PAIRS_CSV}")
    df = pd.read_csv(PAIRS_CSV)
    n = len(df)

    def lookup(smiles_series):
        out = []
        for s in smiles_series:
            s = str(s).strip() if pd.notna(s) else ""
            safe_str = smiles_to_safe.get(s)
            if safe_str is None and s:
                safe_str = canon_to_safe.get(s, "")
            out.append(safe_str if safe_str is not None else "")
        return out

    # assign() 사용으로 pandas 3.0 Copy-on-Write 경고 방지
    df = df.assign(
        toxic_safe=lookup(df["toxic_smiles"]),
        nontoxic_safe=lookup(df["nontoxic_smiles"]),
    )

    toxic_miss = (df["toxic_safe"] == "").sum()
    nontoxic_miss = (df["nontoxic_safe"] == "").sum()
    print(f"Rows with missing toxic_safe: {toxic_miss} / {n}")
    print(f"Rows with missing nontoxic_safe: {nontoxic_miss} / {n}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
