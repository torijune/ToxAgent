"""
commom_frage_pairs.csv에서:
- toxic_safe, nontoxic_safe: 전체 SAFE 문자열을 한 번에 decode → 분자 1개 SMILES
- toxic_safe_fragments, nontoxic_safe_fragments, only_toxic_*, only_nontoxic_*: dot 구분
  각 fragment를 따로 decode → fragment SMILES들을 dot으로 이어 붙임.
결과 컬럼을 추가해 새 CSV 저장.
"""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# safe 패키지 import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "safe"))

import datamol as dm
from safe import decode as safe_decode

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "commom_frage_pairs.csv"
OUTPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles.csv"

SEP = "."

# 출력에 넣을 기존 컬럼 + 새 컬럼
BASE_COLUMNS = [
    "dataset_name",
    "endpoint",
    "toxic_smiles",
    "nontoxic_smiles",
    "toxic_safe",
    "nontoxic_safe",
    "toxic_safe_fragments",
    "nontoxic_safe_fragments",
    "common_safe_fragments",
    "only_toxic_safe_fragments",
    "only_nontoxic_safe_fragments",
    "n_common_safe",
    "n_only_toxic_safe",
    "n_only_nontoxic_safe",
]
# 전체 SAFE 한 번에 decode (분자 1개 SMILES)
FULL_SAFE_DECODED_COLUMNS = [
    "toxic_safe_decoded_smiles",
    "nontoxic_safe_decoded_smiles",
]
# fragment별 decode (조각 SMILES들을 dot으로 이어 붙인 문자열)
NEW_SMILES_COLUMNS = [
    "toxic_safe_fragments_smiles",
    "nontoxic_safe_fragments_smiles",
    "only_toxic_safe_fragments_smiles",
    "only_nontoxic_safe_fragments_smiles",
]


def decode_full_safe_to_smiles(safe_str):
    """
    전체 SAFE 문자열을 한 번에 decode → 분자 1개 SMILES.
    (remove_dummies=True로 더미 제거 후 하나의 분자로 복원)
    """
    if pd.isna(safe_str) or not str(safe_str).strip():
        return ""
    with dm.without_rdkit_log():
        s = safe_decode(
            str(safe_str).strip(),
            as_mol=False,
            fix=True,
            remove_dummies=True,
            remove_added_hs=True,
            ignore_errors=True,
        )
    return s if s else ""


def decode_fragments_to_smiles(fragments_str, sep=SEP):
    """
    dot으로 구분된 SAFE fragment 문자열을 받아,
    각 fragment를 decode한 SMILES를 sep으로 이어 붙여 반환.
    decode 실패 시 해당 fragment는 빈 문자열 또는 실패 표시.
    """
    if pd.isna(fragments_str) or not str(fragments_str).strip():
        return ""
    parts = [s.strip() for s in str(fragments_str).split(sep) if s.strip()]
    decoded = []
    with dm.without_rdkit_log():
        for p in parts:
            s = safe_decode(
                p,
                as_mol=False,
                fix=True,
                remove_dummies=False,
                remove_added_hs=False,
                ignore_errors=True,
            )
            decoded.append(s if s else "")
    return sep.join(decoded)


def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    n = len(df)

    new_columns = {}

    # 전체 SAFE 문자열 한 번에 decode (toxic_safe, nontoxic_safe → 분자 1개 SMILES)
    for col, new_col in [
        ("toxic_safe", "toxic_safe_decoded_smiles"),
        ("nontoxic_safe", "nontoxic_safe_decoded_smiles"),
    ]:
        if col not in df.columns:
            new_columns[new_col] = [""] * n
            continue
        print(f"Decoding full SAFE {col} -> {new_col} (one SMILES per row) ...")
        new_columns[new_col] = [
            decode_full_safe_to_smiles(df[col].iloc[i]) for i in tqdm(range(n), desc=col)
        ]

    # fragment별 decode (dot 구분 조각들을 각각 decode 후 dot으로 이어 붙임)
    for col, new_col in [
        ("toxic_safe_fragments", "toxic_safe_fragments_smiles"),
        ("nontoxic_safe_fragments", "nontoxic_safe_fragments_smiles"),
        ("only_toxic_safe_fragments", "only_toxic_safe_fragments_smiles"),
        ("only_nontoxic_safe_fragments", "only_nontoxic_safe_fragments_smiles"),
    ]:
        if col not in df.columns:
            new_columns[new_col] = [""] * n
            continue
        print(f"Decoding {col} -> {new_col} ...")
        new_columns[new_col] = [
            decode_fragments_to_smiles(df[col].iloc[i]) for i in tqdm(range(n), desc=col)
        ]

    df = df.assign(**new_columns)
    out_cols = (
        [c for c in BASE_COLUMNS if c in df.columns]
        + FULL_SAFE_DECODED_COLUMNS
        + NEW_SMILES_COLUMNS
    )
    df_out = df[out_cols].copy()
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} ({len(out_cols)} columns)")


if __name__ == "__main__":
    main()
