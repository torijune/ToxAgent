"""
pairs_safe.csv의 toxic_safe, nontoxic_safe를 dot으로 split해
공통(중복) fragment와 toxic/nontoxic 각각에만 있는 unique fragment를 추출.
functional group 비교와 유사하게 has_safe_diff, unique_safe 등 컬럼을 추가해 저장.
"""
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PAIRS_SAFE_CSV = SCRIPT_DIR / "pairs_safe.csv"
OUTPUT_CSV = SCRIPT_DIR / "pairs_safe_compared.csv"

# SAFE fragment 구분자 (SAFE는 dot으로 조각 구분)
SEP = "."


def safe_to_fragments(safe_str):
    """SAFE 문자열을 빈 칸 제거 후 dot으로 split한 fragment set 반환. 빈 조각 제외."""
    if pd.isna(safe_str) or not str(safe_str).strip():
        return set()
    return {s.strip() for s in str(safe_str).split(SEP) if s.strip()}


def compare_fragments(toxic_safe, nontoxic_safe):
    """
    toxic_safe, nontoxic_safe 문자열에 대해
    공통 fragment set, toxic 전용, nontoxic 전용을 계산.

    Returns:
        (common_set, only_toxic_set, only_nontoxic_set)
    """
    t_set = safe_to_fragments(toxic_safe)
    n_set = safe_to_fragments(nontoxic_safe)
    common = t_set & n_set
    only_toxic = t_set - n_set
    only_nontoxic = n_set - t_set
    return common, only_toxic, only_nontoxic


def build_unique_safe_json(only_toxic, only_nontoxic):
    """
    FG의 unique_fg처럼, unique한 fragment 목록을 reason과 함께 JSON 리스트로.
    """
    out = []
    for frag in sorted(only_toxic):
        out.append({"fragment": frag, "reason": "only_in_toxic"})
    for frag in sorted(only_nontoxic):
        out.append({"fragment": frag, "reason": "only_in_nontoxic"})
    return json.dumps(out, ensure_ascii=False) if out else "[]"


def main():
    print(f"Loading: {PAIRS_SAFE_CSV}")
    df = pd.read_csv(PAIRS_SAFE_CSV)

    if "toxic_safe" not in df.columns or "nontoxic_safe" not in df.columns:
        raise ValueError("pairs_safe.csv must have columns: toxic_safe, nontoxic_safe")

    n = len(df)
    common_list = []
    only_toxic_list = []
    only_nontoxic_list = []
    has_safe_diff_list = []
    unique_safe_list = []
    n_common_list = []
    n_only_toxic_list = []
    n_only_nontoxic_list = []
    toxic_fragments_str_list = []
    nontoxic_fragments_str_list = []

    for _, row in df.iterrows():
        toxic_safe = row.get("toxic_safe", "")
        nontoxic_safe = row.get("nontoxic_safe", "")
        common, only_toxic, only_nontoxic = compare_fragments(toxic_safe, nontoxic_safe)

        toxic_fragments_str_list.append(SEP.join(sorted(safe_to_fragments(toxic_safe))))
        nontoxic_fragments_str_list.append(SEP.join(sorted(safe_to_fragments(nontoxic_safe))))
        common_list.append(SEP.join(sorted(common)))
        only_toxic_list.append(SEP.join(sorted(only_toxic)))
        only_nontoxic_list.append(SEP.join(sorted(only_nontoxic)))
        has_safe_diff_list.append(len(only_toxic) > 0 or len(only_nontoxic) > 0)
        unique_safe_list.append(build_unique_safe_json(only_toxic, only_nontoxic))
        n_common_list.append(len(common))
        n_only_toxic_list.append(len(only_toxic))
        n_only_nontoxic_list.append(len(only_nontoxic))

    df = df.assign(
        toxic_safe_fragments=toxic_fragments_str_list,
        nontoxic_safe_fragments=nontoxic_fragments_str_list,
        common_safe_fragments=common_list,
        only_toxic_safe_fragments=only_toxic_list,
        only_nontoxic_safe_fragments=only_nontoxic_list,
        n_common_safe=n_common_list,
        n_only_toxic_safe=n_only_toxic_list,
        n_only_nontoxic_safe=n_only_nontoxic_list,
        has_safe_diff=has_safe_diff_list,
        unique_safe=unique_safe_list,
    )

    n_diff = sum(has_safe_diff_list)
    print(f"Rows with has_safe_diff=True: {n_diff} / {n}")

    OUT_COLUMNS = [
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
        "has_safe_diff",
        "unique_safe",
    ]
    df_out = df[[c for c in OUT_COLUMNS if c in df.columns]]
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} ({len(df_out.columns)} columns)")


if __name__ == "__main__":
    main()
