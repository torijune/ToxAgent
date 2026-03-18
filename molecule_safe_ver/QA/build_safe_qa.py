"""
Build QA pairs for:
  - Task 1: toxic_safe_to_nontoxic_safe
  - Task 2: smiles_to_safe
  - Task 3: toxic_fragment_identification

Outputs:
  - Task 1 -> molecule_safe_ver/QA/task1_safe_to_nontoxic/{single_step|multi_step}/
  - Task 2 -> molecule_safe_ver/QA/task2_smiles_to_safe/ (통합, single/multi 구분 없음)
  - Task 3 -> molecule_safe_ver/QA/task3_toxic_fragment_identification/{single_step|multi_step}/
  - Task 4 -> molecule_safe_ver/QA/task4_safe_to_nontoxic_smiles/{single_step|multi_step}/
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_QA_DIR = Path(__file__).resolve().parent
_QA_SRC = _QA_DIR / "src"
_PROJECT_ROOT = _QA_DIR.parent.parent  # ToxAgent 루트 (ICL_template -> utils -> similarity_utils)
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from qa_template import (
    task1_toxic_safe_to_nontoxic_safe,
    task2_smiles_to_safe,
    task3_toxic_fragment_identification,
    task4_safe_to_nontoxic_smiles,
)
# ICL 빌더는 variant가 icl1/icl2/icl4일 때만 로드 (실패 시 원인 예외 보존)
# qa_template이 MolDeTox_bench/src를 sys.path 맨 앞에 넣어서, ICL이 MolDeTox_bench의 utils를
# 로드할 수 있음. ICL import 전에 QA/src를 맨 앞에 두어 molecule_safe_ver/QA/src/utils가 우선 로드되도록 함.
_icl_import_error = None
try:
    sys.path.insert(0, str(_QA_SRC))
    from ICL_template import build_task1_icl, build_task3_icl, build_task4_icl
except Exception as e:
    _icl_import_error = e
    build_task1_icl = build_task3_icl = build_task4_icl = None

RAW_DATASET_PATH = "commom_frage_pairs_with_smiles_no_long_frag_max4frag.csv"

# Data paths
DATA_TASK1 = _QA_DIR.parent / RAW_DATASET_PATH
DATA_TASK2 = _QA_DIR.parent / "smiles_to_safe.csv"
DATA_TASK3 = _QA_DIR.parent / RAW_DATASET_PATH
DATA_TASK4 = _QA_DIR.parent / RAW_DATASET_PATH

# Output directories
OUT_DIR_TASK1 = _QA_DIR / "task1_safe_to_nontoxic"
OUT_DIR_TASK2 = _QA_DIR / "task2_smiles_to_safe"
OUT_DIR_TASK3 = _QA_DIR / "task3_toxic_fragment_identification"
OUT_DIR_TASK4 = _QA_DIR / "task4_safe_to_nontoxic_smiles"

REQUIRED_COLUMNS_TASK = [
    "dataset_name",
    "endpoint",
    "toxic_safe_decoded_smiles",
    "nontoxic_safe_decoded_smiles",
    "toxic_safe",
    "nontoxic_safe",
    "only_toxic_safe_fragments",
    "only_nontoxic_safe_fragments",
]
REQUIRED_COLUMNS_TASK2 = ["smiles", "safe"]


def _str_or_empty(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _count_dot_fragments(dot_separated: str) -> int:
    """dot('.')로 구분된 fragment 문자열의 fragment 개수 (빈 토큰 제외)."""
    s = (dot_separated or "").strip()
    if not s:
        return 0
    parts = [p.strip() for p in s.split(".") if p.strip()]
    return len(parts)


def _classify_step(*frag_strings: str) -> str:
    """
    fragment가 dot 기준으로 2개 이상이면 multi_step, 1개면 single_step.
    여러 컬럼이 주어지면 그 중 최대 fragment 개수를 기준으로 분류한다.
    """
    counts = [_count_dot_fragments(x) for x in frag_strings]
    max_n = max(counts) if counts else 0
    return "multi_step" if max_n >= 2 else "single_step"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_task1():
    """Task 1: toxic_safe_to_nontoxic_safe -> task1_safe_to_nontoxic/{single_step|multi_step}/task1_safe_qa.jsonl"""
    if not DATA_TASK1.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK1}")
    df = pd.read_csv(DATA_TASK1)
    for col in REQUIRED_COLUMNS_TASK:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records_single: list[dict] = []
    records_multi: list[dict] = []
    for idx, row in df.iterrows():
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        # qa_template.task1_toxic_safe_to_nontoxic_safe 시그니처에 맞게 전달 (full molecule SMILES/SAFE 포함, step별 문구)
        question, answer = task1_toxic_safe_to_nontoxic_safe(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
            step=step,
        )
        rec = {"id": int(idx), "question": question, "answer": answer}
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK1 / "single_step" / "task1_safe_qa.jsonl"
    out_multi = OUT_DIR_TASK1 / "multi_step" / "task1_safe_qa.jsonl"
    _write_jsonl(out_single, records_single)
    _write_jsonl(out_multi, records_multi)
    print(f"Task 1: single_step={len(records_single)} -> {out_single}")
    print(f"Task 1: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def build_task2():
    """Task 2: smiles_to_safe -> task2_smiles_to_safe/task2_safe_qa.jsonl (통합, single/multi 구분 없음)"""
    if not DATA_TASK2.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK2}")
    df = pd.read_csv(DATA_TASK2)
    for col in REQUIRED_COLUMNS_TASK2:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records: list[dict] = []
    for idx, row in df.iterrows():
        safe_str = _str_or_empty(row["safe"])
        if not safe_str:
            continue
        question, answer = task2_smiles_to_safe(
            smiles=_str_or_empty(row["smiles"]),
            safe=safe_str,
        )
        rec = {"id": int(idx), "question": question, "answer": answer}
        records.append(rec)

    out_path = OUT_DIR_TASK2 / "task2_safe_qa.jsonl"
    _write_jsonl(out_path, records)
    print(f"Task 2: {len(records)} rows -> {out_path}")
    return out_path


def build_task3():
    """Task 3: toxic_fragment_identification -> task3_toxic_fragment_identification/{single_step|multi_step}/task3_safe_qa.jsonl"""
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    df = pd.read_csv(DATA_TASK3)
    for col in REQUIRED_COLUMNS_TASK:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records_single: list[dict] = []
    records_multi: list[dict] = []
    for idx, row in df.iterrows():
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        step = _classify_step(only_toxic)
        question, answer = task3_toxic_fragment_identification(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            step=step,
        )
        rec = {"id": int(idx), "question": question, "answer": answer}
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK3 / "single_step" / "task3_safe_qa.jsonl"
    out_multi = OUT_DIR_TASK3 / "multi_step" / "task3_safe_qa.jsonl"
    _write_jsonl(out_single, records_single)
    _write_jsonl(out_multi, records_multi)
    print(f"Task 3: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def build_task4():
    """Task 4: safe_to_nontoxic_smiles -> task4_safe_to_nontoxic_smiles/{single_step|multi_step}/task4_safe_qa.jsonl"""
    if not DATA_TASK4.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK4}")
    df = pd.read_csv(DATA_TASK4)
    for col in REQUIRED_COLUMNS_TASK:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records_single: list[dict] = []
    records_multi: list[dict] = []
    for idx, row in df.iterrows():
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        question, answer = task4_safe_to_nontoxic_smiles(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            step=step,
        )
        rec = {"id": int(idx), "question": question, "answer": answer}
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK4 / "single_step" / "task4_safe_qa.jsonl"
    out_multi = OUT_DIR_TASK4 / "multi_step" / "task4_safe_qa.jsonl"
    _write_jsonl(out_single, records_single)
    _write_jsonl(out_multi, records_multi)
    print(f"Task 4: single_step={len(records_single)} -> {out_single}")
    print(f"Task 4: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi

def main():
    ap = argparse.ArgumentParser(description="Build Task 1/2/3 SAFE QA jsonl (base or ICL).")
    ap.add_argument(
        "--task",
        choices=["1", "2", "3", "4", "all"],
        default="all",
        help=(
            "Which task to build: "
            "1 (task1_safe_to_nontoxic), "
            "2 (task2_smiles_to_safe), "
            "3 (task3_toxic_fragment_identification), "
            "4 (task4_safe_to_nontoxic_smiles), "
            "all (default)"
        ),
    )
    ap.add_argument(
        "--variant",
        choices=["base", "icl1", "icl2", "icl4", "all"],
        default="base",
        help=(
            "base: single_step/multi_step QA (no ICL). "
            "icl1/icl2/icl4: few-shot ICL QA for task1 & task3. "
            "all: build base + icl1 + icl2 + icl4. Default: base"
        ),
    )
    args = ap.parse_args()

    variants_to_run = (
        ["base", "icl1", "icl2", "icl4"] if args.variant == "all" else [args.variant]
    )

    for v in variants_to_run:
        if v == "base":
            if args.task in ("1", "all"):
                build_task1()
            if args.task in ("2", "all"):
                build_task2()
            if args.task in ("3", "all"):
                build_task3()
            if args.task in ("4", "all"):
                build_task4()
        else:
            # icl1, icl2, icl4
            if build_task1_icl is None or build_task3_icl is None or build_task4_icl is None:
                msg = "ICL_template import failed; cannot build ICL QA."
                if _icl_import_error is not None:
                    raise RuntimeError(msg) from _icl_import_error
                raise RuntimeError(msg)
            if args.task in ("1", "all"):
                build_task1_icl(variants=[v])
            if args.task in ("3", "all"):
                build_task3_icl(variants=[v])
            if args.task in ("4", "all"):
                build_task4_icl(variants=[v])
            if args.task == "2":
                print("Task 2 has no ICL variant; skipping.")


if __name__ == "__main__":
    main()
