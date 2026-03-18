"""
Build QA pairs for ACE SAFE tasks:
  - subtask1: safe_to_smiles
  - subtask2: smiles_to_safe
  - task1   : toxic_fragment_identification
  - task2   : nontoxic_fragment_generation
  - task3   : nontoxic_smiles_generation

This script converts a SAFE pair CSV (e.g. scaffold-split train/test) into
JSONL QA files.

Data sources (default):
  - Task1/2/3: splits/scaffold_by_endpoint_unseen_ver/merged_train.csv, merged_test.csv
  - Subtask1/2 (smiles↔SAFE): smiles_safe_task_raw.csv (split 컬럼으로 train/test 필터).
    없으면 smiles_to_safe_ace*.csv 사용.

Outputs (ace_safe_ver/QA/<split>/...):
  - subtask1/2 (smiles↔SAFE 변환): molecule_repr 없이 기존 방식
    - subtask1 -> QA/<split>/subtask1_safe_to_smiles/subtask1_safe_to_smiles_qa.jsonl
    - subtask2 -> QA/<split>/subtask2_smiles_to_safe/subtask2_smiles_to_safe_qa.jsonl
  - task1/2/3/task3_instruction: molecule_repr( only_smiles | only_safe | both_repre )별 서브디렉터리
    - task1 -> QA/<split>/task1_toxic_fragment_identification/<molecule_repr>/{single_step|multi_step}/...
    - task2 -> QA/<split>/task2_nontoxic_fragment_generation/<molecule_repr>/{single_step|multi_step}/...
    - task3 -> QA/<split>/task3_nontoxic_smiles_generation/<molecule_repr>/{single_step|multi_step}/...
    - task3_instruction -> QA/<split>/task3_instruction_nontoxic_smiles_generation/<molecule_repr>/{single_step|multi_step}/...

<split>은 train 또는 test.

저장 시: 각 레코드에 dataset_name, endpoint, source_index(원본 행 인덱스)를 넣고,
기본적으로 --shuffle_seed(기본 42)로 샘플 순서를 셔플한 뒤 저장. id는 셔플 후 0..n-1로 부여.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_QA_DIR = Path(__file__).resolve().parent
_QA_SRC = _QA_DIR / "src"
_PROJECT_ROOT = _QA_DIR.parent.parent  # ToxAgent 루트 (ICL_template -> utils -> similarity_utils)
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.qa_template import (
    task1_toxic_fragment_identification,
    task2_nontoxic_fragment_generation,
    task3_nontoxic_smiles_generation,
    task3_instruction_nontoxic_smiles_generation,
    task3_stepwise_cot_nontoxic_smiles_generation,
    subtask1_safe_to_smiles,
    subtask2_smiles_to_safe,
)
from src.task3_instruction_ver import build_cot_instruction
# ICL 빌더는 variant가 icl1/icl2/icl4일 때만 로드 (실패 시 원인 예외 보존)
# qa_template이 MolDeTox_bench/src를 sys.path 맨 앞에 넣어서, ICL이 MolDeTox_bench의 utils를
# 로드할 수 있음. ICL import 전에 QA/src를 맨 앞에 두어 molecule_safe_ver/QA/src/utils가 우선 로드되도록 함.
_icl_import_error = None
try:
    sys.path.insert(0, str(_QA_SRC))
    from src.ICL_template import build_task1_icl, build_task2_icl, build_task3_icl
except Exception as e:
    _icl_import_error = e
    build_task1_icl = build_task2_icl = build_task3_icl = None

# Default data paths (scaffold split train/test)
_DEFAULT_SPLIT_DIR = _QA_DIR.parent / "splits" / "scaffold_by_endpoint_unseen_ver"
_DEFAULT_TRAIN_CSV = _DEFAULT_SPLIT_DIR / "merged_train.csv"
_DEFAULT_TEST_CSV = _DEFAULT_SPLIT_DIR / "merged_test.csv"
# subtask1/subtask2용: smiles_safe_task_raw.csv (split 컬럼으로 train/test 구분)
_DEFAULT_SMILES_SAFE_TASK_RAW = _QA_DIR.parent / "smiles_safe_task_raw.csv"
_DEFAULT_SMILES_TO_SAFE = _QA_DIR.parent / "smiles_to_safe_ace.csv"
_DEFAULT_SMILES_TO_SAFE_TRAIN = _QA_DIR.parent / "smiles_to_safe_ace_train.csv"
_DEFAULT_SMILES_TO_SAFE_TEST = _QA_DIR.parent / "smiles_to_safe_ace_test.csv"

# Data paths (configured at runtime in main())
DATA_TASK1 = _DEFAULT_TEST_CSV         # task1_toxic_fragment_identification
DATA_TASK2 = _DEFAULT_TEST_CSV         # task2_nontoxic_fragment_generation
DATA_TASK3 = _DEFAULT_TEST_CSV         # task3_nontoxic_smiles_generation
DATA_SUBTASK1 = _DEFAULT_SMILES_TO_SAFE  # subtask1_safe_to_smiles (same CSV as subtask2)
DATA_SUBTASK2 = _DEFAULT_SMILES_TO_SAFE  # subtask2_smiles_to_safe
# split=train/test 시 subtask 필터링용 (smiles_safe_task_raw.csv에 split 컬럼 있을 때)
CURRENT_SPLIT = "test"
# question에서 molecule 표시 방식: only_smiles | only_safe | both_repre (각각 서브디렉터리로 저장)
MOLECULE_REPR_CHOICES = ["only_smiles", "only_safe", "both_repre"]
CURRENT_MOLECULE_REPR = "both_repre"
# 셔플 시드 (None이면 셔플 안 함). 기본은 **셔플 안 함**.
BUILD_QA_SHUFFLE_SEED: Optional[int] = None

# Output base directories (configured per split=train/test)
OUT_DIR_TASK1 = _QA_DIR / "test" / "task1_toxic_fragment_identification"
OUT_DIR_TASK2 = _QA_DIR / "test" / "task2_nontoxic_fragment_generation"
OUT_DIR_TASK3 = _QA_DIR / "test" / "task3_nontoxic_smiles_generation"
OUT_DIR_TASK3_INSTRUCTION = _QA_DIR / "test" / "task3_instruction_nontoxic_smiles_generation"
OUT_DIR_TASK3_STEPWISE_COT = _QA_DIR / "test" / "task3_stepwise_cot_nontoxic_smiles_generation"
OUT_DIR_SUBTASK1 = _QA_DIR / "test" / "subtask1_safe_to_smiles"
OUT_DIR_SUBTASK2 = _QA_DIR / "test" / "subtask2_smiles_to_safe"

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
REQUIRED_COLUMNS_SAFE_TO_SMILES = ["smiles", "safe"]  # canonical_smiles optional, fallback to smiles


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


def _shuffle_and_reid(records: list[dict], seed: Optional[int]) -> list[dict]:
    """Shuffle records and reassign id to 0..n-1. Preserves dataset_name, endpoint, source_index."""
    if not records:
        return records
    if seed is not None:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        for i, r in enumerate(shuffled):
            r = dict(r)
            r["id"] = i
            shuffled[i] = r
        return shuffled
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_task2():
    """Task 2: nontoxic_fragment_generation -> task2_nontoxic_fragment_generation/{single_step|multi_step}/task2_nontoxic_fragment_generation_qa.jsonl"""
    if not DATA_TASK2.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK2}")
    df = pd.read_csv(DATA_TASK2)
    for col in REQUIRED_COLUMNS_TASK:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records_single: list[dict] = []
    records_multi: list[dict] = []
    for idx, row in df.iterrows():
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        question, answer = task2_nontoxic_fragment_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
            "common_safe_fragments": _str_or_empty(row.get("common_safe_fragments", "")),
            "nontoxic_safe_decoded_smiles": _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK2 / "single_step" / "task2_nontoxic_fragment_generation_qa.jsonl"
    out_multi = OUT_DIR_TASK2 / "multi_step" / "task2_nontoxic_fragment_generation_qa.jsonl"
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 2: single_step={len(records_single)} -> {out_single}")
    print(f"Task 2: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def build_subtask2():
    """Subtask 2: smiles_to_safe -> subtask2_smiles_to_safe/subtask2_smiles_to_safe_qa.jsonl (단일 파일)"""
    if not DATA_SUBTASK2.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_SUBTASK2}")
    df = pd.read_csv(DATA_SUBTASK2)
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.strip().str.lower() == CURRENT_SPLIT.lower()].reset_index(drop=True)
    for col in REQUIRED_COLUMNS_TASK2:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records: list[dict] = []
    for idx, row in df.iterrows():
        safe_str = _str_or_empty(row["safe"])
        if not safe_str:
            continue
        question, answer = subtask2_smiles_to_safe(
            smiles=_str_or_empty(row["smiles"]),
            safe=safe_str,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
            "split": _str_or_empty(row.get("split", "")) if "split" in row else "",
        }
        records.append(rec)

    out_path = OUT_DIR_SUBTASK2 / "subtask2_smiles_to_safe_qa.jsonl"
    _write_jsonl(out_path, _shuffle_and_reid(records, BUILD_QA_SHUFFLE_SEED))
    print(f"Subtask 2: {len(records)} rows -> {out_path}")
    return out_path


def build_subtask1():
    """Subtask 1: safe_to_smiles — given SAFE string, output SMILES. Uses same CSV as Subtask 2."""
    if not DATA_SUBTASK1.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_SUBTASK1}")
    df = pd.read_csv(DATA_SUBTASK1)
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.strip().str.lower() == CURRENT_SPLIT.lower()].reset_index(drop=True)
    for col in REQUIRED_COLUMNS_SAFE_TO_SMILES:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records: list[dict] = []
    for idx, row in df.iterrows():
        safe_str = _str_or_empty(row["safe"])
        if not safe_str:
            continue
        smiles_out = _str_or_empty(row.get("canonical_smiles") or row["smiles"])
        question, answer = subtask1_safe_to_smiles(safe=safe_str, smiles=smiles_out)
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
            "split": _str_or_empty(row.get("split", "")) if "split" in row else "",
        }
        records.append(rec)

    out_path = OUT_DIR_SUBTASK1 / "subtask1_safe_to_smiles_qa.jsonl"
    _write_jsonl(out_path, _shuffle_and_reid(records, BUILD_QA_SHUFFLE_SEED))
    print(f"Subtask 1: {len(records)} rows -> {out_path}")
    return out_path


def build_task1():
    """Task 1: toxic_fragment_identification -> task1_toxic_fragment_identification/{single_step|multi_step}/task1_toxic_fragment_identification_qa.jsonl"""
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
        step = _classify_step(only_toxic)
        question, answer = task1_toxic_fragment_identification(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK1 / "single_step" / "task1_toxic_fragment_identification_qa.jsonl"
    out_multi = OUT_DIR_TASK1 / "multi_step" / "task1_toxic_fragment_identification_qa.jsonl"
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 1: single_step={len(records_single)} -> {out_single}")
    print(f"Task 1: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def build_task3():
    """Task 3: nontoxic_smiles_generation -> task3_nontoxic_smiles_generation/{single_step|multi_step}/task3_nontoxic_smiles_generation_qa.jsonl"""
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
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        question, answer = task3_nontoxic_smiles_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK3 / "single_step" / "task3_nontoxic_smiles_generation_qa.jsonl"
    out_multi = OUT_DIR_TASK3 / "multi_step" / "task3_nontoxic_smiles_generation_qa.jsonl"
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def build_task3_instruction():
    """Task 3 instruction: nontoxic_smiles_generation with remove/add instruction.

    Uses only_toxic_safe_fragments and only_nontoxic_safe_fragments from the same
    CSV as task3 (merged_train.csv for split=train, merged_test.csv for split=test).
    """
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
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)
        cot_instruction = build_cot_instruction(only_toxic, only_nontoxic, step=step)

        question, answer = task3_instruction_nontoxic_smiles_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            cot_instruction=cot_instruction,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK3_INSTRUCTION / "single_step" / "task3_instruction_nontoxic_smiles_generation_qa.jsonl"
    out_multi = OUT_DIR_TASK3_INSTRUCTION / "multi_step" / "task3_instruction_nontoxic_smiles_generation_qa.jsonl"
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3 instruction: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3 instruction: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def build_task3_stepwise_cot():
    """
    Task 3 stepwise CoT (new task3_CoT version):
    - Single call
    - Output JSON includes Step1/Step2 fragments + natural-language reasoning + final nontoxic SMILES
    - Gold step labels are stored inside QA answer dict for evaluation
    """
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
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        question, answer = task3_stepwise_cot_nontoxic_smiles_generation(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            step=step,
            molecule_repr=CURRENT_MOLECULE_REPR,
        )
        rec = {
            "id": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "source_index": int(idx),
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = (
        OUT_DIR_TASK3_STEPWISE_COT
        / "single_step"
        / "task3_stepwise_cot_nontoxic_smiles_generation_qa.jsonl"
    )
    out_multi = (
        OUT_DIR_TASK3_STEPWISE_COT
        / "multi_step"
        / "task3_stepwise_cot_nontoxic_smiles_generation_qa.jsonl"
    )
    _write_jsonl(out_single, _shuffle_and_reid(records_single, BUILD_QA_SHUFFLE_SEED))
    _write_jsonl(out_multi, _shuffle_and_reid(records_multi, BUILD_QA_SHUFFLE_SEED))
    print(f"Task 3 stepwise CoT: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3 stepwise CoT: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi

# def build_task3_instruction_agentic_flow():
    
#     return out_single, out_multi


def _configure_paths(
    split: str,
    input_csv: Path | None,
    task_raw_csv: Path | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    split ('train' or 'test')과 입력 CSV, molecule_repr에 따라 DATA_* / OUT_DIR_* 전역 변수를 설정한다.
    molecule_repr: only_smiles | only_safe | both_repre → 각각 서브디렉터리로 저장.
    """
    global DATA_TASK1, DATA_TASK2, DATA_TASK3
    global DATA_SUBTASK1, DATA_SUBTASK2
    global OUT_DIR_TASK1, OUT_DIR_TASK2, OUT_DIR_TASK3, OUT_DIR_TASK3_INSTRUCTION, OUT_DIR_TASK3_STEPWISE_COT
    global OUT_DIR_SUBTASK1, OUT_DIR_SUBTASK2
    global CURRENT_SPLIT, CURRENT_MOLECULE_REPR

    CURRENT_SPLIT = split
    repr_dir = (molecule_repr or "both_repre").strip().lower()
    if repr_dir not in MOLECULE_REPR_CHOICES:
        repr_dir = "both_repre"
    CURRENT_MOLECULE_REPR = repr_dir

    if split == "train":
        data_path = input_csv or _DEFAULT_TRAIN_CSV
        split_dir = _QA_DIR / "train"
        smiles_to_safe_path = (
            _DEFAULT_SMILES_TO_SAFE_TRAIN
            if _DEFAULT_SMILES_TO_SAFE_TRAIN.exists()
            else _DEFAULT_SMILES_TO_SAFE
        )
    else:
        data_path = input_csv or _DEFAULT_TEST_CSV
        split_dir = _QA_DIR / "test"
        smiles_to_safe_path = (
            _DEFAULT_SMILES_TO_SAFE_TEST
            if _DEFAULT_SMILES_TO_SAFE_TEST.exists()
            else _DEFAULT_SMILES_TO_SAFE
        )

    DATA_TASK1 = data_path   # toxic_fragment_identification
    DATA_TASK2 = data_path   # nontoxic_fragment_generation
    DATA_TASK3 = data_path   # nontoxic_smiles_generation

    # subtask1/subtask2: task_raw_csv 지정 또는 smiles_safe_task_raw.csv 우선 (split 컬럼으로 train/test 필터)
    use_task_raw = (task_raw_csv and task_raw_csv.exists()) or _DEFAULT_SMILES_SAFE_TASK_RAW.exists()
    if use_task_raw:
        DATA_SUBTASK1 = task_raw_csv if (task_raw_csv and task_raw_csv.exists()) else _DEFAULT_SMILES_SAFE_TASK_RAW
        DATA_SUBTASK2 = DATA_SUBTASK1
    else:
        DATA_SUBTASK1 = smiles_to_safe_path
        DATA_SUBTASK2 = smiles_to_safe_path

    OUT_DIR_TASK1 = split_dir / "task1_toxic_fragment_identification" / repr_dir
    OUT_DIR_TASK2 = split_dir / "task2_nontoxic_fragment_generation" / repr_dir
    OUT_DIR_TASK3 = split_dir / "task3_nontoxic_smiles_generation" / repr_dir
    OUT_DIR_TASK3_INSTRUCTION = split_dir / "task3_instruction_nontoxic_smiles_generation" / repr_dir
    OUT_DIR_TASK3_STEPWISE_COT = split_dir / "task3_stepwise_cot_nontoxic_smiles_generation" / repr_dir
    # subtask1/2는 smiles↔SAFE 변환만 하므로 molecule_repr 서브디렉터리 없음 (기존 방식 유지)
    OUT_DIR_SUBTASK1 = split_dir / "subtask1_safe_to_smiles"
    OUT_DIR_SUBTASK2 = split_dir / "subtask2_smiles_to_safe"


def main():
    ap = argparse.ArgumentParser(description="Build SAFE QA jsonl (base or ICL).")
    ap.add_argument(
        "--task",
        choices=["task1", "task2", "task3", "task3_instruction", "task3_stepwise_cot", "subtask1", "subtask2", "all"],
        default="all",
        help=(
            "Which task to build: "
            "task1 (toxic_fragment_identification), "
            "task2 (nontoxic_fragment_generation), "
            "task3 (nontoxic_smiles_generation), "
            "task3_instruction (task3 with remove/add instruction; uses --molecule_repr), "
            "task3_stepwise_cot (new task3_CoT: stepwise reasoning + step1/2 outputs + final SMILES; uses --molecule_repr), "
            "subtask1 (safe_to_smiles), "
            "subtask2 (smiles_to_safe), "
            "all (default). task1/task2/task3/task3_instruction/task3_stepwise_cot require --molecule_repr (default: both_repre)."
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
    ap.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help=(
            "어떤 split으로 QA 빌드 (train 또는 test). "
            "train/test 각각 한 번씩 실행하면 QA/train/, QA/test/ 모두 생성. 기본: test."
        ),
    )
    ap.add_argument(
        "--input_csv",
        type=Path,
        default=None,
        help=(
            "SAFE pair CSV 경로. 지정하지 않으면 "
            "split=train → merged_train.csv, split=test → merged_test.csv 를 사용."
        ),
    )
    ap.add_argument(
        "--task_raw_csv",
        type=Path,
        default=None,
        help=(
            "Subtask1/2용 smiles_safe_task_raw CSV. 지정하지 않으면 "
            "smiles_safe_task_raw.csv 존재 시 자동 사용."
        ),
    )
    ap.add_argument(
        "--molecule_repr",
        choices=["only_smiles", "only_safe", "both_repre", "all"],
        default="both_repre",
        help=(
            "Question에서 molecule 표시 방식 (task1, task2, task3, task3_instruction에 적용): "
            "only_smiles (SMILES만), only_safe (SAFE만), both_repre (둘 다 + 동일 molecule 명시). "
            "all이면 세 버전 모두 빌드하여 각각 서브디렉터리 저장. 기본: both_repre"
        ),
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="QA 샘플 셔플 시드 (--shuffle 사용 시 적용). 기본: 42",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="저장 전에 QA 샘플 순서를 셔플한다. (--shuffle_seed 사용)",
    )
    ap.add_argument(
        "--no_shuffle",
        action="store_true",
        default=True,
        help="(기본값) 셔플하지 않고 원본 순서대로 저장. --shuffle을 주면 셔플됨.",
    )
    args = ap.parse_args()

    global BUILD_QA_SHUFFLE_SEED
    BUILD_QA_SHUFFLE_SEED = args.shuffle_seed if args.shuffle else None

    if args.task_raw_csv is not None and not args.task_raw_csv.exists():
        raise FileNotFoundError(f"Task raw CSV not found: {args.task_raw_csv}")

    reprs_to_run = (
        list(MOLECULE_REPR_CHOICES) if args.molecule_repr == "all" else [args.molecule_repr]
    )
    variants_to_run = (
        ["base", "icl1", "icl2", "icl4"] if args.variant == "all" else [args.variant]
    )

    for repr_dir in reprs_to_run:
        _configure_paths(
            split=args.split,
            input_csv=args.input_csv,
            task_raw_csv=args.task_raw_csv,
            molecule_repr=repr_dir,
        )
        if len(reprs_to_run) > 1:
            print(f"[molecule_repr={repr_dir}]")

        for v in variants_to_run:
            if v == "base":
                if args.task in ("task1", "all"):
                    build_task1()
                if args.task in ("task2", "all"):
                    build_task2()
                if args.task in ("task3", "all"):
                    build_task3()
                if args.task in ("task3_instruction", "all"):
                    build_task3_instruction()
                if args.task in ("task3_stepwise_cot", "all"):
                    build_task3_stepwise_cot()
                if args.task in ("subtask1", "all"):
                    build_subtask1()
                if args.task in ("subtask2", "all"):
                    build_subtask2()
            else:
                # icl1, icl2, icl4
                if build_task1_icl is None or build_task2_icl is None or build_task3_icl is None:
                    msg = "ICL_template import failed; cannot build ICL QA."
                    if _icl_import_error is not None:
                        raise RuntimeError(msg) from _icl_import_error
                    raise RuntimeError(msg)
                if args.task in ("task1", "all"):
                    build_task1_icl(
                        variants=[v],
                        pairs_csv=DATA_TASK1,
                        out_dir=OUT_DIR_TASK1,
                        molecule_repr=CURRENT_MOLECULE_REPR,
                    )
                if args.task in ("task2", "all"):
                    build_task2_icl(
                        variants=[v],
                        pairs_csv=DATA_TASK2,
                        out_dir=OUT_DIR_TASK2,
                        molecule_repr=CURRENT_MOLECULE_REPR,
                    )
                if args.task in ("task3", "all"):
                    build_task3_icl(
                        variants=[v],
                        pairs_csv=DATA_TASK3,
                        out_dir=OUT_DIR_TASK3,
                        molecule_repr=CURRENT_MOLECULE_REPR,
                    )
                if args.task in ("subtask1", "subtask2"):
                    print(f"{args.task} has no ICL variant; skipping.")


if __name__ == "__main__":
    main()
