"""
ICL (In-Context Learning) style QA builders for SAFE tasks.

현재는 Task 1 (toxic_safe_to_nontoxic_safe)에 대해서만 similarity 기반 ICL을 구현해두고,
다른 task들은 이 파일 안에서 확장할 수 있도록 공통 유틸과 인터페이스를 정리해 둔다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_QA_SRC = Path(__file__).resolve().parent
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))

# Import QA src utils first (before qa_template, which adds MolDeTox_bench to path)
from utils import load_toxic_sim_matrix, DEFAULT_PAIRS_CSV, DEFAULT_SIM_OUT_DIR
from qa_template import (
    task1_toxic_safe_to_nontoxic_safe,
    task3_toxic_fragment_identification,
    task4_safe_to_nontoxic_smiles,
)


def _str_or_empty(val: Any) -> str:
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


def _format_task1_icl_examples(example_pairs: list[tuple[str, str]]) -> str:
    """Format (only_toxic, only_nontoxic) pairs as a few-shot example block (for appending at end of question)."""
    if not example_pairs:
        return ""
    lines = []
    for i, (only_toxic, only_nontoxic) in enumerate(example_pairs, 1):
        lines.append(
            f"Example {i}: only_toxic_safe_fragments = {only_toxic!r} "
            f"-> only_nontoxic_safe_fragments = {only_nontoxic!r}"
        )
    return (
        "Few-shot examples (from similar molecules; only_toxic_safe_fragments -> only_nontoxic_safe_fragments):\n"
        + "\n".join(lines)
        + "\n\nNow output the only_nontoxic_safe_fragments for the task above."
    )


def _format_task3_icl_examples(example_pairs: list[tuple[str, str]]) -> str:
    """Format (toxic_safe, only_toxic) pairs as few-shot examples for Task 3."""
    if not example_pairs:
        return ""
    lines = []
    for i, (toxic_safe, only_toxic) in enumerate(example_pairs, 1):
        lines.append(
            f"Example {i}: toxic_safe = {toxic_safe!r} "
            f"-> only_toxic_safe_fragments = {only_toxic!r}"
        )
    return (
        "Few-shot examples (from similar toxic molecules; toxic_safe -> only_toxic_safe_fragments):\n"
        + "\n".join(lines)
        + "\n\nNow output the only_toxic_safe_fragments for the toxic molecule described above."
    )


def _format_task4_icl_examples(example_pairs: list[tuple[str, str]]) -> str:
    """Format (toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles) pairs as few-shot examples for Task 4."""
    if not example_pairs:
        return ""
    lines = []
    for i, (toxic_smiles, nontoxic_smiles) in enumerate(example_pairs, 1):
        lines.append(
            f"Example {i}: toxic_safe_decoded_smiles = {toxic_smiles!r} "
            f"-> nontoxic_safe_decoded_smiles = {nontoxic_smiles!r}"
        )
    return (
        "Few-shot examples (from similar toxic molecules; toxic SMILES -> nontoxic SMILES):\n"
        + "\n".join(lines)
        + "\n\nNow output the nontoxic_safe_decoded_smiles (single SMILES string) for the toxic molecule described above."
    )


def _build_task1_icl_examples_for_row(
    row_index: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    decoded_to_matrix_idx: dict[str, int],
    decoded_to_row_indices: dict[str, list[int]],
    k: int,
) -> list[tuple[str, str]]:
    """
    For the row at row_index, return up to k example (only_toxic_safe_fragments, only_nontoxic_safe_fragments)
    from other rows whose toxic_safe_decoded_smiles is most similar (by sim_matrix).
    - Same molecule (same toxic_safe_decoded_smiles) is never used (excluded by j != idx).
    - Example pairs identical to the current row's (only_toxic, only_nontoxic) are skipped to avoid answer leakage.
    """
    row = df.iloc[row_index]
    current_only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
    current_only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])

    decoded = _str_or_empty(row["toxic_safe_decoded_smiles"])
    if not decoded:
        return []
    idx = decoded_to_matrix_idx.get(decoded)
    if idx is None:
        return []

    # Similarity row for this molecule; higher = more similar
    sim_row = np.array(sim_matrix[idx], dtype=np.float64)
    # Exclude self (same molecule)
    sorted_indices = np.argsort(sim_row)[::-1]
    candidate_indices = [j for j in sorted_indices if j != idx]

    examples = []
    seen_pairs: set[tuple[str, str]] = set()
    for j in candidate_indices:
        if len(examples) >= k:
            break
        similar_smiles = smiles_list[j]
        row_indices = decoded_to_row_indices.get(similar_smiles, [])
        if not row_indices:
            continue
        example_row_idx = next((r for r in row_indices if r != row_index), row_indices[0])
        only_toxic = _str_or_empty(df.iloc[example_row_idx]["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(df.iloc[example_row_idx]["only_nontoxic_safe_fragments"])
        pair = (only_toxic, only_nontoxic)
        # Skip if identical to current row's (answer leakage)
        if pair == (current_only_toxic, current_only_nontoxic):
            continue
        # Skip if we already used this exact pair (avoid duplicate examples)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        examples.append(pair)
    return examples


def _build_task3_icl_examples_for_row(
    row_index: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    decoded_to_matrix_idx: dict[str, int],
    decoded_to_row_indices: dict[str, list[int]],
    k: int,
) -> list[tuple[str, str]]:
    """
    Task 3용 ICL 예시: (toxic_safe, only_toxic_safe_fragments) 쌍을 k개까지 선택.

    - similarity 기준은 Task 1과 동일하게 toxic_safe_decoded_smiles 기반.
    - 현재 row와 동일한 (toxic_safe, only_toxic_safe_fragments) 쌍은 제외.
    - 중복 쌍도 제외.
    """
    row = df.iloc[row_index]
    current_toxic_safe = _str_or_empty(row["toxic_safe"])
    current_only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])

    decoded = _str_or_empty(row["toxic_safe_decoded_smiles"])
    if not decoded:
        return []
    idx = decoded_to_matrix_idx.get(decoded)
    if idx is None:
        return []

    sim_row = np.array(sim_matrix[idx], dtype=np.float64)
    sorted_indices = np.argsort(sim_row)[::-1]
    candidate_indices = [j for j in sorted_indices if j != idx]

    examples: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for j in candidate_indices:
        if len(examples) >= k:
            break
        similar_smiles = smiles_list[j]
        row_indices = decoded_to_row_indices.get(similar_smiles, [])
        if not row_indices:
            continue
        example_row_idx = next((r for r in row_indices if r != row_index), row_indices[0])
        toxic_safe = _str_or_empty(df.iloc[example_row_idx]["toxic_safe"])
        only_toxic = _str_or_empty(df.iloc[example_row_idx]["only_toxic_safe_fragments"])
        pair = (toxic_safe, only_toxic)
        if pair == (current_toxic_safe, current_only_toxic):
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        examples.append(pair)
    return examples


def _build_task4_icl_examples_for_row(
    row_index: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    decoded_to_matrix_idx: dict[str, int],
    decoded_to_row_indices: dict[str, list[int]],
    k: int,
) -> list[tuple[str, str]]:
    """
    Task 4용 ICL 예시: (toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles) 쌍을 k개까지 선택.
    유사한 toxic molecule 기준은 Task 1/3과 동일. 현재 row와 동일한 (toxic_smiles, nontoxic_smiles) 쌍은 제외.
    """
    row = df.iloc[row_index]
    current_toxic = _str_or_empty(row["toxic_safe_decoded_smiles"])
    current_nontoxic = _str_or_empty(row["nontoxic_safe_decoded_smiles"])

    decoded = current_toxic
    if not decoded:
        return []
    idx = decoded_to_matrix_idx.get(decoded)
    if idx is None:
        return []

    sim_row = np.array(sim_matrix[idx], dtype=np.float64)
    sorted_indices = np.argsort(sim_row)[::-1]
    candidate_indices = [j for j in sorted_indices if j != idx]

    examples: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for j in candidate_indices:
        if len(examples) >= k:
            break
        similar_smiles = smiles_list[j]
        row_indices = decoded_to_row_indices.get(similar_smiles, [])
        if not row_indices:
            continue
        example_row_idx = next((r for r in row_indices if r != row_index), row_indices[0])
        toxic_smiles = _str_or_empty(df.iloc[example_row_idx]["toxic_safe_decoded_smiles"])
        nontoxic_smiles = _str_or_empty(df.iloc[example_row_idx]["nontoxic_safe_decoded_smiles"])
        pair = (toxic_smiles, nontoxic_smiles)
        if pair == (current_toxic, current_nontoxic):
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        examples.append(pair)
    return examples


def task1_toxic_safe_to_nontoxic_safe_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str]],
    example_pairs_2: list[tuple[str, str]],
    example_pairs_4: list[tuple[str, str]],
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    step: str = "multi_step",
) -> tuple[str, str, str, dict]:
    """
    Build Task 1 question/answer with ICL-1, ICL-2, ICL-4 variants.
    Returns (question_icl1, question_icl2, question_icl4, answer).
    """
    # qa_template.task1_toxic_safe_to_nontoxic_safe 시그니처에 맞춰 전달 (full molecule SMILES/SAFE 포함, step별 문구)
    base_question, answer = task1_toxic_safe_to_nontoxic_safe(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        only_toxic_safe_fragments=_str_or_empty(row["only_toxic_safe_fragments"]),
        only_nontoxic_safe_fragments=_str_or_empty(row["only_nontoxic_safe_fragments"]),
        dataset_name=dataset_name or _str_or_empty(row.get("dataset_name")) or None,
        endpoint=endpoint or _str_or_empty(row.get("endpoint")) or None,
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
        step=step,
    )

    suffix_1 = _format_task1_icl_examples(example_pairs_1)
    suffix_2 = _format_task1_icl_examples(example_pairs_2)
    suffix_4 = _format_task1_icl_examples(example_pairs_4)

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def task3_toxic_fragment_identification_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str]],
    example_pairs_2: list[tuple[str, str]],
    example_pairs_4: list[tuple[str, str]],
    step: str = "multi_step",
) -> tuple[str, str, str, dict]:
    """
    Build Task 3 question/answer with ICL-1, ICL-2, ICL-4 variants.
    Returns (question_icl1, question_icl2, question_icl4, answer).
    """
    base_question, answer = task3_toxic_fragment_identification(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        only_toxic_safe_fragments=_str_or_empty(row["only_toxic_safe_fragments"]),
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        step=step,
    )

    suffix_1 = _format_task3_icl_examples(example_pairs_1)
    suffix_2 = _format_task3_icl_examples(example_pairs_2)
    suffix_4 = _format_task3_icl_examples(example_pairs_4)

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def task4_safe_to_nontoxic_smiles_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str]],
    example_pairs_2: list[tuple[str, str]],
    example_pairs_4: list[tuple[str, str]],
    step: str = "multi_step",
) -> tuple[str, str, str, dict]:
    """
    Build Task 4 question/answer with ICL-1, ICL-2, ICL-4 variants.
    Returns (question_icl1, question_icl2, question_icl4, answer).
    """
    base_question, answer = task4_safe_to_nontoxic_smiles(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        dataset_name=_str_or_empty(row.get("dataset_name")) or None,
        endpoint=_str_or_empty(row.get("endpoint")) or None,
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        step=step,
    )

    suffix_1 = _format_task4_icl_examples(example_pairs_1)
    suffix_2 = _format_task4_icl_examples(example_pairs_2)
    suffix_4 = _format_task4_icl_examples(example_pairs_4)

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def build_task1_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
) -> None:
    """
    Load pairs CSV and similarity matrix, build ICL-1/2/4 questions per row for Task 1,
    and save under the Task 1 directory (step별로 분리, base와 동일):

        molecule_safe_ver/QA/task1_safe_to_nontoxic/single_step/task1_safe_qa_icl{1,2,4}.jsonl
        molecule_safe_ver/QA/task1_safe_to_nontoxic/multi_step/task1_safe_qa_icl{1,2,4}.jsonl

    variants: if provided, only write these (e.g. ["icl1"], ["icl2"], ["icl4"]).
              None = write all icl1, icl2, icl4.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    # 기본 out_dir: QA/task1_safe_to_nontoxic
    default_out_dir = _QA_SRC.parent / "task1_safe_to_nontoxic"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in ["toxic_safe_decoded_smiles", "only_toxic_safe_fragments", "only_nontoxic_safe_fragments"]:
        if col not in df.columns:
            raise ValueError(f"CSV must have column: {col}")

    sim_matrix, smiles_list = load_toxic_sim_matrix(out_dir=sim_dir)
    decoded_to_matrix_idx = {s: i for i, s in enumerate(smiles_list)}

    # decoded_smiles -> list of row indices (for picking example rows)
    decoded_to_row_indices: dict[str, list[int]] = {}
    for i in range(len(df)):
        decoded = _str_or_empty(df.iloc[i]["toxic_safe_decoded_smiles"])
        if decoded:
            decoded_to_row_indices.setdefault(decoded, []).append(i)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(df)):
        row = df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        ex1 = _build_task1_icl_examples_for_row(
            row_index, df, sim_matrix, smiles_list,
            decoded_to_matrix_idx, decoded_to_row_indices, k=1,
        )
        ex2 = _build_task1_icl_examples_for_row(
            row_index, df, sim_matrix, smiles_list,
            decoded_to_matrix_idx, decoded_to_row_indices, k=2,
        )
        ex4 = _build_task1_icl_examples_for_row(
            row_index, df, sim_matrix, smiles_list,
            decoded_to_matrix_idx, decoded_to_row_indices, k=4,
        )
        q1, q2, q4, answer = task1_toxic_safe_to_nontoxic_safe_icl(
            row, ex1, ex2, ex4, step=step,
        )
        rec_id = int(row_index)
        rec1 = {"id": rec_id, "question": q1, "answer": answer}
        rec2 = {"id": rec_id, "question": q2, "answer": answer}
        rec4 = {"id": rec_id, "question": q4, "answer": answer}
        if step == "single_step":
            records_single_icl1.append(rec1)
            records_single_icl2.append(rec2)
            records_single_icl4.append(rec4)
        else:
            records_multi_icl1.append(rec1)
            records_multi_icl2.append(rec2)
            records_multi_icl4.append(rec4)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "single_step").mkdir(parents=True, exist_ok=True)
    (out_dir / "multi_step").mkdir(parents=True, exist_ok=True)
    which = variants if variants is not None else ["icl1", "icl2", "icl4"]
    name_to_single_multi = [
        ("icl1", records_single_icl1, records_multi_icl1),
        ("icl2", records_single_icl2, records_multi_icl2),
        ("icl4", records_single_icl4, records_multi_icl4),
    ]
    for name, rec_single, rec_multi in name_to_single_multi:
        if name not in which:
            continue
        path_single = out_dir / "single_step" / f"task1_safe_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"task1_safe_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Task1 ICL] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[Task1 ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}")


def build_task3_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
) -> None:
    """
    Task 3 (toxic_fragment_identification) ICL-1/2/4 빌더.

    toxic_safe_decoded_smiles similarity를 사용해 유사한 toxic molecule들을 찾고,
    (toxic_safe -> only_toxic_safe_fragments) 예시를 few-shot으로 붙인다.

    출력 (step별로 분리, base와 동일):
        molecule_safe_ver/QA/task3_toxic_fragment_identification/single_step/task3_safe_qa_icl{1,2,4}.jsonl
        molecule_safe_ver/QA/task3_toxic_fragment_identification/multi_step/task3_safe_qa_icl{1,2,4}.jsonl

    variants: if provided, only write these (e.g. ["icl1"], ["icl2"], ["icl4"]). None = all.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task3_toxic_fragment_identification"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in ["toxic_safe_decoded_smiles", "toxic_safe", "only_toxic_safe_fragments"]:
        if col not in df.columns:
            raise ValueError(f"CSV must have column: {col}")

    sim_matrix, smiles_list = load_toxic_sim_matrix(out_dir=sim_dir)
    decoded_to_matrix_idx = {s: i for i, s in enumerate(smiles_list)}

    decoded_to_row_indices: dict[str, list[int]] = {}
    for i in range(len(df)):
        decoded = _str_or_empty(df.iloc[i]["toxic_safe_decoded_smiles"])
        if decoded:
            decoded_to_row_indices.setdefault(decoded, []).append(i)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(df)):
        row = df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        step = _classify_step(only_toxic)

        ex1 = _build_task3_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=1,
        )
        ex2 = _build_task3_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=2,
        )
        ex4 = _build_task3_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=4,
        )
        q1, q2, q4, answer = task3_toxic_fragment_identification_icl(
            row, ex1, ex2, ex4, step=step,
        )
        rec_id = int(row_index)
        rec1 = {"id": rec_id, "question": q1, "answer": answer}
        rec2 = {"id": rec_id, "question": q2, "answer": answer}
        rec4 = {"id": rec_id, "question": q4, "answer": answer}
        if step == "single_step":
            records_single_icl1.append(rec1)
            records_single_icl2.append(rec2)
            records_single_icl4.append(rec4)
        else:
            records_multi_icl1.append(rec1)
            records_multi_icl2.append(rec2)
            records_multi_icl4.append(rec4)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "single_step").mkdir(parents=True, exist_ok=True)
    (out_dir / "multi_step").mkdir(parents=True, exist_ok=True)
    which = variants if variants is not None else ["icl1", "icl2", "icl4"]
    name_to_single_multi = [
        ("icl1", records_single_icl1, records_multi_icl1),
        ("icl2", records_single_icl2, records_multi_icl2),
        ("icl4", records_single_icl4, records_multi_icl4),
    ]
    for name, rec_single, rec_multi in name_to_single_multi:
        if name not in which:
            continue
        path_single = out_dir / "single_step" / f"task3_safe_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"task3_safe_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Task3 ICL] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[Task3 ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}")


def build_task4_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
) -> None:
    """
    Task 4 (safe_to_nontoxic_smiles) ICL-1/2/4 빌더.

    toxic_safe_decoded_smiles similarity로 유사한 toxic molecule을 찾고,
    (toxic_safe_decoded_smiles -> nontoxic_safe_decoded_smiles) 예시를 few-shot으로 붙인다.

    출력 (step별로 분리, base와 동일):
        molecule_safe_ver/QA/task4_safe_to_nontoxic_smiles/single_step/task4_safe_qa_icl{1,2,4}.jsonl
        molecule_safe_ver/QA/task4_safe_to_nontoxic_smiles/multi_step/task4_safe_qa_icl{1,2,4}.jsonl

    variants: if provided, only write these (e.g. ["icl1"], ["icl2"], ["icl4"]). None = all.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task4_safe_to_nontoxic_smiles"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in ["toxic_safe_decoded_smiles", "nontoxic_safe_decoded_smiles", "only_toxic_safe_fragments", "only_nontoxic_safe_fragments"]:
        if col not in df.columns:
            raise ValueError(f"CSV must have column: {col}")

    sim_matrix, smiles_list = load_toxic_sim_matrix(out_dir=sim_dir)
    decoded_to_matrix_idx = {s: i for i, s in enumerate(smiles_list)}

    decoded_to_row_indices: dict[str, list[int]] = {}
    for i in range(len(df)):
        decoded = _str_or_empty(df.iloc[i]["toxic_safe_decoded_smiles"])
        if decoded:
            decoded_to_row_indices.setdefault(decoded, []).append(i)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(df)):
        row = df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        ex1 = _build_task4_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=1,
        )
        ex2 = _build_task4_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=2,
        )
        ex4 = _build_task4_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=4,
        )
        q1, q2, q4, answer = task4_safe_to_nontoxic_smiles_icl(
            row, ex1, ex2, ex4, step=step,
        )
        rec_id = int(row_index)
        rec1 = {"id": rec_id, "question": q1, "answer": answer}
        rec2 = {"id": rec_id, "question": q2, "answer": answer}
        rec4 = {"id": rec_id, "question": q4, "answer": answer}
        if step == "single_step":
            records_single_icl1.append(rec1)
            records_single_icl2.append(rec2)
            records_single_icl4.append(rec4)
        else:
            records_multi_icl1.append(rec1)
            records_multi_icl2.append(rec2)
            records_multi_icl4.append(rec4)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "single_step").mkdir(parents=True, exist_ok=True)
    (out_dir / "multi_step").mkdir(parents=True, exist_ok=True)
    which = variants if variants is not None else ["icl1", "icl2", "icl4"]
    name_to_single_multi = [
        ("icl1", records_single_icl1, records_multi_icl1),
        ("icl2", records_single_icl2, records_multi_icl2),
        ("icl4", records_single_icl4, records_multi_icl4),
    ]
    for name, rec_single, rec_multi in name_to_single_multi:
        if name not in which:
            continue
        path_single = out_dir / "single_step" / f"task4_safe_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"task4_safe_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Task4 ICL] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[Task4 ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}")
