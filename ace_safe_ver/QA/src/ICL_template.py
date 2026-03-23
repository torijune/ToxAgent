"""
ICL (In-Context Learning) style QA builders for SAFE tasks.

현재는 Task 2 (nontoxic_fragment_generation), Task 1 (toxic_fragment_identification),
Task 3 (nontoxic_smiles_generation / nontoxic_safe_generation / stepwise_cot)에 대해
  - similarity 기반 ICL (기존)
  - `icl_train_topk_indices.json`에 저장된 train row index 기반 ICL (신규)
를 지원한다.

Few-shot 블록은 `molecule_repr` (only_smiles / only_safe / both_repre)에 맞춰
본문의 `_smiles_safe_matching`과 동일한 규칙으로 train 예시 분자를 표시한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

_QA_SRC = Path(__file__).resolve().parent
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))

# Import QA src utils first (before qa_template, which adds MolDeTox_bench to path)
from utils import load_toxic_sim_matrix, DEFAULT_PAIRS_CSV, DEFAULT_SIM_OUT_DIR
from qa_template import (
    task1_toxic_fragment_identification,
    task2_nontoxic_fragment_generation,
    task3_nontoxic_smiles_generation,
    task3_nontoxic_safe_generation,
    task3_stepwise_cot_nontoxic_smiles_generation,
    task3_stepwise_cot_nontoxic_safe_generation,
    toxic_molecule_content_for_repr,
)

# ---------------------------------------------------------------------------
# Paths: ICL index JSON & default scaffold split (property outlier dropped)
# ---------------------------------------------------------------------------
_ACE_ROOT = _QA_SRC.parent.parent
DEFAULT_ICL_INDEX_JSON = _QA_SRC / "icl_train_topk_indices.json"
DEFAULT_PROPERTY_OUTLIER_TRAIN = (
    _ACE_ROOT
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped"
    / "merged_train.csv"
)
DEFAULT_PROPERTY_OUTLIER_UNSEEN_TEST = (
    _ACE_ROOT
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped"
    / "merged_unseen_test.csv"
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


def _format_task2_nontoxic_fragment_generation_icl_examples(
    example_rows: list[tuple[str, str, str, str]],
    molecule_repr: str = "both_repre",
) -> str:
    """
    task2_nontoxic_fragment_generation ICL.
    각 튜플: (toxic_safe, toxic_safe_decoded_smiles, only_toxic_safe_fragments, only_nontoxic_safe_fragments).
    molecule_repr에 본문의 _smiles_safe_matching과 동일하게 toxic 전체 분자 표현을 맞춘다.
    """
    if not example_rows:
        return ""
    lines = []
    for i, (toxic_safe, toxic_smiles, only_toxic, only_nontoxic) in enumerate(example_rows, 1):
        mol = toxic_molecule_content_for_repr(toxic_safe, toxic_smiles, molecule_repr)
        if mol:
            lines.append(
                f"Example {i}: toxic molecule ({mol}); only_toxic_safe_fragments = {only_toxic!r} "
                f"-> only_nontoxic_safe_fragments = {only_nontoxic!r}"
            )
        else:
            lines.append(
                f"Example {i}: only_toxic_safe_fragments = {only_toxic!r} "
                f"-> only_nontoxic_safe_fragments = {only_nontoxic!r}"
            )
    return (
        "Few-shot examples (from similar molecules; only_toxic_safe_fragments -> only_nontoxic_safe_fragments):\n"
        + "\n".join(lines)
        + "\n\nNow output the only_nontoxic_safe_fragments for the task above."
    )


def _format_task1_toxic_fragment_identification_icl_examples(
    example_rows: list[tuple[str, str, str]],
    molecule_repr: str = "both_repre",
) -> str:
    """
    task1_toxic_fragment_identification ICL.
    각 튜플: (toxic_safe, toxic_safe_decoded_smiles, only_toxic_safe_fragments).
    """
    if not example_rows:
        return ""
    lines = []
    for i, (toxic_safe, toxic_smiles, only_toxic) in enumerate(example_rows, 1):
        mol = toxic_molecule_content_for_repr(toxic_safe, toxic_smiles, molecule_repr)
        if mol:
            lines.append(
                f"Example {i}: toxic molecule ({mol}) -> only_toxic_safe_fragments = {only_toxic!r}"
            )
        else:
            lines.append(
                f"Example {i}: toxic_safe = {toxic_safe!r} -> only_toxic_safe_fragments = {only_toxic!r}"
            )
    return (
        "Few-shot examples (from similar toxic molecules; toxic molecule -> only_toxic_safe_fragments):\n"
        + "\n".join(lines)
        + "\n\nNow output the only_toxic_safe_fragments for the toxic molecule described above."
    )


def _format_task3_nontoxic_smiles_generation_icl_examples(
    example_rows: list[tuple[str, str, str, str]],
    molecule_repr: str = "both_repre",
) -> str:
    """
    task3_nontoxic_smiles_generation ICL.
    각 튜플: (toxic_safe, nontoxic_safe, toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles).
    answer는 여전히 SMILES이지만, few-shot은 본문과 동일한 molecule representation으로 toxic/nontoxic 쌍을 묘사한다.
    """
    if not example_rows:
        return ""
    lines = []
    repr_type = (molecule_repr or "both_repre").strip().lower()
    for i, (toxic_safe, nontoxic_safe, toxic_smiles, nontoxic_smiles) in enumerate(example_rows, 1):
        if repr_type == "only_smiles":
            lines.append(
                f"Example {i}: toxic SMILES = {toxic_smiles!r} -> nontoxic SMILES = {nontoxic_smiles!r}"
            )
        elif repr_type == "only_safe":
            lines.append(
                f"Example {i}: toxic SAFE = {toxic_safe!r} -> nontoxic SAFE = {nontoxic_safe!r}"
            )
        else:
            left = toxic_molecule_content_for_repr(toxic_safe, toxic_smiles, "both_repre")
            right = toxic_molecule_content_for_repr(nontoxic_safe, nontoxic_smiles, "both_repre")
            lines.append(
                f"Example {i}: toxic molecule ({left}) -> nontoxic molecule ({right})"
            )
    return (
        "Few-shot examples (from similar molecules; toxic -> nontoxic reference pairs):\n"
        + "\n".join(lines)
        + "\n\nNow output the nontoxic_safe_decoded_smiles (single SMILES string) for the toxic molecule described above."
    )


def _format_task3_nontoxic_safe_generation_icl_examples(
    example_rows: list[tuple[str, str, str, str]],
    molecule_repr: str = "both_repre",
) -> str:
    """
    task3_nontoxic_safe_generation ICL: 출력이 **전체 nontoxic SAFE**이므로 few-shot도
    toxic(표현은 molecule_repr) -> nontoxic SAFE 참조 쌍으로 맞춘다.
    각 튜플: (toxic_safe, nontoxic_safe, toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles).
    """
    if not example_rows:
        return ""
    lines: list[str] = []
    repr_type = (molecule_repr or "both_repre").strip().lower()
    for i, (toxic_safe, nontoxic_safe, toxic_smiles, nontoxic_smiles) in enumerate(example_rows, 1):
        if repr_type == "only_smiles":
            lines.append(
                f"Example {i}: toxic SMILES = {toxic_smiles!r} -> "
                f"nontoxic SAFE (full molecule, reference) = {nontoxic_safe!r}"
            )
        elif repr_type == "only_safe":
            lines.append(
                f"Example {i}: toxic SAFE = {toxic_safe!r} -> "
                f"nontoxic SAFE (reference) = {nontoxic_safe!r}"
            )
        else:
            left = toxic_molecule_content_for_repr(toxic_safe, toxic_smiles, "both_repre")
            lines.append(
                f"Example {i}: toxic molecule ({left}) -> "
                f"nontoxic SAFE (full molecule, reference) = {nontoxic_safe!r}"
            )
    return (
        "Few-shot examples (similar training pairs; output form is nontoxic SAFE string):\n"
        + "\n".join(lines)
        + "\n\nNow output the nontoxic SAFE string (full molecule SAFE) for the toxic molecule described above."
    )


def _format_task3_stepwise_cot_nontoxic_smiles_generation_icl_examples(
    example_rows: list[tuple[str, str, str, str, str, str]],
    molecule_repr: str = "both_repre",
) -> str:
    """
    task3_stepwise_cot ICL: 각 train 예시에 Step1/Step2 **정답(gold)** fragment와 최종 SMILES를 함께 제시.
    튜플: (toxic_safe, nontoxic_safe, toxic_smiles, nontoxic_smiles, only_toxic_frags, only_nontoxic_frags).
    """
    if not example_rows:
        return ""
    lines: list[str] = []
    repr_type = (molecule_repr or "both_repre").strip().lower()
    for i, (ts, ns, tsmi, nsmi, ot, on) in enumerate(example_rows, 1):
        if repr_type == "only_smiles":
            head = f"Example {i}: toxic SMILES = {tsmi!r}"
        elif repr_type == "only_safe":
            head = f"Example {i}: toxic SAFE = {ts!r}"
        else:
            mol = toxic_molecule_content_for_repr(ts, tsmi, "both_repre")
            head = f"Example {i}: toxic molecule ({mol})"
        lines.append(
            f"{head}\n"
            f"  Step 1 (gold): only_toxic_safe_fragments = {ot!r}\n"
            f"  Step 2 (gold): only_nontoxic_safe_fragments = {on!r}\n"
            f"  Step 3 (gold final SMILES): {nsmi!r}"
        )
    return (
        "Few-shot examples (reference training pairs with gold Step 1/2 fragments and final SMILES):\n"
        + "\n".join(lines)
        + "\n\nNow solve the task above in one JSON object following the output format."
    )


def _build_task2_nontoxic_fragment_generation_icl_examples_for_row(
    row_index: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    decoded_to_matrix_idx: dict[str, int],
    decoded_to_row_indices: dict[str, list[int]],
    k: int,
) -> list[tuple[str, str, str, str]]:
    """
    task2_nontoxic_fragment_generation ICL: row_index 행에 대해 유사 toxic_safe_decoded_smiles 기준으로
    최대 k개의 (toxic_safe, toxic_decoded_smiles, only_toxic, only_nontoxic) 예시를 다른 행에서 선택.
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

    examples: list[tuple[str, str, str, str]] = []
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
        ex = df.iloc[example_row_idx]
        examples.append(
            (
                _str_or_empty(ex.get("toxic_safe")),
                _str_or_empty(ex.get("toxic_safe_decoded_smiles")),
                only_toxic,
                only_nontoxic,
            )
        )
    return examples


def _build_task1_toxic_fragment_identification_icl_examples_for_row(
    row_index: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    decoded_to_matrix_idx: dict[str, int],
    decoded_to_row_indices: dict[str, list[int]],
    k: int,
) -> list[tuple[str, str, str]]:
    """
    task1_toxic_fragment_identification ICL: (toxic_safe, toxic_decoded_smiles, only_toxic_safe_fragments) 쌍을 k개까지.

    - similarity 기준: toxic_safe_decoded_smiles 기반 (유사도 행렬).
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

    examples: list[tuple[str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for j in candidate_indices:
        if len(examples) >= k:
            break
        similar_smiles = smiles_list[j]
        row_indices = decoded_to_row_indices.get(similar_smiles, [])
        if not row_indices:
            continue
        example_row_idx = next((r for r in row_indices if r != row_index), row_indices[0])
        ex = df.iloc[example_row_idx]
        toxic_safe = _str_or_empty(ex["toxic_safe"])
        only_toxic = _str_or_empty(ex["only_toxic_safe_fragments"])
        toxic_dec = _str_or_empty(ex.get("toxic_safe_decoded_smiles"))
        pair = (toxic_safe, only_toxic)
        if pair == (current_toxic_safe, current_only_toxic):
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        examples.append((toxic_safe, toxic_dec, only_toxic))
    return examples


def _build_task3_nontoxic_smiles_generation_icl_examples_for_row(
    row_index: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    decoded_to_matrix_idx: dict[str, int],
    decoded_to_row_indices: dict[str, list[int]],
    k: int,
) -> list[tuple[str, str, str, str]]:
    """
    task3_nontoxic_smiles_generation ICL: train 행의 (toxic_safe, nontoxic_safe, toxic_smiles, nontoxic_smiles) 쌍을 k개까지.
    유사한 toxic molecule 기준은 toxic_safe_decoded_smiles 기반. 현재 row와 동일한 (SMILES, SMILES) 쌍은 제외.
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

    examples: list[tuple[str, str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for j in candidate_indices:
        if len(examples) >= k:
            break
        similar_smiles = smiles_list[j]
        row_indices = decoded_to_row_indices.get(similar_smiles, [])
        if not row_indices:
            continue
        example_row_idx = next((r for r in row_indices if r != row_index), row_indices[0])
        ex = df.iloc[example_row_idx]
        toxic_smiles = _str_or_empty(ex["toxic_safe_decoded_smiles"])
        nontoxic_smiles = _str_or_empty(ex["nontoxic_safe_decoded_smiles"])
        pair = (toxic_smiles, nontoxic_smiles)
        if pair == (current_toxic, current_nontoxic):
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        examples.append(
            (
                _str_or_empty(ex.get("toxic_safe")),
                _str_or_empty(ex.get("nontoxic_safe")),
                toxic_smiles,
                nontoxic_smiles,
            )
        )
    return examples


def _build_task3_stepwise_cot_icl_examples_for_row(
    row_index: int,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    smiles_list: list[str],
    decoded_to_matrix_idx: dict[str, int],
    decoded_to_row_indices: dict[str, list[int]],
    k: int,
) -> list[tuple[str, str, str, str, str, str]]:
    """
    task3_stepwise_cot ICL: task3와 동일 유사도로 train 행을 고르고,
    few-shot에 Step1/Step2 gold fragment + 최종 SMILES를 포함한다.
    튜플: (toxic_safe, nontoxic_safe, toxic_smiles, nontoxic_smiles, only_toxic, only_nontoxic).
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

    examples: list[tuple[str, str, str, str, str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for j in candidate_indices:
        if len(examples) >= k:
            break
        similar_smiles = smiles_list[j]
        row_indices = decoded_to_row_indices.get(similar_smiles, [])
        if not row_indices:
            continue
        example_row_idx = next((r for r in row_indices if r != row_index), row_indices[0])
        ex = df.iloc[example_row_idx]
        toxic_smiles = _str_or_empty(ex["toxic_safe_decoded_smiles"])
        nontoxic_smiles = _str_or_empty(ex["nontoxic_safe_decoded_smiles"])
        pair = (toxic_smiles, nontoxic_smiles)
        if pair == (current_toxic, current_nontoxic):
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        examples.append(
            (
                _str_or_empty(ex.get("toxic_safe")),
                _str_or_empty(ex.get("nontoxic_safe")),
                toxic_smiles,
                nontoxic_smiles,
                _str_or_empty(ex.get("only_toxic_safe_fragments")),
                _str_or_empty(ex.get("only_nontoxic_safe_fragments")),
            )
        )
    return examples


def task2_nontoxic_fragment_generation_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str, str, str]],
    example_pairs_2: list[tuple[str, str, str, str]],
    example_pairs_4: list[tuple[str, str, str, str]],
    dataset_name: Optional[str] = None,
    endpoint: Optional[str] = None,
    step: str = "multi_step",
    molecule_repr: str = "both_repre",
) -> tuple[str, str, str, dict]:
    """
    Build Task 2 (nontoxic_fragment_generation) question/answer with ICL-1, ICL-2, ICL-4 variants.
    각 example 튜플: (toxic_safe, toxic_safe_decoded_smiles, only_toxic_safe_fragments, only_nontoxic_safe_fragments).
    Returns (question_icl1, question_icl2, question_icl4, answer).
    """
    base_question, answer = task2_nontoxic_fragment_generation(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        only_toxic_safe_fragments=_str_or_empty(row["only_toxic_safe_fragments"]),
        only_nontoxic_safe_fragments=_str_or_empty(row["only_nontoxic_safe_fragments"]),
        dataset_name=dataset_name or _str_or_empty(row.get("dataset_name")) or None,
        endpoint=endpoint or _str_or_empty(row.get("endpoint")) or None,
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
        step=step,
        molecule_repr=molecule_repr,
    )

    suffix_1 = _format_task2_nontoxic_fragment_generation_icl_examples(
        example_pairs_1, molecule_repr=molecule_repr,
    )
    suffix_2 = _format_task2_nontoxic_fragment_generation_icl_examples(
        example_pairs_2, molecule_repr=molecule_repr,
    )
    suffix_4 = _format_task2_nontoxic_fragment_generation_icl_examples(
        example_pairs_4, molecule_repr=molecule_repr,
    )

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def task1_toxic_fragment_identification_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str, str]],
    example_pairs_2: list[tuple[str, str, str]],
    example_pairs_4: list[tuple[str, str, str]],
    step: str = "multi_step",
    molecule_repr: str = "both_repre",
) -> tuple[str, str, str, dict]:
    """
    Build Task 1 (toxic_fragment_identification) question/answer with ICL-1, ICL-2, ICL-4 variants.
    각 example 튜플: (toxic_safe, toxic_safe_decoded_smiles, only_toxic_safe_fragments).
    Returns (question_icl1, question_icl2, question_icl4, answer).
    """
    base_question, answer = task1_toxic_fragment_identification(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        only_toxic_safe_fragments=_str_or_empty(row["only_toxic_safe_fragments"]),
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        step=step,
        molecule_repr=molecule_repr,
    )

    suffix_1 = _format_task1_toxic_fragment_identification_icl_examples(
        example_pairs_1, molecule_repr=molecule_repr,
    )
    suffix_2 = _format_task1_toxic_fragment_identification_icl_examples(
        example_pairs_2, molecule_repr=molecule_repr,
    )
    suffix_4 = _format_task1_toxic_fragment_identification_icl_examples(
        example_pairs_4, molecule_repr=molecule_repr,
    )

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def task3_nontoxic_smiles_generation_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str, str, str]],
    example_pairs_2: list[tuple[str, str, str, str]],
    example_pairs_4: list[tuple[str, str, str, str]],
    step: str = "multi_step",
    molecule_repr: str = "both_repre",
) -> tuple[str, str, str, dict]:
    """
    Build Task 3 (nontoxic_smiles_generation) question/answer with ICL-1, ICL-2, ICL-4 variants.
    각 example 튜플: (toxic_safe, nontoxic_safe, toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles).
    Returns (question_icl1, question_icl2, question_icl4, answer).
    """
    base_question, answer = task3_nontoxic_smiles_generation(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        dataset_name=_str_or_empty(row.get("dataset_name")) or None,
        endpoint=_str_or_empty(row.get("endpoint")) or None,
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        step=step,
        molecule_repr=molecule_repr,
    )

    suffix_1 = _format_task3_nontoxic_smiles_generation_icl_examples(
        example_pairs_1, molecule_repr=molecule_repr,
    )
    suffix_2 = _format_task3_nontoxic_smiles_generation_icl_examples(
        example_pairs_2, molecule_repr=molecule_repr,
    )
    suffix_4 = _format_task3_nontoxic_smiles_generation_icl_examples(
        example_pairs_4, molecule_repr=molecule_repr,
    )

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def task3_nontoxic_safe_generation_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str, str, str]],
    example_pairs_2: list[tuple[str, str, str, str]],
    example_pairs_4: list[tuple[str, str, str, str]],
    step: str = "multi_step",
    molecule_repr: str = "both_repre",
) -> tuple[str, str, str, dict]:
    """
    task3_nontoxic_safe_generation + ICL. 예시 튜플은 task3_nontoxic_smiles_generation_icl과 동일 4-tuple
    (유사 train 쌍); 포맷만 nontoxic SAFE 출력에 맞춘다.
    """
    base_question, answer = task3_nontoxic_safe_generation(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
        dataset_name=_str_or_empty(row.get("dataset_name")) or None,
        endpoint=_str_or_empty(row.get("endpoint")) or None,
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        step=step,
        molecule_repr=molecule_repr,
    )

    suffix_1 = _format_task3_nontoxic_safe_generation_icl_examples(
        example_pairs_1, molecule_repr=molecule_repr,
    )
    suffix_2 = _format_task3_nontoxic_safe_generation_icl_examples(
        example_pairs_2, molecule_repr=molecule_repr,
    )
    suffix_4 = _format_task3_nontoxic_safe_generation_icl_examples(
        example_pairs_4, molecule_repr=molecule_repr,
    )

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def task3_stepwise_cot_nontoxic_smiles_generation_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str, str, str, str, str]],
    example_pairs_2: list[tuple[str, str, str, str, str, str]],
    example_pairs_4: list[tuple[str, str, str, str, str, str]],
    step: str = "multi_step",
    molecule_repr: str = "both_repre",
) -> tuple[str, str, str, dict]:
    """
    task3_stepwise_cot + ICL. 예시에 Step1/Step2 gold fragment 및 최종 SMILES 포함.
    """
    base_question, answer = task3_stepwise_cot_nontoxic_smiles_generation(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        dataset_name=_str_or_empty(row.get("dataset_name")) or None,
        endpoint=_str_or_empty(row.get("endpoint")) or None,
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
        only_toxic_safe_fragments=_str_or_empty(row.get("only_toxic_safe_fragments", "")),
        only_nontoxic_safe_fragments=_str_or_empty(row.get("only_nontoxic_safe_fragments", "")),
        step=step,
        molecule_repr=molecule_repr,
    )

    suffix_1 = _format_task3_stepwise_cot_nontoxic_smiles_generation_icl_examples(
        example_pairs_1, molecule_repr=molecule_repr,
    )
    suffix_2 = _format_task3_stepwise_cot_nontoxic_smiles_generation_icl_examples(
        example_pairs_2, molecule_repr=molecule_repr,
    )
    suffix_4 = _format_task3_stepwise_cot_nontoxic_smiles_generation_icl_examples(
        example_pairs_4, molecule_repr=molecule_repr,
    )

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


def _format_task3_stepwise_cot_nontoxic_safe_generation_icl_examples(
    example_rows: list[tuple[str, str, str, str, str, str]],
    molecule_repr: str = "both_repre",
) -> str:
    """task3_stepwise_cot (SAFE 최종) ICL: Step3 gold는 전체 nontoxic SAFE 문자열."""
    if not example_rows:
        return ""
    lines: list[str] = []
    repr_type = (molecule_repr or "both_repre").strip().lower()
    for i, (ts, ns, tsmi, nsmi, ot, on) in enumerate(example_rows, 1):
        if repr_type == "only_smiles":
            head = f"Example {i}: toxic SMILES = {tsmi!r}"
        elif repr_type == "only_safe":
            head = f"Example {i}: toxic SAFE = {ts!r}"
        else:
            mol = toxic_molecule_content_for_repr(ts, tsmi, "both_repre")
            head = f"Example {i}: toxic molecule ({mol})"
        lines.append(
            f"{head}\n"
            f"  Step 1 (gold): only_toxic_safe_fragments = {ot!r}\n"
            f"  Step 2 (gold): only_nontoxic_safe_fragments = {on!r}\n"
            f"  Step 3 (gold final SAFE): {ns!r}"
        )
    return (
        "Few-shot examples (reference training pairs with gold Step 1/2 fragments and final SAFE):\n"
        + "\n".join(lines)
        + "\n\nNow solve the task above in one JSON object following the output format."
    )


def task3_stepwise_cot_nontoxic_safe_generation_icl(
    row: pd.Series,
    example_pairs_1: list[tuple[str, str, str, str, str, str]],
    example_pairs_2: list[tuple[str, str, str, str, str, str]],
    example_pairs_4: list[tuple[str, str, str, str, str, str]],
    step: str = "multi_step",
    molecule_repr: str = "both_repre",
) -> tuple[str, str, str, dict]:
    """task3_stepwise_cot (SAFE 최종) + ICL."""
    base_question, answer = task3_stepwise_cot_nontoxic_safe_generation(
        toxic_safe=_str_or_empty(row["toxic_safe"]),
        dataset_name=_str_or_empty(row.get("dataset_name")) or None,
        endpoint=_str_or_empty(row.get("endpoint")) or None,
        toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
        nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
        only_toxic_safe_fragments=_str_or_empty(row.get("only_toxic_safe_fragments", "")),
        only_nontoxic_safe_fragments=_str_or_empty(row.get("only_nontoxic_safe_fragments", "")),
        step=step,
        molecule_repr=molecule_repr,
    )

    suffix_1 = _format_task3_stepwise_cot_nontoxic_safe_generation_icl_examples(
        example_pairs_1, molecule_repr=molecule_repr,
    )
    suffix_2 = _format_task3_stepwise_cot_nontoxic_safe_generation_icl_examples(
        example_pairs_2, molecule_repr=molecule_repr,
    )
    suffix_4 = _format_task3_stepwise_cot_nontoxic_safe_generation_icl_examples(
        example_pairs_4, molecule_repr=molecule_repr,
    )

    def build_question(suffix: str) -> str:
        if not suffix:
            return base_question
        return base_question.strip() + "\n\n" + suffix.strip()

    q1 = build_question(suffix_1)
    q2 = build_question(suffix_2)
    q4 = build_question(suffix_4)
    return q1, q2, q4, answer


# ---------------------------------------------------------------------------
# ICL from precomputed train indices (icl_train_topk_indices.json + merged_train.csv)
# ---------------------------------------------------------------------------


def load_icl_index_payload(icl_json: str | Path | None = None) -> dict[str, Any]:
    """Load `icl_train_topk_indices.json` (meta + jobs[] with entries per test row)."""
    path = Path(icl_json or DEFAULT_ICL_INDEX_JSON)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_icl_job(
    payload: dict[str, Any],
    *,
    test_csv: str | Path,
    train_csv: str | Path | None = None,
    job_name: str | None = None,
) -> dict[str, Any] | None:
    """
    Find the job in `payload["jobs"]` matching `test_csv` (and optionally `train_csv` / `job_name`).

    If several jobs share the same test CSV, pass `train_csv` or `job_name` to disambiguate
    (e.g. `job_name="scaffold_property_outlier_unseen_test"`).
    """
    te = Path(test_csv).expanduser().resolve()
    tr = Path(train_csv).expanduser().resolve() if train_csv else None
    matches: list[dict[str, Any]] = []
    for job in payload.get("jobs", []):
        try:
            jte = Path(job.get("test_csv", "")).expanduser().resolve()
        except Exception:
            continue
        if jte != te:
            continue
        if job_name is not None and job.get("name") != job_name:
            continue
        if tr is not None:
            try:
                jtr = Path(job.get("train_csv", "")).expanduser().resolve()
            except Exception:
                continue
            if jtr != tr:
                continue
        matches.append(job)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if tr is not None or job_name is not None:
        return matches[0]
    names = [j.get("name") for j in matches]
    raise ValueError(
        f"Multiple ICL jobs match test_csv={test_csv!r}: {names}. "
        "Pass train_csv=... or job_name=... to disambiguate."
    )


def icl_entries_by_test_row_index(job: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Map test_row_index -> entry dict for fast lookup."""
    out: dict[int, dict[str, Any]] = {}
    for e in job.get("entries", []):
        idx = e.get("test_row_index")
        if idx is not None:
            out[int(idx)] = e
    return out


def _task2_nontoxic_fragment_generation_icl_pairs_from_train_indices(
    train_df: pd.DataFrame,
    train_indices: Sequence[int],
    test_row: pd.Series,
    k: int,
) -> list[tuple[str, str, str, str]]:
    """(toxic_safe, toxic_decoded_smiles, only_toxic, only_nontoxic) up to k; skip leakage / duplicates."""
    cur_ot = _str_or_empty(test_row.get("only_toxic_safe_fragments"))
    cur_on = _str_or_empty(test_row.get("only_nontoxic_safe_fragments"))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str, str]] = []
    for raw_i in train_indices:
        if len(out) >= k:
            break
        i = int(raw_i)
        if i < 0 or i >= len(train_df):
            continue
        r = train_df.iloc[i]
        ot = _str_or_empty(r.get("only_toxic_safe_fragments"))
        on = _str_or_empty(r.get("only_nontoxic_safe_fragments"))
        pair = (ot, on)
        if pair == (cur_ot, cur_on):
            continue
        if pair in seen:
            continue
        seen.add(pair)
        out.append(
            (
                _str_or_empty(r.get("toxic_safe")),
                _str_or_empty(r.get("toxic_safe_decoded_smiles")),
                ot,
                on,
            )
        )
    return out


def _task1_toxic_fragment_identification_icl_pairs_from_train_indices(
    train_df: pd.DataFrame,
    train_indices: Sequence[int],
    test_row: pd.Series,
    k: int,
) -> list[tuple[str, str, str]]:
    """(toxic_safe, toxic_decoded_smiles, only_toxic_safe_fragments) for task1 ICL."""
    cur_ts = _str_or_empty(test_row.get("toxic_safe"))
    cur_ot = _str_or_empty(test_row.get("only_toxic_safe_fragments"))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for raw_i in train_indices:
        if len(out) >= k:
            break
        i = int(raw_i)
        if i < 0 or i >= len(train_df):
            continue
        r = train_df.iloc[i]
        ts = _str_or_empty(r.get("toxic_safe"))
        ot = _str_or_empty(r.get("only_toxic_safe_fragments"))
        pair = (ts, ot)
        if pair == (cur_ts, cur_ot):
            continue
        if pair in seen:
            continue
        seen.add(pair)
        out.append((ts, _str_or_empty(r.get("toxic_safe_decoded_smiles")), ot))
    return out


def _task3_nontoxic_smiles_generation_icl_pairs_from_train_indices(
    train_df: pd.DataFrame,
    train_indices: Sequence[int],
    test_row: pd.Series,
    k: int,
) -> list[tuple[str, str, str, str]]:
    """(toxic_safe, nontoxic_safe, toxic_decoded_smiles, nontoxic_decoded_smiles) for task3 ICL."""
    cur_tox = _str_or_empty(test_row.get("toxic_safe_decoded_smiles"))
    cur_non = _str_or_empty(test_row.get("nontoxic_safe_decoded_smiles"))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str, str]] = []
    for raw_i in train_indices:
        if len(out) >= k:
            break
        i = int(raw_i)
        if i < 0 or i >= len(train_df):
            continue
        r = train_df.iloc[i]
        tox = _str_or_empty(r.get("toxic_safe_decoded_smiles"))
        non = _str_or_empty(r.get("nontoxic_safe_decoded_smiles"))
        pair = (tox, non)
        if pair == (cur_tox, cur_non):
            continue
        if pair in seen:
            continue
        seen.add(pair)
        out.append(
            (
                _str_or_empty(r.get("toxic_safe")),
                _str_or_empty(r.get("nontoxic_safe")),
                tox,
                non,
            )
        )
    return out


def _task3_stepwise_cot_icl_pairs_from_train_indices(
    train_df: pd.DataFrame,
    train_indices: Sequence[int],
    test_row: pd.Series,
    k: int,
) -> list[tuple[str, str, str, str, str, str]]:
    """
    task3_stepwise_cot ICL: task3와 동일 유사도로 train 행을 고르고,
    Step1/Step2 gold fragment + 최종 SMILES를 포함한 6-tuple.
    """
    cur_tox = _str_or_empty(test_row.get("toxic_safe_decoded_smiles"))
    cur_non = _str_or_empty(test_row.get("nontoxic_safe_decoded_smiles"))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str, str, str, str]] = []
    for raw_i in train_indices:
        if len(out) >= k:
            break
        i = int(raw_i)
        if i < 0 or i >= len(train_df):
            continue
        r = train_df.iloc[i]
        tox = _str_or_empty(r.get("toxic_safe_decoded_smiles"))
        non = _str_or_empty(r.get("nontoxic_safe_decoded_smiles"))
        pair = (tox, non)
        if pair == (cur_tox, cur_non):
            continue
        if pair in seen:
            continue
        seen.add(pair)
        out.append(
            (
                _str_or_empty(r.get("toxic_safe")),
                _str_or_empty(r.get("nontoxic_safe")),
                tox,
                non,
                _str_or_empty(r.get("only_toxic_safe_fragments")),
                _str_or_empty(r.get("only_nontoxic_safe_fragments")),
            )
        )
    return out


def _write_icl_variant_jsonl(
    out_dir: Path,
    task_qa_basename: str,
    *,
    records_single_icl1: list[dict[str, Any]],
    records_multi_icl1: list[dict[str, Any]],
    records_single_icl2: list[dict[str, Any]],
    records_multi_icl2: list[dict[str, Any]],
    records_single_icl4: list[dict[str, Any]],
    records_multi_icl4: list[dict[str, Any]],
    variants: list[str] | None,
    log_prefix: str,
) -> None:
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
        path_single = out_dir / "single_step" / f"{task_qa_basename}_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"{task_qa_basename}_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{log_prefix}] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[{log_prefix}] {name}: multi_step ={len(rec_multi)} -> {path_multi}")


def build_task2_nontoxic_fragment_generation_icl_from_index_json(
    test_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    icl_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    job_name: str | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    Task 2 ICL: `icl_train_topk_indices.json`의 `top_train_row_indices`로
    `merged_train.csv`(job의 train_csv)에서 행을 읽어 few-shot 예시를 구성한다.

    기본값은 property-outlier split의 unseen test (`merged_unseen_test.csv`)와
    대응 train (`merged_train.csv`), job `scaffold_property_outlier_unseen_test`.

    출력: `build_task2_nontoxic_fragment_generation_icl`과 동일한 파일명·디렉터리 구조.
    """
    test_csv = Path(test_csv or DEFAULT_PROPERTY_OUTLIER_UNSEEN_TEST)
    payload = load_icl_index_payload(icl_json)
    job = resolve_icl_job(
        payload,
        test_csv=test_csv,
        train_csv=train_csv,
        job_name=job_name,
    )
    if job is None:
        raise ValueError(
            f"No ICL job for test_csv={test_csv!r}, train_csv={train_csv!r}, job_name={job_name!r}"
        )
    train_path = Path(job["train_csv"])
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_csv)

    for col in ("toxic_safe", "only_toxic_safe_fragments", "only_nontoxic_safe_fragments"):
        if col not in test_df.columns:
            raise ValueError(f"test CSV missing column: {col}")
    for col in (
        "toxic_safe",
        "toxic_safe_decoded_smiles",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
    ):
        if col not in train_df.columns:
            raise ValueError(f"train CSV missing column: {col}")

    entry_map = icl_entries_by_test_row_index(job)
    default_out = _QA_SRC.parent / "task2_nontoxic_fragment_generation"
    out_dir = Path(out_dir or default_out)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(test_df)):
        row = test_df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        ent = entry_map.get(row_index)
        idxs = list(ent.get("top_train_row_indices") or []) if ent else []
        full = _task2_nontoxic_fragment_generation_icl_pairs_from_train_indices(train_df, idxs, row, k=4)
        ex1, ex2, ex4 = full[:1], full[:2], full[:4]

        q1, q2, q4, answer = task2_nontoxic_fragment_generation_icl(
            row,
            ex1,
            ex2,
            ex4,
            dataset_name=_str_or_empty(row.get("dataset_name")) or None,
            endpoint=_str_or_empty(row.get("endpoint")) or None,
            step=step,
            molecule_repr=molecule_repr,
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

    _write_icl_variant_jsonl(
        out_dir,
        "task2_nontoxic_fragment_generation",
        records_single_icl1=records_single_icl1,
        records_multi_icl1=records_multi_icl1,
        records_single_icl2=records_single_icl2,
        records_multi_icl2=records_multi_icl2,
        records_single_icl4=records_single_icl4,
        records_multi_icl4=records_multi_icl4,
        variants=variants,
        log_prefix="Task2 ICL (index JSON)",
    )


def build_task1_toxic_fragment_identification_icl_from_index_json(
    test_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    icl_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    job_name: str | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """Task 1 ICL: train index 기반 few-shot. Same layout as `build_task1_toxic_fragment_identification_icl`."""
    test_csv = Path(test_csv or DEFAULT_PROPERTY_OUTLIER_UNSEEN_TEST)
    payload = load_icl_index_payload(icl_json)
    job = resolve_icl_job(
        payload,
        test_csv=test_csv,
        train_csv=train_csv,
        job_name=job_name,
    )
    if job is None:
        raise ValueError(
            f"No ICL job for test_csv={test_csv!r}, train_csv={train_csv!r}, job_name={job_name!r}"
        )
    train_df = pd.read_csv(Path(job["train_csv"]))
    test_df = pd.read_csv(test_csv)

    for col in ("toxic_safe", "only_toxic_safe_fragments", "toxic_safe_decoded_smiles"):
        if col not in test_df.columns:
            raise ValueError(f"test CSV missing column: {col}")
    for col in ("toxic_safe", "toxic_safe_decoded_smiles", "only_toxic_safe_fragments"):
        if col not in train_df.columns:
            raise ValueError(f"train CSV missing column: {col}")

    entry_map = icl_entries_by_test_row_index(job)
    default_out = _QA_SRC.parent / "task1_toxic_fragment_identification"
    out_dir = Path(out_dir or default_out)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(test_df)):
        row = test_df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        step = _classify_step(only_toxic)

        ent = entry_map.get(row_index)
        idxs = list(ent.get("top_train_row_indices") or []) if ent else []
        full = _task1_toxic_fragment_identification_icl_pairs_from_train_indices(train_df, idxs, row, k=4)
        ex1, ex2, ex4 = full[:1], full[:2], full[:4]

        q1, q2, q4, answer = task1_toxic_fragment_identification_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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

    _write_icl_variant_jsonl(
        out_dir,
        "task1_toxic_fragment_identification",
        records_single_icl1=records_single_icl1,
        records_multi_icl1=records_multi_icl1,
        records_single_icl2=records_single_icl2,
        records_multi_icl2=records_multi_icl2,
        records_single_icl4=records_single_icl4,
        records_multi_icl4=records_multi_icl4,
        variants=variants,
        log_prefix="Task1 ICL (index JSON)",
    )


def build_task3_nontoxic_smiles_generation_icl_from_index_json(
    test_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    icl_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    job_name: str | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """Task 3 ICL: train index 기반 few-shot (molecule_repr에 맞는 toxic/nontoxic 표현)."""
    test_csv = Path(test_csv or DEFAULT_PROPERTY_OUTLIER_UNSEEN_TEST)
    payload = load_icl_index_payload(icl_json)
    job = resolve_icl_job(
        payload,
        test_csv=test_csv,
        train_csv=train_csv,
        job_name=job_name,
    )
    if job is None:
        raise ValueError(
            f"No ICL job for test_csv={test_csv!r}, train_csv={train_csv!r}, job_name={job_name!r}"
        )
    train_df = pd.read_csv(Path(job["train_csv"]))
    test_df = pd.read_csv(test_csv)

    for col in (
        "toxic_safe",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in test_df.columns:
            raise ValueError(f"test CSV missing column: {col}")
    for col in (
        "toxic_safe",
        "nontoxic_safe",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in train_df.columns:
            raise ValueError(f"train CSV missing column: {col}")

    entry_map = icl_entries_by_test_row_index(job)
    default_out = _QA_SRC.parent / "task3_nontoxic_smiles_generation"
    out_dir = Path(out_dir or default_out)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(test_df)):
        row = test_df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        ent = entry_map.get(row_index)
        idxs = list(ent.get("top_train_row_indices") or []) if ent else []
        full = _task3_nontoxic_smiles_generation_icl_pairs_from_train_indices(train_df, idxs, row, k=4)
        ex1, ex2, ex4 = full[:1], full[:2], full[:4]

        q1, q2, q4, answer = task3_nontoxic_smiles_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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

    _write_icl_variant_jsonl(
        out_dir,
        "task3_nontoxic_smiles_generation",
        records_single_icl1=records_single_icl1,
        records_multi_icl1=records_multi_icl1,
        records_single_icl2=records_single_icl2,
        records_multi_icl2=records_multi_icl2,
        records_single_icl4=records_single_icl4,
        records_multi_icl4=records_multi_icl4,
        variants=variants,
        log_prefix="Task3 ICL (index JSON)",
    )


def build_task3_nontoxic_safe_generation_icl_from_index_json(
    test_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    icl_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    job_name: str | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """task3_nontoxic_safe_generation ICL (train index). 출력 SAFE에 맞춘 few-shot."""
    test_csv = Path(test_csv or DEFAULT_PROPERTY_OUTLIER_UNSEEN_TEST)
    payload = load_icl_index_payload(icl_json)
    job = resolve_icl_job(
        payload,
        test_csv=test_csv,
        train_csv=train_csv,
        job_name=job_name,
    )
    if job is None:
        raise ValueError(
            f"No ICL job for test_csv={test_csv!r}, train_csv={train_csv!r}, job_name={job_name!r}"
        )
    train_df = pd.read_csv(Path(job["train_csv"]))
    test_df = pd.read_csv(test_csv)

    for col in (
        "toxic_safe",
        "nontoxic_safe",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in test_df.columns:
            raise ValueError(f"test CSV missing column: {col}")
    for col in (
        "toxic_safe",
        "nontoxic_safe",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in train_df.columns:
            raise ValueError(f"train CSV missing column: {col}")

    entry_map = icl_entries_by_test_row_index(job)
    default_out = _QA_SRC.parent / "task3_nontoxic_safe_generation"
    out_dir = Path(out_dir or default_out)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(test_df)):
        row = test_df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        ent = entry_map.get(row_index)
        idxs = list(ent.get("top_train_row_indices") or []) if ent else []
        full = _task3_nontoxic_smiles_generation_icl_pairs_from_train_indices(train_df, idxs, row, k=4)
        ex1, ex2, ex4 = full[:1], full[:2], full[:4]

        q1, q2, q4, answer = task3_nontoxic_safe_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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

    _write_icl_variant_jsonl(
        out_dir,
        "task3_nontoxic_safe_generation",
        records_single_icl1=records_single_icl1,
        records_multi_icl1=records_multi_icl1,
        records_single_icl2=records_single_icl2,
        records_multi_icl2=records_multi_icl2,
        records_single_icl4=records_single_icl4,
        records_multi_icl4=records_multi_icl4,
        variants=variants,
        log_prefix="Task3 nontoxic SAFE ICL (index JSON)",
    )


def build_task3_stepwise_cot_nontoxic_smiles_generation_icl_from_index_json(
    test_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    icl_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    job_name: str | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """task3_stepwise_cot ICL: few-shot에 Step1/Step2 gold + 최종 SMILES."""
    test_csv = Path(test_csv or DEFAULT_PROPERTY_OUTLIER_UNSEEN_TEST)
    payload = load_icl_index_payload(icl_json)
    job = resolve_icl_job(
        payload,
        test_csv=test_csv,
        train_csv=train_csv,
        job_name=job_name,
    )
    if job is None:
        raise ValueError(
            f"No ICL job for test_csv={test_csv!r}, train_csv={train_csv!r}, job_name={job_name!r}"
        )
    train_df = pd.read_csv(Path(job["train_csv"]))
    test_df = pd.read_csv(test_csv)

    for col in (
        "toxic_safe",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in test_df.columns:
            raise ValueError(f"test CSV missing column: {col}")
    for col in (
        "toxic_safe",
        "nontoxic_safe",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in train_df.columns:
            raise ValueError(f"train CSV missing column: {col}")

    entry_map = icl_entries_by_test_row_index(job)
    default_out = _QA_SRC.parent / "task3_stepwise_cot_nontoxic_smiles_generation"
    out_dir = Path(out_dir or default_out)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(test_df)):
        row = test_df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        ent = entry_map.get(row_index)
        idxs = list(ent.get("top_train_row_indices") or []) if ent else []
        full = _task3_stepwise_cot_icl_pairs_from_train_indices(train_df, idxs, row, k=4)
        ex1, ex2, ex4 = full[:1], full[:2], full[:4]

        q1, q2, q4, answer = task3_stepwise_cot_nontoxic_smiles_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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

    _write_icl_variant_jsonl(
        out_dir,
        "task3_stepwise_cot_nontoxic_smiles_generation",
        records_single_icl1=records_single_icl1,
        records_multi_icl1=records_multi_icl1,
        records_single_icl2=records_single_icl2,
        records_multi_icl2=records_multi_icl2,
        records_single_icl4=records_single_icl4,
        records_multi_icl4=records_multi_icl4,
        variants=variants,
        log_prefix="Task3 stepwise CoT ICL (index JSON)",
    )


def build_task3_stepwise_cot_nontoxic_safe_generation_icl_from_index_json(
    test_csv: str | Path | None = None,
    train_csv: str | Path | None = None,
    icl_json: str | Path | None = None,
    out_dir: str | Path | None = None,
    job_name: str | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """task3_stepwise_cot (SAFE 최종) ICL (index JSON)."""
    test_csv = Path(test_csv or DEFAULT_PROPERTY_OUTLIER_UNSEEN_TEST)
    payload = load_icl_index_payload(icl_json)
    job = resolve_icl_job(
        payload,
        test_csv=test_csv,
        train_csv=train_csv,
        job_name=job_name,
    )
    if job is None:
        raise ValueError(
            f"No ICL job for test_csv={test_csv!r}, train_csv={train_csv!r}, job_name={job_name!r}"
        )
    train_df = pd.read_csv(Path(job["train_csv"]))
    test_df = pd.read_csv(test_csv)

    for col in (
        "toxic_safe",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in test_df.columns:
            raise ValueError(f"test CSV missing column: {col}")
    for col in (
        "toxic_safe",
        "nontoxic_safe",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ):
        if col not in train_df.columns:
            raise ValueError(f"train CSV missing column: {col}")

    entry_map = icl_entries_by_test_row_index(job)
    default_out = _QA_SRC.parent / "task3_stepwise_cot_nontoxic_safe_generation"
    out_dir = Path(out_dir or default_out)

    records_single_icl1: list[dict[str, Any]] = []
    records_multi_icl1: list[dict[str, Any]] = []
    records_single_icl2: list[dict[str, Any]] = []
    records_multi_icl2: list[dict[str, Any]] = []
    records_single_icl4: list[dict[str, Any]] = []
    records_multi_icl4: list[dict[str, Any]] = []

    for row_index in range(len(test_df)):
        row = test_df.iloc[row_index]
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        ent = entry_map.get(row_index)
        idxs = list(ent.get("top_train_row_indices") or []) if ent else []
        full = _task3_stepwise_cot_icl_pairs_from_train_indices(train_df, idxs, row, k=4)
        ex1, ex2, ex4 = full[:1], full[:2], full[:4]

        q1, q2, q4, answer = task3_stepwise_cot_nontoxic_safe_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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

    _write_icl_variant_jsonl(
        out_dir,
        "task3_stepwise_cot_nontoxic_safe_generation",
        records_single_icl1=records_single_icl1,
        records_multi_icl1=records_multi_icl1,
        records_single_icl2=records_single_icl2,
        records_multi_icl2=records_multi_icl2,
        records_single_icl4=records_single_icl4,
        records_multi_icl4=records_multi_icl4,
        variants=variants,
        log_prefix="Task3 stepwise CoT (SAFE) ICL (index JSON)",
    )


def build_task2_nontoxic_fragment_generation_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    Load pairs CSV and similarity matrix, build ICL-1/2/4 questions per row for
    Task 2 (nontoxic_fragment_generation), and save under the task directory (step별로 분리):

        QA/<split>/task2_nontoxic_fragment_generation/single_step/task2_nontoxic_fragment_generation_qa_icl{1,2,4}.jsonl
        QA/<split>/task2_nontoxic_fragment_generation/multi_step/task2_nontoxic_fragment_generation_qa_icl{1,2,4}.jsonl

    variants: if provided, only write these (e.g. ["icl1"], ["icl2"], ["icl4"]).
              None = write all icl1, icl2, icl4.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task2_nontoxic_fragment_generation"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in [
        "toxic_safe",
        "toxic_safe_decoded_smiles",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
    ]:
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

        ex1 = _build_task2_nontoxic_fragment_generation_icl_examples_for_row(
            row_index, df, sim_matrix, smiles_list,
            decoded_to_matrix_idx, decoded_to_row_indices, k=1,
        )
        ex2 = _build_task2_nontoxic_fragment_generation_icl_examples_for_row(
            row_index, df, sim_matrix, smiles_list,
            decoded_to_matrix_idx, decoded_to_row_indices, k=2,
        )
        ex4 = _build_task2_nontoxic_fragment_generation_icl_examples_for_row(
            row_index, df, sim_matrix, smiles_list,
            decoded_to_matrix_idx, decoded_to_row_indices, k=4,
        )
        q1, q2, q4, answer = task2_nontoxic_fragment_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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
        path_single = out_dir / "single_step" / f"task2_nontoxic_fragment_generation_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"task2_nontoxic_fragment_generation_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Task2 ICL] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[Task2 ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}")


def build_task1_toxic_fragment_identification_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    Task 1 (toxic_fragment_identification) ICL-1/2/4 빌더.

    toxic_safe_decoded_smiles similarity로 유사 train 행을 고르고,
    few-shot에는 본문과 동일한 molecule_repr로 toxic 분자 + only_toxic_safe_fragments를 표시한다.

    출력 (step별로 분리, base와 동일):
        QA/<split>/task1_toxic_fragment_identification/single_step/task1_toxic_fragment_identification_qa_icl{1,2,4}.jsonl
        QA/<split>/task1_toxic_fragment_identification/multi_step/task1_toxic_fragment_identification_qa_icl{1,2,4}.jsonl

    variants: if provided, only write these (e.g. ["icl1"], ["icl2"], ["icl4"]). None = all.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task1_toxic_fragment_identification"
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

        ex1 = _build_task1_toxic_fragment_identification_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=1,
        )
        ex2 = _build_task1_toxic_fragment_identification_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=2,
        )
        ex4 = _build_task1_toxic_fragment_identification_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=4,
        )
        q1, q2, q4, answer = task1_toxic_fragment_identification_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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
        path_single = out_dir / "single_step" / f"task1_toxic_fragment_identification_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"task1_toxic_fragment_identification_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Task1 ICL] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[Task1 ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}")


def build_task3_nontoxic_smiles_generation_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    Task 3 (nontoxic_smiles_generation) ICL-1/2/4 빌더.

    toxic_safe_decoded_smiles similarity로 유사 train 행을 고르고,
    few-shot에는 molecule_repr에 맞춰 toxic/nontoxic 분자 쌍을 표시한다 (답은 여전히 nontoxic SMILES).

    출력 (step별로 분리, base와 동일):
        QA/<split>/task3_nontoxic_smiles_generation/single_step/task3_nontoxic_smiles_generation_qa_icl{1,2,4}.jsonl
        QA/<split>/task3_nontoxic_smiles_generation/multi_step/task3_nontoxic_smiles_generation_qa_icl{1,2,4}.jsonl

    variants: if provided, only write these (e.g. ["icl1"], ["icl2"], ["icl4"]). None = all.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task3_nontoxic_smiles_generation"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in [
        "toxic_safe",
        "nontoxic_safe",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
    ]:
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

        ex1 = _build_task3_nontoxic_smiles_generation_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=1,
        )
        ex2 = _build_task3_nontoxic_smiles_generation_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=2,
        )
        ex4 = _build_task3_nontoxic_smiles_generation_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=4,
        )
        q1, q2, q4, answer = task3_nontoxic_smiles_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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
        path_single = out_dir / "single_step" / f"task3_nontoxic_smiles_generation_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"task3_nontoxic_smiles_generation_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Task3 ICL] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[Task3 ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}")


def build_task3_nontoxic_safe_generation_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    task3_nontoxic_safe_generation ICL. 유사도·train 데이터는 task3 SMILES ICL과 동일하게 사용하며,
    few-shot 문구만 **nontoxic SAFE** 출력에 맞춘다.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task3_nontoxic_safe_generation"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in [
        "toxic_safe",
        "nontoxic_safe",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
    ]:
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

        ex1 = _build_task3_nontoxic_smiles_generation_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=1,
        )
        ex2 = _build_task3_nontoxic_smiles_generation_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=2,
        )
        ex4 = _build_task3_nontoxic_smiles_generation_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=4,
        )
        q1, q2, q4, answer = task3_nontoxic_safe_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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
        path_single = out_dir / "single_step" / f"task3_nontoxic_safe_generation_qa_{name}.jsonl"
        path_multi = out_dir / "multi_step" / f"task3_nontoxic_safe_generation_qa_{name}.jsonl"
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Task3 nontoxic SAFE ICL] {name}: single_step={len(rec_single)} -> {path_single}")
        print(f"[Task3 nontoxic SAFE ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}")


def build_task3_stepwise_cot_nontoxic_smiles_generation_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """
    task3_stepwise_cot ICL: 각 유사 train 예시에 Step1/Step2 gold fragment 및 최종 SMILES를 few-shot으로 붙인다.
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task3_stepwise_cot_nontoxic_smiles_generation"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in [
        "toxic_safe",
        "nontoxic_safe",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
    ]:
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

        ex1 = _build_task3_stepwise_cot_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=1,
        )
        ex2 = _build_task3_stepwise_cot_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=2,
        )
        ex4 = _build_task3_stepwise_cot_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=4,
        )
        q1, q2, q4, answer = task3_stepwise_cot_nontoxic_smiles_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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
        path_single = (
            out_dir / "single_step" / f"task3_stepwise_cot_nontoxic_smiles_generation_qa_{name}.jsonl"
        )
        path_multi = (
            out_dir / "multi_step" / f"task3_stepwise_cot_nontoxic_smiles_generation_qa_{name}.jsonl"
        )
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"[Task3 stepwise CoT ICL] {name}: single_step={len(rec_single)} -> {path_single}"
        )
        print(
            f"[Task3 stepwise CoT ICL] {name}: multi_step ={len(rec_multi)} -> {path_multi}"
        )


def build_task3_stepwise_cot_nontoxic_safe_generation_icl(
    pairs_csv: str | Path | None = None,
    sim_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    variants: list[str] | None = None,
    molecule_repr: str = "both_repre",
) -> None:
    """task3_stepwise_cot (SAFE 최종) ICL: 유사 train 예시에 Step1/2 gold + 최종 SAFE."""
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    sim_dir = Path(sim_dir or DEFAULT_SIM_OUT_DIR)
    default_out_dir = _QA_SRC.parent / "task3_stepwise_cot_nontoxic_safe_generation"
    out_dir = Path(out_dir or default_out_dir)

    df = pd.read_csv(pairs_csv)
    for col in [
        "toxic_safe",
        "nontoxic_safe",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
        "only_toxic_safe_fragments",
        "only_nontoxic_safe_fragments",
    ]:
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

        ex1 = _build_task3_stepwise_cot_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=1,
        )
        ex2 = _build_task3_stepwise_cot_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=2,
        )
        ex4 = _build_task3_stepwise_cot_icl_examples_for_row(
            row_index,
            df,
            sim_matrix,
            smiles_list,
            decoded_to_matrix_idx,
            decoded_to_row_indices,
            k=4,
        )
        q1, q2, q4, answer = task3_stepwise_cot_nontoxic_safe_generation_icl(
            row, ex1, ex2, ex4, step=step, molecule_repr=molecule_repr,
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
        path_single = (
            out_dir / "single_step" / f"task3_stepwise_cot_nontoxic_safe_generation_qa_{name}.jsonl"
        )
        path_multi = (
            out_dir / "multi_step" / f"task3_stepwise_cot_nontoxic_safe_generation_qa_{name}.jsonl"
        )
        with open(path_single, "w", encoding="utf-8") as f:
            for r in rec_single:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path_multi, "w", encoding="utf-8") as f:
            for r in rec_multi:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"[Task3 stepwise CoT ICL (SAFE)] {name}: single_step={len(rec_single)} -> {path_single}"
        )
        print(
            f"[Task3 stepwise CoT ICL (SAFE)] {name}: multi_step ={len(rec_multi)} -> {path_multi}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="ICL QA 빌더: 유사도 기반(기본) 또는 icl_train_topk_indices.json 기반(--from-index-json)."
    )
    ap.add_argument(
        "--from-index-json",
        action="store_true",
        help="JSON에 저장된 train 행 인덱스 + job의 train_csv에서 few-shot 구성",
    )
    ap.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument(
        "--task3-flavor",
        choices=("smiles", "nontoxic_safe", "stepwise_cot", "stepwise_cot_safe"),
        default="smiles",
        help="task=3일 때만: smiles(기본) | nontoxic_safe | stepwise_cot | stepwise_cot_safe",
    )
    ap.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="--from-index-json: 테스트 CSV (기본 merged_unseen_test). "
        "유사도 모드: pairs_csv로 전달 (미지정 시 utils 기본값).",
    )
    ap.add_argument("--train-csv", type=str, default=None, help="job과 함께 쓰면 매칭에 사용")
    ap.add_argument("--icl-json", type=str, default=None, help=f"기본: {DEFAULT_ICL_INDEX_JSON}")
    ap.add_argument("--job-name", type=str, default=None, help="예: scaffold_property_outlier_unseen_test")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument(
        "--variants",
        type=str,
        default="icl1,icl2,icl4",
        help="쉼표 구분: icl1,icl2,icl4",
    )
    ap.add_argument("--molecule-repr", type=str, default="both_repre")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    kwargs: dict[str, Any] = {
        "icl_json": args.icl_json,
        "out_dir": args.out_dir,
        "job_name": args.job_name,
        "variants": variants,
        "molecule_repr": args.molecule_repr,
    }
    if args.test_csv:
        kwargs["test_csv"] = args.test_csv
    if args.train_csv:
        kwargs["train_csv"] = args.train_csv

    if args.from_index_json:
        if args.task == 1:
            build_task1_toxic_fragment_identification_icl_from_index_json(**kwargs)
        elif args.task == 2:
            build_task2_nontoxic_fragment_generation_icl_from_index_json(**kwargs)
        elif args.task3_flavor == "nontoxic_safe":
            build_task3_nontoxic_safe_generation_icl_from_index_json(**kwargs)
        elif args.task3_flavor == "stepwise_cot":
            build_task3_stepwise_cot_nontoxic_smiles_generation_icl_from_index_json(**kwargs)
        elif args.task3_flavor == "stepwise_cot_safe":
            build_task3_stepwise_cot_nontoxic_safe_generation_icl_from_index_json(**kwargs)
        else:
            build_task3_nontoxic_smiles_generation_icl_from_index_json(**kwargs)
    else:
        sim_kwargs = {
            "pairs_csv": args.test_csv,
            "out_dir": args.out_dir,
            "variants": variants,
            "molecule_repr": args.molecule_repr,
        }
        if args.task == 1:
            build_task1_toxic_fragment_identification_icl(**sim_kwargs)
        elif args.task == 2:
            build_task2_nontoxic_fragment_generation_icl(**sim_kwargs)
        elif args.task3_flavor == "nontoxic_safe":
            build_task3_nontoxic_safe_generation_icl(**sim_kwargs)
        elif args.task3_flavor == "stepwise_cot":
            build_task3_stepwise_cot_nontoxic_smiles_generation_icl(**sim_kwargs)
        elif args.task3_flavor == "stepwise_cot_safe":
            build_task3_stepwise_cot_nontoxic_safe_generation_icl(**sim_kwargs)
        else:
            build_task3_nontoxic_smiles_generation_icl(**sim_kwargs)
