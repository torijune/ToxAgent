"""
Build MCQA (Multiple Choice QA) datasets from the same tasks and raw data as open-ended QA.
LLM answering format only: question ends with choices A/B/C/D and answer is a letter.

Distractor (오답) selection:
  - Task 1: Same endpoint first; if fewer than 3 unique candidates, use full df.
  - Task 2: Tanimoto sim (question SMILES vs others) <= 0.9, then top-3 by sim as distractors.
  - Task 3: Same endpoint first; if fewer than 3 unique candidates, use full df.

Outputs (same directory structure as build_safe_qa, filename suffix _mcqa):
  - Task 1 -> task1_safe_to_nontoxic/{single_step|multi_step}/task1_safe_qa_mcqa.jsonl
  - Task 2 -> task2_smiles_to_safe/task2_safe_qa_mcqa.jsonl
  - Task 3 -> task3_toxic_fragment_identification/{single_step|multi_step}/task3_safe_qa_mcqa.jsonl

Note (multi_step / 4+ correct options): Currently each item has one correct answer string and
4 choice slots (A–D). For multi_step, the "answer" is still one string (e.g. "frag1.frag2");
distractors are other rows' answer strings. If in some setting the correct answer set has
4+ options, the current format (exactly 4 choices, 1 correct) cannot hold; options include
allowing more than 4 choices (A–H), or treating multi_step separately (e.g. multiple-select MCQA).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

_QA_SRC = Path(__file__).resolve().parent
_QA_DIR = _QA_SRC.parent  # molecule_safe_ver/QA
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

from qa_template import (
    task1_toxic_safe_to_nontoxic_safe,
    task2_smiles_to_safe,
    task3_toxic_fragment_identification,
)
from make_choices import get_choices

# Data paths (same as build_safe_qa)
DATA_TASK1 = _QA_DIR.parent / "commom_frage_pairs_with_smiles.csv"
DATA_TASK2 = _QA_DIR.parent / "smiles_to_safe.csv"
DATA_TASK3 = DATA_TASK1

OUT_DIR_TASK1 = _QA_DIR / "task1_safe_to_nontoxic"
OUT_DIR_TASK2 = _QA_DIR / "task2_smiles_to_safe"
OUT_DIR_TASK3 = _QA_DIR / "task3_toxic_fragment_identification"

REQUIRED_COLUMNS_TASK1 = [
    "dataset_name", "endpoint", "toxic_safe_decoded_smiles", "nontoxic_safe_decoded_smiles",
    "toxic_safe", "nontoxic_safe", "only_toxic_safe_fragments", "only_nontoxic_safe_fragments",
]

MCQA_OUTPUT_INSTRUCTION = (
    'Output format: a single JSON object with key "answer" and value the letter (A, B, C, or D). '
    'Example: {"answer": "A"}'
)

# 오답 풀 최소 크기: 같은 endpoint에서 이만큼 없으면 전체 df 사용
MIN_POOL_SAME_ENDPOINT = 3


def _tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2, nbits: int = 2048) -> Optional[float]:
    """두 SMILES 간 Morgan fingerprint Tanimoto 유사도. 실패 시 None."""
    if not _HAS_RDKIT:
        return None
    s1, s2 = (smiles1 or "").strip(), (smiles2 or "").strip()
    if not s1 or not s2:
        return None
    try:
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 is None or m2 is None:
            return None
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits=nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius, nBits=nbits)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None


def _pool_task1(df: pd.DataFrame, idx: int, correct_str: str) -> List[str]:
    """Task 1: 같은 endpoint 내에서 오답 후보; 부족하면 전체 df."""
    row = df.iloc[idx]
    col = "only_nontoxic_safe_fragments"
    if "endpoint" not in df.columns:
        pool_all = [
            _str_or_empty(s) for i, s in df[col].items()
            if i != idx and _str_or_empty(s) != correct_str and _str_or_empty(s)
        ]
        return list(dict.fromkeys(pool_all))

    endpoint = _str_or_empty(row.get("endpoint", ""))
    same_ep = df[df["endpoint"].astype(str).str.strip() == endpoint] if endpoint else pd.DataFrame()
    if len(same_ep) > 0:
        pool_same = [
            _str_or_empty(s)
            for i, s in same_ep[col].items()
            if i != idx and _str_or_empty(s) != correct_str and _str_or_empty(s)
        ]
        unique_same = list(dict.fromkeys(pool_same))
        if len(unique_same) >= MIN_POOL_SAME_ENDPOINT:
            return unique_same

    pool_all = [
        _str_or_empty(s)
        for i, s in df[col].items()
        if i != idx and _str_or_empty(s) != correct_str and _str_or_empty(s)
    ]
    return list(dict.fromkeys(pool_all))


def _pool_task2_by_tanimoto(
    df: pd.DataFrame,
    idx: int,
    correct_str: str,
    question_smiles: str,
    max_sim: float = 0.9,
    top_k: int = 3,
) -> List[str]:
    """Task 2: question의 SMILES와 Tanimoto <= max_sim인 다른 행 중 유사도 상위 top_k개의 SAFE를 오답 후보."""
    col_smiles = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
    col_safe = "safe"
    candidates: List[tuple[float, str]] = []  # (sim, safe)

    for i in range(len(df)):
        if i == idx:
            continue
        other_safe = _str_or_empty(df.iloc[i][col_safe])
        if not other_safe or other_safe == correct_str:
            continue
        other_smiles = _str_or_empty(df.iloc[i][col_smiles])
        if not other_smiles:
            continue
        sim = _tanimoto_similarity(question_smiles, other_smiles)
        if sim is not None and sim <= max_sim:
            candidates.append((sim, other_safe))

    if not candidates:
        pool_fallback = [
            _str_or_empty(df.iloc[i][col_safe])
            for i in range(len(df))
            if i != idx and _str_or_empty(df.iloc[i][col_safe]) != correct_str
        ]
        return list(dict.fromkeys(pool_fallback))

    candidates.sort(key=lambda x: -x[0])  # sim 내림차순 → 유사도 높은 순
    seen: set[str] = set()
    out: List[str] = []
    for _sim, safe in candidates:
        if safe not in seen and safe != correct_str:
            seen.add(safe)
            out.append(safe)
            if len(out) >= top_k:
                break
    return out


def _pool_task3(df: pd.DataFrame, idx: int, correct_str: str) -> List[str]:
    """Task 3: 같은 endpoint 내에서 오답 후보; 부족하면 전체 df."""
    row = df.iloc[idx]
    col = "only_toxic_safe_fragments"
    if "endpoint" not in df.columns:
        pool_all = [
            _str_or_empty(s) for i, s in df[col].items()
            if i != idx and _str_or_empty(s) != correct_str and _str_or_empty(s)
        ]
        return list(dict.fromkeys(pool_all))

    endpoint = _str_or_empty(row.get("endpoint", ""))
    same_ep = df[df["endpoint"].astype(str).str.strip() == endpoint] if endpoint else pd.DataFrame()
    if len(same_ep) > 0:
        pool_same = [
            _str_or_empty(s)
            for i, s in same_ep[col].items()
            if i != idx and _str_or_empty(s) != correct_str and _str_or_empty(s)
        ]
        unique_same = list(dict.fromkeys(pool_same))
        if len(unique_same) >= MIN_POOL_SAME_ENDPOINT:
            return unique_same

    pool_all = [
        _str_or_empty(s)
        for i, s in df[col].items()
        if i != idx and _str_or_empty(s) != correct_str and _str_or_empty(s)
    ]
    return list(dict.fromkeys(pool_all))


def _str_or_empty(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _count_dot_fragments(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    return len([p for p in s.replace(" ", "").split(".") if p.strip()])


def _classify_step(*frag_strings: str) -> str:
    counts = [_count_dot_fragments(x) for x in frag_strings]
    max_n = max(counts) if counts else 0
    return "multi_step" if max_n >= 2 else "single_step"


def _format_mcqa_question(question_body: str, options: list[str]) -> str:
    lines = [question_body, "", "Choose one of the following options:"]
    for i, opt in enumerate(options):
        lines.append(f"{chr(65 + i)}) {opt}")
    lines.append("")
    lines.append(MCQA_OUTPUT_INSTRUCTION)
    return "\n".join(lines).strip()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_task1_mcqa() -> tuple[Path, Path]:
    if not DATA_TASK1.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK1}")
    df = pd.read_csv(DATA_TASK1)
    for col in REQUIRED_COLUMNS_TASK1:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records_single: list[dict] = []
    records_multi: list[dict] = []

    for idx, row in df.iterrows():
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        only_nontoxic = _str_or_empty(row["only_nontoxic_safe_fragments"])
        step = _classify_step(only_toxic, only_nontoxic)

        question_body, answer_dict = task1_toxic_safe_to_nontoxic_safe(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            only_nontoxic_safe_fragments=only_nontoxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
            step=step,
            include_output_format=False,
        )
        correct_str = (answer_dict.get("answer") or "").strip()
        pool = _pool_task1(df, idx, correct_str)
        options, correct_idx = get_choices(correct_str, pool, n_choices=4, seed=idx)
        letter = chr(65 + correct_idx)
        question_mcqa = _format_mcqa_question(question_body, options)

        rec = {
            "id": int(idx),
            "question": question_mcqa,
            "answer": {"answer": letter},
            "choices": options,
            "correct_index": correct_idx,
            "answer_value": correct_str,
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK1 / "single_step" / "task1_safe_qa_mcqa.jsonl"
    out_multi = OUT_DIR_TASK1 / "multi_step" / "task1_safe_qa_mcqa.jsonl"
    _write_jsonl(out_single, records_single)
    _write_jsonl(out_multi, records_multi)
    print(f"Task 1 MCQA: single_step={len(records_single)} -> {out_single}")
    print(f"Task 1 MCQA: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def build_task2_mcqa() -> Path:
    if not DATA_TASK2.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK2}")
    df = pd.read_csv(DATA_TASK2)
    for col in ["smiles", "safe"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records: list[dict] = []
    q_smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"

    for idx, row in df.iterrows():
        safe_str = _str_or_empty(row["safe"])
        if not safe_str:
            continue
        question_smiles = _str_or_empty(row[q_smiles_col]) or _str_or_empty(row["smiles"])
        question_body, answer_dict = task2_smiles_to_safe(
            smiles=_str_or_empty(row["smiles"]),
            safe=safe_str,
            include_output_format=False,
        )
        correct_str = (answer_dict.get("answer") or "").strip()
        pool = _pool_task2_by_tanimoto(
            df, idx, correct_str,
            question_smiles=question_smiles,
            max_sim=0.9,
            top_k=3,
        )
        options, correct_idx = get_choices(correct_str, pool, n_choices=4, seed=idx)
        letter = chr(65 + correct_idx)
        question_mcqa = _format_mcqa_question(question_body, options)

        rec = {
            "id": int(idx),
            "question": question_mcqa,
            "answer": {"answer": letter},
            "choices": options,
            "correct_index": correct_idx,
            "answer_value": correct_str,
        }
        records.append(rec)

    out_path = OUT_DIR_TASK2 / "task2_safe_qa_mcqa.jsonl"
    _write_jsonl(out_path, records)
    print(f"Task 2 MCQA: {len(records)} -> {out_path}")
    return out_path


def build_task3_mcqa() -> tuple[Path, Path]:
    if not DATA_TASK3.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_TASK3}")
    df = pd.read_csv(DATA_TASK3)
    for col in ["toxic_safe", "only_toxic_safe_fragments"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    records_single: list[dict] = []
    records_multi: list[dict] = []

    for idx, row in df.iterrows():
        only_toxic = _str_or_empty(row["only_toxic_safe_fragments"])
        step = _classify_step(only_toxic)

        question_body, answer_dict = task3_toxic_fragment_identification(
            toxic_safe=_str_or_empty(row["toxic_safe"]),
            only_toxic_safe_fragments=only_toxic,
            dataset_name=_str_or_empty(row["dataset_name"]) or None,
            endpoint=_str_or_empty(row["endpoint"]) or None,
            toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
            step=step,
            include_output_format=False,
        )
        correct_str = (answer_dict.get("answer") or "").strip()
        pool = _pool_task3(df, idx, correct_str)
        options, correct_idx = get_choices(correct_str, pool, n_choices=4, seed=idx)
        letter = chr(65 + correct_idx)
        question_mcqa = _format_mcqa_question(question_body, options)

        rec = {
            "id": int(idx),
            "question": question_mcqa,
            "answer": {"answer": letter},
            "choices": options,
            "correct_index": correct_idx,
            "answer_value": correct_str,
        }
        (records_multi if step == "multi_step" else records_single).append(rec)

    out_single = OUT_DIR_TASK3 / "single_step" / "task3_safe_qa_mcqa.jsonl"
    out_multi = OUT_DIR_TASK3 / "multi_step" / "task3_safe_qa_mcqa.jsonl"
    _write_jsonl(out_single, records_single)
    _write_jsonl(out_multi, records_multi)
    print(f"Task 3 MCQA: single_step={len(records_single)} -> {out_single}")
    print(f"Task 3 MCQA: multi_step ={len(records_multi)} -> {out_multi}")
    return out_single, out_multi


def main():
    ap = argparse.ArgumentParser(description="Build Task 1/2/3 MCQA jsonl (same data as open-ended).")
    ap.add_argument(
        "--task",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which task to build: 1, 2, 3, or all.",
    )
    args = ap.parse_args()

    if args.task in ("1", "all"):
        build_task1_mcqa()
    if args.task in ("2", "all"):
        build_task2_mcqa()
    if args.task in ("3", "all"):
        build_task3_mcqa()


if __name__ == "__main__":
    main()
