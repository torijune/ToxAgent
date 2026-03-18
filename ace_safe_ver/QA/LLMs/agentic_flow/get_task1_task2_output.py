"""
Task1/Task2 agentic inference utilities.

Provides two main functions used in the agentic flow:

  build_and_infer_task1(df, client, model, ...)
    - Dynamically builds task1 (toxic fragment identification) questions
      from CSV rows using qa_template.task1_toxic_fragment_identification
    - Runs inference in parallel batches
    - Returns per-sample prediction dicts with:
        source_index, toxic_safe, dataset_name, endpoint,
        gold_toxic_fragments, pred_toxic_fragments,
        step, question, gold, pred, raw, metrics (fragment_EM, ...)

  build_and_infer_task2(df, task1_preds, client, model, ...)
    - Dynamically builds task2 (nontoxic fragment generation) questions
      using PREDICTED toxic fragments from task1 (not gold)
    - Runs inference in parallel batches
    - Returns per-sample prediction dicts with:
        source_index, pred_toxic_fragments (from task1),
        gold_nontoxic_fragments, pred_nontoxic_fragments,
        step, question, gold, pred, raw, metrics (fragment_EM, molecule_EM, ...)

Also exposes _infer_rows_batch() used by inference_agentic_task3.py for task3_instruction.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

# Path setup: add LLMs/ for inference_gpt, QA/src/ for qa_template / eval_metric
_AGENTIC_DIR = Path(__file__).resolve().parent
_LLM_DIR = _AGENTIC_DIR.parent
_QA_DIR = _LLM_DIR.parent
_QA_SRC = _QA_DIR / "src"
for _p in [str(_LLM_DIR), str(_QA_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference_gpt import (  # noqa: E402
    call_model,
    read_jsonl,
    _system_instruction_for_task,
    _get_metrics_for_task,
)
from qa_template import (  # noqa: E402
    task1_toxic_fragment_identification,
    task2_nontoxic_fragment_generation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str_or_empty(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def _count_dot_fragments(dot_separated: str) -> int:
    s = (dot_separated or "").strip()
    if not s:
        return 0
    return len([p for p in s.split(".") if p.strip()])


def _classify_step(*frag_strings: str) -> str:
    """single_step if max fragment count == 1, multi_step otherwise."""
    counts = [_count_dot_fragments(x) for x in frag_strings]
    return "multi_step" if max(counts, default=0) >= 2 else "single_step"


def _extract_gold_str(answer: Any) -> str:
    if isinstance(answer, dict):
        return str(answer.get("answer", "")).strip()
    return str(answer or "").strip()


# ---------------------------------------------------------------------------
# Core batch inference (shared with inference_agentic_task3.py)
# ---------------------------------------------------------------------------

def _call_row(
    client,
    model: str,
    row: dict,
    system_instruction: str,
    max_retries: int,
    sleep_s: float,
) -> Tuple[dict, Optional[str], str]:
    """Call the model for a single QA row and return (row, pred, raw)."""
    q = str(row.get("question", ""))
    pred, raw = call_model(
        client=client,
        model=model,
        question=q,
        system_instruction=system_instruction,
        max_retries=max_retries,
        sleep_s=sleep_s,
    )
    return (row, pred, raw)


def _infer_rows_batch(
    rows: List[dict],
    client,
    model: str,
    system_instruction: str,
    batch_size: int,
    sleep_s: float,
    max_retries: int = 3,
) -> List[Tuple[dict, Optional[str], str]]:
    """
    Run inference on a list of QA rows in parallel batches.
    Returns a list of (row, pred, raw) in the same order as the input rows.
    """
    if not rows:
        return []
    batch_size = max(batch_size, 1)
    results: List[Optional[Tuple[dict, Optional[str], str]]] = [None] * len(rows)

    for batch_start in tqdm(
        range(0, len(rows), batch_size),
        desc=f"  [{model}]",
        total=(len(rows) + batch_size - 1) // batch_size,
    ):
        batch = rows[batch_start : batch_start + batch_size]
        batch_results: List[Optional[Tuple]] = [None] * len(batch)

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_idx = {
                executor.submit(
                    _call_row,
                    client, model, row, system_instruction, max_retries, sleep_s,
                ): i
                for i, row in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                batch_results[i] = future.result()

        for i, r in enumerate(batch_results):
            results[batch_start + i] = r

        if sleep_s > 0:
            time.sleep(sleep_s)

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Task1 / Task2: inference from pre-built QA JSONL (QA/test/task1_..., task2_...)
# ---------------------------------------------------------------------------

def run_task1_inference_from_qa(
    qa_path: Path,
    client,
    model: str,
    step_norm: str,
    batch_size: int = 10,
    sleep_s: float = 0.2,
    max_retries: int = 3,
    num_samples: int = 0,
    source_indices: Optional[List[int]] = None,
) -> List[dict]:
    """
    Load pre-built task1 QA from qa_path, run inference, return list of pred dicts.
    If num_samples > 0, only the first num_samples rows are run (so API is called at most num_samples times per step).
    """
    if not Path(qa_path).exists():
        return []
    rows = read_jsonl(str(qa_path))
    if not rows:
        return []
    if source_indices is not None:
        wanted = list(source_indices)
        by_idx: Dict[int, dict] = {}
        for r in rows:
            si = r.get("source_index")
            if si is None:
                continue
            try:
                by_idx[int(si)] = r
            except Exception:
                continue
        rows = [by_idx[i] for i in wanted if i in by_idx]
    if num_samples and num_samples > 0:
        rows = rows[:num_samples]
    system_instruction = _system_instruction_for_task("task1")
    inference_results = _infer_rows_batch(
        rows, client, model, system_instruction, batch_size, sleep_s, max_retries
    )
    preds = []
    for row, pred, raw in inference_results:
        gold_str = _extract_gold_str(row.get("answer"))
        pred_str = (pred or "").strip()
        is_correct = int(pred_str == gold_str)
        metrics = _get_metrics_for_task(
            "task1",
            row.get("answer"),
            {"answer": pred_str},
            row_id=row.get("source_index"),
        )
        out = {
            **{k: v for k, v in row.items() if k not in ("question", "answer", "step")},
            "model": model,
            "id": row.get("id"),
            "dataset_name": row.get("dataset_name", ""),
            "endpoint": row.get("endpoint", ""),
            "source_index": row.get("source_index"),
            "gold": gold_str,
            "pred": pred_str,
            "pred_toxic_fragments": pred_str,
            "raw": raw,
            "correct": is_correct,
            "step": step_norm,
        }
        out.update(metrics)
        preds.append(out)
    return preds


def run_task2_inference_from_qa(
    qa_path: Path,
    client,
    model: str,
    step_norm: str,
    batch_size: int = 10,
    sleep_s: float = 0.2,
    max_retries: int = 3,
    num_samples: int = 0,
    source_indices: Optional[List[int]] = None,
) -> List[dict]:
    """
    Load pre-built task2 QA from qa_path, run inference, return list of pred dicts.
    If num_samples > 0, only the first num_samples rows are run.
    """
    if not Path(qa_path).exists():
        return []
    rows = read_jsonl(str(qa_path))
    if not rows:
        return []
    if source_indices is not None:
        wanted = list(source_indices)
        by_idx: Dict[int, dict] = {}
        for r in rows:
            si = r.get("source_index")
            if si is None:
                continue
            try:
                by_idx[int(si)] = r
            except Exception:
                continue
        rows = [by_idx[i] for i in wanted if i in by_idx]
    if num_samples and num_samples > 0:
        rows = rows[:num_samples]
    system_instruction = _system_instruction_for_task("task2")
    inference_results = _infer_rows_batch(
        rows, client, model, system_instruction, batch_size, sleep_s, max_retries
    )
    preds = []
    for row, pred, raw in inference_results:
        gold_str = _extract_gold_str(row.get("answer"))
        pred_str = (pred or "").strip()
        is_correct = int(pred_str == gold_str)
        metrics = _get_metrics_for_task(
            "task2",
            row.get("answer"),
            {"answer": pred_str},
            row_id=row.get("source_index"),
        )
        out = {
            **{k: v for k, v in row.items() if k not in ("question", "answer", "step")},
            "model": model,
            "id": row.get("id"),
            "dataset_name": row.get("dataset_name", ""),
            "endpoint": row.get("endpoint", ""),
            "source_index": row.get("source_index"),
            "gold": gold_str,
            "pred": pred_str,
            "pred_nontoxic_fragments": pred_str,
            "raw": raw,
            "correct": is_correct,
            "step": step_norm,
        }
        out.update(metrics)
        preds.append(out)
    return preds


# ---------------------------------------------------------------------------
# Task1: toxic fragment identification (build QA from CSV — legacy / optional)
# ---------------------------------------------------------------------------

def build_and_infer_task1(
    df: pd.DataFrame,
    client,
    model: str,
    molecule_repr: str = "both_repre",
    batch_size: int = 10,
    sleep_s: float = 0.2,
    max_retries: int = 3,
) -> List[dict]:
    """
    Dynamically build task1 QA questions from df and run inference.

    The question format (single_step vs multi_step) is determined from
    gold only_toxic_safe_fragments (the target answer count).

    Returns list of dicts, one per CSV row, including:
      source_index, toxic_safe, gold_toxic_fragments,
      pred_toxic_fragments, step, question, gold, pred, raw,
      correct, fragment_EM, fragment_BLEU1, fragment_Precision,
      fragment_Recall, fragment_F1
    """
    rows = []
    for idx, row in df.iterrows():
        toxic_safe = _str_or_empty(row.get("toxic_safe", ""))
        toxic_smiles = _str_or_empty(row.get("toxic_safe_decoded_smiles", ""))
        dataset_name = _str_or_empty(row.get("dataset_name", "")) or None
        endpoint = _str_or_empty(row.get("endpoint", "")) or None
        gold_toxic = _str_or_empty(row.get("only_toxic_safe_fragments", ""))
        step = _classify_step(gold_toxic)

        question, answer = task1_toxic_fragment_identification(
            toxic_safe=toxic_safe,
            only_toxic_safe_fragments=gold_toxic,
            dataset_name=dataset_name,
            endpoint=endpoint,
            toxic_safe_decoded_smiles=toxic_smiles,
            step=step,
            molecule_repr=molecule_repr,
        )
        rows.append({
            "id": int(idx),
            "source_index": int(idx),
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "toxic_safe": toxic_safe,
            "toxic_smiles": toxic_smiles,
            "gold_toxic_fragments": gold_toxic,
            "gold_nontoxic_fragments": _str_or_empty(row.get("only_nontoxic_safe_fragments", "")),
            "gold_nontoxic_smiles": _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
            "step": step,
        })

    system_instruction = _system_instruction_for_task("task1")
    inference_results = _infer_rows_batch(
        rows, client, model, system_instruction, batch_size, sleep_s, max_retries
    )

    preds = []
    for row, pred, raw in inference_results:
        gold_str = _extract_gold_str(row["answer"])
        pred_str = (pred or "").strip()
        is_correct = int(pred_str == gold_str)
        metrics = _get_metrics_for_task(
            "task1",
            row["answer"],
            {"answer": pred_str},
            row_id=row["source_index"],
        )
        out = {
            "model": model,
            "id": row["id"],
            "dataset_name": row["dataset_name"],
            "endpoint": row["endpoint"],
            "source_index": row["source_index"],
            "gold": gold_str,
            "pred": pred_str,
            "correct": is_correct,
            "raw": raw,
            **row,
            "pred_toxic_fragments": pred_str,
        }
        out.update(metrics)
        preds.append(out)

    return preds


# ---------------------------------------------------------------------------
# Task2: nontoxic fragment generation (with predicted toxic fragments)
# ---------------------------------------------------------------------------

def build_and_infer_task2(
    df: pd.DataFrame,
    task1_preds: List[dict],
    client,
    model: str,
    molecule_repr: str = "both_repre",
    batch_size: int = 10,
    sleep_s: float = 0.2,
    max_retries: int = 3,
) -> List[dict]:
    """
    Dynamically build task2 QA questions using PREDICTED toxic fragments from task1,
    then run inference.

    The `only_toxic_safe_fragments` slot in the question is filled with the
    task1 model prediction (not the gold value), completing the agentic chain.

    The question format (single_step vs multi_step) is determined from the
    predicted toxic fragment count.

    Returns list of dicts, one per CSV row, including:
      source_index, pred_toxic_fragments (from task1),
      gold_nontoxic_fragments, pred_nontoxic_fragments,
      step, question, gold, pred, raw,
      correct, fragment_EM, fragment_BLEU1, ..., molecule_EM, molecule_morganFT, ...
    """
    task1_by_idx = {p["source_index"]: p for p in task1_preds}

    rows = []
    for idx, row in df.iterrows():
        source_idx = int(idx)
        t1 = task1_by_idx.get(source_idx, {})
        # Use predicted toxic fragment from task1; fall back to empty string
        predicted_toxic = (t1.get("pred_toxic_fragments") or "").strip()

        toxic_safe = _str_or_empty(row.get("toxic_safe", ""))
        toxic_smiles = _str_or_empty(row.get("toxic_safe_decoded_smiles", ""))
        nontoxic_safe = _str_or_empty(row.get("nontoxic_safe", ""))
        nontoxic_smiles = _str_or_empty(row.get("nontoxic_safe_decoded_smiles", ""))
        dataset_name = _str_or_empty(row.get("dataset_name", "")) or None
        endpoint = _str_or_empty(row.get("endpoint", "")) or None
        gold_nontoxic = _str_or_empty(row.get("only_nontoxic_safe_fragments", ""))

        # Step based on predicted toxic fragment count (the actual input to this task)
        step = _classify_step(predicted_toxic)

        question, answer = task2_nontoxic_fragment_generation(
            toxic_safe=toxic_safe,
            only_toxic_safe_fragments=predicted_toxic,
            only_nontoxic_safe_fragments=gold_nontoxic,
            dataset_name=dataset_name,
            endpoint=endpoint,
            toxic_safe_decoded_smiles=toxic_smiles,
            nontoxic_safe_decoded_smiles=nontoxic_smiles,
            nontoxic_safe=nontoxic_safe,
            step=step,
            molecule_repr=molecule_repr,
        )
        rows.append({
            "id": source_idx,
            "source_index": source_idx,
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "toxic_safe": toxic_safe,
            "toxic_smiles": toxic_smiles,
            "pred_toxic_fragments": predicted_toxic,
            "gold_nontoxic_fragments": gold_nontoxic,
            "gold_nontoxic_smiles": nontoxic_smiles,
            "step": step,
        })

    system_instruction = _system_instruction_for_task("task2")
    inference_results = _infer_rows_batch(
        rows, client, model, system_instruction, batch_size, sleep_s, max_retries
    )

    preds = []
    for row, pred, raw in inference_results:
        gold_str = _extract_gold_str(row["answer"])
        pred_str = (pred or "").strip()
        is_correct = int(pred_str == gold_str)
        metrics = _get_metrics_for_task(
            "task2",
            row["answer"],
            {"answer": pred_str},
            row_id=row["source_index"],
        )
        out = {
            "model": model,
            "id": row["id"],
            "dataset_name": row["dataset_name"],
            "endpoint": row["endpoint"],
            "source_index": row["source_index"],
            "gold": gold_str,
            "pred": pred_str,
            "correct": is_correct,
            "raw": raw,
            **row,
            "pred_nontoxic_fragments": pred_str,
        }
        out.update(metrics)
        preds.append(out)

    return preds
