"""
Task3 instruction QA builder for the agentic flow.

Given predicted task1 (toxic fragments) and task2 (nontoxic fragments) outputs,
builds task3_instruction QA records whose instruction is derived from the MODEL's
own predictions — not gold labels.

Main function:
  build_task3_instruction_records(df, task1_preds, task2_preds, molecule_repr)
    - For each row in df, looks up the corresponding task1 and task2 predictions
      by source_index.
    - Builds the CoT instruction:
        "Remove [predicted_toxic_fragments]; add [predicted_nontoxic_fragments]."
    - Calls task3_instruction_nontoxic_smiles_generation to produce the final question.
    - Returns a list of QA record dicts ready for inference (question + answer).
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Path setup: QA/src/ for qa_template and task3_instruction_ver
_AGENTIC_DIR = Path(__file__).resolve().parent
_LLM_DIR = _AGENTIC_DIR.parent
_QA_DIR = _LLM_DIR.parent
_QA_SRC = _QA_DIR / "src"
for _p in [str(_QA_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qa_template import task3_instruction_nontoxic_smiles_generation  # noqa: E402
from task3_instruction_ver import build_cot_instruction  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrored from get_task1_task2_output to keep files independent)
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
    counts = [_count_dot_fragments(x) for x in frag_strings]
    return "multi_step" if max(counts, default=0) >= 2 else "single_step"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_task3_instruction_records(
    df: pd.DataFrame,
    task1_preds: List[dict],
    task2_preds: List[dict],
    molecule_repr: str = "both_repre",
) -> List[dict]:
    """
    Build task3_instruction QA records from predicted task1/task2 fragments.

    For each row in df:
      1. Look up task1 prediction  → pred_toxic_fragments
      2. Look up task2 prediction  → pred_nontoxic_fragments
      3. Build CoT instruction:
           "Remove [pred_toxic]; add [pred_nontoxic]. Output nontoxic SMILES."
      4. Build the full question via task3_instruction_nontoxic_smiles_generation.
      5. The gold answer is nontoxic_safe_decoded_smiles from the CSV.

    Args:
        df: source DataFrame (merged_test.csv or merged_train.csv), index reset.
        task1_preds: list of dicts from build_and_infer_task1(), keyed by source_index.
        task2_preds: list of dicts from build_and_infer_task2(), keyed by source_index.
        molecule_repr: "only_smiles" | "only_safe" | "both_repre"

    Returns:
        List of QA record dicts with keys:
          id, source_index, question, answer,
          dataset_name, endpoint,
          toxic_safe, toxic_smiles,
          pred_toxic_fragments, pred_nontoxic_fragments,
          gold_nontoxic_smiles, cot_instruction, step
    """
    task1_by_idx = {p["source_index"]: p for p in task1_preds}
    task2_by_idx = {p["source_index"]: p for p in task2_preds}

    records = []
    for idx, row in df.iterrows():
        source_idx = int(row.get("source_index", idx))
        t1 = task1_by_idx.get(source_idx, {})
        t2 = task2_by_idx.get(source_idx, {})

        pred_toxic = (t1.get("pred_toxic_fragments") or "").strip()
        pred_nontoxic = (t2.get("pred_nontoxic_fragments") or "").strip()
        gold_toxic = _str_or_empty(row.get("only_toxic_safe_fragments", ""))
        gold_nontoxic = _str_or_empty(row.get("only_nontoxic_safe_fragments", ""))

        # CoT에는 remove/add가 둘 다 있어야 함. 예측이 비어 있으면 gold로 채움
        effective_toxic = pred_toxic if pred_toxic else gold_toxic
        effective_nontoxic = pred_nontoxic if pred_nontoxic else gold_nontoxic

        # step: 예측이 있으면 예측 기준, 없으면 gold 기준
        step = _classify_step(
            effective_toxic or pred_toxic or gold_toxic,
            effective_nontoxic or pred_nontoxic or gold_nontoxic,
        )
        cot_instruction = build_cot_instruction(effective_toxic, effective_nontoxic, step=step)

        toxic_safe = _str_or_empty(row.get("toxic_safe", ""))
        toxic_smiles = _str_or_empty(row.get("toxic_safe_decoded_smiles", ""))
        gold_nontoxic_smiles = _str_or_empty(row.get("nontoxic_safe_decoded_smiles", ""))

        question, answer = task3_instruction_nontoxic_smiles_generation(
            toxic_safe=toxic_safe,
            cot_instruction=cot_instruction,
            dataset_name=_str_or_empty(row.get("dataset_name", "")) or None,
            endpoint=_str_or_empty(row.get("endpoint", "")) or None,
            toxic_safe_decoded_smiles=toxic_smiles,
            nontoxic_safe_decoded_smiles=gold_nontoxic_smiles,
            step=step,
            molecule_repr=molecule_repr,
        )

        records.append({
            "id": source_idx,
            "source_index": source_idx,
            "question": question,
            "answer": answer,
            "dataset_name": _str_or_empty(row.get("dataset_name", "")),
            "endpoint": _str_or_empty(row.get("endpoint", "")),
            "toxic_safe": toxic_safe,
            "toxic_smiles": toxic_smiles,
            "pred_toxic_fragments": pred_toxic,
            "pred_nontoxic_fragments": pred_nontoxic,
            "gold_nontoxic_smiles": gold_nontoxic_smiles,
            "cot_instruction": cot_instruction,
            "step": step,
            "cot_used_fallback": bool(not pred_toxic or not pred_nontoxic),
        })

    return records
