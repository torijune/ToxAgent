#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build *agentic-flow-only* QA/context files from the same raw CSV used by build_safe_qa.py.

Goal
----
For an agentic pipeline (Task1 → Task2 → Task3 instruction), we want all tasks to operate on
the *same* samples, aligned by `source_index`.

What this script builds
-----------------------
1) Task1 QA JSONL (same format as build_safe_qa.py)
2) Task2 QA JSONL (same format as build_safe_qa.py)
3) Task3 instruction *context* JSONL (NO question generation here):
   - stores all fields required to later build the Task3 instruction question
   - intentionally excludes the remove/add fragments (only_*_safe_fragments), because
     those will be filled with Task1/Task2 *predictions* during inference.

Key property
------------
Within each (split, molecule_repr, step) group, Task1 and Task2 are generated using
the *same ordered list of source_index* (optionally shuffled with a fixed seed).
This guarantees that downstream agentic inference can align samples by source_index.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup (mirror build_safe_qa.py behavior)
# ---------------------------------------------------------------------------
_AGENTIC_DIR = Path(__file__).resolve().parent
_LLM_DIR = _AGENTIC_DIR.parent
_QA_DIR = _LLM_DIR.parent
_QA_SRC = _QA_DIR / "src"
_PROJECT_ROOT = _QA_DIR.parent.parent

for _p in [str(_QA_SRC), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qa_template import (  # noqa: E402
    task1_toxic_fragment_identification,
    task2_nontoxic_fragment_generation,
    task3_nontoxic_smiles_generation,
    task3_instruction_nontoxic_smiles_generation,
)
from task3_instruction_ver import build_cot_instruction  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_SPLIT_DIR = _QA_DIR.parent / "splits" / "scaffold_by_endpoint_unseen_ver"
_DEFAULT_TRAIN_CSV = _DEFAULT_SPLIT_DIR / "merged_train.csv"
_DEFAULT_TEST_CSV = _DEFAULT_SPLIT_DIR / "merged_test.csv"

MOLECULE_REPR_CHOICES = ["only_smiles", "only_safe", "both_repre"]


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


def _write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _build_index_order(df: pd.DataFrame, seed: Optional[int]) -> Dict[str, List[int]]:
    """
    Return ordered source_index lists per step: {"single_step": [...], "multi_step": [...]}.
    Ordering is by ascending source_index, optionally shuffled by seed (same shuffle per step).
    """
    single: List[int] = []
    multi: List[int] = []
    for idx, row in df.iterrows():
        only_toxic = _str_or_empty(row.get("only_toxic_safe_fragments", ""))
        only_nontoxic = _str_or_empty(row.get("only_nontoxic_safe_fragments", ""))
        step = _classify_step(only_toxic, only_nontoxic)
        (multi if step == "multi_step" else single).append(int(row["source_index"]))

    single.sort()
    multi.sort()
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(single)
        rng.shuffle(multi)
    return {"single_step": single, "multi_step": multi}


def _subset_df_by_indices(df: pd.DataFrame, indices: List[int]) -> pd.DataFrame:
    by_idx = df.set_index("source_index", drop=False)
    rows = [by_idx.loc[i] for i in indices if i in by_idx.index]
    if not rows:
        return df.head(0).copy()
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_task1_task2_for_repr(
    df: pd.DataFrame,
    out_root: Path,
    molecule_repr: str,
    seed: Optional[int],
) -> Tuple[Path, Path, Path, Path]:
    """
    Build Task1 + Task2 QA for a molecule_repr, split into single/multi step.
    Returns (task1_single, task1_multi, task2_single, task2_multi) paths.
    """
    order = _build_index_order(df, seed)

    out_task1 = out_root / "task1_toxic_fragment_identification" / molecule_repr
    out_task2 = out_root / "task2_nontoxic_fragment_generation" / molecule_repr

    built_paths = []
    for step_norm in ["single_step", "multi_step"]:
        indices = order[step_norm]
        df_step = _subset_df_by_indices(df, indices)

        # Task1
        recs1: List[dict] = []
        for i, row in df_step.iterrows():
            q, a = task1_toxic_fragment_identification(
                toxic_safe=_str_or_empty(row.get("toxic_safe", "")),
                only_toxic_safe_fragments=_str_or_empty(row.get("only_toxic_safe_fragments", "")),
                dataset_name=_str_or_empty(row.get("dataset_name", "")) or None,
                endpoint=_str_or_empty(row.get("endpoint", "")) or None,
                toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
                step=step_norm,
                molecule_repr=molecule_repr,
            )
            recs1.append(
                {
                    "id": i,
                    "question": q,
                    "answer": a,
                    "dataset_name": _str_or_empty(row.get("dataset_name", "")),
                    "endpoint": _str_or_empty(row.get("endpoint", "")),
                    "source_index": int(row["source_index"]),
                }
            )

        p1 = out_task1 / step_norm / "task1_toxic_fragment_identification_qa.jsonl"
        _write_jsonl(p1, recs1)

        # Task2
        recs2: List[dict] = []
        for i, row in df_step.iterrows():
            q, a = task2_nontoxic_fragment_generation(
                toxic_safe=_str_or_empty(row.get("toxic_safe", "")),
                only_toxic_safe_fragments=_str_or_empty(row.get("only_toxic_safe_fragments", "")),
                only_nontoxic_safe_fragments=_str_or_empty(row.get("only_nontoxic_safe_fragments", "")),
                dataset_name=_str_or_empty(row.get("dataset_name", "")) or None,
                endpoint=_str_or_empty(row.get("endpoint", "")) or None,
                toxic_safe_decoded_smiles=_str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
                nontoxic_safe_decoded_smiles=_str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
                nontoxic_safe=_str_or_empty(row.get("nontoxic_safe", "")),
                step=step_norm,
                molecule_repr=molecule_repr,
            )
            recs2.append(
                {
                    "id": i,
                    "question": q,
                    "answer": a,
                    "dataset_name": _str_or_empty(row.get("dataset_name", "")),
                    "endpoint": _str_or_empty(row.get("endpoint", "")),
                    "source_index": int(row["source_index"]),
                    "common_safe_fragments": _str_or_empty(row.get("common_safe_fragments", "")),
                    "nontoxic_safe_decoded_smiles": _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
                }
            )

        p2 = out_task2 / step_norm / "task2_nontoxic_fragment_generation_qa.jsonl"
        _write_jsonl(p2, recs2)

        built_paths.extend([p1, p2])

    return (
        out_task1 / "single_step" / "task1_toxic_fragment_identification_qa.jsonl",
        out_task1 / "multi_step" / "task1_toxic_fragment_identification_qa.jsonl",
        out_task2 / "single_step" / "task2_nontoxic_fragment_generation_qa.jsonl",
        out_task2 / "multi_step" / "task2_nontoxic_fragment_generation_qa.jsonl",
    )


def build_task3_instruction_context(
    df: pd.DataFrame,
    out_root: Path,
    molecule_repr: str,
    seed: Optional[int],
) -> Tuple[Path, Path]:
    """
    Build Task3 instruction *context* (no question) JSONL per step.

    The record contains everything needed to later construct the Task3_CoT question,
    but does NOT include only_toxic_safe_fragments / only_nontoxic_safe_fragments.
    Those should be filled with Task1/Task2 *predictions* during agentic inference.
    """
    order = _build_index_order(df, seed)
    out_dir = out_root / "task3_instruction_context" / molecule_repr

    out_paths: Dict[str, Path] = {}
    for step_norm in ["single_step", "multi_step"]:
        indices = order[step_norm]
        df_step = _subset_df_by_indices(df, indices)
        recs: List[dict] = []
        for i, row in df_step.iterrows():
            recs.append(
                {
                    "id": i,
                    "dataset_name": _str_or_empty(row.get("dataset_name", "")),
                    "endpoint": _str_or_empty(row.get("endpoint", "")),
                    "source_index": int(row["source_index"]),
                    # representations + gold answer fields
                    "toxic_safe": _str_or_empty(row.get("toxic_safe", "")),
                    "toxic_safe_decoded_smiles": _str_or_empty(row.get("toxic_safe_decoded_smiles", "")),
                    "nontoxic_safe_decoded_smiles": _str_or_empty(row.get("nontoxic_safe_decoded_smiles", "")),
                    "nontoxic_safe": _str_or_empty(row.get("nontoxic_safe", "")),
                    # for later sanity checks / optional use
                    "step": step_norm,
                    "molecule_repr": molecule_repr,
                }
            )

        out_path = out_dir / step_norm / "task3_instruction_context.jsonl"
        _write_jsonl(out_path, recs)
        out_paths[step_norm] = out_path

    return out_paths["single_step"], out_paths["multi_step"]


def build_task3_and_task3_instruction_gold(
    df: pd.DataFrame,
    out_root: Path,
    molecule_repr: str,
    seed: Optional[int],
) -> Tuple[Path, Path, Path, Path]:
    """
    Build aligned Task3 QA and Task3 instruction QA (with GOLD remove/add fragments) per step.

    - task3: standard end-to-end prompt (no remove/add)
    - task3_instruction_gold: prompt that includes remove/add using gold only_*_safe_fragments

    Both are built on the exact same ordered source_index list as task1/2, so they can be
    compared apples-to-apples on identical samples.
    """
    order = _build_index_order(df, seed)
    out_task3 = out_root / "task3_nontoxic_smiles_generation" / molecule_repr
    out_task3_instruction = out_root / "task3_instruction_nontoxic_smiles_generation" / molecule_repr

    paths: Dict[str, Path] = {}
    for step_norm in ["single_step", "multi_step"]:
        indices = order[step_norm]
        df_step = _subset_df_by_indices(df, indices)

        recs3: List[dict] = []
        recs3c: List[dict] = []
        for i, row in df_step.iterrows():
            toxic_safe = _str_or_empty(row.get("toxic_safe", ""))
            toxic_smiles = _str_or_empty(row.get("toxic_safe_decoded_smiles", ""))
            gold_nontoxic_smiles = _str_or_empty(row.get("nontoxic_safe_decoded_smiles", ""))
            dataset_name = _str_or_empty(row.get("dataset_name", "")) or None
            endpoint = _str_or_empty(row.get("endpoint", "")) or None

            # task3
            q3, a3 = task3_nontoxic_smiles_generation(
                toxic_safe=toxic_safe,
                dataset_name=dataset_name,
                endpoint=endpoint,
                toxic_safe_decoded_smiles=toxic_smiles,
                nontoxic_safe_decoded_smiles=gold_nontoxic_smiles,
                step=step_norm,
                molecule_repr=molecule_repr,
            )
            recs3.append(
                {
                    "id": i,
                    "question": q3,
                    "answer": a3,
                    "dataset_name": _str_or_empty(row.get("dataset_name", "")),
                    "endpoint": _str_or_empty(row.get("endpoint", "")),
                    "source_index": int(row["source_index"]),
                }
            )

            # task3_instruction (gold remove/add)
            gold_toxic_frag = _str_or_empty(row.get("only_toxic_safe_fragments", ""))
            gold_nontoxic_frag = _str_or_empty(row.get("only_nontoxic_safe_fragments", ""))
            cot_instruction = build_cot_instruction(gold_toxic_frag, gold_nontoxic_frag, step=step_norm)
            q3c, a3c = task3_instruction_nontoxic_smiles_generation(
                toxic_safe=toxic_safe,
                cot_instruction=cot_instruction,
                dataset_name=dataset_name,
                endpoint=endpoint,
                toxic_safe_decoded_smiles=toxic_smiles,
                nontoxic_safe_decoded_smiles=gold_nontoxic_smiles,
                step=step_norm,
                molecule_repr=molecule_repr,
            )
            recs3c.append(
                {
                    "id": i,
                    "question": q3c,
                    "answer": a3c,
                    "dataset_name": _str_or_empty(row.get("dataset_name", "")),
                    "endpoint": _str_or_empty(row.get("endpoint", "")),
                    "source_index": int(row["source_index"]),
                    "only_toxic_safe_fragments": gold_toxic_frag,
                    "only_nontoxic_safe_fragments": gold_nontoxic_frag,
                    "cot_instruction": cot_instruction,
                }
            )

        p3 = out_task3 / step_norm / "task3_nontoxic_smiles_generation_qa.jsonl"
        p3c = out_task3_instruction / step_norm / "task3_instruction_nontoxic_smiles_generation_qa.jsonl"
        _write_jsonl(p3, recs3)
        _write_jsonl(p3c, recs3c)
        paths[f"task3_{step_norm}"] = p3
        paths[f"task3_instruction_{step_norm}"] = p3c

    return (
        paths["task3_single_step"],
        paths["task3_multi_step"],
        paths["task3_instruction_single_step"],
        paths["task3_instruction_multi_step"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build aligned QA/context for agentic flow (Task1/2 QA + Task3_CoT context).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--split", choices=["train", "test"], default="test", help="Which split CSV to use.")
    ap.add_argument(
        "--split_dir",
        type=str,
        default=str(_DEFAULT_SPLIT_DIR),
        help="Directory containing merged_train.csv and merged_test.csv.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_QA_DIR / "test" / "agentic_flow_qa"),
        help="Output root directory for agentic QA/context (will be overwritten per --split default).",
    )
    ap.add_argument(
        "--molecule_repr",
        choices=MOLECULE_REPR_CHOICES + ["all"],
        default="both_repre",
        help="Molecule representation. 'all' builds for all three representations.",
    )
    ap.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle seed applied to source_index ordering within each step. Set to -1 to disable shuffling.",
    )
    args = ap.parse_args()

    split_dir = Path(args.split_dir)
    csv_path = split_dir / ("merged_train.csv" if args.split == "train" else "merged_test.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    seed: Optional[int] = None if int(args.shuffle_seed) < 0 else int(args.shuffle_seed)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    if "source_index" not in df.columns:
        df["source_index"] = df.index

    out_root = Path(args.out_dir)
    # Default out_dir should be split-aware if user didn't override
    if str(out_root).endswith(str(Path("test") / "agentic_flow_qa")) or str(out_root).endswith(str(Path("train") / "agentic_flow_qa")):
        out_root = _QA_DIR / args.split / "agentic_flow_qa"
    else:
        # If custom out_root is given, still nest by split for cleanliness
        out_root = out_root / args.split

    repres = MOLECULE_REPR_CHOICES if args.molecule_repr == "all" else [args.molecule_repr]

    for r in repres:
        if r not in MOLECULE_REPR_CHOICES:
            raise ValueError(f"Invalid molecule_repr: {r}")

        t1_single, t1_multi, t2_single, t2_multi = build_task1_task2_for_repr(
            df=df,
            out_root=out_root,
            molecule_repr=r,
            seed=seed,
        )
        c_single, c_multi = build_task3_instruction_context(
            df=df,
            out_root=out_root,
            molecule_repr=r,
            seed=seed,
        )
        t3_single, t3_multi, t3c_single, t3c_multi = build_task3_and_task3_instruction_gold(
            df=df,
            out_root=out_root,
            molecule_repr=r,
            seed=seed,
        )

        print(f"\n[agentic_flow_qa] split={args.split} repr={r}")
        print(f"  task1 single_step -> {t1_single}")
        print(f"  task1 multi_step  -> {t1_multi}")
        print(f"  task2 single_step -> {t2_single}")
        print(f"  task2 multi_step  -> {t2_multi}")
        print(f"  task3_instruction_context single_step -> {c_single}")
        print(f"  task3_instruction_context multi_step  -> {c_multi}")
        print(f"  task3 (aligned) single_step -> {t3_single}")
        print(f"  task3 (aligned) multi_step  -> {t3_multi}")
        print(f"  task3_instruction_gold (aligned) single_step -> {t3c_single}")
        print(f"  task3_instruction_gold (aligned) multi_step  -> {t3c_multi}")


if __name__ == "__main__":
    main()