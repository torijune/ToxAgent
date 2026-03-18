#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full agentic flow: Task1 → Task2 → Task3 instruction.

Flow
----
  Step 1 (Task1): Load pre-built QA from QA/<split>/task1_toxic_fragment_identification/<repre>/<step>/, run inference.
  Step 2 (Task2): Load pre-built QA from QA/<split>/task2_nontoxic_fragment_generation/<repre>/<step>/, run inference.
  Step 3 (Task3 instruction): Build QA from task1 + task2 *outputs* only (remove pred_toxic, add pred_nontoxic), run inference.
                       single/multi step is considered when building and when saving results.

Output structure (results/ + evaluation/ per task and step, same as general inference)
--------------------------------------------------------------------------------
  <out_dir>/agentic/<split>/<molecule_repr>/
    task1/single_step/  and  task1/multi_step/
      results/predictions_<model>.jsonl
      evaluation/evaluation_summary_<model>.json
    task2/single_step/  and  task2/multi_step/
      ...
    task3_instruction/single_step/  and  task3_instruction/multi_step/
      ...

Usage
-----
  python inference_agentic_task3.py --model gpt-4o --split test --molecule_repr both_repre --step all
  python inference_agentic_task3.py --model gpt-4o --step multi_step --num_samples 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_AGENTIC_DIR = Path(__file__).resolve().parent
_LLM_DIR = _AGENTIC_DIR.parent
_QA_DIR = _LLM_DIR.parent
_QA_SRC = _QA_DIR / "src"
_PROJECT_ROOT = _QA_DIR.parent.parent

for _p in [str(_AGENTIC_DIR), str(_LLM_DIR), str(_QA_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Pre-built QA dir: QA/test/task1_toxic_fragment_identification, task2_nontoxic_fragment_generation
_QA_BASE = _LLM_DIR.parent

# Agentic utilities
from get_task1_task2_output import (  # noqa: E402
    run_task1_inference_from_qa,
    run_task2_inference_from_qa,
    _infer_rows_batch,
    _extract_gold_str,
)
from generate_task3_instruction import build_task3_instruction_records  # noqa: E402

# Inference / evaluation utilities from inference_gpt.py
from inference_gpt import (  # noqa: E402
    _system_instruction_for_task,
    _get_metrics_for_task,
    read_jsonl,
)
try:
    from eval_metric import TASK_METRIC_KEYS
except ImportError:
    TASK_METRIC_KEYS = {}

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
_DEFAULT_SPLIT_DIR = _QA_DIR.parent / "splits" / "scaffold_by_endpoint_unseen_ver"
_DEFAULT_ENV_PATH = _PROJECT_ROOT / ".env"
_DEFAULT_OUT_DIR = _LLM_DIR / "safe_qa_outputs"

REPRE_CHOICES = ["only_safe", "only_smiles", "both_repre"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _save_and_summarize(
    preds: List[dict],
    task: str,
    model: str,
    split: str,
    molecule_repr: str,
    out_dir: Path,
    step_norm: str = "multi_step",
) -> None:
    """
    Write predictions JSONL and evaluation summary. step_norm = single_step | multi_step (same layout as inference_gpt).
    """
    if not preds:
        return
    safe_model = model.replace("/", "_")
    task_dir = out_dir / "agentic" / split / molecule_repr / task / step_norm
    results_dir = task_dir / "results"
    evaluation_dir = task_dir / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"predictions_{safe_model}.jsonl"
    _write_jsonl(pred_path, preds)

    total = len(preds)
    correct = sum(int(p.get("correct", 0)) for p in preds)
    acc = correct / max(total, 1)

    task_keys = TASK_METRIC_KEYS.get(task, [])
    metric_sums: Dict[str, float] = {}
    for p in preds:
        for k in task_keys:
            v = p.get(k)
            if isinstance(v, (int, float)):
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
    metric_means = {}
    for k in task_keys:
        metric_means[k] = metric_sums[k] / max(total, 1) if k in metric_sums else None

    summary = {
        "task": task,
        "variant": "agentic",
        "step": step_norm,
        "split": split,
        "repre": molecule_repr,
        "run": None,
        "model": model,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "metrics_mean": metric_means,
    }

    summary_path = evaluation_dir / f"evaluation_summary_{safe_model}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== [agentic/{task}/{step_norm}] {model} | {split} | {molecule_repr} ===")
    print(f"  total={total}, correct={correct}, acc={acc:.4f}")
    for k, v in metric_means.items():
        if v is not None:
            arrow = "↑" if k not in ("levenshtein", "levenshtein_dist", "levenshtein_norm") else "↓"
            print(f"  {k}: {v:.4f} {arrow}")
    print(f"  results    → {pred_path}")
    print(f"  evaluation → {summary_path}")


# ---------------------------------------------------------------------------
# Core agentic runner
# ---------------------------------------------------------------------------

def _task1_qa_path(split: str, molecule_repr: str, step_norm: str) -> Path:
    """Pre-built task1 QA: QA/<split>/task1_toxic_fragment_identification/<repre>/<step>/..."""
    return (
        _QA_BASE
        / split
        / "task1_toxic_fragment_identification"
        / molecule_repr
        / step_norm
        / "task1_toxic_fragment_identification_qa.jsonl"
    )


def _task2_qa_path(split: str, molecule_repr: str, step_norm: str) -> Path:
    """Pre-built task2 QA: QA/<split>/task2_nontoxic_fragment_generation/<repre>/<step>/..."""
    return (
        _QA_BASE
        / split
        / "task2_nontoxic_fragment_generation"
        / molecule_repr
        / step_norm
        / "task2_nontoxic_fragment_generation_qa.jsonl"
    )


def _resolve_qa_base(qa_root: Optional[Path], split: str) -> Path:
    """
    Resolve the directory that contains task QA folders.

    - Default (qa_root=None): use legacy QA layout under `_QA_BASE/<split>/...`
    - If qa_root points to `.../QA/<split>`: use qa_root directly.
    - If qa_root points to `.../QA/<split>/agentic_flow_qa`: use that directly.
    - If qa_root points to `.../QA` (or similar): use `qa_root/<split>`.
    """
    if qa_root is None:
        return _QA_BASE / split
    qr = Path(qa_root)
    if not qr.exists():
        raise FileNotFoundError(f"qa_root not found: {qr}")
    if qr.name in ("train", "test"):
        return qr
    if qr.name == "agentic_flow_qa":
        # already split-specific folder: QA/<split>/agentic_flow_qa
        return qr
    # Otherwise treat as QA root-like folder
    cand = qr / split
    if cand.exists():
        return cand
    return qr


def _task1_qa_path_in(base: Path, molecule_repr: str, step_norm: str) -> Path:
    return (
        base
        / "task1_toxic_fragment_identification"
        / molecule_repr
        / step_norm
        / "task1_toxic_fragment_identification_qa.jsonl"
    )


def _task2_qa_path_in(base: Path, molecule_repr: str, step_norm: str) -> Path:
    return (
        base
        / "task2_nontoxic_fragment_generation"
        / molecule_repr
        / step_norm
        / "task2_nontoxic_fragment_generation_qa.jsonl"
    )


def _task3_instruction_context_path_in(base: Path, molecule_repr: str, step_norm: str) -> Path:
    return (
        base
        / "task3_instruction_context"
        / molecule_repr
        / step_norm
        / "task3_instruction_context.jsonl"
    )


def _task3_instruction_qa_path_legacy(split: str, molecule_repr: str, step_norm: str) -> Path:
    """
    Legacy (non-agentic) task3_instruction QA path under ace_safe_ver/QA/<split>/task3_instruction_nontoxic_smiles_generation/...
    This is used as an *anchor* to force agentic flow to run on the exact same samples.
    """
    return (
        _QA_BASE
        / split
        / "task3_instruction_nontoxic_smiles_generation"
        / molecule_repr
        / step_norm
        / "task3_instruction_nontoxic_smiles_generation_qa.jsonl"
    )


def _load_source_indices_from_jsonl(path: Path) -> List[int]:
    if not path.exists():
        return []
    rows = read_jsonl(str(path))
    out: List[int] = []
    for r in rows:
        si = r.get("source_index")
        if si is None:
            continue
        try:
            out.append(int(si))
        except Exception:
            continue
    return out


def run_agentic_flow(
    csv_path: Path,
    client: OpenAI,
    model: str,
    molecule_repr: str = "both_repre",
    num_samples: int = 0,
    out_dir: Path = _DEFAULT_OUT_DIR,
    batch_size: int = 10,
    sleep_s: float = 0.2,
    split: str = "test",
    max_retries: int = 3,
    step_filter: str = "all",
    qa_root: Optional[Path] = None,
    align_to_task3_instruction: bool = True,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    - Task1/Task2: Load QA from QA/<split>/task1_..., task2_... (<repre>/<step>/), run inference.
    - Task3 CoT: Build QA from task1+2 outputs only (remove pred_toxic, add pred_nontoxic), run inference; single/multi when building and saving.
    step_filter: single_step | multi_step | all.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    steps_to_run = ["single_step", "multi_step"] if step_filter == "all" else [step_filter]
    qa_base = _resolve_qa_base(qa_root, split)

    # Anchor indices: force the whole flow to run on the same samples as legacy task3_instruction QA.
    anchor_indices_by_step: Dict[str, List[int]] = {}
    if align_to_task3_instruction:
        for step_norm in steps_to_run:
            anchor_path = _task3_instruction_qa_path_legacy(split, molecule_repr, step_norm)
            anchor_indices_by_step[step_norm] = _load_source_indices_from_jsonl(anchor_path)
            if not anchor_indices_by_step[step_norm]:
                print(f"  [warn] anchor task3_instruction QA not found or empty: {anchor_path}")

    # ------------------------------------------------------------------
    # Step 1 — Task1: pre-built QA → inference
    # ------------------------------------------------------------------
    print("\n[Step 1/3] Task1: toxic fragment identification (from pre-built QA)")
    task1_preds: List[dict] = []
    for step_norm in steps_to_run:
        qa1_path = _task1_qa_path_in(qa_base, molecule_repr, step_norm)
        qa2_path = _task2_qa_path_in(qa_base, molecule_repr, step_norm)
        if not qa1_path.exists():
            print(f"  Skip (not found): {qa1_path}")
            continue
        if not qa2_path.exists():
            print(f"  Skip (task2 QA missing for alignment): {qa2_path}")
            continue

        # Align task1/task2 QA samples by source_index (same sample set for this step)
        task1_rows = read_jsonl(str(qa1_path))
        task2_rows = read_jsonl(str(qa2_path))
        task1_indices: List[int] = []
        for r in task1_rows:
            si = r.get("source_index")
            if si is None:
                continue
            try:
                task1_indices.append(int(si))
            except Exception:
                continue
        task2_index_set = set()
        for r in task2_rows:
            si = r.get("source_index")
            if si is None:
                continue
            try:
                task2_index_set.add(int(si))
            except Exception:
                continue

        common_indices = [i for i in task1_indices if i in task2_index_set]
        if align_to_task3_instruction:
            anchor_set = set(anchor_indices_by_step.get(step_norm, []))
            common_indices = [i for i in common_indices if i in anchor_set]
        if num_samples and num_samples > 0:
            common_indices = common_indices[:num_samples]

        preds = run_task1_inference_from_qa(
            qa1_path,
            client,
            model,
            step_norm,
            batch_size,
            sleep_s,
            max_retries,
            num_samples=0,
            source_indices=common_indices,
        )
        task1_preds.extend(preds)
        _save_and_summarize(preds, "task1", model, split, molecule_repr, out_dir, step_norm=step_norm)

    if not task1_preds:
        print("[agentic] No task1 predictions; abort.")
        return [], [], []

    # ------------------------------------------------------------------
    # Step 2 — Task2: pre-built QA → inference
    # ------------------------------------------------------------------
    print("\n[Step 2/3] Task2: nontoxic fragment generation (from pre-built QA)")
    task2_preds: List[dict] = []
    for step_norm in steps_to_run:
        qa1_path = _task1_qa_path_in(qa_base, molecule_repr, step_norm)
        qa2_path = _task2_qa_path_in(qa_base, molecule_repr, step_norm)
        if not qa2_path.exists():
            print(f"  Skip (not found): {qa2_path}")
            continue
        if not qa1_path.exists():
            print(f"  Skip (task1 QA missing for alignment): {qa1_path}")
            continue

        # Align task1/task2 QA samples by source_index (same sample set for this step)
        task1_rows = read_jsonl(str(qa1_path))
        task2_rows = read_jsonl(str(qa2_path))
        task1_indices: List[int] = []
        for r in task1_rows:
            si = r.get("source_index")
            if si is None:
                continue
            try:
                task1_indices.append(int(si))
            except Exception:
                continue
        task2_index_set = set()
        for r in task2_rows:
            si = r.get("source_index")
            if si is None:
                continue
            try:
                task2_index_set.add(int(si))
            except Exception:
                continue

        common_indices = [i for i in task1_indices if i in task2_index_set]
        if align_to_task3_instruction:
            anchor_set = set(anchor_indices_by_step.get(step_norm, []))
            common_indices = [i for i in common_indices if i in anchor_set]
        if num_samples and num_samples > 0:
            common_indices = common_indices[:num_samples]

        preds = run_task2_inference_from_qa(
            qa2_path,
            client,
            model,
            step_norm,
            batch_size,
            sleep_s,
            max_retries,
            num_samples=0,
            source_indices=common_indices,
        )
        task2_preds.extend(preds)
        _save_and_summarize(preds, "task2", model, split, molecule_repr, out_dir, step_norm=step_norm)

    # ------------------------------------------------------------------
    # Step 3 — Task3 instruction: build QA from task1+2 outputs → inference
    # ------------------------------------------------------------------
    print("\n[Step 3/3] Task3 instruction: nontoxic SMILES (instruction from task1+2 outputs)")

    # Prefer agentic_flow_qa's task3_instruction_context (guaranteed same sample set by source_index)
    ctx_rows: List[dict] = []
    for step_norm in steps_to_run:
        ctx_path = _task3_instruction_context_path_in(qa_base, molecule_repr, step_norm)
        if ctx_path.exists():
            ctx_rows.extend(read_jsonl(str(ctx_path)))
    if ctx_rows:
        df = pd.DataFrame(ctx_rows)
    else:
        df = pd.read_csv(csv_path).reset_index(drop=True)
        if "source_index" not in df.columns:
            df["source_index"] = df.index
    source_indices = set(p["source_index"] for p in task1_preds) & set(p["source_index"] for p in task2_preds)
    if align_to_task3_instruction:
        anchor_all = set()
        for step_norm in steps_to_run:
            anchor_all |= set(anchor_indices_by_step.get(step_norm, []))
        source_indices &= anchor_all
    df = df[df["source_index"].isin(source_indices)].copy()
    if df.empty:
        print("[agentic] No overlapping source_indices for task3; skip.")
        return task1_preds, task2_preds, []

    task3_rows = build_task3_instruction_records(
        df=df,
        task1_preds=task1_preds,
        task2_preds=task2_preds,
        molecule_repr=molecule_repr,
    )

    system_instruction = _system_instruction_for_task("task3_instruction")
    inference_results = _infer_rows_batch(
        task3_rows,
        client,
        model,
        system_instruction,
        batch_size,
        sleep_s,
        max_retries,
    )

    task3_preds = []
    for row, pred, raw in inference_results:
        gold_str = _extract_gold_str(row["answer"])
        pred_str = (pred or "").strip()
        is_correct = int(pred_str == gold_str)
        metrics = _get_metrics_for_task(
            "task3_instruction",
            row["answer"],
            {"answer": pred_str},
            row_id=row["source_index"],
        )
        out = {
            "model": model,
            "id": row.get("id"),
            "dataset_name": row.get("dataset_name", ""),
            "endpoint": row.get("endpoint", ""),
            "source_index": row.get("source_index"),
            "gold": gold_str,
            "pred": pred_str,
            "correct": is_correct,
            "raw": raw,
            **row,
        }
        out.update(metrics)
        task3_preds.append(out)

    # Save task3_instruction by step (single_step / multi_step)
    by_step: Dict[str, List[dict]] = {}
    for p in task3_preds:
        s = p.get("step", "multi_step")
        by_step.setdefault(s, []).append(p)
    for step_norm, preds in by_step.items():
        _save_and_summarize(preds, "task3_instruction", model, split, molecule_repr, out_dir, step_norm=step_norm)

    return task1_preds, task2_preds, task3_preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Agentic flow: Task1 → Task2 → Task3 instruction (sequential, model-driven).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--env",
        type=str,
        default=str(_DEFAULT_ENV_PATH),
        help="Path to .env file containing OPENAI_API_KEY.",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test"],
        default="test",
        help="Data split. Selects merged_train.csv or merged_test.csv.",
    )
    ap.add_argument(
        "--molecule_repr",
        choices=REPRE_CHOICES + ["all"],
        default="both_repre",
        help=(
            "Molecule representation in prompts. "
            "'all' runs all three representations sequentially."
        ),
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model name (e.g. gpt-4o). Overrides --models.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="gpt-4o",
        help="Comma-separated model names.",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of rows to process (0 = all).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=(
            "Output root directory. "
            "Default: <QA_DIR>/LLMs/safe_qa_outputs. "
            "Results saved under <out_dir>/agentic/<split>/<molecule_repr>/."
        ),
    )
    ap.add_argument(
        "--sleep_s",
        type=float,
        default=0.2,
        help="Sleep seconds between API call batches.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of rows to process in parallel per batch.",
    )
    ap.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max API call retries per sample.",
    )
    ap.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Override the CSV path (default: merged_<split>.csv).",
    )
    ap.add_argument(
        "--step",
        type=str,
        choices=["single_step", "multi_step", "all"],
        default="all",
        help="Which step QA to run: single_step, multi_step, or all (both). Default: all.",
    )
    ap.add_argument(
        "--qa_root",
        type=str,
        default=None,
        help=(
            "Optional QA root override for aligned agentic QA.\n"
            "Examples:\n"
            "  - ace_safe_ver/QA/test\n"
            "  - ace_safe_ver/QA/test/agentic_flow_qa\n"
            "If not set, uses legacy QA under ace_safe_ver/QA/<split>/."
        ),
    )
    ap.add_argument(
        "--no_align_to_task3_instruction",
        action="store_true",
        help="Do not anchor agentic samples to legacy task3_instruction QA source_index set.",
    )
    args = ap.parse_args()

    load_dotenv(args.env, override=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or set it as an environment variable."
        )
    client = OpenAI(api_key=api_key)

    models = [args.model.strip()] if args.model else [
        m.strip() for m in args.models.split(",") if m.strip()
    ]
    reprs = REPRE_CHOICES if args.molecule_repr == "all" else [args.molecule_repr]

    if args.input_csv:
        csv_path = Path(args.input_csv)
    else:
        csv_fname = "merged_train.csv" if args.split == "train" else "merged_test.csv"
        csv_path = _DEFAULT_SPLIT_DIR / csv_fname

    out_dir = Path(args.out_dir) if args.out_dir else _DEFAULT_OUT_DIR

    for model in models:
        for repr_mode in reprs:
            run_agentic_flow(
                csv_path=csv_path,
                client=client,
                model=model,
                molecule_repr=repr_mode,
                num_samples=args.num_samples,
                out_dir=out_dir,
                batch_size=args.batch_size,
                sleep_s=args.sleep_s,
                split=args.split,
                max_retries=args.max_retries,
                step_filter=args.step,
                qa_root=Path(args.qa_root) if args.qa_root else None,
                align_to_task3_instruction=not args.no_align_to_task3_instruction,
            )


if __name__ == "__main__":
    main()
