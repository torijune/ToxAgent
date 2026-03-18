#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 저장된 prediction 결과에 대해 평가만 다시 수행하는 스크립트.
LLM inference는 하지 않고, 기존 gold/pred를 eval_metric으로 재계산하여 덮어쓴다.

inference_gpt.py와 동일한 인자(--task, --variant, --model/--models, --out_dir)로
저장 경로를 찾아 해당 prediction 파일을 재평가한다.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None):
        return iterable

# 스크립트 위치: safe_qa_outputs/evaluation_results.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_LLM_DIR = _SCRIPT_DIR.parent
_QA_DIR = _LLM_DIR.parent
_QA_SRC = _QA_DIR / "src"

import sys
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))

try:
    from eval_metric import (
        task1_toxic_safe_to_nontoxic_safe_eval,
        task2_smiles_to_safe_eval,
        task3_toxic_fragment_identification_eval,
        task4_safe_to_nontoxic_smiles_eval,
        TASK_METRIC_KEYS,
    )
except ImportError as e:
    task1_toxic_safe_to_nontoxic_safe_eval = None
    task2_smiles_to_safe_eval = None
    task3_toxic_fragment_identification_eval = None
    task4_safe_to_nontoxic_smiles_eval = None
    TASK_METRIC_KEYS = {}
    import warnings
    warnings.warn(
        f"eval_metric import failed: {e}. Run from project root (ToxAgent) with PYTHONPATH including molecule_safe_ver/QA/src so that eval_metric and its deps (rdkit, safe, pandas) are available."
    )

# inference_gpt와 동일한 기본값
DEFAULT_OUT_DIR = _SCRIPT_DIR
VARIANT_CHOICES = ["base", "icl1", "icl2", "icl4"]


def _task_subdir(task: int) -> str:
    if task == 1:
        return "task1_safe_to_nontoxic"
    if task == 2:
        return "task2_smiles_to_safe"
    if task == 3:
        return "task3_toxic_fragment_identification"
    return "task4_safe_to_nontoxic_smiles"


def _prediction_filename(model: str, variant: str) -> str:
    safe_model = model.replace("/", "_")
    if variant == "base":
        return f"predictions_{safe_model}.jsonl"
    return f"predictions_{safe_model}_{variant}.jsonl"


def _summary_filename(model: str, variant: str, task: int) -> str:
    safe_model = model.replace("/", "_")
    if variant == "base":
        return f"summary_{safe_model}.json"
    return f"summary_{safe_model}_{variant}_task{task}.json"


def normalize_answer(ans: Optional[str]) -> str:
    return (ans or "").strip()


def _get_metrics_for_task(
    task: int,
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int],
) -> Dict[str, Any]:
    """eval_metric으로 gold/pred 평가 후 출력용 dict 반환."""
    if task == 1 and task1_toxic_safe_to_nontoxic_safe_eval is not None:
        (
            fragment_EM,
            fragment_BLEU1,
            fragment_Precision,
            fragment_Recall,
            fragment_F1,
            molecule_EM,
            molecule_morganFT,
            molecule_validity,
        ) = task1_toxic_safe_to_nontoxic_safe_eval(gold_answer, llm_answer, row_id=row_id)
        return {
            "fragment_EM": fragment_EM,
            "fragment_BLEU1": fragment_BLEU1,
            "fragment_Precision": fragment_Precision,
            "fragment_Recall": fragment_Recall,
            "fragment_F1": fragment_F1,
            "molecule_EM": molecule_EM,
            "molecule_morganFT": molecule_morganFT,
            "molecule_validity": molecule_validity,
        }
    if task == 2 and task2_smiles_to_safe_eval is not None:
        (
            EM,
            BLEU1,
            validity,
            lev_dist,
            lev_norm,
            molecule_EM,
            molecule_morganFT,
            molecule_validity,
        ) = task2_smiles_to_safe_eval(gold_answer, llm_answer, row_id=row_id)
        return {
            "EM": EM,
            "BLEU1": BLEU1,
            "validity": validity,
            "levenshtein_dist": lev_dist,
            "levenshtein_norm": lev_norm,
            "molecule_EM": molecule_EM,
            "molecule_morganFT": molecule_morganFT,
            "molecule_validity": molecule_validity,
        }
    if task == 3 and task3_toxic_fragment_identification_eval is not None:
        (
            fragment_EM,
            fragment_BLEU1,
            fragment_Precision,
            fragment_Recall,
            fragment_F1,
        ) = task3_toxic_fragment_identification_eval(gold_answer, llm_answer)
        return {
            "fragment_EM": fragment_EM,
            "fragment_BLEU1": fragment_BLEU1,
            "fragment_Precision": fragment_Precision,
            "fragment_Recall": fragment_Recall,
            "fragment_F1": fragment_F1,
        }
    if task == 4 and task4_safe_to_nontoxic_smiles_eval is not None:
        (
            exact_match,
            bleu,
            levenshtein,
            rdk_fts,
            maccs_fts,
            morgan_fts,
            validity,
        ) = task4_safe_to_nontoxic_smiles_eval(gold_answer, llm_answer)
        return {
            "exact_match": exact_match,
            "bleu": bleu,
            "levenshtein": levenshtein,
            "rdk_fts": rdk_fts,
            "maccs_fts": maccs_fts,
            "morgan_fts": morgan_fts,
            "validity": validity,
        }
    return {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_reeval(
    out_dir: Path,
    task: int,
    variant: str,
    models: List[str],
    num_samples: int = 0,
    step: str = "single_step",
) -> None:
    out_dir = Path(out_dir)
    task_subdir = _task_subdir(task)
    # Task 2는 step 구분 없음. Task 1/3은 step 서브디렉터리.
    if task == 2:
        task_out_dir = out_dir / task_subdir
    else:
        task_out_dir = out_dir / task_subdir / step
    if not task_out_dir.exists():
        print(f"Task output dir not found: {task_out_dir}")
        return

    for model in models:
        pred_path = task_out_dir / _prediction_filename(model, variant)
        if not pred_path.exists():
            print(f"Skip (file not found): {pred_path}")
            continue

        all_rows = read_jsonl(pred_path)
        to_process = all_rows[:num_samples] if num_samples and num_samples > 0 else all_rows

        correct = 0
        total = 0
        metric_sums: Dict[str, float] = {}
        updated_rows: List[Dict[str, Any]] = []

        for row in tqdm(to_process, desc=f"Re-eval [{model}] {variant}", total=len(to_process)):
            gold = row.get("gold", "")
            pred = row.get("pred", "")
            if isinstance(gold, dict):
                gold = gold.get("answer", "")
            if isinstance(pred, dict):
                pred = pred.get("answer", "")
            gold = (gold or "").strip()
            pred = (pred or "").strip()
            row_id = row.get("id")

            gold_answer = {"answer": gold}
            llm_answer = {"answer": pred}
            metrics = _get_metrics_for_task(task, gold_answer, llm_answer, row_id=row_id)

            is_correct = 1 if normalize_answer(pred) == normalize_answer(gold) else 0
            correct += is_correct
            total += 1
            for k, v in metrics.items():
                if v is not None and isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

            out_row = {
                "model": row.get("model", model),
                "id": row_id,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw": row.get("raw", ""),
            }
            out_row.update(metrics)
            updated_rows.append(out_row)

        # num_samples 사용 시 나머지 행은 그대로 유지
        if num_samples and num_samples > 0 and len(all_rows) > len(updated_rows):
            updated_rows.extend(all_rows[len(updated_rows) :])

        with open(pred_path, "w", encoding="utf-8") as wf:
            for r in updated_rows:
                wf.write(json.dumps(r, ensure_ascii=False) + "\n")

        acc = correct / max(total, 1)
        task_keys = TASK_METRIC_KEYS.get(task, list(metric_sums.keys()))
        metric_means = {}
        for k in task_keys:
            if k in metric_sums:
                metric_means[k] = metric_sums[k] / max(total, 1)
            else:
                metric_means[k] = None
        summary = {
            "task": task,
            "variant": variant,
            "model": model,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "metrics_mean": metric_means,
        }
        summary_path = task_out_dir / _summary_filename(model, variant, task)
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, ensure_ascii=False, indent=2)

        print(f"\n=== {model} (task={task}, variant={variant}) ===")
        print(f"total={total}, correct={correct}, acc={acc:.4f}")
        print(f"predictions (updated) -> {pred_path}")
        print(f"summary (updated) -> {summary_path}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Re-run evaluation on existing prediction files (no LLM inference)."
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory containing task subdirs (task1_safe_to_nontoxic, etc.). Default: script dir.",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="1",
        help="Task: 1, 2, 3, 4, or all.",
    )
    ap.add_argument(
        "--variant",
        type=str,
        choices=VARIANT_CHOICES + ["all"],
        default="base",
        help="Variant: base, icl1, icl2, icl4, or all.",
    )
    ap.add_argument(
        "--step",
        type=str,
        choices=["single_step", "multi_step", "all"],
        default="single_step",
        help="Step: single_step, multi_step, or all (Task 2는 step 미적용).",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model name (overrides --models).",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="gpt-5,gpt-5-mini,gpt-4o-mini",
        help="Comma-separated model names to re-eval.",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of samples to re-eval (0 = all).",
    )
    args = ap.parse_args()

    if args.model:
        models = [args.model.strip()]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    tasks = [1, 2, 3, 4] if args.task == "all" else [int(args.task)]
    variants = ["base", "icl1", "icl2", "icl4"] if args.variant == "all" else [args.variant]
    steps = ["single_step", "multi_step"] if args.step == "all" else [args.step]

    for task in tasks:
        if task == 2:
            run_reeval(
                out_dir=Path(args.out_dir),
                task=task,
                variant=variants[0],
                models=models,
                num_samples=args.num_samples,
                step="single_step",
            )
            continue
        for variant in variants:
            for step in steps:
                run_reeval(
                    out_dir=Path(args.out_dir),
                    task=task,
                    variant=variant,
                    models=models,
                    num_samples=args.num_samples,
                    step=step,
                )


if __name__ == "__main__":
    main()
