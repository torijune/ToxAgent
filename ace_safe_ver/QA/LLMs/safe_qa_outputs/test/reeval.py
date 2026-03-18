#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수정된 메트릭(예: task2의 merged_test.csv + source_index 기반 분자 평가)으로
기존 predictions_*.jsonl을 다시 평가해 evaluation_summary_*.json을 갱신한다.
LLM 호출 없이 예측 결과만 읽어서 메트릭만 재계산한다.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# QA/src에서 eval_metric import (reeval.py 위치: .../safe_qa_outputs/test/reeval.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
_QA_DIR = _SCRIPT_DIR.parent.parent.parent  # test -> safe_qa_outputs -> LLMs -> QA
_QA_SRC = _QA_DIR / "src"
assert _QA_SRC.exists(), f"QA src 없음: {_QA_SRC}"
import sys
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))

try:
    from eval_metric import (
        task1_toxic_fragment_identification_eval,
        task2_nontoxic_fragment_generation_eval,
        task3_nontoxic_smiles_generation_eval,
        subtask1_safe_to_smiles_eval,
        subtask2_smiles_to_safe_eval,
        TASK_METRIC_KEYS,
    )
except ImportError as e:
    raise RuntimeError(f"eval_metric import 실패 (QA/src 경로 확인): {e}") from e


def _get_metrics_for_task(
    task: str,
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int],
) -> Dict[str, Any]:
    """inference_gpt와 동일한 메트릭 계산."""
    if task == "task1" and task1_toxic_fragment_identification_eval is not None:
        t = task1_toxic_fragment_identification_eval(gold_answer, llm_answer)
        return dict(zip(["fragment_EM", "fragment_BLEU1", "fragment_Precision", "fragment_Recall", "fragment_F1"], t))
    if task == "task2" and task2_nontoxic_fragment_generation_eval is not None:
        t = task2_nontoxic_fragment_generation_eval(gold_answer, llm_answer, row_id=row_id)
        return dict(zip([
            "fragment_EM", "fragment_BLEU1", "fragment_Precision", "fragment_Recall", "fragment_F1",
            "molecule_EM", "molecule_morganFT", "molecule_validity",
        ], t))
    if task == "task3" and task3_nontoxic_smiles_generation_eval is not None:
        t = task3_nontoxic_smiles_generation_eval(gold_answer, llm_answer)
        return dict(zip(["exact_match", "bleu", "levenshtein", "rdk_fts", "maccs_fts", "morgan_fts", "validity"], t))
    if task == "subtask1" and subtask1_safe_to_smiles_eval is not None:
        t = subtask1_safe_to_smiles_eval(gold_answer, llm_answer)
        return dict(zip(["exact_match", "bleu", "levenshtein", "rdk_fts", "maccs_fts", "morgan_fts", "validity"], t))
    if task == "subtask2" and subtask2_smiles_to_safe_eval is not None:
        t = subtask2_smiles_to_safe_eval(gold_answer, llm_answer, row_id=row_id)
        return dict(zip([
            "EM", "BLEU1", "validity", "levenshtein_dist", "levenshtein_norm",
            "molecule_EM", "molecule_morganFT", "molecule_validity",
        ], t))
    return {}


def _parse_prediction_path(pred_path: Path, root: Path) -> Optional[Dict[str, str]]:
    """results/predictions_<model>.jsonl 경로에서 task, step, repres, split, model 추출."""
    try:
        rel = pred_path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if "results" not in parts or not pred_path.name.startswith("predictions_") or not pred_path.name.endswith(".jsonl"):
        return None
    # root가 .../test 이면 rel = task2/both_repre/single_step/results/... → split은 root 이름
    idx = parts.index("results")
    name = pred_path.stem  # predictions_gpt-4o
    model = name.replace("predictions_", "", 1) if name.startswith("predictions_") else "unknown"
    split = root.name if root.name in ("train", "test") else "test"
    task = parts[0] if idx >= 1 else ""
    if task in ("subtask1", "subtask2"):
        repres = ""
        step = ""
    else:
        # task1, task2, task3: .../task2/both_repre/single_step/results/...
        if idx >= 4:
            step = parts[idx - 1]
            repres = parts[idx - 2]
        else:
            repres = ""
            step = ""
    return {"task": task, "step": step, "repre": repres, "split": split, "model": model}


def reeval_one(pred_path: Path, root: Path, variant: str = "base", run_idx: Optional[int] = None) -> None:
    """단일 predictions_*.jsonl에 대해 메트릭 재계산 후 evaluation_summary_*.json 저장."""
    meta = _parse_prediction_path(pred_path, root)
    if not meta or not meta["task"]:
        print(f"  skip (경로 파싱 불가): {pred_path}")
        return

    task = meta["task"]
    step = meta["step"] or "single_step"
    repres = meta["repre"] or "both_repre"
    split = meta["split"]
    model = meta["model"]

    task_keys = TASK_METRIC_KEYS.get(task, [])
    metric_sums: Dict[str, float] = {}
    correct = 0
    total = 0

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            gold = row.get("gold", "")
            pred = row.get("pred", "")
            gold_answer = row.get("answer", gold)
            if isinstance(gold_answer, str):
                gold_answer = {"answer": gold_answer}
            llm_answer = {"answer": pred or ""}
            row_id = row.get("source_index", row.get("id"))

            metrics = _get_metrics_for_task(task, gold_answer, llm_answer, row_id=row_id)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                elif v is None and k in task_keys:
                    # None은 합산 제외 (평균 시에도 제외하려면 카운트 따로 관리해야 함)
                    pass

            # correct: fragment/answer 문자열 일치 (task에 따라 gold vs pred)
            gold_norm = (gold or "").strip()
            pred_norm = (pred or "").strip()
            correct += 1 if gold_norm == pred_norm else 0

    acc = correct / max(total, 1)
    metric_means: Dict[str, Optional[float]] = {}
    for k in task_keys:
        if k in metric_sums:
            metric_means[k] = metric_sums[k] / max(total, 1)
        else:
            metric_means[k] = None

    summary = {
        "task": task,
        "variant": variant,
        "step": step,
        "split": split,
        "repre": repres,
        "run": run_idx,
        "model": model,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "metrics_mean": metric_means,
    }

    evaluation_dir = pred_path.parent.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    summary_name = f"evaluation_summary_{safe_model}.json"
    summary_path = evaluation_dir / summary_name
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"  {task} {step} {model}: total={total}, correct={correct}, acc={acc:.4f} -> {summary_path}")


def find_prediction_files(root: Path, task_filter: Optional[str] = None) -> List[Path]:
    """root 아래 results/predictions_*.jsonl 목록. task_filter가 있으면 해당 task만."""
    out: List[Path] = []
    for pred_path in root.rglob("predictions_*.jsonl"):
        if pred_path.parent.name != "results":
            continue
        if task_filter:
            rel = pred_path.resolve().relative_to(root.resolve())
            parts = rel.parts
            # rel = task2/both_repre/single_step/results/... → task는 parts[0]
            if len(parts) >= 1:
                t = parts[0]
                if t != task_filter:
                    continue
        out.append(pred_path)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description="수정된 메트릭으로 predictions 재평가 → evaluation_summary 갱신")
    ap.add_argument(
        "--root",
        type=Path,
        default=_SCRIPT_DIR,
        help="출력 루트 (기본: reeval.py가 있는 디렉터리, 즉 safe_qa_outputs/test)",
    )
    ap.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["task1", "task2", "task3", "subtask1", "subtask2", "all"],
        help="재평가할 task (기본: all = 모든 task/step 재평가)",
    )
    ap.add_argument("--variant", type=str, default="base", help="variant 라벨 (summary에만 기록)")
    args = ap.parse_args()

    root = args.root.resolve()
    task_filter = None if args.task == "all" else args.task
    pred_files = find_prediction_files(root, task_filter=task_filter)
    if not pred_files:
        print(f"예측 파일 없음: {root} (task={args.task})")
        return

    print(f"재평가: root={root}, task={args.task}, 파일 {len(pred_files)}개\n")
    for p in pred_files:
        reeval_one(p, root, variant=args.variant)
    print("\n완료.")


if __name__ == "__main__":
    main()
