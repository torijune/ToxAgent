#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFE QA에 대해 GPT inference 및 evaluation 수행.
build_safe_qa.py와 동일한 인자(--split, --task, --variant, --molecule_repr, --step)로 동일한 QA 데이터 경로 사용.

출력 디렉터리 구조 (build_qa와 동일한 트리):
  out_dir / <split> / <task> / [<molecule_repr>] / <step> /
    results/          <- 샘플별 결과 (predictions_<model>.jsonl)
    evaluation/       <- 총 evaluation 요약 (evaluation_summary_<model>.json)
"""

import os
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# QA 디렉터리 (LLMs의 상위)
_LLM_DIR = Path(__file__).resolve().parent
_QA_DIR = _LLM_DIR.parent
_QA_SRC = _QA_DIR / "src"
_PROJECT_ROOT = _QA_DIR.parent.parent

# eval_metric import (QA/src)
import sys
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))
try:
    from eval_metric import (
        task1_toxic_fragment_identification_eval,
        task2_nontoxic_fragment_generation_eval,
        task3_nontoxic_smiles_generation_eval,
        task3_stepwise_cot_nontoxic_smiles_generation_eval,
        subtask1_safe_to_smiles_eval,
        subtask2_smiles_to_safe_eval,
        TASK_METRIC_KEYS,
    )
except ImportError:
    task1_toxic_fragment_identification_eval = None
    task2_nontoxic_fragment_generation_eval = None
    task3_nontoxic_smiles_generation_eval = None
    task3_stepwise_cot_nontoxic_smiles_generation_eval = None
    subtask1_safe_to_smiles_eval = None
    subtask2_smiles_to_safe_eval = None
    TASK_METRIC_KEYS = {}

DEFAULT_QA_DIR = _QA_DIR
DEFAULT_DATA_PATH = _QA_DIR / "test" / "task1_toxic_fragment_identification" / "both_repre" / "single_step" / "task1_toxic_fragment_identification_qa.jsonl"
DEFAULT_ENV_PATH = _PROJECT_ROOT / ".env"

REPRE_CHOICES = ["only_safe", "only_smiles", "both_repre"]
REPRE_CHOICES_WITH_ALL = REPRE_CHOICES + ["all"]


def _normalize_step(step: str) -> str:
    if step in ("single", "single_step"):
        return "single_step"
    if step in ("multi", "multi_step"):
        return "multi_step"
    return step


def _data_path_for(
    task: str,
    variant: str,
    step: str,
    split: str = "test",
    repres: str = "both_repre",
) -> Path:
    qa_base = _QA_DIR / split
    if task == "subtask1":
        return qa_base / "subtask1_safe_to_smiles" / "subtask1_safe_to_smiles_qa.jsonl"
    if task == "subtask2":
        return qa_base / "subtask2_smiles_to_safe" / "subtask2_smiles_to_safe_qa.jsonl"
    step_norm = _normalize_step(step)
    if task == "task1":
        base = qa_base / "task1_toxic_fragment_identification" / repres / step_norm
        fname = "task1_toxic_fragment_identification_qa.jsonl" if variant == "base" else f"task1_toxic_fragment_identification_qa_{variant}.jsonl"
        return base / fname
    if task == "task2":
        base = qa_base / "task2_nontoxic_fragment_generation" / repres / step_norm
        fname = "task2_nontoxic_fragment_generation_qa.jsonl" if variant == "base" else f"task2_nontoxic_fragment_generation_qa_{variant}.jsonl"
        return base / fname
    # task3
    if task == "task3_instruction":
        base = qa_base / "task3_instruction_nontoxic_smiles_generation" / repres / step_norm
        # NOTE: build_safe_qa 쪽 산출물 파일명이 task3_CoT_* 로 되어 있음 (historical naming)
        fname = "task3_CoT_nontoxic_smiles_generation_qa.jsonl"
        return base / fname
    if task == "task3_stepwise_cot":
        base = qa_base / "task3_stepwise_cot_nontoxic_smiles_generation" / repres / step_norm
        fname = "task3_stepwise_cot_nontoxic_smiles_generation_qa.jsonl"
        return base / fname
    base = qa_base / "task3_nontoxic_smiles_generation" / repres / step_norm
    fname = "task3_nontoxic_smiles_generation_qa.jsonl" if variant == "base" else f"task3_nontoxic_smiles_generation_qa_{variant}.jsonl"
    return base / fname


JSON_SCHEMA = {
    "name": "only_nontoxic_safe_fragments",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}

JSON_SCHEMA_STEPWISE_COT = {
    "name": "task3_stepwise_cot_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "step1_only_toxic_safe_fragments": {"type": "string"},
            "step1_reasoning": {"type": "string"},
            "step2_only_nontoxic_safe_fragments": {"type": "string"},
            "step2_reasoning": {"type": "string"},
            "step3_reasoning": {"type": "string"},
        },
        # OpenAI strict schema requires `required` to include every key in `properties`.
        "required": [
            "answer",
            "step1_only_toxic_safe_fragments",
            "step1_reasoning",
            "step2_only_nontoxic_safe_fragments",
            "step2_reasoning",
            "step3_reasoning",
        ],
        # OpenAI json_schema response_format requires additionalProperties to be present and false at the root.
        "additionalProperties": False,
    },
}


def _system_instruction_for_task(task: str) -> str:
    base = (
        "You are a strict evaluator for SAFE QA.\n"
        "Return ONLY a JSON object matching the schema: {\"answer\": \"...\"}\n"
        "No extra keys, no prose, no markdown.\n"
    )
    if task == "task1":
        return base + "The value must be the only_toxic_safe_fragments string exactly (dot-separated if multiple).\n"
    if task == "task2":
        return base + "The value must be the only_nontoxic_safe_fragments string exactly (dot-separated if multiple).\n"
    if task == "task3" or task == "task3_instruction":
        return base + "The value must be the single nontoxic molecule SMILES string (nontoxic_safe_decoded_smiles).\n"
    if task == "task3_stepwise_cot":
        return (
            "You are a strict evaluator for SAFE QA.\n"
            "Return ONLY a single JSON object.\n"
            "No extra text, no markdown.\n"
            "The JSON must include key \"answer\" with value the final non-toxic SMILES string.\n"
            "Also include step1/step2 fragment fields and reasoning fields as instructed by the prompt.\n"
        )
    if task == "subtask1":
        return base + "The value must be the SMILES string of the molecule reconstructed from the SAFE representation.\n"
    if task == "subtask2":
        return base + "The value must be the SAFE representation string exactly (dot-separated if multiple).\n"
    return base


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_done_ids(predictions_path: Path) -> set:
    """이미 저장된 예측 파일에서 row id 집합 반환 (이어하기용)."""
    done: set = set()
    if not predictions_path.exists():
        return done
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                i = obj.get("id")
                # If a previous run recorded an API/schema error, allow re-try by NOT marking it done.
                raw = str(obj.get("raw", "") or "")
                pred = obj.get("pred", None)
                is_error = raw.startswith("ERROR:")
                is_empty_pred = pred is None or pred == "" or pred == {}
                if i is not None and (not is_error) and (not is_empty_pred):
                    done.add(i)
            except (json.JSONDecodeError, TypeError):
                continue
    return done


def extract_gold(row: Dict[str, Any]) -> str:
    a = row.get("answer", "")
    if isinstance(a, dict):
        return str(a.get("answer", "")).strip()
    return str(a).strip()


def extract_question(row: Dict[str, Any]) -> str:
    return str(row.get("question", ""))


def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(text[l:r + 1])
            except json.JSONDecodeError:
                return None
        return None


def call_model(
    client: OpenAI,
    model: str,
    question: str,
    system_instruction: str,
    max_retries: int = 3,
    sleep_s: float = 0.5,
    json_schema: Optional[dict] = None,
) -> Tuple[Optional[Any], str]:
    last_err = None
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": question},
    ]
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": (json_schema or JSON_SCHEMA),
                    },
                )
            except TypeError as te:
                if "response_format" in str(te):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                else:
                    raise

            raw = (resp.choices[0].message.content if resp.choices else "") or ""
            obj = parse_model_json(raw)
            if obj and "answer" in obj:
                return obj, raw

            return (raw.strip() if raw else None), raw

        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))

    return None, f"ERROR: {last_err}"


def _call_model_for_row(
    client: OpenAI,
    model: str,
    row: dict,
    system_instruction: str,
    max_retries: int,
    sleep_s: float,
    json_schema: Optional[dict] = None,
) -> Tuple[dict, Optional[Any], str]:
    """한 행에 대해 call_model 호출. (row, pred, raw) 반환. 배치 병렬용."""
    q = extract_question(row)
    pred, raw = call_model(
        client=client,
        model=model,
        question=q,
        system_instruction=system_instruction,
        max_retries=max_retries,
        sleep_s=sleep_s,
        json_schema=json_schema,
    )
    return (row, pred, raw)


def normalize_answer(ans: Any) -> str:
    if isinstance(ans, dict):
        return str(ans.get("answer", "") or "").strip()
    return str(ans or "").strip()


def _get_metrics_for_task(
    task: str,
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int],
) -> Dict[str, Any]:
    if task == "task1" and task1_toxic_fragment_identification_eval is not None:
        (
            fragment_EM,
            fragment_BLEU1,
            fragment_Precision,
            fragment_Recall,
            fragment_F1,
        ) = task1_toxic_fragment_identification_eval(gold_answer, llm_answer)
        return {
            "fragment_EM": fragment_EM,
            "fragment_BLEU1": fragment_BLEU1,
            "fragment_Precision": fragment_Precision,
            "fragment_Recall": fragment_Recall,
            "fragment_F1": fragment_F1,
        }

    if task == "task2" and task2_nontoxic_fragment_generation_eval is not None:
        (
            fragment_EM,
            fragment_BLEU1,
            fragment_Precision,
            fragment_Recall,
            fragment_F1,
            molecule_EM,
            molecule_morganFT,
            molecule_validity,
        ) = task2_nontoxic_fragment_generation_eval(gold_answer, llm_answer, row_id=row_id)
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

    if task in ("task3", "task3_instruction") and task3_nontoxic_smiles_generation_eval is not None:
        (
            exact_match,
            bleu,
            levenshtein,
            rdk_fts,
            maccs_fts,
            morgan_fts,
            validity,
        ) = task3_nontoxic_smiles_generation_eval(gold_answer, llm_answer)
        return {
            "exact_match": exact_match,
            "bleu": bleu,
            "levenshtein": levenshtein,
            "rdk_fts": rdk_fts,
            "maccs_fts": maccs_fts,
            "morgan_fts": morgan_fts,
            "validity": validity,
        }

    if task == "task3_stepwise_cot" and task3_stepwise_cot_nontoxic_smiles_generation_eval is not None:
        return task3_stepwise_cot_nontoxic_smiles_generation_eval(
            gold_answer=gold_answer,
            llm_answer=llm_answer,
            row_id=row_id,
        )

    if task == "subtask1" and subtask1_safe_to_smiles_eval is not None:
        (
            exact_match,
            bleu,
            levenshtein,
            rdk_fts,
            maccs_fts,
            morgan_fts,
            validity,
        ) = subtask1_safe_to_smiles_eval(gold_answer, llm_answer)
        return {
            "exact_match": exact_match,
            "bleu": bleu,
            "levenshtein": levenshtein,
            "rdk_fts": rdk_fts,
            "maccs_fts": maccs_fts,
            "morgan_fts": morgan_fts,
            "validity": validity,
        }

    if task == "subtask2" and subtask2_smiles_to_safe_eval is not None:
        (
            EM,
            BLEU1,
            validity,
            lev_dist,
            lev_norm,
            molecule_EM,
            molecule_morganFT,
            molecule_validity,
        ) = subtask2_smiles_to_safe_eval(gold_answer, llm_answer, row_id=row_id)
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

    return {}


def run_eval(
    data_path: str | Path,
    models: List[str],
    num_samples: int,
    out_dir: str | Path,
    sleep_s: float,
    variant: str = "base",
    task: str = "task1",
    step: str = "single_step",
    run_idx: Optional[int] = None,
    split: str = "test",
    repres: str = "both_repre",
    batch_size: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    step_norm = _normalize_step(step)
    print(f"Task: {task} | Variant: {variant} | Step: {step_norm} | Split: {split} | Repre: {repres} | Data: {data_path} | Samples: {num_samples or 'all'} | batch_size: {batch_size}")

    rows = read_jsonl(str(data_path))
    if num_samples and num_samples > 0:
        rows = rows[:num_samples]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. .env에 OPENAI_API_KEY=...를 넣어주세요.")

    client = OpenAI(api_key=api_key)
    system_instruction = _system_instruction_for_task(task)

    name_parts = []
    if variant != "base":
        name_parts.append(variant)
    if run_idx is not None:
        name_parts.append(f"run{run_idx}")

    model_suffix = "_".join(name_parts)
    out_name_template = (
        f"predictions_{{model}}.jsonl"
        if not model_suffix
        else f"predictions_{{model}}_{model_suffix}.jsonl"
    )

    for model in models:
        safe_model = model.replace("/", "_")

        # build_safe_qa와 동일한 디렉터리 구조: out_dir / split / task / [repre] / [step]
        task_out_dir = out_dir / split / task
        if task not in ("subtask1", "subtask2"):
            task_out_dir = task_out_dir / repres / step_norm
        task_out_dir.mkdir(parents=True, exist_ok=True)
        results_dir = task_out_dir / "results"
        evaluation_dir = task_out_dir / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        # 샘플별 결과: results/predictions_<model>.jsonl
        task_out_path = results_dir / out_name_template.format(model=safe_model)

        # 이어하기: 이미 저장된 id는 건너뛰기
        done_ids = _load_done_ids(task_out_path)
        rows_to_do = [r for r in rows if r.get("id") not in done_ids]
        if done_ids:
            print(f"  이어하기: {len(done_ids)}개 이미 완료, {len(rows_to_do)}개 남음")

        mode = "a" if task_out_path.exists() and done_ids else "w"
        with open(task_out_path, mode, encoding="utf-8") as wf:
            for batch_start in tqdm(range(0, len(rows_to_do), batch_size), desc=f"[{model}] {variant}", total=(len(rows_to_do) + batch_size - 1) // max(batch_size, 1)):
                batch = rows_to_do[batch_start : batch_start + batch_size]
                # 완료되는 대로 순서 유지하며 즉시 저장
                results_by_idx: List[Optional[Tuple[dict, Optional[str], str]]] = [None] * len(batch)
                next_to_write = 0
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    future_to_idx = {
                        executor.submit(
                            _call_model_for_row,
                            client,
                            model,
                            row,
                            system_instruction,
                            3,
                            sleep_s,
                            JSON_SCHEMA_STEPWISE_COT if task == "task3_stepwise_cot" else None,
                        ): i
                        for i, row in enumerate(batch)
                    }
                    for future in as_completed(future_to_idx):
                        i = future_to_idx[future]
                        row, pred, raw = future.result()
                        results_by_idx[i] = (row, pred, raw)
                        while next_to_write < len(batch) and results_by_idx[next_to_write] is not None:
                            row, pred, raw = results_by_idx[next_to_write]
                            gold = extract_gold(row)
                            pred_norm = normalize_answer(pred)
                            gold_norm = normalize_answer(gold)
                            is_correct = int(pred_norm == gold_norm)
                            gold_answer = row.get("answer", gold)
                            llm_answer = pred if isinstance(pred, dict) else {"answer": pred or ""}
                            row_id = row.get("source_index", row.get("id", None))
                            metrics = _get_metrics_for_task(task, gold_answer, llm_answer, row_id=row_id)
                            out_row = {
                                "model": model,
                                "id": row.get("id", None),
                                "dataset_name": row.get("dataset_name", ""),
                                "endpoint": row.get("endpoint", ""),
                                "source_index": row.get("source_index", None),
                                "gold": gold,
                                "pred": pred,
                                "correct": is_correct,
                                "raw": raw,
                            }
                            out_row.update(metrics)
                            wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                            wf.flush()
                            next_to_write += 1
                if sleep_s > 0:
                    time.sleep(sleep_s)

        # 전체 파일 기준으로 요약 재계산 (이어하기 포함)
        all_lines = read_jsonl(str(task_out_path))
        correct = sum(int(line.get("correct", 0)) for line in all_lines)
        total = len(all_lines)
        metric_sums: Dict[str, float] = {}
        task_keys = TASK_METRIC_KEYS.get(task, [])
        for line in all_lines:
            for k in task_keys:
                v = line.get(k)
                if isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
        acc = correct / max(total, 1)
        metric_means = {}
        for k in task_keys:
            if k in metric_sums:
                metric_means[k] = metric_sums[k] / max(total, 1)
            else:
                metric_means[k] = None

        # 총 evaluation 결과: evaluation/evaluation_summary_<model>.json
        summary = {
            "task": task,
            "variant": variant,
            "step": step_norm,
            "split": split,
            "repre": repres,
            "run": run_idx,
            "model": model,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "metrics_mean": metric_means,
        }

        summary_name_parts = [f"evaluation_summary_{safe_model}"]
        if model_suffix:
            summary_name_parts.append(model_suffix)
        summary_name = "_".join(summary_name_parts) + ".json"
        summary_path = evaluation_dir / summary_name
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, ensure_ascii=False, indent=2)

        print(f"\n=== {model} ===")
        print(f"total={total}, correct={correct}, acc={acc:.4f}")
        print(f"  results (per-sample) -> {task_out_path}")
        print(f"  evaluation (summary) -> {summary_path}\n")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run GPT inference & evaluation on SAFE QA. "
            "Arguments aligned with build_safe_qa.py (--split, --task, --variant, --molecule_repr, --step) so the same QA dataset is used."
        ),
    )
    ap.add_argument("--env", type=str, default=str(DEFAULT_ENV_PATH), help="Path to .env for OPENAI_API_KEY")
    ap.add_argument("--data", type=str, default=None, help="Path to QA jsonl (overrides --split/--task/--variant/--molecule_repr/--step)")
    ap.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Split: train or test. QA/<split>/... 와 동일. 기본: test (build_safe_qa와 동일)",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=["task1", "task2", "task3", "task3_instruction", "task3_stepwise_cot", "subtask1", "subtask2", "all"],
        default="task1",
        help="Task: task1, task2, task3, task3_instruction, task3_stepwise_cot, subtask1, subtask2, all (build_safe_qa와 동일). 기본: task1",
    )
    ap.add_argument(
        "--variant",
        type=str,
        choices=["base", "icl1", "icl2", "icl4", "all"],
        default="base",
        help="QA variant: base, icl1, icl2, icl4, all (build_safe_qa와 동일). 기본: base",
    )
    ap.add_argument(
        "--molecule_repr",
        type=str,
        dest="repre",
        choices=REPRE_CHOICES_WITH_ALL,
        default="both_repre",
        help="Molecule representation: only_safe, only_smiles, both_repre, all. all이면 모든 representation을 자동으로 inference하고 각 경로에 저장. 기본: both_repre",
    )
    ap.add_argument(
        "--step",
        type=str,
        choices=["single", "multi", "all", "single_step", "multi_step"],
        default="single",
        help="Step: single, multi, all (task1/2/3만). 기본: single",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="단일 모델명 (e.g. gpt-4o, gpt-5). 지정 시 --models 무시.",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="gpt-4o-mini,gpt-4o",
        help="쉼표 구분 모델명. 기본: gpt-4o-mini,gpt-4o",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="상위 N개 샘플만 inference (0=전체). 기본: 0",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./safe_qa_outputs",
        help="출력 루트. 구조: out_dir/<split>/<task>/[<molecule_repr>/]<step>/results/ 및 evaluation/",
    )
    ap.add_argument("--sleep_s", type=float, default=0.2, help="API 호출 간 sleep(초)")
    ap.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="한 번에 병렬로 호출할 샘플 수 (배치 크기). 기본: 10",
    )
    ap.add_argument(
        "--run",
        type=int,
        default=None,
        help="실험 run 인덱스 (파일명에 run<N> 추가)",
    )
    args = ap.parse_args()

    load_dotenv(args.env, override=True)

    if args.model:
        models = [args.model.strip()]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    step_choices = ["single_step", "multi_step"]
    if args.step in ("single", "single_step"):
        steps = ["single_step"]
    elif args.step in ("multi", "multi_step"):
        steps = ["multi_step"]
    else:
        steps = step_choices

    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        p = str(data_path)
        if "subtask1" in p:
            task = "subtask1"
        elif "subtask2" in p:
            task = "subtask2"
        elif "task1" in p and "task3" not in p:
            task = "task1"
        elif "task2" in p:
            task = "task2"
        elif "task3_instruction" in p or "task3_Instruction" in p:
            task = "task3_instruction"
        elif "task3" in p:
            task = "task3"
        else:
            task = "task1"

        step_for_eval = "single_step"
        if task not in ("subtask1", "subtask2") and "multi_step" in p:
            step_for_eval = "multi_step"
        elif task not in ("subtask1", "subtask2") and args.step in ("multi", "multi_step"):
            step_for_eval = "multi_step"

        run_eval(
            data_path=data_path,
            models=models,
            num_samples=args.num_samples,
            out_dir=args.out_dir,
            sleep_s=args.sleep_s,
            variant=args.variant if args.variant != "all" else "base",
            task=task,
            step=step_for_eval,
            run_idx=args.run,
            split=args.split,
            repres=args.repre,
            batch_size=args.batch_size,
        )
        return

    _ALL_TASKS = ["task1", "task2", "task3", "task3_instruction", "task3_stepwise_cot", "subtask1", "subtask2"]
    tasks = _ALL_TASKS if args.task == "all" else [args.task]
    variants = ["base", "icl1", "icl2", "icl4"] if args.variant == "all" else [args.variant]
    split = args.split
    repres_list = REPRE_CHOICES if args.repre == "all" else [args.repre]

    runs: List[Tuple[Path, str, str, str, str]] = []  # (data_path, task, variant, step, repres)
    for repres in repres_list:
        for task in tasks:
            # subtask1/2는 data_path 및 출력 디렉터리 구조에 repres가 없음 → 중복 실행 방지
            if task in ("subtask1", "subtask2"):
                if repres != repres_list[0]:
                    continue
                path = _data_path_for(task, variants[0], "single_step", split=split, repres=repres)
                if path.exists():
                    runs.append((path, task, variants[0], "", repres))
                else:
                    print(f"Skip (not found): {path}")
                continue

            for variant in variants:
                for step in steps:
                    path = _data_path_for(task, variant, step, split=split, repres=repres)
                    if path.exists():
                        runs.append((path, task, variant, step, repres))
                    else:
                        print(f"Skip (not found): {path}")

    if not runs:
        raise FileNotFoundError("No QA data files found for the given --split/--task/--variant/--repre/--step.")

    for data_path, task, variant, step, repres in runs:
        run_eval(
            data_path=data_path,
            models=models,
            num_samples=args.num_samples,
            out_dir=args.out_dir,
            sleep_s=args.sleep_s,
            variant=variant,
            task=task,
            step=step or "single_step",
            run_idx=args.run,
            split=split,
            repres=repres,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()