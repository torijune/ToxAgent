#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
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
        task1_toxic_safe_to_nontoxic_safe_eval,
        task2_smiles_to_safe_eval,
        task3_toxic_fragment_identification_eval,
        task4_safe_to_nontoxic_smiles_eval,
        TASK_METRIC_KEYS,
    )
except ImportError:
    task1_toxic_safe_to_nontoxic_safe_eval = None
    task2_smiles_to_safe_eval = None
    task3_toxic_fragment_identification_eval = None
    task4_safe_to_nontoxic_smiles_eval = None
    TASK_METRIC_KEYS = {}

DEFAULT_QA_DIR = _QA_DIR
DEFAULT_DATA_PATH = _QA_DIR / "task1_safe_to_nontoxic" / "single_step" / "task1_safe_qa.jsonl"
DEFAULT_ENV_PATH = _PROJECT_ROOT / ".env"


def _data_path_for(task: int, variant: str, step: str) -> Path:
    qa = DEFAULT_QA_DIR
    if task == 2:
        return qa / "task2_smiles_to_safe" / "task2_safe_qa.jsonl"
    if task == 1:
        base = qa / "task1_safe_to_nontoxic" / step
        fname = "task1_safe_qa.jsonl" if variant == "base" else f"task1_safe_qa_{variant}.jsonl"
        return base / fname
    if task == 3:
        base = qa / "task3_toxic_fragment_identification" / step
        fname = "task3_safe_qa.jsonl" if variant == "base" else f"task3_safe_qa_{variant}.jsonl"
        return base / fname
    base = qa / "task4_safe_to_nontoxic_smiles" / step
    fname = "task4_safe_qa.jsonl" if variant == "base" else f"task4_safe_qa_{variant}.jsonl"
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


def _system_instruction_for_task(task: int) -> str:
    base = (
        "You are a strict evaluator for SAFE QA.\n"
        "Return ONLY a JSON object matching the schema: {\"answer\": \"...\"}\n"
        "No extra keys, no prose, no markdown.\n"
    )
    if task == 1:
        return base + "The value must be the only_nontoxic_safe_fragments string exactly (dot-separated if multiple).\n"
    if task == 2:
        return base + "The value must be the SAFE representation string exactly (dot-separated if multiple).\n"
    if task == 3:
        return base + "The value must be the only_toxic_safe_fragments string exactly (dot-separated if multiple).\n"
    if task == 4:
        return base + "The value must be the single nontoxic molecule SMILES string (nontoxic_safe_decoded_smiles).\n"
    return base


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
) -> Tuple[Optional[str], str]:
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
                        "json_schema": JSON_SCHEMA,
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
                return str(obj["answer"]).strip(), raw

            return (raw.strip() if raw else None), raw

        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))

    return None, f"ERROR: {last_err}"


def normalize_answer(ans: Optional[str]) -> str:
    return (ans or "").strip()


def _get_metrics_for_task(
    task: int,
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int],
) -> Dict[str, Any]:
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


def run_eval(
    data_path: str | Path,
    models: List[str],
    num_samples: int,
    out_dir: str | Path,
    sleep_s: float,
    variant: str = "base",
    task: int = 1,
    step: str = "single_step",
    run_idx: Optional[int] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    print(f"Task: {task} | Variant: {variant} | Step: {step} | Data: {data_path} | Samples: {num_samples or 'all'}")

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

        if task == 1:
            task_subdir_name = "task1_safe_to_nontoxic"
        elif task == 2:
            task_subdir_name = "task2_smiles_to_safe"
        elif task == 3:
            task_subdir_name = "task3_toxic_fragment_identification"
        else:
            task_subdir_name = "task4_safe_to_nontoxic_smiles"

        if task == 2:
            task_out_dir = out_dir / task_subdir_name
        else:
            task_out_dir = out_dir / task_subdir_name / step
        task_out_dir.mkdir(parents=True, exist_ok=True)

        task_out_path = task_out_dir / out_name_template.format(model=safe_model)

        correct = 0
        total = 0
        metric_sums: Dict[str, float] = {}

        with open(task_out_path, "w", encoding="utf-8") as wf:
            for row in tqdm(rows, desc=f"[{model}] {variant}", total=len(rows)):
                q = extract_question(row)
                gold = extract_gold(row)

                pred, raw = call_model(
                    client=client,
                    model=model,
                    question=q,
                    system_instruction=system_instruction,
                    max_retries=3,
                    sleep_s=sleep_s,
                )

                pred_norm = normalize_answer(pred)
                gold_norm = normalize_answer(gold)
                is_correct = int(pred_norm == gold_norm)

                gold_answer = row.get("answer", gold)
                llm_answer = pred if isinstance(pred, dict) else {"answer": pred or ""}
                row_id = row.get("id", None)
                metrics = _get_metrics_for_task(task, gold_answer, llm_answer, row_id=row_id)

                correct += is_correct
                total += 1

                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

                out_row = {
                    "model": model,
                    "id": row.get("id", None),
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                    "raw": raw,
                }
                out_row.update(metrics)
                wf.write(json.dumps(out_row, ensure_ascii=False) + "\n")

                if sleep_s > 0:
                    time.sleep(sleep_s)

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
            "run": run_idx,
            "model": model,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "metrics_mean": metric_means,
        }

        summary_name_parts = [f"summary_{safe_model}"]
        if model_suffix:
            summary_name_parts.append(model_suffix)
        summary_name_parts.append(f"task{task}")
        summary_name = "_".join(summary_name_parts) + ".json"

        summary_path = task_out_dir / summary_name
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, ensure_ascii=False, indent=2)

        print(f"\n=== {model} ===")
        print(f"total={total}, correct={correct}, acc={acc:.4f}")
        print(f"predictions -> {task_out_path}")
        print(f"summary -> {summary_path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default=str(DEFAULT_ENV_PATH))
    ap.add_argument("--data", type=str, default=None, help="path to QA jsonl (overrides task/variant/step)")
    ap.add_argument(
        "--task",
        type=str,
        choices=["1", "2", "3", "4", "all"],
        default="1",
        help="Task: 1, 2, 3, 4, or all",
    )
    ap.add_argument(
        "--variant",
        type=str,
        choices=["base", "icl1", "icl2", "icl4", "all"],
        default="base",
        help="QA variant: base, icl1/icl2/icl4, all",
    )
    ap.add_argument(
        "--step",
        type=str,
        choices=["single_step", "multi_step", "all"],
        default="single_step",
        help="Step: single_step, multi_step, or all (Task 2 ignores step)",
    )
    ap.add_argument("--model", type=str, default=None, help="single model name (overrides --models)")
    ap.add_argument("--models", type=str, default="gpt-5,gpt-5-mini,gpt-4o-mini", help="comma-separated model names")
    ap.add_argument("--num_samples", type=int, default=0, help="number of samples (0 = all)")
    ap.add_argument("--out_dir", type=str, default="./safe_qa_outputs")
    ap.add_argument("--sleep_s", type=float, default=0.2)
    ap.add_argument(
        "--run",
        type=int,
        default=None,
        help="Optional run index appended to output filenames, e.g. --run 1 -> *_run1.jsonl/json",
    )
    args = ap.parse_args()

    load_dotenv(args.env, override=True)

    if args.model:
        models = [args.model.strip()]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        p = str(data_path)
        if "task2_smiles_to_safe" in p:
            task = 2
        elif "task3" in p:
            task = 3
        elif "task4" in p:
            task = 4
        else:
            task = 1

        run_eval(
            data_path=data_path,
            models=models,
            num_samples=args.num_samples,
            out_dir=args.out_dir,
            sleep_s=args.sleep_s,
            variant=args.variant if args.variant != "all" else "base",
            task=task,
            step=args.step if args.step != "all" else "single_step",
            run_idx=args.run,
        )
        return

    tasks = [1, 2, 3, 4] if args.task == "all" else [int(args.task)]
    variants = ["base", "icl1", "icl2", "icl4"] if args.variant == "all" else [args.variant]
    steps = ["single_step", "multi_step"] if args.step == "all" else [args.step]

    runs: List[Tuple[Path, int, str, str]] = []
    for task in tasks:
        if task == 2:
            path = _data_path_for(task, variants[0], "single_step")
            if path.exists():
                runs.append((path, task, variants[0], ""))
            continue

        for variant in variants:
            for step in steps:
                path = _data_path_for(task, variant, step)
                if path.exists():
                    runs.append((path, task, variant, step))
                else:
                    print(f"Skip (not found): {path}")

    if not runs:
        raise FileNotFoundError("No QA data files found for the given --task/--variant/--step.")

    for data_path, task, variant, step in runs:
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
        )


if __name__ == "__main__":
    main()