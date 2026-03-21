#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수정된 eval_metric으로 기존 predictions_*.jsonl을 다시 평가해 evaluation_summary_*.json을 갱신한다.
LLM 호출 없이 예측 결과만 읽어 메트릭만 재계산한다.

지원 task: task1, task2, task3, task3_instruction, task3_nontoxic_safe_generation,
          task3_stepwise_cot, subtask1, subtask2  (또는 all)

task3_stepwise_cot: 단계별 gold fragment는 QA jsonl에서 id로 조회해 병합한다
(inference_gpt._data_path_for 와 동일한 경로 규칙).

validity_diagnostics (기본 on): task/step/representation/model별로 RDKit validity(및 SAFE 디코딩) 실패 시
이유 문자열(reason_counts)과 stderr 캡처 예시(examples)를 evaluation_summary_*.json 에 포함한다.
끄려면 --no-validity-diagnostics.

safe_decode: eval_metric 의 safe_decode 를 그대로 쓴다. 예전에 reeval 만 별도 import 하면서
sys.path 에 저장소 루트가 없어 safe_decode_unavailable 이 전부 찍히는 문제가 있었음
(저장소 루트 삽입 + eval_metric.safe_decode 재사용으로 정렬).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

# QA/src에서 eval_metric import (reeval.py 위치: .../safe_qa_outputs/test/reeval.py)
_SCRIPT_DIR = Path(__file__).resolve().parent
_QA_DIR = _SCRIPT_DIR.parent.parent.parent  # test -> safe_qa_outputs -> LLMs -> QA
_QA_SRC = _QA_DIR / "src"
# eval_metric.py 와 동일: 저장소 루트를 넣어야 `safe` 패키지(safe.safe.converter) import 가능
_PROJECT_ROOT = _QA_DIR.parent.parent
assert _QA_SRC.exists(), f"QA src 없음: {_QA_SRC}"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))

try:
    from eval_metric import (
        TASK_METRIC_KEYS,
        task1_toxic_fragment_identification_eval,
        task2_nontoxic_fragment_generation_eval,
        task3_nontoxic_smiles_generation_eval,
        task3_nontoxic_safe_generation_eval,
        task3_stepwise_cot_nontoxic_smiles_generation_eval,
        subtask1_safe_to_smiles_eval,
        subtask2_smiles_to_safe_eval,
        _decode_safe_to_smiles,
        _extract_answer,
        _get_merged_test_row_by_id,
        _get_row_smiles_to_safe_by_id,
        _get_step_field,
        _join_safe_fragments,
        safe_decode,  # eval_metric 과 동일 인스턴스 (별도 import 실패로 None 되지 않도록)
    )
except ImportError as e:
    raise RuntimeError(f"eval_metric import 실패 (QA/src 경로 확인): {e}") from e


def _normalize_answer(ans: Any) -> str:
    """inference_gpt.normalize_answer 와 동일."""
    if isinstance(ans, dict):
        return str(ans.get("answer", "") or "").strip()
    return str(ans or "").strip()


def _try_parse_json_object(s: str) -> Optional[Dict[str, Any]]:
    """문자열이 JSON 객체이면 dict로, 아니면 None."""
    s = (s or "").strip()
    if len(s) < 2 or s[0] != "{":
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _llm_answer_from_pred(pred: Any) -> Any:
    """predictions 줄의 pred 필드 → eval용 llm_answer.

    - pred가 str인데 내용이 JSON 객체이면 파싱해 dict로 넘긴다.
      (그렇지 않으면 eval_metric._extract_answer 가 전체 문자열을 SMILES로 취급해
      RDKit SMILES Parse Error: '{' … 가 난다.)
    - pred가 dict인데 'answer' 값만 JSON 문자열로 들어온 경우도 병합한다.
    """
    if isinstance(pred, str):
        parsed = _try_parse_json_object(pred)
        if parsed is not None:
            return parsed
        return {"answer": pred or ""}
    if isinstance(pred, dict):
        a = pred.get("answer")
        if isinstance(a, str):
            inner = _try_parse_json_object(a)
            if inner is not None:
                merged = {**pred, **inner}
                return merged
        return pred
    return {"answer": pred or ""}


def _collapse_stderr(s: str) -> str:
    """RDKit stderr 여러 줄을 한 줄로 합쳐 집계 키로 쓰기 좋게 만든다."""
    return " ".join((s or "").split()).strip()


def _mol_from_smiles_with_capture(smiles: str) -> Tuple[Optional[Chem.Mol], str]:
    """
    MolFromSmiles 호출 시 RDKit이 stderr로 출력하는 메시지를 캡처한다.
    (C++ 로거 일부는 잡히지 않을 수 있으나, 일반적인 SMILES Parse Error는 대부분 잡힌다.)
    """
    s = (smiles or "").strip()
    if not s:
        return None, ""
    buf = io.StringIO()
    with redirect_stderr(buf):
        mol = Chem.MolFromSmiles(s)
    return mol, buf.getvalue()


def _decode_safe_with_reason(safe_str: str) -> Tuple[Optional[str], str]:
    """eval_metric._decode_safe_to_smiles 와 동일 조건 + 실패 이유 문자열."""
    safe_str = (safe_str or "").strip()
    if not safe_str:
        return None, "empty_safe_string"
    if safe_decode is None:
        return None, "safe_decode_unavailable"
    try:
        decoded = safe_decode(safe_str)
    except Exception as e:  # noqa: BLE001 — 디코딩 실패 사유 전달
        return None, f"safe_decode_exception:{type(e).__name__}:{e!s}"
    decoded = (decoded or "").strip()
    if not decoded:
        return None, "safe_decode_empty_result"
    return decoded, ""


def _validity_diag_for_smiles_string(smiles: str) -> Dict[str, Any]:
    """단일 SMILES 문자열에 대한 RDKit validity 진단 (eval_metric validity와 동일 입력)."""
    s = (smiles or "").strip()
    if not s:
        return {
            "applies": True,
            "valid": False,
            "reason": "empty_smiles",
            "rdkit_stderr": "",
        }
    mol, stderr = _mol_from_smiles_with_capture(s)
    if mol is not None:
        return {"applies": True, "valid": True, "reason": "", "rdkit_stderr": ""}
    rk = _collapse_stderr(stderr)
    if not rk:
        rk = "MolFromSmiles_returned_None_no_stderr"
    return {
        "applies": True,
        "valid": False,
        "reason": f"rdkit:{rk}",
        "rdkit_stderr": (stderr or "").strip(),
    }


def _task2_molecule_validity_diag(llm_answer: Any, row_id: Any) -> Optional[Dict[str, Any]]:
    """task2 / stepwise step2 molecule_validity 와 동일 경로."""
    pred = (_extract_answer(llm_answer) or "").strip()
    row = _get_merged_test_row_by_id(row_id)
    if row is None or not pred:
        return None
    common_safe = str(row.get("common_safe_fragments", "") or "").strip()
    pred_full_safe = _join_safe_fragments(common_safe, pred)
    pred_smiles = _decode_safe_to_smiles(pred_full_safe)
    if pred_smiles is None:
        dec_err = ""
        if safe_decode is not None:
            _, dec_err = _decode_safe_with_reason(pred_full_safe)
        else:
            dec_err = "safe_decode_unavailable"
        return {
            "applies": True,
            "valid": False,
            "reason": f"safe_decode_failed:{dec_err}",
            "rdkit_stderr": "",
        }
    return _validity_diag_for_smiles_string(pred_smiles)


def collect_validity_diagnostics_for_sample(
    task: str,
    llm_answer: Any,
    row_id: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    eval_metric 에서 RDKit validity(및 SAFE→SMILES)와 동일한 기준으로 샘플별 진단.
    반환 키: validity, molecule_validity, step2_molecule_validity (해당 task에서만).
    """
    out: Dict[str, Dict[str, Any]] = {}

    if task == "task1":
        return out

    if task == "task2":
        d = _task2_molecule_validity_diag(llm_answer, row_id)
        if d is not None:
            out["molecule_validity"] = d
        return out

    if task in ("task3", "task3_instruction", "subtask1"):
        pred_s = (_extract_answer(llm_answer) or "").strip()
        out["validity"] = _validity_diag_for_smiles_string(pred_s)
        return out

    if task == "task3_nontoxic_safe_generation":
        pred_safe = (_extract_answer(llm_answer) or "").strip()
        if not pred_safe:
            out["validity"] = {
                "applies": True,
                "valid": False,
                "reason": "empty_pred_safe",
                "rdkit_stderr": "",
            }
            return out
        decoded, dec_reason = _decode_safe_with_reason(pred_safe)
        if decoded is None:
            out["validity"] = {
                "applies": True,
                "valid": False,
                "reason": f"safe_decode:{dec_reason}",
                "rdkit_stderr": "",
            }
            return out
        vd = _validity_diag_for_smiles_string(decoded)
        out["validity"] = vd
        return out

    if task == "task3_stepwise_cot":
        pred_smiles = (_extract_answer(llm_answer) or "").strip()
        out["validity"] = _validity_diag_for_smiles_string(pred_smiles)
        pred_step2 = _get_step_field(llm_answer, "step2_only_nontoxic_safe_fragments")
        d2 = _task2_molecule_validity_diag({"answer": pred_step2}, row_id)
        if d2 is not None:
            out["step2_molecule_validity"] = d2
        return out

    if task == "subtask2":
        pred_safe = (_extract_answer(llm_answer) or "").strip()
        if not pred_safe.strip():
            out["validity"] = {
                "applies": True,
                "valid": False,
                "reason": "empty_pred_safe",
                "rdkit_stderr": "",
            }
        else:
            decoded, dec_reason = _decode_safe_with_reason(pred_safe)
            if decoded is None:
                out["validity"] = {
                    "applies": True,
                    "valid": False,
                    "reason": f"safe_decode:{dec_reason}",
                    "rdkit_stderr": "",
                }
            else:
                out["validity"] = _validity_diag_for_smiles_string(decoded)

        row = _get_row_smiles_to_safe_by_id(row_id)
        if row is not None and pred_safe:
            ref_smiles = str(row.get("canonical_smiles", "") or row.get("smiles", "") or "").strip()
            pred_smiles = _decode_safe_to_smiles(pred_safe)
            if not ref_smiles:
                out["molecule_validity"] = {
                    "applies": True,
                    "valid": False,
                    "reason": "missing_reference_smiles_in_csv",
                    "rdkit_stderr": "",
                }
            elif pred_smiles is None:
                _, dec_reason = _decode_safe_with_reason(pred_safe)
                out["molecule_validity"] = {
                    "applies": True,
                    "valid": False,
                    "reason": f"safe_decode:{dec_reason}",
                    "rdkit_stderr": "",
                }
            else:
                out["molecule_validity"] = _validity_diag_for_smiles_string(pred_smiles)
        return out

    return out


def _aggregate_validity_diagnostics(
    acc: Dict[str, Any],
    diag: Dict[str, Dict[str, Any]],
    sample_id: Any,
) -> None:
    """acc에 reason_counts / applies / invalid / examples 누적."""
    for metric_name, info in diag.items():
        if not info.get("applies"):
            continue
        m = acc.setdefault(
            metric_name,
            {
                "applies_rows": 0,
                "invalid_rows": 0,
                "reason_counts": defaultdict(int),
                "examples": [],
            },
        )
        m["applies_rows"] += 1
        if info.get("valid"):
            continue
        m["invalid_rows"] += 1
        reason = str(info.get("reason") or "unknown")
        m["reason_counts"][reason] += 1
        examples: List[Dict[str, Any]] = m["examples"]
        if len(examples) >= 24:
            continue
        examples.append(
            {
                "id": sample_id,
                "reason": reason,
                "rdkit_stderr": (info.get("rdkit_stderr") or "")[:4000],
            },
        )


def _finalize_validity_diagnostics(acc: Dict[str, Any]) -> Dict[str, Any]:
    """defaultdict 등을 JSON 직렬화 가능한 dict로 변환."""
    out: Dict[str, Any] = {}
    for metric_name, m in acc.items():
        rc = m.get("reason_counts") or {}
        if hasattr(rc, "items"):
            rc_sorted = dict(sorted(rc.items(), key=lambda x: (-x[1], x[0])))
        else:
            rc_sorted = dict(rc)
        out[metric_name] = {
            "applies_rows": m.get("applies_rows", 0),
            "invalid_rows": m.get("invalid_rows", 0),
            "reason_counts": rc_sorted,
            "examples": m.get("examples", []),
        }
    return out


def _gold_answer_from_row(
    task: str,
    row: Dict[str, Any],
    qa_by_id: Optional[Dict[int, Dict[str, Any]]],
) -> Any:
    """
    eval에 넘길 gold_answer.
    - task3_stepwise_cot: QA jsonl의 answer dict( gold fragment 포함 ) 우선.
    - 그 외: row['gold'] 문자열 또는 row['answer'].
    """
    if task == "task3_stepwise_cot" and qa_by_id is not None:
        rid = row.get("id")
        if rid is not None and int(rid) in qa_by_id:
            return qa_by_id[int(rid)].get("answer", row.get("gold", ""))
    if row.get("answer") is not None:
        return row["answer"]
    g = row.get("gold", "")
    return {"answer": g} if not isinstance(g, dict) else g


def _qa_jsonl_path(split: str, task: str, repre: str, step: str, variant: str = "base") -> Path:
    """inference_gpt._data_path_for 와 동일한 QA jsonl 경로."""
    qa_base = _QA_DIR / split
    step_norm = step if step in ("single_step", "multi_step") else (
        "single_step" if step in ("single",) else "multi_step"
    )
    if task == "subtask1":
        return qa_base / "subtask1_safe_to_smiles" / "subtask1_safe_to_smiles_qa.jsonl"
    if task == "subtask2":
        return qa_base / "subtask2_smiles_to_safe" / "subtask2_smiles_to_safe_qa.jsonl"
    if task == "task1":
        base = qa_base / "task1_toxic_fragment_identification" / repre / step_norm
        fname = (
            "task1_toxic_fragment_identification_qa.jsonl"
            if variant == "base"
            else f"task1_toxic_fragment_identification_qa_{variant}.jsonl"
        )
        return base / fname
    if task == "task2":
        base = qa_base / "task2_nontoxic_fragment_generation" / repre / step_norm
        fname = (
            "task2_nontoxic_fragment_generation_qa.jsonl"
            if variant == "base"
            else f"task2_nontoxic_fragment_generation_qa_{variant}.jsonl"
        )
        return base / fname
    if task == "task3_instruction":
        base = qa_base / "task3_instruction_nontoxic_smiles_generation" / repre / step_norm
        return base / "task3_CoT_nontoxic_smiles_generation_qa.jsonl"
    if task == "task3_nontoxic_safe_generation":
        base = qa_base / "task3_nontoxic_safe_generation" / repre / step_norm
        fname = (
            "task3_nontoxic_safe_generation_qa.jsonl"
            if variant == "base"
            else f"task3_nontoxic_safe_generation_qa_{variant}.jsonl"
        )
        return base / fname
    if task == "task3_stepwise_cot":
        base = qa_base / "task3_stepwise_cot_nontoxic_smiles_generation" / repre / step_norm
        return base / "task3_stepwise_cot_nontoxic_smiles_generation_qa.jsonl"
    # task3
    base = qa_base / "task3_nontoxic_smiles_generation" / repre / step_norm
    fname = (
        "task3_nontoxic_smiles_generation_qa.jsonl"
        if variant == "base"
        else f"task3_nontoxic_smiles_generation_qa_{variant}.jsonl"
    )
    return base / fname


def _load_qa_by_id(qa_path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not qa_path.is_file():
        print(f"  [WARN] QA 파일 없음 (stepwise 등 fragment gold 불가): {qa_path}", file=sys.stderr)
        return out
    with open(qa_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] QA jsonl skip line {lineno}: {e}", file=sys.stderr)
                continue
            i = rec.get("id")
            if i is not None:
                out[int(i)] = rec
    return out


def _get_metrics_for_task(
    task: str,
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int],
) -> Dict[str, Any]:
    """inference_gpt._get_metrics_for_task 와 동일 분기."""
    if task == "task1" and task1_toxic_fragment_identification_eval is not None:
        t = task1_toxic_fragment_identification_eval(gold_answer, llm_answer)
        return dict(
            zip(
                ["fragment_EM", "fragment_BLEU1", "fragment_Precision", "fragment_Recall", "fragment_F1"],
                t,
            )
        )
    if task == "task2" and task2_nontoxic_fragment_generation_eval is not None:
        t = task2_nontoxic_fragment_generation_eval(gold_answer, llm_answer, row_id=row_id)
        return dict(
            zip(
                [
                    "fragment_EM",
                    "fragment_BLEU1",
                    "fragment_Precision",
                    "fragment_Recall",
                    "fragment_F1",
                    "molecule_EM",
                    "molecule_morganFT",
                    "molecule_validity",
                ],
                t,
            )
        )
    if task in ("task3", "task3_instruction") and task3_nontoxic_smiles_generation_eval is not None:
        t = task3_nontoxic_smiles_generation_eval(gold_answer, llm_answer)
        return dict(
            zip(
                ["exact_match", "bleu", "levenshtein", "rdk_fts", "maccs_fts", "morgan_fts", "validity"],
                t,
            )
        )
    if task == "task3_nontoxic_safe_generation" and task3_nontoxic_safe_generation_eval is not None:
        t = task3_nontoxic_safe_generation_eval(gold_answer, llm_answer)
        return dict(
            zip(
                [
                    "safe_EM",
                    "exact_match",
                    "bleu",
                    "levenshtein",
                    "rdk_fts",
                    "maccs_fts",
                    "morgan_fts",
                    "validity",
                ],
                t,
            )
        )
    if task == "task3_stepwise_cot" and task3_stepwise_cot_nontoxic_smiles_generation_eval is not None:
        return task3_stepwise_cot_nontoxic_smiles_generation_eval(
            gold_answer,
            llm_answer,
            row_id=row_id,
        )
    if task == "subtask1" and subtask1_safe_to_smiles_eval is not None:
        t = subtask1_safe_to_smiles_eval(gold_answer, llm_answer)
        return dict(
            zip(
                ["exact_match", "bleu", "levenshtein", "rdk_fts", "maccs_fts", "morgan_fts", "validity"],
                t,
            )
        )
    if task == "subtask2" and subtask2_smiles_to_safe_eval is not None:
        t = subtask2_smiles_to_safe_eval(gold_answer, llm_answer, row_id=row_id)
        return dict(
            zip(
                [
                    "EM",
                    "BLEU1",
                    "validity",
                    "levenshtein_dist",
                    "levenshtein_norm",
                    "molecule_EM",
                    "molecule_morganFT",
                    "molecule_validity",
                ],
                t,
            )
        )
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
    idx = parts.index("results")
    name = pred_path.stem
    model = name.replace("predictions_", "", 1) if name.startswith("predictions_") else "unknown"
    split = root.name if root.name in ("train", "test") else "test"
    task = parts[0] if idx >= 1 else ""
    if task in ("subtask1", "subtask2"):
        repres = ""
        step = ""
    else:
        if idx >= 4:
            step = parts[idx - 1]
            repres = parts[idx - 2]
        else:
            repres = ""
            step = ""
    return {"task": task, "step": step, "repre": repres, "split": split, "model": model}


def reeval_one(
    pred_path: Path,
    root: Path,
    variant: str = "base",
    run_idx: Optional[int] = None,
    collect_validity: bool = True,
) -> None:
    """단일 predictions_*.jsonl에 대해 메트릭 재계산 후 evaluation_summary_*.json 저장."""
    meta = _parse_prediction_path(pred_path, root)
    if not meta or not meta["task"]:
        print(f"  skip (경로 파싱 불가): {pred_path}")
        return

    task = meta["task"]
    step = meta["step"] or "single_step"
    repres = meta["repre"] or "both_repre"
    split = meta["split"]

    qa_by_id: Optional[Dict[int, Dict[str, Any]]] = None
    if task == "task3_stepwise_cot":
        qa_path = _qa_jsonl_path(split, task, repres, step, variant=variant)
        qa_by_id = _load_qa_by_id(qa_path)

    task_keys = TASK_METRIC_KEYS.get(task, [])
    metric_sums: Dict[str, float] = {}
    correct = 0
    total = 0
    validity_acc: Dict[str, Any] = {}

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] predictions skip (bad JSON): {pred_path}: {e}", file=sys.stderr)
                continue
            total += 1
            gold_str = row.get("gold", "")
            pred = row.get("pred")
            gold_answer = _gold_answer_from_row(task, row, qa_by_id)
            llm_answer = _llm_answer_from_pred(pred)
            row_id = row.get("source_index", row.get("id"))

            metrics = _get_metrics_for_task(task, gold_answer, llm_answer, row_id=row_id)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

            if collect_validity:
                diag = collect_validity_diagnostics_for_sample(task, llm_answer, row_id)
                _aggregate_validity_diagnostics(validity_acc, diag, row.get("id"))

            pred_norm = _normalize_answer(llm_answer)
            gold_norm = (gold_str or "").strip() if isinstance(gold_str, str) else _normalize_answer(gold_str)
            correct += 1 if gold_norm == pred_norm else 0

    model = meta["model"]
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
    if collect_validity and validity_acc:
        summary["validity_diagnostics"] = _finalize_validity_diagnostics(validity_acc)

    evaluation_dir = pred_path.parent.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    summary_name = f"evaluation_summary_{safe_model}.json"
    summary_path = evaluation_dir / summary_name
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    print(f"  {task} {step} {model}: total={total}, correct={correct}, acc={acc:.4f} -> {summary_path}")


def find_prediction_files(root: Path, task_filter: Optional[str] = None) -> List[Path]:
    """root 아래 results/predictions_*.jsonl 목록. task_filter가 있으면 해당 task 폴더만."""
    out: List[Path] = []
    for pred_path in root.rglob("predictions_*.jsonl"):
        if pred_path.parent.name != "results":
            continue
        if task_filter:
            try:
                rel = pred_path.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) >= 1 and parts[0] != task_filter:
                continue
        out.append(pred_path)
    return sorted(out)


_TASK_CHOICES = [
    "task1",
    "task2",
    "task3",
    "task3_instruction",
    "task3_nontoxic_safe_generation",
    "task3_stepwise_cot",
    "subtask1",
    "subtask2",
    "all",
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="수정된 eval_metric으로 predictions 재평가 → evaluation_summary 갱신 (LLM 없음)",
    )
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
        choices=_TASK_CHOICES,
        help="재평가할 task 폴더명 (기본: all = 모든 task)",
    )
    ap.add_argument(
        "--variant",
        type=str,
        default="base",
        help="QA jsonl 파일명용 variant (task1/2/3 등, summary의 variant 라벨)",
    )
    ap.add_argument(
        "--no-validity-diagnostics",
        action="store_true",
        help="RDKit validity 실패 사유 집계·rdkit_stderr 예시 수집 생략 (약간 빠름)",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    task_filter = None if args.task == "all" else args.task
    pred_files = find_prediction_files(root, task_filter=task_filter)
    if not pred_files:
        print(f"예측 파일 없음: {root} (task={args.task})")
        return

    collect_vd = not args.no_validity_diagnostics
    print(f"재평가: root={root}, task={args.task}, 파일 {len(pred_files)}개")
    print(f"  validity_diagnostics={'on' if collect_vd else 'off'}\n")
    for p in pred_files:
        reeval_one(p, root, variant=args.variant, collect_validity=collect_vd)
    print("\n완료.")


if __name__ == "__main__":
    main()
