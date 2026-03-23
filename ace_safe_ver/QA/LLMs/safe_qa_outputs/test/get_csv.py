#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safe_qa_outputs/test 아래 evaluation_summary_*.json 을 읽어
Task1 / Task2 / Task3 및 Task3 Step-wise CoT(최종 SMILES + CoT Step1/Step2) 성능표를 하나의 CSV로 저장합니다.

경로 규칙 (inference_gpt.py 출력과 동일):
  {base_dir}/task{1|2|3}/{both_repre|only_smiles|only_safe}/{single_step|multi_step}/evaluation/evaluation_summary_{model_slug}.json
  Task3 CoT: {base_dir}/task3_stepwise_cot/.../evaluation/evaluation_summary_{model_slug}.json

행 순서:
  - 고정: GPT-4o×3, GPT-5.2×3 (각 Single/Multi Step)
  - 자동: 아래 허용 슬러그만(gemini-3-flash, gemini-3.1-pro, gemini-3.1-flash-lite[/-preview])
    Both / Only SMILES / Only SAFE 3행 추가 (알파벳 순). 그 외 모델·클로드는 제외.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# 스크립트 위치: .../safe_qa_outputs/test/get_csv.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_BASE = _SCRIPT_DIR

_STEP_SINGLE = "Single Step"
_STEP_MULTI = "Multi Step"
_STEP_FOLDER = {"Single Step": "single_step", "Multi Step": "multi_step"}

# (CSV Model 열 표시명, Mol Represent, repre 폴더명, 파일 슬러그 후보들)
RowSpec = Tuple[str, str, str, Tuple[str, ...]]

# 고정 GPT 행 (슬러그는 inference_gpt.py 출력 파일명과 일치)
_ROW_TEMPLATE_GPT: Sequence[RowSpec] = (
    ("GPT-4o", "Both", "both_repre", ("gpt-4o",)),
    ("GPT-4o", "Only SMILES", "only_smiles", ("gpt-4o",)),
    ("GPT-4o", "Only SAFE", "only_safe", ("gpt-4o",)),
    ("GPT-5.2", "Both", "both_repre", ("gpt-5.2", "gpt-5_2")),
    ("GPT-5.2", "Only SMILES", "only_smiles", ("gpt-5.2", "gpt-5_2")),
    ("GPT-5.2", "Only SAFE", "only_safe", ("gpt-5.2", "gpt-5_2")),
)

# CoT 블록: GPT-4o / GPT-5.2 만 (고정 행과 동일 개수)
_ROW_TEMPLATE_GPT_COT: Sequence[RowSpec] = tuple(_ROW_TEMPLATE_GPT)

# 고정 행에서 이미 다루는 슬러그 (파일 스캔 시 중복 추가 방지)
_STATIC_SLUGS: Set[str] = set()
for _, _, _, slugs in _ROW_TEMPLATE_GPT:
    _STATIC_SLUGS.update(slugs)

# 자동 행: 파일명 베이스 슬러그가 이 집합에 있을 때만 (ICL 접미사는 제거 후 비교)
_ALLOWED_PROVIDER_BASE_SLUGS = frozenset(
    {
        "gemini-3-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
    }
)


def _base_slug_strip_icl(slug: str) -> str:
    for suf in ("_icl4", "_icl2", "_icl1"):
        if slug.endswith(suf):
            return slug[: -len(suf)]
    return slug


def _has_icl_suffix(slug: str) -> bool:
    return any(slug.endswith(suf) for suf in ("_icl4", "_icl2", "_icl1"))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:.6g}"
    return str(v)


def _load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _discover_slugs_for_task(base: Path, task_folder: str) -> Set[str]:
    """task 아래 모든 repre × single/multi 의 evaluation_summary_*.json 슬러그 수집."""
    found: Set[str] = set()
    for step in ("single_step", "multi_step"):
        pattern = f"{task_folder}/*/{step}/evaluation/evaluation_summary_*.json"
        for p in base.glob(pattern):
            stem = p.stem  # evaluation_summary_gpt-4o
            if stem.startswith("evaluation_summary_"):
                found.add(stem.replace("evaluation_summary_", "", 1))
    return found


def _extra_provider_rows(all_slugs: Set[str]) -> List[RowSpec]:
    """허용된 Gemini 슬러그만 3행(Both/SMILES/SAFE) 생성. 파일명 슬러그를 Model 열에 그대로 사용."""
    extra_slugs = sorted(
        s
        for s in all_slugs
        if s not in _STATIC_SLUGS
        and _base_slug_strip_icl(s) in _ALLOWED_PROVIDER_BASE_SLUGS
        and not _has_icl_suffix(s)
    )
    rows: List[RowSpec] = []
    for slug in extra_slugs:
        rows.append((slug, "Both", "both_repre", (slug,)))
        rows.append((slug, "Only SMILES", "only_smiles", (slug,)))
        rows.append((slug, "Only SAFE", "only_safe", (slug,)))
    return rows


def _build_row_template_for_task(base: Path, task_folder: str, cot: bool = False) -> List[RowSpec]:
    base_rows: List[RowSpec] = list(_ROW_TEMPLATE_GPT_COT if cot else _ROW_TEMPLATE_GPT)
    discovered = _discover_slugs_for_task(base, task_folder)
    base_rows.extend(_extra_provider_rows(discovered))
    return base_rows


def _find_summary(
    base: Path, task: str, repre: str, step_folder: str, slug_candidates: Tuple[str, ...]
) -> Optional[Dict[str, Any]]:
    eval_dir = base / task / repre / step_folder / "evaluation"
    if not eval_dir.is_dir():
        return None
    for slug in slug_candidates:
        p = eval_dir / f"evaluation_summary_{slug}.json"
        data = _load_summary(p)
        if data is not None:
            return data
    return None


def _metrics_mean(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {}
    mm = data.get("metrics_mean")
    return mm if isinstance(mm, dict) else {}


def _row_task1(data: Optional[Dict[str, Any]]) -> List[str]:
    m = _metrics_mean(data)
    return [
        _fmt(m.get("fragment_EM")),
        _fmt(m.get("fragment_BLEU1")),
        _fmt(m.get("fragment_Precision")),
        _fmt(m.get("fragment_Recall")),
        _fmt(m.get("fragment_F1")),
    ]


def _row_task2(data: Optional[Dict[str, Any]]) -> List[str]:
    m = _metrics_mean(data)
    return [
        _fmt(m.get("fragment_EM")),
        _fmt(m.get("fragment_BLEU1")),
        _fmt(m.get("fragment_Precision")),
        _fmt(m.get("fragment_Recall")),
        _fmt(m.get("fragment_F1")),
        _fmt(m.get("molecule_EM")),
        _fmt(m.get("molecule_morganFT")),
        _fmt(m.get("molecule_validity")),
    ]


def _row_task3(data: Optional[Dict[str, Any]]) -> List[str]:
    m = _metrics_mean(data)
    return [
        _fmt(m.get("exact_match")),
        _fmt(m.get("bleu")),
        _fmt(m.get("levenshtein")),
        _fmt(m.get("rdk_fts")),
        _fmt(m.get("maccs_fts")),
        _fmt(m.get("morgan_fts")),
        _fmt(m.get("validity")),
    ]


def _row_cot_step1(data: Optional[Dict[str, Any]]) -> List[str]:
    m = _metrics_mean(data)
    return [
        _fmt(m.get("step1_fragment_EM")),
        _fmt(m.get("step1_fragment_BLEU1")),
        _fmt(m.get("step1_fragment_Precision")),
        _fmt(m.get("step1_fragment_Recall")),
        _fmt(m.get("step1_fragment_F1")),
    ]


def _row_cot_step2(data: Optional[Dict[str, Any]]) -> List[str]:
    m = _metrics_mean(data)
    return [
        _fmt(m.get("step2_fragment_EM")),
        _fmt(m.get("step2_fragment_BLEU1")),
        _fmt(m.get("step2_fragment_Precision")),
        _fmt(m.get("step2_fragment_Recall")),
        _fmt(m.get("step2_fragment_F1")),
        _fmt(m.get("step2_molecule_EM")),
        _fmt(m.get("step2_molecule_morganFT")),
        _fmt(m.get("step2_molecule_validity")),
    ]


def _row_task3_stepwise_cot_safe_final(data: Optional[Dict[str, Any]]) -> List[str]:
    """task3_stepwise_cot_safe_generation: Step3 = full SAFE + task3_nontoxic_safe_generation 메트릭."""
    m = _metrics_mean(data)
    return [
        _fmt(m.get("safe_EM")),
        _fmt(m.get("exact_match")),
        _fmt(m.get("bleu")),
        _fmt(m.get("levenshtein")),
        _fmt(m.get("rdk_fts")),
        _fmt(m.get("maccs_fts")),
        _fmt(m.get("morgan_fts")),
        _fmt(m.get("validity")),
    ]


def _write_task_block(
    writer: csv.writer,
    title: str,
    header: List[str],
    row_fn,
    base: Path,
    task_folder: str,
    row_template: Sequence[RowSpec],
) -> None:
    writer.writerow([title] + [""] * (len(header) - 1))
    writer.writerow(header)
    for step_label in (_STEP_SINGLE, _STEP_MULTI):
        step_folder = _STEP_FOLDER[step_label]
        for model_disp, mol_rep_disp, repre, slug_candidates in row_template:
            data = _find_summary(base, task_folder, repre, step_folder, slug_candidates)
            metrics = row_fn(data)
            writer.writerow([step_label, model_disp, mol_rep_disp] + metrics)
    writer.writerow([])


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate evaluation_summary JSON -> one CSV (Task1-3 + Task3 CoT).")
    ap.add_argument(
        "--base_dir",
        type=Path,
        default=_DEFAULT_BASE,
        help="safe_qa_outputs/test 와 동일 레벨 (task1/, task2/, task3/ 하위 포함)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_BASE / "evaluation_summary_tasks_1_2_3.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()
    base = Path(args.base_dir).resolve()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    tpl_task1 = _build_row_template_for_task(base, "task1", cot=False)
    tpl_task2 = _build_row_template_for_task(base, "task2", cot=False)
    tpl_task3 = _build_row_template_for_task(base, "task3", cot=False)
    tpl_task3_safe = _build_row_template_for_task(base, "task3_nontoxic_safe_generation", cot=False)
    tpl_cot = _build_row_template_for_task(base, "task3_stepwise_cot", cot=True)
    tpl_cot_safe = _build_row_template_for_task(base, "task3_stepwise_cot_safe_generation", cot=True)

    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        _write_task_block(
            writer,
            "Task1 - Toxic Fragment Identification",
            ["Step", "Model", "Mol Represent", "EM", "BLEU1", "Precision", "Recall", "F1"],
            _row_task1,
            base,
            "task1",
            tpl_task1,
        )
        _write_task_block(
            writer,
            "Task2 - NonToxic Fragment Generation",
            [
                "Step",
                "Model",
                "Mol Represent",
                "EM",
                "BLEU1",
                "Precision",
                "Recall",
                "F1",
                "Molecule EM",
                "Morgan FTS",
                "Validity",
            ],
            _row_task2,
            base,
            "task2",
            tpl_task2,
        )
        _write_task_block(
            writer,
            "Task3 - Nontoxic SMILES Generation",
            [
                "Step",
                "Model",
                "Mol Represent",
                "EM",
                "BLEU1",
                "Levenshtein",
                "RDK FTS",
                "MACCS FTS",
                "Morgan FTS",
                "Validity",
            ],
            _row_task3,
            base,
            "task3",
            tpl_task3,
        )
        _write_task_block(
            writer,
            "Task3 - Nontoxic Safe Generation (SAFE answer -> SMILES eval)",
            [
                "Step",
                "Model",
                "Mol Represent",
                "EM",
                "BLEU1",
                "Levenshtein",
                "RDK FTS",
                "MACCS FTS",
                "Morgan FTS",
                "Validity",
            ],
            _row_task3,
            base,
            "task3_nontoxic_safe_generation",
            tpl_task3_safe,
        )
        _write_task_block(
            writer,
            "Task3 Step-wise CoT - Nontoxic SMILES Generation",
            [
                "Step",
                "Model",
                "Mol Represent",
                "EM",
                "BLEU1",
                "Levenshtein",
                "RDK FTS",
                "MACCS FTS",
                "Morgan FTS",
                "Validity",
            ],
            _row_task3,
            base,
            "task3_stepwise_cot",
            tpl_cot,
        )
        _write_task_block(
            writer,
            "Task3 CoT - Step 1 - Toxic Fragment Identification",
            ["Step", "Model", "Mol Represent", "EM", "BLEU1", "Precision", "Recall", "F1"],
            _row_cot_step1,
            base,
            "task3_stepwise_cot",
            tpl_cot,
        )
        _write_task_block(
            writer,
            "Task3 CoT - Step 2 - NonToxic Fragment Generation",
            [
                "Step",
                "Model",
                "Mol Represent",
                "EM",
                "BLEU1",
                "Precision",
                "Recall",
                "F1",
                "Molecule EM",
                "Morgan FTS",
                "Validity",
            ],
            _row_cot_step2,
            base,
            "task3_stepwise_cot",
            tpl_cot,
        )
        _write_task_block(
            writer,
            "Task3 Step-wise CoT (SAFE final) - Nontoxic SAFE Generation",
            [
                "Step",
                "Model",
                "Mol Represent",
                "SAFE EM",
                "SMILES EM",
                "BLEU1",
                "Levenshtein",
                "RDK FTS",
                "MACCS FTS",
                "Morgan FTS",
                "Validity",
            ],
            _row_task3_stepwise_cot_safe_final,
            base,
            "task3_stepwise_cot_safe_generation",
            tpl_cot_safe,
        )
        _write_task_block(
            writer,
            "Task3 CoT (SAFE) - Step 1 - Toxic Fragment Identification",
            ["Step", "Model", "Mol Represent", "EM", "BLEU1", "Precision", "Recall", "F1"],
            _row_cot_step1,
            base,
            "task3_stepwise_cot_safe_generation",
            tpl_cot_safe,
        )
        _write_task_block(
            writer,
            "Task3 CoT (SAFE) - Step 2 - NonToxic Fragment Generation",
            [
                "Step",
                "Model",
                "Mol Represent",
                "EM",
                "BLEU1",
                "Precision",
                "Recall",
                "F1",
                "Molecule EM",
                "Morgan FTS",
                "Validity",
            ],
            _row_cot_step2,
            base,
            "task3_stepwise_cot_safe_generation",
            tpl_cot_safe,
        )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
