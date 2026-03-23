#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICL 실험용: safe_qa_outputs/test 아래 evaluation_summary_*.json 을 읽어
Task1~3 및 Task3 Step-wise CoT 성능표를 CSV로 저장합니다.

기존 get_csv.py 와의 차이:
  - 행을 Mol Represent(Both / Only SMILES / Only SAFE) 대신 **ICL-Mode**
    (0-Shot, ICL-1, ICL-2, ICL-4) 로 펼칩니다.
  - QA/추론 경로는 ICL 실험에서 보통 **both_repre** 고정이므로,
    기본적으로 `.../task*/both_repre/{single_step|multi_step}/evaluation/` 만 조회합니다.

파일명 규칙 (inference_gpt.py 와 동일):
  - 0-Shot (base): evaluation_summary_<model>.json
  - ICL-k: evaluation_summary_<model>_icl<k>.json  (예: gpt-4o_icl1)

행 순서:
  - 고정: GPT-4o, GPT-5.2 각각 × (0-Shot, ICL-1, ICL-2, ICL-4) × Single/Multi Step
  - 자동: both_repre 아래 허용 Gemini 슬러그만 (gemini-3-flash, gemini-3.1-pro, gemini-3.1-flash-lite[/-preview])
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# 스크립트 위치: .../safe_qa_outputs/test/get_csv_icl.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_BASE = _SCRIPT_DIR

_STEP_SINGLE = "Single Step"
_STEP_MULTI = "Multi Step"
_STEP_FOLDER = {"Single Step": "single_step", "Multi Step": "multi_step"}

# ICL variant 키 -> CSV 표시명
_ICL_MODE_LABELS: Dict[str, str] = {
    "base": "0-Shot",
    "icl1": "ICL-1",
    "icl2": "ICL-2",
    "icl4": "ICL-4",
}

_ICL_VARIANT_ORDER = ("base", "icl1", "icl2", "icl4")

# (CSV Model 열, ICL-Mode 표시, slug 후보 튜플) — 항상 both_repre 경로만 사용
IclRowSpec = Tuple[str, str, Tuple[str, ...]]


def _slugs_for_icl_variant(base_slugs: Tuple[str, ...], variant: str) -> Tuple[str, ...]:
    if variant == "base":
        return base_slugs
    return tuple(f"{s}_{variant}" for s in base_slugs)


# 고정 GPT 행 (베이스 슬러그는 inference_gpt 출력 파일명과 일치)
_STATIC_MODELS: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("GPT-4o", ("gpt-4o",)),
    ("GPT-5.2", ("gpt-5.2", "gpt-5_2")),
)

# CoT 블록: 고정 모델과 동일 (GPT-4o / GPT-5.2 만)
_STATIC_MODELS_COT: Sequence[Tuple[str, Tuple[str, ...]]] = _STATIC_MODELS

_STATIC_BASE_SLUGS: Set[str] = set()
for _, slugs in _STATIC_MODELS:
    _STATIC_BASE_SLUGS.update(slugs)

# 파일 슬러그 -> 표시용 이름 (스프레드시트 정렬용; 없으면 슬러그 그대로)
_MODEL_DISPLAY: Dict[str, str] = {
    "gemini-3-flash": "Gemini-Flash",
    "gemini-3.1-flash-lite": "Gemini-Flash-Lite",
    "gemini-3.1-flash-lite-preview": "Gemini-Flash-Lite",
    "gemini-3.1-pro": "Gemini-Pro",
}

# both_repre 자동 행에 포함할 베이스 슬러그만 (gemini-3.1-flash 비·lite 제외)
_ALLOWED_EXTRA_BASE_SLUGS = frozenset(
    {
        "gemini-3-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
    }
)


def _model_display_name(slug: str) -> str:
    return _MODEL_DISPLAY.get(slug, slug)


def _expand_static_icl_rows(models: Sequence[Tuple[str, Tuple[str, ...]]]) -> List[IclRowSpec]:
    rows: List[IclRowSpec] = []
    for model_disp, base_slugs in models:
        for vk in _ICL_VARIANT_ORDER:
            label = _ICL_MODE_LABELS[vk]
            slugs = _slugs_for_icl_variant(base_slugs, vk)
            rows.append((model_disp, label, slugs))
    return rows


def _strip_icl_suffix(stem_after_prefix: str) -> str:
    """evaluation_summary_ 이후 문자열에서 _icl1/_icl2/_icl4 제거."""
    for suf in ("_icl4", "_icl2", "_icl1"):
        if stem_after_prefix.endswith(suf):
            return stem_after_prefix[: -len(suf)]
    return stem_after_prefix


def _discover_base_slugs_both_repre(base: Path, task_folder: str) -> Set[str]:
    """both_repre 아래 evaluation_summary_*.json 에서 베이스 슬러그 수집."""
    found: Set[str] = set()
    for step in ("single_step", "multi_step"):
        pattern = f"{task_folder}/both_repre/{step}/evaluation/evaluation_summary_*.json"
        for p in base.glob(pattern):
            stem = p.stem
            if not stem.startswith("evaluation_summary_"):
                continue
            rest = stem[len("evaluation_summary_") :]
            found.add(_strip_icl_suffix(rest))
    return found


def _extra_provider_rows_icl(all_base_slugs: Set[str]) -> List[IclRowSpec]:
    """허용된 Gemini 베이스 슬러그마다 4 ICL 모드 행."""
    extra_bases = sorted(
        s
        for s in all_base_slugs
        if s not in _STATIC_BASE_SLUGS and s in _ALLOWED_EXTRA_BASE_SLUGS
    )
    rows: List[IclRowSpec] = []
    for base_slug in extra_bases:
        disp = _model_display_name(base_slug)
        slugs_base = (base_slug,)
        for vk in _ICL_VARIANT_ORDER:
            label = _ICL_MODE_LABELS[vk]
            slugs = _slugs_for_icl_variant(slugs_base, vk)
            rows.append((disp, label, slugs))
    return rows


def _build_icl_row_template(base: Path, task_folder: str, cot: bool = False) -> List[IclRowSpec]:
    models = _STATIC_MODELS_COT if cot else _STATIC_MODELS
    rows = _expand_static_icl_rows(models)
    discovered = _discover_base_slugs_both_repre(base, task_folder)
    rows.extend(_extra_provider_rows_icl(discovered))
    return rows


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


def _find_summary(
    base: Path,
    task: str,
    repre: str,
    step_folder: str,
    slug_candidates: Tuple[str, ...],
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
    row_template: Sequence[IclRowSpec],
    repre: str,
) -> None:
    writer.writerow([title] + [""] * (len(header) - 1))
    writer.writerow(header)
    for step_label in (_STEP_SINGLE, _STEP_MULTI):
        step_folder = _STEP_FOLDER[step_label]
        for model_disp, icl_mode_disp, slug_candidates in row_template:
            data = _find_summary(base, task_folder, repre, step_folder, slug_candidates)
            metrics = row_fn(data)
            writer.writerow([step_label, model_disp, icl_mode_disp] + metrics)
    writer.writerow([])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate evaluation_summary JSON -> one CSV (ICL modes: 0-Shot / ICL-1/2/4, default both_repre)."
    )
    ap.add_argument(
        "--base_dir",
        type=Path,
        default=_DEFAULT_BASE,
        help="safe_qa_outputs/test 와 동일 레벨 (task1/, task2/, … 하위 포함)",
    )
    ap.add_argument(
        "--molecule_repr",
        type=str,
        default="both_repre",
        help="ICL 결과가 저장된 molecule_repr 폴더명 (기본: both_repre)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_DEFAULT_BASE / "evaluation_summary_tasks_1_2_3_icl.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()
    base = Path(args.base_dir).resolve()
    repre = args.molecule_repr.strip() or "both_repre"
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    tpl_task1 = _build_icl_row_template(base, "task1", cot=False)
    tpl_task2 = _build_icl_row_template(base, "task2", cot=False)
    tpl_task3 = _build_icl_row_template(base, "task3", cot=False)
    tpl_task3_safe = _build_icl_row_template(base, "task3_nontoxic_safe_generation", cot=False)
    tpl_cot = _build_icl_row_template(base, "task3_stepwise_cot", cot=True)
    tpl_cot_safe = _build_icl_row_template(base, "task3_stepwise_cot_safe_generation", cot=True)

    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        _write_task_block(
            writer,
            "Task1 - Toxic Fragment Identification",
            ["Step", "Model", "ICL-Mode", "EM", "BLEU1", "Precision", "Recall", "F1"],
            _row_task1,
            base,
            "task1",
            tpl_task1,
            repre,
        )
        _write_task_block(
            writer,
            "Task2 - NonToxic Fragment Generation",
            [
                "Step",
                "Model",
                "ICL-Mode",
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
            repre,
        )
        _write_task_block(
            writer,
            "Task3 - Nontoxic SMILES Generation",
            [
                "Step",
                "Model",
                "ICL-Mode",
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
            repre,
        )
        _write_task_block(
            writer,
            "Task3 - Nontoxic Safe Generation (SAFE answer -> SMILES eval)",
            [
                "Step",
                "Model",
                "ICL-Mode",
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
            repre,
        )
        _write_task_block(
            writer,
            "Task3 Step-wise CoT - Nontoxic SMILES Generation",
            [
                "Step",
                "Model",
                "ICL-Mode",
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
            repre,
        )
        _write_task_block(
            writer,
            "Task3 CoT - Step 1 - Toxic Fragment Identification",
            ["Step", "Model", "ICL-Mode", "EM", "BLEU1", "Precision", "Recall", "F1"],
            _row_cot_step1,
            base,
            "task3_stepwise_cot",
            tpl_cot,
            repre,
        )
        _write_task_block(
            writer,
            "Task3 CoT - Step 2 - NonToxic Fragment Generation",
            [
                "Step",
                "Model",
                "ICL-Mode",
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
            repre,
        )
        _write_task_block(
            writer,
            "Task3 Step-wise CoT (SAFE final) - Nontoxic SAFE Generation",
            [
                "Step",
                "Model",
                "ICL-Mode",
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
            repre,
        )
        _write_task_block(
            writer,
            "Task3 CoT (SAFE) - Step 1 - Toxic Fragment Identification",
            ["Step", "Model", "ICL-Mode", "EM", "BLEU1", "Precision", "Recall", "F1"],
            _row_cot_step1,
            base,
            "task3_stepwise_cot_safe_generation",
            tpl_cot_safe,
            repre,
        )
        _write_task_block(
            writer,
            "Task3 CoT (SAFE) - Step 2 - NonToxic Fragment Generation",
            [
                "Step",
                "Model",
                "ICL-Mode",
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
            repre,
        )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
