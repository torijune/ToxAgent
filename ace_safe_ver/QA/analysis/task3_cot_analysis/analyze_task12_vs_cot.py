#!/usr/bin/env python3
"""
일반 Task1 / Task2의 fragment EM과 task3_stepwise_cot의 Step1·Step2 fragment EM을
동일 테스트 샘플(id 기준)으로 맞춰 비교한다.

데이터는 기본적으로 아래 세 곳의 both_repre 결과를 쓴다 (동일 --test-root):
  task1/, task2/, task3_stepwise_cot/

모델이 적게 보이는 이유: 비교하려면 세 파일이 모두 있어야 하며(`predictions_<model>.jsonl`),
실제로는 task3_stepwise_cot에만 돌린 모델·task1에만 있는 모델 등이 섞여 있다.
스크립트는 세 태스크의 (single_step|multi_step, model) 합집합을 나열하고,
없는 쪽은 summary CSV에서 missing_parts로 표시한다.

입력 (기본):
  QA/LLMs/safe_qa_outputs/test/{task1,task2,task3_stepwise_cot}/both_repre/
    {single_step,multi_step}/results/predictions_<model>.jsonl

출력 (--out):
  - task12_vs_cot_summary.csv: model·prompt_variant별 평균 EM·표본 수·일치율
  - task12_vs_cot_paired_<model>_<variant>.csv: 샘플 단위 병합 (모델별 파일이 너무 많으면 --paired-one 모델만)

사용 예:
  python analyze_task12_vs_cot.py
  python analyze_task12_vs_cot.py --out ./outputs --paired-model gpt-4o --paired-variant single_step
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def _repo_qa_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_task_predictions(path: Path) -> dict[int, dict[str, Any]]:
    """id -> row dict"""
    out: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        k = row.get("id")
        if k is None:
            continue
        out[int(k)] = row
    return out


def prediction_path(
    test_root: Path, task: str, variant: str, model: str
) -> Path:
    return (
        test_root
        / task
        / "both_repre"
        / variant
        / "results"
        / f"predictions_{model}.jsonl"
    )


def discover_variant_model_union(test_root: Path) -> list[tuple[str, str]]:
    """
    task1 / task2 / task3_stepwise_cot 각각에 존재하는
    (prompt_variant, model)의 합집합. 한쪽에만 있는 모델도 포함된다.
    """
    s: set[tuple[str, str]] = set()
    for task in ("task1", "task2", "task3_stepwise_cot"):
        for variant in ("single_step", "multi_step"):
            res = test_root / task / "both_repre" / variant / "results"
            if not res.is_dir():
                continue
            for p in res.glob("predictions_*.jsonl"):
                stem = p.stem
                if not stem.startswith("predictions_"):
                    continue
                model = stem[len("predictions_") :]
                s.add((variant, model))
    return sorted(s)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _parse_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def merge_three(
    t1: dict[int, dict[str, Any]],
    t2: dict[int, dict[str, Any]],
    t3: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """공통 id에 대해 한 행으로."""
    common = set(t1.keys()) & set(t2.keys()) & set(t3.keys())
    rows: list[dict[str, Any]] = []
    for i in sorted(common):
        r1, r2, r3 = t1[i], t2[i], t3[i]
        # 스키마 일관성 검사 (선택)
        ds = (r1.get("dataset_name"), r2.get("dataset_name"), r3.get("dataset_name"))
        ep = (r1.get("endpoint"), r2.get("endpoint"), r3.get("endpoint"))
        si = (r1.get("source_index"), r2.get("source_index"), r3.get("source_index"))
        mismatch = ds[0] != ds[1] or ds[1] != ds[2] or ep[0] != ep[1] or ep[1] != ep[2]
        if si[0] != si[1] or si[1] != si[2]:
            mismatch = True

        rows.append(
            {
                "id": i,
                "dataset_name": r1.get("dataset_name"),
                "endpoint": r1.get("endpoint"),
                "source_index": r1.get("source_index"),
                "meta_mismatch": int(mismatch),
                "task1_fragment_EM": _parse_float(r1.get("fragment_EM")),
                "task2_fragment_EM": _parse_float(r2.get("fragment_EM")),
                "cot_step1_fragment_EM": _parse_float(r3.get("step1_fragment_EM")),
                "cot_step2_fragment_EM": _parse_float(r3.get("step2_fragment_EM")),
                "cot_exact_match": _parse_float(r3.get("exact_match")),
            }
        )
    return rows


def summarize_paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}

    def col_mean(key: str) -> float:
        xs = [float(r[key]) for r in rows if r.get(key) is not None]
        return _mean(xs)

    def agreement(a: str, b: str) -> float:
        ok = 0
        tot = 0
        for r in rows:
            va, vb = r.get(a), r.get(b)
            if va is None or vb is None:
                continue
            tot += 1
            if int(round(float(va))) == int(round(float(vb))):
                ok += 1
        return ok / tot if tot else float("nan")

    n = len(rows)
    return {
        "n": n,
        "mean_task1_EM": col_mean("task1_fragment_EM"),
        "mean_task2_EM": col_mean("task2_fragment_EM"),
        "mean_cot_step1_EM": col_mean("cot_step1_fragment_EM"),
        "mean_cot_step2_EM": col_mean("cot_step2_fragment_EM"),
        "mean_cot_final_EM": col_mean("cot_exact_match"),
        "agreement_task1_vs_cot_step1": agreement("task1_fragment_EM", "cot_step1_fragment_EM"),
        "agreement_task2_vs_cot_step2": agreement("task2_fragment_EM", "cot_step2_fragment_EM"),
        "mean_delta_step1_cot_minus_task1": col_mean("cot_step1_fragment_EM")
        - col_mean("task1_fragment_EM"),
        "mean_delta_step2_cot_minus_task2": col_mean("cot_step2_fragment_EM")
        - col_mean("task2_fragment_EM"),
    }


def main() -> int:
    qa = _repo_qa_root()
    default_test_root = qa / "LLMs" / "safe_qa_outputs" / "test"
    default_out = Path(__file__).resolve().parent / "outputs"

    ap = argparse.ArgumentParser(
        description="Task1/2 standalone vs task3_stepwise_cot Step1/2 EM 비교"
    )
    ap.add_argument(
        "--test-root",
        type=Path,
        default=default_test_root,
        help="safe_qa_outputs/test 디렉터리",
    )
    ap.add_argument("--out", type=Path, default=default_out, help="CSV 출력 디렉터리")
    ap.add_argument(
        "--paired-model",
        type=str,
        default=None,
        help="이 모델에 대해서만 샘플 단위 paired CSV 저장",
    )
    ap.add_argument(
        "--paired-variant",
        type=str,
        choices=["single_step", "multi_step", "both"],
        default="both",
        help="paired CSV를 저장할 prompt variant",
    )
    args = ap.parse_args()

    test_root: Path = args.test_root
    if not test_root.is_dir():
        print(f"ERROR: --test-root not found: {test_root}", file=sys.stderr)
        return 1

    pairs = discover_variant_model_union(test_root)
    if not pairs:
        print(
            f"ERROR: no predictions_*.jsonl under "
            f"{test_root}/{{task1,task2,task3_stepwise_cot}}/both_repre/",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for variant, model in pairs:
        t1_path = prediction_path(test_root, "task1", variant, model)
        t2_path = prediction_path(test_root, "task2", variant, model)
        cot_path = prediction_path(test_root, "task3_stepwise_cot", variant, model)

        missing: list[str] = []
        if not t1_path.is_file():
            missing.append("task1")
        if not t2_path.is_file():
            missing.append("task2")
        if not cot_path.is_file():
            missing.append("task3_stepwise_cot")

        if missing:
            summary_rows.append(
                {
                    "model": model,
                    "prompt_variant": variant,
                    "status": "incomplete",
                    "missing_parts": ",".join(missing),
                    "task1_path": str(t1_path) if t1_path.is_file() else "",
                    "task2_path": str(t2_path) if t2_path.is_file() else "",
                    "cot_path": str(cot_path) if cot_path.is_file() else "",
                }
            )
            continue

        t1 = load_task_predictions(t1_path)
        t2 = load_task_predictions(t2_path)
        t3 = load_task_predictions(cot_path)
        paired = merge_three(t1, t2, t3)
        stat = summarize_paired(paired)
        n_paired = int(stat.pop("n", 0))
        row = {
            "model": model,
            "prompt_variant": variant,
            "status": "ok",
            "missing_parts": "",
            "n_task1": len(t1),
            "n_task2": len(t2),
            "n_cot": len(t3),
            "n_paired": n_paired,
        }
        row.update(stat)
        if paired:
            row["meta_mismatch_rate"] = sum(r["meta_mismatch"] for r in paired) / len(paired)
        else:
            row["meta_mismatch_rate"] = float("nan")
        summary_rows.append(row)

        do_paired = args.paired_model == model and (
            args.paired_variant == "both" or args.paired_variant == variant
        )
        if do_paired and paired:
            paired_path = (
                args.out / f"task12_vs_cot_paired_{model.replace('.', '_')}_{variant}.csv"
            )
            fieldnames = list(paired[0].keys())
            with paired_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(paired)
            print(f"Wrote paired: {paired_path}")

    # flatten summary for CSV
    if summary_rows:
        keys: set[str] = set()
        for r in summary_rows:
            keys.update(r.keys())
        fieldnames = sorted(keys)
        # prefer logical order
        preferred = [
            "model",
            "prompt_variant",
            "status",
            "missing_parts",
            "n_paired",
            "n_task1",
            "n_task2",
            "n_cot",
            "mean_task1_EM",
            "mean_cot_step1_EM",
            "mean_delta_step1_cot_minus_task1",
            "agreement_task1_vs_cot_step1",
            "mean_task2_EM",
            "mean_cot_step2_EM",
            "mean_delta_step2_cot_minus_task2",
            "agreement_task2_vs_cot_step2",
            "mean_cot_final_EM",
            "task1_path",
            "task2_path",
            "cot_path",
        ]
        fieldnames = [c for c in preferred if c in keys] + [c for c in fieldnames if c not in preferred]

        out_csv = args.out / "task12_vs_cot_summary.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote: {out_csv}")

    if not args.paired_model:
        print(
            "\nTip: 샘플 단위 CSV를 남기려면 "
            "--paired-model gpt-4o --paired-variant single_step (또는 both) 를 지정하세요.",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
