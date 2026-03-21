#!/usr/bin/env python3
"""
task3 (단일 응답 nontoxic SMILES) vs task3_stepwise_cot (CoT) — 최종 성능만 비교.

동일 테스트 id에 대해 `exact_match` 및 (선택) 보조 지표(bleu, validity 등)를 맞춘다.

입력 (기본):
  QA/LLMs/safe_qa_outputs/test/{task3,task3_stepwise_cot}/both_repre/
    {single_step,multi_step}/results/predictions_<model>.jsonl

출력 (--out):
  - task3_vs_task3_cot_summary.csv: model·prompt_variant별 평균·일치율·혼동 요약
  - task3_vs_task3_cot_paired_<model>_<variant>.csv: --paired-model 지정 시

사용 예:
  python analyze_task3_vs_task3_cot.py
  python analyze_task3_vs_task3_cot.py --paired-model gpt-4o --paired-variant both
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
    out: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        k = row.get("id")
        if k is None:
            continue
        out[int(k)] = row
    return out


def discover_cot_predictions(root: Path) -> list[tuple[str, str, Path]]:
    """task3_stepwise_cot 기준으로 (prompt_variant, model, path) 나열."""
    found: list[tuple[str, str, Path]] = []
    for variant in ("single_step", "multi_step"):
        res = root / "task3_stepwise_cot" / "both_repre" / variant / "results"
        if not res.is_dir():
            continue
        for p in sorted(res.glob("predictions_*.jsonl")):
            stem = p.stem
            if not stem.startswith("predictions_"):
                continue
            model = stem[len("predictions_") :]
            found.append((variant, model, p))
    return found


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _parse_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def merge_pair(
    r3: dict[str, Any],
    rc: dict[str, Any],
    id_: int,
) -> dict[str, Any]:
    ds = (r3.get("dataset_name"), rc.get("dataset_name"))
    ep = (r3.get("endpoint"), rc.get("endpoint"))
    si = (r3.get("source_index"), rc.get("source_index"))
    mismatch = ds[0] != ds[1] or ep[0] != ep[1] or si[0] != si[1]

    return {
        "id": id_,
        "dataset_name": r3.get("dataset_name"),
        "endpoint": r3.get("endpoint"),
        "source_index": r3.get("source_index"),
        "meta_mismatch": int(mismatch),
        "task3_exact_match": _parse_float(r3.get("exact_match")),
        "cot_exact_match": _parse_float(rc.get("exact_match")),
        "task3_bleu": _parse_float(r3.get("bleu")),
        "cot_bleu": _parse_float(rc.get("bleu")),
        "task3_validity": _parse_float(r3.get("validity")),
        "cot_validity": _parse_float(rc.get("validity")),
    }


def merge_all(
    t3: dict[int, dict[str, Any]],
    tc: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    common = set(t3.keys()) & set(tc.keys())
    return [merge_pair(t3[i], tc[i], i) for i in sorted(common)]


def summarize_final(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}

    def col_mean(key: str) -> float:
        xs = [float(r[key]) for r in rows if r.get(key) is not None]
        return _mean(xs)

    def agreement_em() -> float:
        ok = tot = 0
        for r in rows:
            a, b = r.get("task3_exact_match"), r.get("cot_exact_match")
            if a is None or b is None:
                continue
            tot += 1
            if int(round(float(a))) == int(round(float(b))):
                ok += 1
        return ok / tot if tot else float("nan")

    n = len(rows)
    # exact_match confusion: task3 x cot (binary)
    both_1 = both_0 = t3_only_1 = cot_only_1 = 0
    for r in rows:
        a = r.get("task3_exact_match")
        b = r.get("cot_exact_match")
        if a is None or b is None:
            continue
        ia, ib = int(round(float(a))), int(round(float(b)))
        if ia == 1 and ib == 1:
            both_1 += 1
        elif ia == 0 and ib == 0:
            both_0 += 1
        elif ia == 1 and ib == 0:
            t3_only_1 += 1
        else:
            cot_only_1 += 1

    m3 = col_mean("task3_exact_match")
    mc = col_mean("cot_exact_match")

    return {
        "n": n,
        "mean_task3_exact_match": m3,
        "mean_cot_exact_match": mc,
        "mean_delta_cot_minus_task3": mc - m3,
        "agreement_exact_match_01": agreement_em(),
        "count_both_em1": both_1,
        "count_both_em0": both_0,
        "count_task3_em1_cot_em0": t3_only_1,
        "count_task3_em0_cot_em1": cot_only_1,
        "mean_task3_bleu": col_mean("task3_bleu"),
        "mean_cot_bleu": col_mean("cot_bleu"),
        "mean_delta_bleu_cot_minus_task3": col_mean("cot_bleu") - col_mean("task3_bleu"),
        "mean_task3_validity": col_mean("task3_validity"),
        "mean_cot_validity": col_mean("cot_validity"),
    }


def main() -> int:
    qa = _repo_qa_root()
    default_test_root = qa / "LLMs" / "safe_qa_outputs" / "test"
    default_out = Path(__file__).resolve().parent / "outputs"

    ap = argparse.ArgumentParser(
        description="task3 vs task3_stepwise_cot 최종 성능(exact_match 등) 비교"
    )
    ap.add_argument("--test-root", type=Path, default=default_test_root)
    ap.add_argument("--out", type=Path, default=default_out)
    ap.add_argument("--paired-model", type=str, default=None)
    ap.add_argument(
        "--paired-variant",
        type=str,
        choices=["single_step", "multi_step", "both"],
        default="both",
    )
    args = ap.parse_args()

    test_root: Path = args.test_root
    if not test_root.is_dir():
        print(f"ERROR: --test-root not found: {test_root}", file=sys.stderr)
        return 1

    discovered = discover_cot_predictions(test_root)
    if not discovered:
        print(
            f"ERROR: no task3_stepwise_cot predictions under {test_root}",
            file=sys.stderr,
        )
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for variant, model, cot_path in discovered:
        t3_path = test_root / "task3" / "both_repre" / variant / "results" / f"predictions_{model}.jsonl"
        if not t3_path.is_file():
            summary_rows.append(
                {
                    "model": model,
                    "prompt_variant": variant,
                    "status": "missing_task3",
                    "task3_path": "",
                    "cot_path": str(cot_path),
                }
            )
            continue

        t3 = load_task_predictions(t3_path)
        tc = load_task_predictions(cot_path)
        paired = merge_all(t3, tc)
        stat = summarize_final(paired)
        n_paired = int(stat.pop("n", 0))
        row: dict[str, Any] = {
            "model": model,
            "prompt_variant": variant,
            "status": "ok",
            "n_task3": len(t3),
            "n_cot": len(tc),
            "n_paired": n_paired,
        }
        row.update(stat)
        if paired:
            row["meta_mismatch_rate"] = sum(r["meta_mismatch"] for r in paired) / len(paired)
        else:
            row["meta_mismatch_rate"] = float("nan")
        row["task3_path"] = str(t3_path)
        row["cot_path"] = str(cot_path)
        summary_rows.append(row)

        do_paired = args.paired_model == model and (
            args.paired_variant == "both" or args.paired_variant == variant
        )
        if do_paired and paired:
            paired_path = (
                args.out / f"task3_vs_task3_cot_paired_{model.replace('.', '_')}_{variant}.csv"
            )
            fieldnames = list(paired[0].keys())
            with paired_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(paired)
            print(f"Wrote paired: {paired_path}")

    if summary_rows:
        keys: set[str] = set()
        for r in summary_rows:
            keys.update(r.keys())
        preferred = [
            "model",
            "prompt_variant",
            "status",
            "n_paired",
            "n_task3",
            "n_cot",
            "mean_task3_exact_match",
            "mean_cot_exact_match",
            "mean_delta_cot_minus_task3",
            "agreement_exact_match_01",
            "count_both_em1",
            "count_both_em0",
            "count_task3_em1_cot_em0",
            "count_task3_em0_cot_em1",
            "mean_task3_bleu",
            "mean_cot_bleu",
            "mean_delta_bleu_cot_minus_task3",
            "mean_task3_validity",
            "mean_cot_validity",
            "meta_mismatch_rate",
            "task3_path",
            "cot_path",
        ]
        fieldnames = [c for c in preferred if c in keys] + sorted(
            c for c in keys if c not in preferred
        )

        out_csv = args.out / "task3_vs_task3_cot_summary.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Wrote: {out_csv}")

    if not args.paired_model:
        print(
            "\nTip: 샘플 단위 CSV는 "
            "--paired-model gpt-4o --paired-variant both 로 저장 가능합니다."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
