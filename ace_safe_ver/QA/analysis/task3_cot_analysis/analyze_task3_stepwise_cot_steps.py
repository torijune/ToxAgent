#!/usr/bin/env python3
"""
task3_stepwise_cot: 최종 nontoxic SMILES EM(exact_match)이 0/1일 때
Step1·Step2 fragment EM 분포를 model·representation(both_repre)·prompt variant별로 집계한다.

데이터 기본 경로:
  ace_safe_ver/QA/LLMs/safe_qa_outputs/test/task3_stepwise_cot/both_repre/
    {single_step,multi_step}/results/predictions_<model>.jsonl

사용 예:
  python analyze_task3_stepwise_cot_steps.py
  python analyze_task3_stepwise_cot_steps.py --root /path/to/both_repre --out ./cot_step_tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _parse_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _em_to_bin(x: float | None) -> int | None:
    """0/1 또는 0.0/1.0 → 0 또는 1. None이면 None."""
    if x is None:
        return None
    if x >= 0.5:
        return 1
    return 0


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def aggregate_file(path: Path) -> dict[str, Any]:
    """단일 predictions jsonl에 대한 집계."""
    # final_em -> (s1, s2) joint counts
    joint: dict[int, dict[tuple[int, int], int]] = defaultdict(lambda: defaultdict(int))
    # final_em -> step1 counts
    s1_marginal: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    s2_marginal: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    n_skip = 0
    n_total = 0

    for row in iter_jsonl(path):
        n_total += 1
        final_em = _em_to_bin(_parse_float(row.get("exact_match")))
        s1 = _em_to_bin(_parse_float(row.get("step1_fragment_EM")))
        s2 = _em_to_bin(_parse_float(row.get("step2_fragment_EM")))

        if final_em is None:
            n_skip += 1
            continue
        if s1 is None or s2 is None:
            n_skip += 1
            continue

        joint[final_em][(s1, s2)] += 1
        s1_marginal[final_em][s1] += 1
        s2_marginal[final_em][s2] += 1

    return {
        "path": str(path),
        "n_total": n_total,
        "n_used": n_total - n_skip,
        "n_skip_missing": n_skip,
        "joint": {k: dict(v) for k, v in joint.items()},
        "s1_marginal": {k: dict(v) for k, v in s1_marginal.items()},
        "s2_marginal": {k: dict(v) for k, v in s2_marginal.items()},
    }


def joint_to_rows(
    model: str,
    variant: str,
    agg: dict[str, Any],
) -> list[dict[str, Any]]:
    """CSV 행: joint 분포를 long 형태로."""
    rows: list[dict[str, Any]] = []
    joint = agg.get("joint", {})
    for final_em in (0, 1):
        if final_em not in joint:
            continue
        total = sum(joint[final_em].values())
        for (s1, s2), cnt in sorted(joint[final_em].items()):
            rows.append(
                {
                    "model": model,
                    "prompt_variant": variant,
                    "final_exact_match": final_em,
                    "step1_fragment_EM": s1,
                    "step2_fragment_EM": s2,
                    "count": cnt,
                    "frac_within_final_em": round(cnt / total, 6) if total else 0.0,
                }
            )
    return rows


def marginal_rows(
    model: str,
    variant: str,
    agg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Step1 / Step2 단독 EM 비율 (final_em 그룹 안에서)."""
    rows: list[dict[str, Any]] = []
    for final_em in (0, 1):
        for step_name, key in (
            ("step1", "s1_marginal"),
            ("step2", "s2_marginal"),
        ):
            m = agg.get(key, {}).get(final_em, {})
            total = sum(m.values())
            for em_val in (0, 1):
                cnt = m.get(em_val, 0)
                rows.append(
                    {
                        "model": model,
                        "prompt_variant": variant,
                        "final_exact_match": final_em,
                        "step": step_name,
                        "fragment_EM": em_val,
                        "count": cnt,
                        "frac_within_final_em": round(cnt / total, 6) if total else 0.0,
                    }
                )
    return rows


def discover_models(root: Path) -> list[tuple[str, Path]]:
    """(model_name, predictions_path) — single_step / multi_step 각각."""
    out: list[tuple[str, Path]] = []
    for variant in ("single_step", "multi_step"):
        res_dir = root / variant / "results"
        if not res_dir.is_dir():
            continue
        for p in sorted(res_dir.glob("predictions_*.jsonl")):
            stem = p.stem  # predictions_gemini-3-flash
            if not stem.startswith("predictions_"):
                continue
            model = stem[len("predictions_") :]
            out.append((f"{variant}::{model}", p))
    return out


def print_human_summary(
    model: str,
    variant: str,
    agg: dict[str, Any],
) -> None:
    print(f"\n=== {model} | {variant} ===")
    print(f"  n_total={agg['n_total']} used={agg['n_used']} skip={agg['n_skip_missing']}")
    joint = agg.get("joint", {})
    for final_em in (0, 1):
        if final_em not in joint:
            continue
        total = sum(joint[final_em].values())
        label = "final EM=1 (success)" if final_em == 1 else "final EM=0 (fail)"
        print(f"  [{label}] n={total}")
        for (s1, s2), cnt in sorted(joint[final_em].items()):
            frac = cnt / total if total else 0.0
            print(f"    step1={s1}, step2={s2}: {cnt} ({frac:.1%})")


def main() -> int:
    repo_qa = Path(__file__).resolve().parents[2]
    default_root = (
        repo_qa
        / "LLMs"
        / "safe_qa_outputs"
        / "test"
        / "task3_stepwise_cot"
        / "both_repre"
    )

    ap = argparse.ArgumentParser(description="task3_stepwise_cot: final EM vs step1/2 EM 분포")
    ap.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="both_repre 루트 (single_step, multi_step 하위 포함)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="CSV 출력 디렉터리",
    )
    ap.add_argument("--quiet", action="store_true", help="표준 출력 요약 생략")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        return 1

    discovered = discover_models(root)
    if not discovered:
        print(f"ERROR: no predictions_*.jsonl under {root}/{{single_step,multi_step}}/results", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    all_joint_rows: list[dict[str, Any]] = []
    all_marginal_rows: list[dict[str, Any]] = []

    for key, path in discovered:
        variant, _, model = key.partition("::")
        agg = aggregate_file(path)
        if not args.quiet:
            print_human_summary(model, variant, agg)

        for r in joint_to_rows(model, variant, agg):
            all_joint_rows.append(r)
        for r in marginal_rows(model, variant, agg):
            all_marginal_rows.append(r)

    joint_csv = args.out / "task3_stepwise_cot_joint_step_em_by_final_em.csv"
    marg_csv = args.out / "task3_stepwise_cot_marginal_step_em_by_final_em.csv"

    if all_joint_rows:
        fieldnames = list(all_joint_rows[0].keys())
        with joint_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_joint_rows)

    if all_marginal_rows:
        fieldnames = list(all_marginal_rows[0].keys())
        with marg_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_marginal_rows)

    print(f"\nWrote: {joint_csv}")
    print(f"Wrote: {marg_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
