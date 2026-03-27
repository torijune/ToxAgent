#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case-study용 "best cases" 샘플을 고른다.

요구사항
--------
- 결과 저장 디렉터리: ace_safe_ver/QA/analysis/case_study/best_case (여기)
- 각 task 하위: both_repre에서 single_step, multi_step 각각 2개씩
- "여기에 있는 모델들": predictions_*.jsonl로 존재하는 모델(파일)별로 선택
- 정답을 맞춘(=correct==1 또는 gold==pred_answer) 샘플이 없는 경우: 패스
- optimal pair 선정 로직(select_optimal_pair.py)과 완전 동일하게 엄격하진 않지만,
  어느 정도 "이쁜" toxic/nontoxic pair를 선호하도록 스코어링해서 상위 샘플 선택

출력
----
각 predictions 파일에 대해 다음 경로에 JSONL 저장:
  ace_safe_ver/QA/analysis/case_study/best_case/selected/test/<task>/both_repre/<step>/best_cases_<model>.jsonl

한 줄 레코드에는:
- task/step/model/source_index
- question(해당 QA jsonl에서)
- gold/pred(+raw)
- merged_test.csv의 toxic/nontoxic SMILES/SAFE/fragment 필드 일부
- pretty_score(완화된 optimal score)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CASE_STUDY_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = CASE_STUDY_DIR.parent.parent.parent.parent

SAFE_QA_OUTPUTS = PROJECT_ROOT / "ace_safe_ver" / "QA" / "LLMs" / "safe_qa_outputs"
QA_DIR = PROJECT_ROOT / "ace_safe_ver" / "QA"
BEST_CASE_OUT_ROOT = SCRIPT_DIR / "selected"
MERGED_TEST_CSV = (
    PROJECT_ROOT
    / "ace_safe_ver"
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)


TASKS = ("task1", "task2", "task3")
STEPs = ("single_step", "multi_step")


def _normalize_answer(ans: Any) -> str:
    if isinstance(ans, dict):
        return str(ans.get("answer", "") or "").strip()
    return str(ans or "").strip()


def _count_dot_fragments(dot_separated: str) -> int:
    s = (dot_separated or "").strip()
    if not s:
        return 0
    return len([p for p in s.split(".") if p.strip()])


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    s = (smiles or "").strip()
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def _morgan_tanimoto(smiles1: str, smiles2: str, radius: int = 2, fp_size: int = 2048) -> Optional[float]:
    m1 = _mol_from_smiles(smiles1)
    m2 = _mol_from_smiles(smiles2)
    if m1 is None or m2 is None:
        return None
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size, includeChirality=True)
    fp1 = fpgen.GetFingerprint(m1)
    fp2 = fpgen.GetFingerprint(m2)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def _mcs_ratios(m1: Chem.Mol, m2: Chem.Mol, timeout_sec: int = 2) -> Tuple[float, float]:
    """MCS atom/bond ratio (간단/빠르게). 실패 시 0."""
    try:
        mcs = rdFMCS.FindMCS(
            [m1, m2],
            timeout=timeout_sec,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=False,
        )
        mcs_atoms = int(mcs.numAtoms or 0)
        mcs_bonds = int(mcs.numBonds or 0)
        max_atoms = max(int(m1.GetNumHeavyAtoms()), int(m2.GetNumHeavyAtoms()), 1)
        max_bonds = max(int(m1.GetNumBonds()), int(m2.GetNumBonds()), 1)
        return mcs_atoms / max_atoms, mcs_bonds / max_bonds
    except Exception:
        return 0.0, 0.0


@dataclass
class Pretty:
    score: float
    tanimoto: float
    mcs_atom_ratio: float
    mcs_bond_ratio: float
    atom_diff_abs: int
    bond_diff_abs: int
    ring_diff_abs: int


def _pretty_score_for_row(merged_row: Dict[str, str]) -> Optional[Pretty]:
    """
    select_optimal_pair.py와 유사한 점수지만 완화:
    - tanimoto를 최우선(유사해야 예쁨)
    - atom/bond diff는 너무 크면 패널티
    - MCS ratio로 'local change' 선호
    """
    tox = str(merged_row.get("toxic_safe_decoded_smiles", "") or "").strip()
    non = str(merged_row.get("nontoxic_safe_decoded_smiles", "") or "").strip()
    m1 = _mol_from_smiles(tox)
    m2 = _mol_from_smiles(non)
    if m1 is None or m2 is None:
        return None

    tan = _morgan_tanimoto(tox, non)
    if tan is None:
        return None

    # 완화된 필터: 너무 다르면 case study로 별로
    if tan < 0.62:
        return None

    tox_atoms = int(m1.GetNumHeavyAtoms())
    non_atoms = int(m2.GetNumHeavyAtoms())
    tox_bonds = int(m1.GetNumBonds())
    non_bonds = int(m2.GetNumBonds())
    atom_diff_abs = abs(tox_atoms - non_atoms)
    bond_diff_abs = abs(tox_bonds - non_bonds)
    ring_diff_abs = abs(int(m1.GetRingInfo().NumRings()) - int(m2.GetRingInfo().NumRings()))

    # MCS ratio (빠른 timeout)
    mcs_atom_ratio, mcs_bond_ratio = _mcs_ratios(m1, m2, timeout_sec=2)

    max_atoms = max(tox_atoms, non_atoms, 1)
    max_bonds = max(tox_bonds, non_bonds, 1)
    atom_diff_ratio = atom_diff_abs / max_atoms
    bond_diff_ratio = bond_diff_abs / max_bonds

    score = (
        0.55 * tan
        + 0.30 * mcs_atom_ratio
        + 0.15 * mcs_bond_ratio
        - 0.12 * atom_diff_ratio
        - 0.06 * bond_diff_ratio
        - 0.03 * float(ring_diff_abs)
    )
    return Pretty(
        score=float(score),
        tanimoto=float(tan),
        mcs_atom_ratio=float(mcs_atom_ratio),
        mcs_bond_ratio=float(mcs_bond_ratio),
        atom_diff_abs=int(atom_diff_abs),
        bond_diff_abs=int(bond_diff_abs),
        ring_diff_abs=int(ring_diff_abs),
    )


def _qa_path(task: str, step: str, repre: str = "both_repre") -> Path:
    """QA jsonl 경로(테스트 split 기준)."""
    base = QA_DIR / "test"
    if task == "task1":
        return base / "task1_toxic_fragment_identification" / repre / step / "task1_toxic_fragment_identification_qa.jsonl"
    if task == "task2":
        return base / "task2_nontoxic_fragment_generation" / repre / step / "task2_nontoxic_fragment_generation_qa.jsonl"
    # task3
    return base / "task3_nontoxic_smiles_generation" / repre / step / "task3_nontoxic_smiles_generation_qa.jsonl"


def _load_qa_by_source_index(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            si = obj.get("source_index", obj.get("id"))
            if si is None:
                continue
            try:
                out[int(si)] = obj
            except Exception:
                continue
    return out


def _load_merged_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _iter_prediction_files() -> List[Path]:
    out: List[Path] = []
    root = SAFE_QA_OUTPUTS / "test"
    for task in TASKS:
        for step in STEPs:
            p = root / task / "both_repre" / step / "results"
            if not p.is_dir():
                continue
            out.extend(sorted(p.glob("predictions_*.jsonl")))
    return out


def _model_slug_from_pred_path(pred_path: Path) -> str:
    name = pred_path.stem  # predictions_xxx
    return name.replace("predictions_", "", 1) if name.startswith("predictions_") else name


def _task_step_from_pred_path(pred_path: Path) -> Tuple[str, str]:
    # .../test/<task>/both_repre/<step>/results/predictions_*.jsonl
    parts = pred_path.parts
    idx = parts.index("test")
    task = parts[idx + 1]
    step = parts[idx + 3]  # both_repre, step
    return task, step


def select_and_write_for_predictions(
    pred_path: Path,
    merged_rows: List[Dict[str, str]],
    qa_cache: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]],
    *,
    k_per_step: int = 2,
) -> Optional[Path]:
    task, step = _task_step_from_pred_path(pred_path)
    model = _model_slug_from_pred_path(pred_path)

    qa_key = (task, step)
    if qa_key not in qa_cache:
        qa_cache[qa_key] = _load_qa_by_source_index(_qa_path(task, step, "both_repre"))
    qa_by_si = qa_cache[qa_key]

    scored_rows: List[Tuple[float, Dict[str, Any]]] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            si = row.get("source_index", row.get("id"))
            if si is None:
                continue
            try:
                si_int = int(si)
            except Exception:
                continue

            gold = _normalize_answer(row.get("gold", ""))
            pred_ans = _normalize_answer(row.get("pred", ""))
            correct_flag = row.get("correct")
            correct = bool(int(correct_flag)) if correct_flag is not None else (gold == pred_ans)
            if not correct:
                continue

            if si_int < 0 or si_int >= len(merged_rows):
                continue
            mrow = merged_rows[si_int]
            pretty = _pretty_score_for_row(mrow)
            if pretty is None:
                continue

            # 너무 복잡한 multi-fragment 케이스는 제외(완화된 "예쁨")
            if step == "single_step":
                if _count_dot_fragments(str(mrow.get("only_toxic_safe_fragments", ""))) != 1:
                    continue
                if _count_dot_fragments(str(mrow.get("only_nontoxic_safe_fragments", ""))) != 1:
                    continue

            qa_obj = qa_by_si.get(si_int, {})
            out_obj = {
                "task": task,
                "step": step,
                "model": model,
                "source_index": si_int,
                "question": qa_obj.get("question", ""),
                "gold": row.get("gold", ""),
                "pred": row.get("pred", ""),
                "raw": row.get("raw", ""),
                "correct": 1,
                "pretty": {
                    "score": pretty.score,
                    "tanimoto": pretty.tanimoto,
                    "mcs_atom_ratio": pretty.mcs_atom_ratio,
                    "mcs_bond_ratio": pretty.mcs_bond_ratio,
                    "atom_diff_abs": pretty.atom_diff_abs,
                    "bond_diff_abs": pretty.bond_diff_abs,
                    "ring_diff_abs": pretty.ring_diff_abs,
                },
                "merged": {
                    "dataset_name": mrow.get("dataset_name", ""),
                    "endpoint": mrow.get("endpoint", ""),
                    "toxic_safe": mrow.get("toxic_safe", ""),
                    "nontoxic_safe": mrow.get("nontoxic_safe", ""),
                    "toxic_smiles": mrow.get("toxic_smiles", ""),
                    "nontoxic_smiles": mrow.get("nontoxic_smiles", ""),
                    "toxic_safe_decoded_smiles": mrow.get("toxic_safe_decoded_smiles", ""),
                    "nontoxic_safe_decoded_smiles": mrow.get("nontoxic_safe_decoded_smiles", ""),
                    "only_toxic_safe_fragments": mrow.get("only_toxic_safe_fragments", ""),
                    "only_nontoxic_safe_fragments": mrow.get("only_nontoxic_safe_fragments", ""),
                },
            }
            scored_rows.append((pretty.score, out_obj))

    if not scored_rows:
        return None

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    picked = [obj for _, obj in scored_rows[:k_per_step]]

    # 저장 위치는 case_study/best_case 아래로 고정 (요청사항)
    out_dir = BEST_CASE_OUT_ROOT / "test" / task / "both_repre" / step
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_cases_{model}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for obj in picked:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Select best case-study QA samples from safe_qa_outputs predictions.")
    ap.add_argument("--k", type=int, default=2, help="step당 선택 개수 (기본 2)")
    ap.add_argument("--dry-run", action="store_true", help="파일 저장 없이 선택 결과만 출력")
    args = ap.parse_args()

    if not SAFE_QA_OUTPUTS.is_dir():
        raise SystemExit(f"safe_qa_outputs not found: {SAFE_QA_OUTPUTS}")
    if not MERGED_TEST_CSV.is_file():
        raise SystemExit(f"merged_test.csv not found: {MERGED_TEST_CSV}")

    merged_rows = _load_merged_rows(MERGED_TEST_CSV)
    qa_cache: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}

    pred_files = _iter_prediction_files()
    if not pred_files:
        raise SystemExit("predictions_*.jsonl not found under safe_qa_outputs/test/*")

    wrote = 0
    for p in pred_files:
        out = select_and_write_for_predictions(p, merged_rows, qa_cache, k_per_step=args.k)
        if out is None:
            continue
        wrote += 1
        if args.dry_run:
            print(f"[DRY] would write: {out}")
        else:
            print(f"wrote: {out}")

    print(f"done. wrote_files={wrote}")


if __name__ == "__main__":
    main()

