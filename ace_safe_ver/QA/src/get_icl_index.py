#!/usr/bin/env python3
"""
Test 분자쌍(merged_test / merged_unseen_test 등)마다, Train merged_train의 분자쌍과
Toxic SMILES 기준 Tanimoto 유사도(Morgan FP, canonical toxic)로 가장 가까운 Top-K Train 행 index를 저장.

규칙
  1) 유사도: RDKit Morgan fingerprint + Tanimoto. Toxic SMILES는 MolToSmiles 로 canonical 후 FP 생성.
  2) Train 행 중 toxic_canonical == nontoxic_canonical 이면 ICL 후보에서 제외.
  3) (선택) query toxic 와 train toxic canonical 이 동일한 Train 행은 제외(동일 앵커 중복 방지).

출력: JSON (기본) — 이후 ICL QA 빌드에서 train_row_index 로 merged_train 행을 조회하면 됨.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

_QA_SRC = Path(__file__).resolve().parent
_ACE_VER = _QA_SRC.parent.parent  # ace_safe_ver
_SPLITS = _ACE_VER / "splits"


def _canonical_smiles(smiles: str | None) -> str | None:
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    s = str(smiles).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _build_fp_list(
    smiles_list: list[str],
    radius: int,
    fp_size: int,
) -> tuple[list[Any | None], list[bool]]:
    """각 SMILES에 대해 Morgan FP; 실패 시 None."""
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fps: list[Any | None] = []
    ok: list[bool] = []
    for s in tqdm(smiles_list, desc="Train toxic FP (canonical)"):
        if not s:
            fps.append(None)
            ok.append(False)
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(None)
            ok.append(False)
            continue
        fps.append(fpgen.GetFingerprint(mol))
        ok.append(True)
    return fps, ok


def _top_k_train_indices(
    query_fp: Any,
    train_fp_compact: list[Any],
    valid_train_row_indices: list[int],
    train_tox_canon_by_row: list[str | None],
    query_tox_canon: str,
    top_k: int,
    exclude_same_toxic: bool,
) -> tuple[list[int], list[float]]:
    """compact train FP 리스트에 대해 유사도 상위 top_k → 원본 train 행 index."""
    if not train_fp_compact:
        return [], []

    sims = np.asarray(
        DataStructs.BulkTanimotoSimilarity(query_fp, train_fp_compact),
        dtype=np.float64,
    )

    if exclude_same_toxic:
        for j, tri in enumerate(valid_train_row_indices):
            tc = train_tox_canon_by_row[tri]
            if tc is not None and tc == query_tox_canon:
                sims[j] = -1.0

    k = min(top_k, len(sims))
    if k == 0:
        return [], []

    # 상위 k 인덱스 (유사도 내림차순)
    if len(sims) <= k:
        order = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k - 1)[:k]
        order = part[np.argsort(-sims[part])]

    out_idx: list[int] = []
    out_sim: list[float] = []
    for j in order[:k]:
        if sims[j] < 0:
            continue
        tri = valid_train_row_indices[int(j)]
        out_idx.append(tri)
        out_sim.append(float(sims[j]))
    return out_idx, out_sim


def _top_k_train_indices_unique_toxic(
    query_fp: Any,
    train_fp_compact: list[Any],
    valid_train_row_indices: list[int],
    train_tox_canon_by_row: list[str | None],
    query_tox_canon: str,
    top_k: int,
    exclude_same_toxic: bool,
) -> tuple[list[int], list[float]]:
    """
    상위 유사도 순으로 고르되, 이미 선택한 ICL과 **canonical toxic SMILES가 같은** train 행은 건너뜀.
    (few-shot에 동일 toxic 앵커가 중복되는 것 방지)
    """
    if not train_fp_compact:
        return [], []

    sims = np.asarray(
        DataStructs.BulkTanimotoSimilarity(query_fp, train_fp_compact),
        dtype=np.float64,
    )

    if exclude_same_toxic:
        for j, tri in enumerate(valid_train_row_indices):
            tc = train_tox_canon_by_row[tri]
            if tc is not None and tc == query_tox_canon:
                sims[j] = -1.0

    order = np.argsort(-sims)
    out_idx: list[int] = []
    out_sim: list[float] = []
    seen_toxic: set[str] = set()

    for j in order:
        if sims[j] < 0:
            continue
        tri = valid_train_row_indices[int(j)]
        tc = train_tox_canon_by_row[tri]
        if not tc:
            continue
        if tc in seen_toxic:
            continue
        seen_toxic.add(tc)
        out_idx.append(tri)
        out_sim.append(float(sims[j]))
        if len(out_idx) >= top_k:
            break

    return out_idx, out_sim


def _prepare_train(
    train_df: pd.DataFrame,
    radius: int,
    fp_size: int,
) -> dict[str, Any]:
    """Train 행별 canonical toxic/nontoxic, ICL 가능 여부, compact FP."""
    n = len(train_df)
    tox_c: list[str | None] = [None] * n
    non_c: list[str | None] = [None] * n
    icl_ok: list[bool] = [False] * n

    for i in range(n):
        row = train_df.iloc[i]
        tc = _canonical_smiles(row.get("toxic_smiles"))
        nc = _canonical_smiles(row.get("nontoxic_smiles"))
        tox_c[i] = tc
        non_c[i] = nc
        if tc and nc and tc != nc:
            icl_ok[i] = True

    # canonical toxic 로만 FP (요구사항)
    tox_for_fp = [tox_c[i] if icl_ok[i] else "" for i in range(n)]
    fps_all, fp_ok = _build_fp_list(tox_for_fp, radius=radius, fp_size=fp_size)

    valid_train_row_indices: list[int] = []
    train_fp_compact: list[Any] = []
    for i in range(n):
        if not icl_ok[i]:
            continue
        if not fp_ok[i] or fps_all[i] is None:
            continue
        valid_train_row_indices.append(i)
        train_fp_compact.append(fps_all[i])

    return {
        "tox_c": tox_c,
        "non_c": non_c,
        "icl_ok": icl_ok,
        "valid_train_row_indices": valid_train_row_indices,
        "train_fp_compact": train_fp_compact,
    }


def _run_job(
    name: str,
    train_path: Path,
    test_path: Path,
    top_k: int,
    radius: int,
    fp_size: int,
    exclude_same_toxic: bool,
) -> dict[str, Any]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for col in ("toxic_smiles", "nontoxic_smiles"):
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"{name}: missing column {col} in train or test")

    prep = _prepare_train(train_df, radius=radius, fp_size=fp_size)
    tox_c_train: list[str | None] = prep["tox_c"]
    valid_train_row_indices: list[int] = prep["valid_train_row_indices"]
    train_fp_compact: list[Any] = prep["train_fp_compact"]

    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)

    entries: list[dict[str, Any]] = []
    for test_i in tqdm(range(len(test_df)), desc=f"Query test [{name}]"):
        row = test_df.iloc[test_i]
        q_tc = _canonical_smiles(row.get("toxic_smiles"))
        if not q_tc:
            entries.append(
                {
                    "test_row_index": test_i,
                    "query_toxic_canonical": None,
                    "error": "invalid_or_unparsable_toxic_smiles",
                    "top_train_row_indices": [],
                    "similarities": [],
                }
            )
            continue
        q_mol = Chem.MolFromSmiles(q_tc)
        if q_mol is None:
            entries.append(
                {
                    "test_row_index": test_i,
                    "query_toxic_canonical": q_tc,
                    "error": "mol_from_canonical_failed",
                    "top_train_row_indices": [],
                    "similarities": [],
                }
            )
            continue
        q_fp = fpgen.GetFingerprint(q_mol)

        idxs, sims = _top_k_train_indices(
            query_fp=q_fp,
            train_fp_compact=train_fp_compact,
            valid_train_row_indices=valid_train_row_indices,
            train_tox_canon_by_row=tox_c_train,
            query_tox_canon=q_tc,
            top_k=top_k,
            exclude_same_toxic=exclude_same_toxic,
        )

        rec: dict[str, Any] = {
            "test_row_index": test_i,
            "query_toxic_canonical": q_tc,
            "top_train_row_indices": idxs,
            "similarities": sims,
        }
        if "source_index" in test_df.columns:
            v = row.get("source_index")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                try:
                    rec["source_index"] = int(v)
                except (TypeError, ValueError):
                    rec["source_index"] = v
        if "dataset_name" in test_df.columns:
            rec["dataset_name"] = row.get("dataset_name")
        if "endpoint" in test_df.columns:
            rec["endpoint"] = row.get("endpoint")
        entries.append(rec)

    return {
        "name": name,
        "train_csv": str(train_path.resolve()),
        "test_csv": str(test_path.resolve()),
        "n_train_rows": len(train_df),
        "n_train_icl_candidates": len(valid_train_row_indices),
        "n_test_rows": len(test_df),
        "top_k": top_k,
        "radius": radius,
        "fp_size": fp_size,
        "exclude_same_toxic": exclude_same_toxic,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ICL용 Train Top-K row index 검색")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", type=int, default=1024)
    parser.add_argument(
        "--keep_same_toxic",
        action="store_true",
        help="query와 train의 canonical toxic 이 같아도 후보에 포함 (기본: 동일 toxic train 행 제외)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_QA_SRC / "icl_train_topk_indices.json",
        help="출력 JSON 경로",
    )
    parser.add_argument(
        "--train",
        type=Path,
        nargs="*",
        default=None,
        help="지정 시 jobs 기본값 대신 (train, test) 쌍을 순서대로 두 개씩 전달",
    )
    args = parser.parse_args()
    exclude_same = not args.keep_same_toxic

    default_jobs = [
        {
            "name": "scaffold_property_outlier_moved_many",
            "train": _SPLITS
            / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
            / "merged_train.csv",
            "test": _SPLITS
            / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
            / "merged_test.csv",
        },
        {
            "name": "scaffold_property_outlier_unseen_test",
            "train": _SPLITS / "scaffold_by_endpoint_property_outlier_dropped" / "merged_train.csv",
            "test": _SPLITS / "scaffold_by_endpoint_property_outlier_dropped" / "merged_unseen_test.csv",
        },
    ]

    if args.train:
        pairs = list(zip(args.train[0::2], args.train[1::2]))
        jobs = [
            {"name": f"custom_{i}", "train": Path(t), "test": Path(te)}
            for i, (t, te) in enumerate(pairs)
        ]
    else:
        jobs = default_jobs

    payload: dict[str, Any] = {
        "meta": {
            "description": "Per test row: top-K train row indices in corresponding merged_train.csv",
            "top_k": args.top_k,
            "radius": args.radius,
            "fp_size": args.fp_size,
            "tanimoto": "Morgan + Tanimoto on canonical toxic SMILES",
            "train_pair_filter": "exclude train rows where canonical(toxic)==canonical(nontoxic)",
            "exclude_same_toxic_as_query": exclude_same,
        },
        "jobs": [],
    }

    for job in jobs:
        tp = Path(job["train"])
        sp = Path(job["test"])
        if not tp.is_file():
            raise FileNotFoundError(f"train not found: {tp}")
        if not sp.is_file():
            raise FileNotFoundError(f"test not found: {sp}")
        payload["jobs"].append(
            _run_job(
                name=job["name"],
                train_path=tp,
                test_path=sp,
                top_k=args.top_k,
                radius=args.radius,
                fp_size=args.fp_size,
                exclude_same_toxic=exclude_same,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_path}")
    for j in payload["jobs"]:
        print(
            f"  [{j['name']}] test={j['n_test_rows']}  train_icl_candidates={j['n_train_icl_candidates']}"
        )


if __name__ == "__main__":
    main()
