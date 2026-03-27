#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merged_test.csv에서 특정 row_index(기본 row_858)와 '스타일'이 비슷한 toxic/nontoxic pair 5개를 찾는다.

스타일 가정(필터):
- only_toxic_safe_fragments, only_nontoxic_safe_fragments 각각 fragment 개수가 1인 row만 (single fragment 변경 케이스 위주)
- (선택) only_*_safe_fragment를 decode해서 heavy atom 수가 reference 범위와 비슷한 row만 (기본 ±3)

유사도 점수:
- Morgan(chiral) Tanimoto similarity:
  - sim(toxic_candidate, toxic_ref)
  - sim(nontoxic_candidate, nontoxic_ref)
- 최종 score = (sim_tox + sim_non) / 2

출력:
1) 화면에 상위 K개 요약 출력
2) 이미지 생성에 바로 쓰기 좋은 CSV:
   - optimal_pair_fragment_highlight.py 가 요구하는 컬럼 스키마에 맞춰 생성
     (rank,row_index,dataset_name,endpoint,toxic_smiles,nontoxic_smiles,only_toxic_safe_fragment,only_nontoxic_safe_fragment,...)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safe.safe.converter import decode as safe_decode  # noqa: E402
import datamol as dm  # noqa: E402


DEFAULT_MERGED_TEST_CSV = (
    PROJECT_ROOT
    / "ace_safe_ver"
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)
DEFAULT_OUT_CSV = SCRIPT_DIR / "optimal_pairs_similar_to_row858_top5.csv"


def _count_dot_fragments(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 0
    return len([p for p in s.split(".") if p.strip()])


def _safe_to_mol(smiles_or_safe: str, *, is_safe: bool) -> Optional[Chem.Mol]:
    x = (smiles_or_safe or "").strip()
    if not x:
        return None
    try:
        if is_safe:
            decoded = safe_decode(x, as_mol=False, remove_dummies=True, ignore_errors=True)
            if not decoded:
                return None
            x = str(decoded).strip()
        return Chem.MolFromSmiles(x)
    except Exception:
        return None


def _morgan_tanimoto(smiles1: str, smiles2: str, *, radius: int = 2, fp_size: int = 2048) -> Optional[float]:
    m1 = Chem.MolFromSmiles(smiles1)
    m2 = Chem.MolFromSmiles(smiles2)
    if m1 is None or m2 is None:
        return None
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size, includeChirality=True)
    fp1 = fpgen.GetFingerprint(m1)
    fp2 = fpgen.GetFingerprint(m2)
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def _fragment_heavy_atom_count(fragment_safe: str) -> Optional[int]:
    frag = (fragment_safe or "").strip()
    if not frag:
        return None
    decoded = safe_decode(frag, as_mol=False, remove_dummies=True, ignore_errors=True)
    if not decoded:
        return None
    mol = Chem.MolFromSmiles(str(decoded).strip())
    if mol is None:
        return None
    return int(mol.GetNumHeavyAtoms())


def _read_merged_row_by_index(merged_csv: Path, row_index: int) -> Dict[str, str]:
    with merged_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == row_index:
                return row
    raise SystemExit(f"row_index={row_index} not found in {merged_csv}")


@dataclass
class Candidate:
    row_index: int
    dataset_name: str
    endpoint: str
    toxic_smiles: str
    nontoxic_smiles: str
    only_toxic_frag: str
    only_nontoxic_frag: str
    sim_tox: float
    sim_non: float
    score: float


def main() -> None:
    ap = argparse.ArgumentParser(description="Find similar toxic/nontoxic pairs to a given row_index.")
    ap.add_argument("--merged-csv", type=Path, default=DEFAULT_MERGED_TEST_CSV)
    ap.add_argument("--ref-row-index", type=int, default=858)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument(
        "--max-frag-heavy-atom-delta",
        type=int,
        default=3,
        help="reference only fragments heavy atom count 기준 ±N로 필터",
    )
    ap.add_argument(
        "--require-single-fragment-change",
        action="store_true",
        default=True,
        help="only_toxic_safe_fragments, only_nontoxic_safe_fragments 각각 fragment=1만 허용",
    )
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fp-size", type=int, default=2048)
    args = ap.parse_args()

    ref = _read_merged_row_by_index(args.merged_csv, args.ref_row_index)
    ref_tox_smiles = str(ref.get("toxic_safe_decoded_smiles", "") or "").strip()
    ref_non_smiles = str(ref.get("nontoxic_safe_decoded_smiles", "") or "").strip()
    if not ref_tox_smiles or not ref_non_smiles:
        raise SystemExit("ref toxic/nontoxic smiles decode가 비어 있습니다.")

    ref_only_tox = str(ref.get("only_toxic_safe_fragments", "") or "").strip()
    ref_only_non = str(ref.get("only_nontoxic_safe_fragments", "") or "").strip()

    ref_tox_frag_cnt = _count_dot_fragments(ref_only_tox)
    ref_non_frag_cnt = _count_dot_fragments(ref_only_non)

    ref_heavy_tox = _fragment_heavy_atom_count(ref_only_tox)
    ref_heavy_non = _fragment_heavy_atom_count(ref_only_non)

    if ref_heavy_tox is None or ref_heavy_non is None:
        print("[WARN] reference only fragments heavy atom count decode 실패. (heavy atom 필터는 비활성화)", file=sys.stderr)

    candidates: List[Candidate] = []

    with args.merged_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i == args.ref_row_index:
                continue

            toxic_smiles = str(row.get("toxic_safe_decoded_smiles", "") or "").strip()
            nontoxic_smiles = str(row.get("nontoxic_safe_decoded_smiles", "") or "").strip()
            if not toxic_smiles or not nontoxic_smiles:
                continue

            if args.require_single_fragment_change:
                only_tox_frag = str(row.get("only_toxic_safe_fragments", "") or "").strip()
                only_non_frag = str(row.get("only_nontoxic_safe_fragments", "") or "").strip()
                if _count_dot_fragments(only_tox_frag) != 1 or _count_dot_fragments(only_non_frag) != 1:
                    continue
            else:
                only_tox_frag = str(row.get("only_toxic_safe_fragments", "") or "").strip()
                only_non_frag = str(row.get("only_nontoxic_safe_fragments", "") or "").strip()

            sim_tox = _morgan_tanimoto(ref_tox_smiles, toxic_smiles, radius=args.radius, fp_size=args.fp_size)
            sim_non = _morgan_tanimoto(ref_non_smiles, nontoxic_smiles, radius=args.radius, fp_size=args.fp_size)
            if sim_tox is None or sim_non is None:
                continue
            score = (sim_tox + sim_non) / 2.0

            # heavy atom delta 필터(선택)
            if (
                ref_heavy_tox is not None
                and ref_heavy_non is not None
                and _count_dot_fragments(only_tox_frag) == 1
                and _count_dot_fragments(only_non_frag) == 1
            ):
                cand_heavy_tox = _fragment_heavy_atom_count(only_tox_frag)
                cand_heavy_non = _fragment_heavy_atom_count(only_non_frag)
                if cand_heavy_tox is None or cand_heavy_non is None:
                    continue
                if abs(cand_heavy_tox - ref_heavy_tox) > args.max_frag_heavy_atom_delta:
                    continue
                if abs(cand_heavy_non - ref_heavy_non) > args.max_frag_heavy_atom_delta:
                    continue

            candidates.append(
                Candidate(
                    row_index=i,
                    dataset_name=str(row.get("dataset_name", "") or "").strip(),
                    endpoint=str(row.get("endpoint", "") or "").strip(),
                    toxic_smiles=toxic_smiles,
                    nontoxic_smiles=nontoxic_smiles,
                    only_toxic_frag=only_tox_frag,
                    only_nontoxic_frag=only_non_frag,
                    sim_tox=float(sim_tox),
                    sim_non=float(sim_non),
                    score=float(score),
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    top = candidates[: args.topk]

    if not top:
        raise SystemExit("후보를 찾지 못했습니다. 필터를 완화해 주세요.")

    print(f"Reference row_index={args.ref_row_index}")
    print(f"  only_toxic fragment count={ref_tox_frag_cnt}, only_nontoxic fragment count={ref_non_frag_cnt}")
    print(f"  ref heavy atom counts: tox={ref_heavy_tox}, non={ref_heavy_non}")
    print()
    for idx, c in enumerate(top, 1):
        print(
            f"{idx}. row_index={c.row_index} score={c.score:.4f} "
            f"sim_tox={c.sim_tox:.4f} sim_non={c.sim_non:.4f} "
            f"dataset={c.dataset_name} endpoint={c.endpoint}"
        )

    # output CSV for image generation
    out_path = args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # optimal_pair_fragment_highlight.py가 쓰는 컬럼 위주(추가 컬럼은 ignore)
        writer.writerow(
            [
                "rank",
                "row_index",
                "dataset_name",
                "endpoint",
                "toxic_smiles",
                "nontoxic_smiles",
                "only_toxic_safe_fragment",
                "only_nontoxic_safe_fragment",
                "only_toxic_fragment_count",
                "only_nontoxic_fragment_count",
                "optimal_score",
            ]
        )
        for k, c in enumerate(top, 1):
            writer.writerow(
                [
                    k,
                    c.row_index,
                    c.dataset_name,
                    c.endpoint,
                    c.toxic_smiles,
                    c.nontoxic_smiles,
                    c.only_toxic_frag,
                    c.only_nontoxic_frag,
                    float(_count_dot_fragments(c.only_toxic_frag)),
                    float(_count_dot_fragments(c.only_nontoxic_frag)),
                    c.score,
                ]
            )

    print(f"\nSaved CSV for visualization: {out_path}")


if __name__ == "__main__":
    main()

