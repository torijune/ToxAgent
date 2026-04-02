#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merged_test.csv에서 Toxic/NonToxic pair 중
"구조적으로 매우 유사하고(local change), 국소 부위만 다른" 케이스를 점수화하여
상위 K개 optimal pair를 고릅니다.

선정 기준(기본):
1) Morgan Tanimoto similarity가 높을수록 좋음
2) MCS(Maximum Common Substructure) 비율이 높을수록 좋음
3) 원자/결합 차이 수가 적을수록 좋음

출력:
- 전체 유효 pair + 지표 + 점수 CSV
- 상위 K개 optimal pair CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS


SCRIPT_DIR = Path(__file__).resolve().parent
_CASE_STUDY = SCRIPT_DIR
if str(_CASE_STUDY) not in sys.path:
    sys.path.insert(0, str(_CASE_STUDY))
from paths_bundle import setup_bundle_paths  # noqa: E402

_ACE = setup_bundle_paths(with_safe_pkg=True)
from safe.safe.converter import decode as safe_decode  # noqa: E402

DEFAULT_INPUT = (
    _ACE
    / "splits"
    / "scaffold_by_endpoint_property_outlier_dropped_moved_many_to_train"
    / "merged_test.csv"
)
DEFAULT_OUTPUT_TOPK = SCRIPT_DIR / "optimal_pairs_top10.csv"
DEFAULT_OUTPUT_ALL = SCRIPT_DIR / "optimal_pairs_scored_all.csv"


def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of columns exist: {candidates}")


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


def _count_safe_fragments(safe_str: str) -> int:
    s = str(safe_str or "").strip()
    if not s:
        return 0
    return len([p for p in s.split(".") if p.strip()])


def _fragment_size_from_safe_fragment(safe_fragment: str) -> Optional[int]:
    """
    SAFE fragment 1개를 decode하여 heavy atom 수를 반환.
    decode 실패 시 None.
    """
    frag = str(safe_fragment or "").strip()
    if not frag:
        return None
    decoded = safe_decode(
        frag,
        as_mol=False,
        remove_dummies=False,
        ignore_errors=True,
    )
    if not decoded:
        return None
    mol = Chem.MolFromSmiles(str(decoded))
    if mol is None:
        return None
    return int(mol.GetNumHeavyAtoms())


def _pair_metrics(
    toxic_smiles: str,
    nontoxic_smiles: str,
    radius: int = 2,
    fp_size: int = 2048,
    mcs_timeout_sec: int = 5,
    prefilter_tanimoto: float = 0.70,
) -> Optional[Dict[str, float]]:
    tox = _mol_from_smiles(toxic_smiles)
    non = _mol_from_smiles(nontoxic_smiles)
    if tox is None or non is None:
        return None

    # 1) Fingerprint Tanimoto (빠른 필터링용 핵심 지표)
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size, includeChirality=True)
    fp_tox = fpgen.GetFingerprint(tox)
    fp_non = fpgen.GetFingerprint(non)
    tanimoto = float(DataStructs.TanimotoSimilarity(fp_tox, fp_non))

    # 저유사도 pair는 MCS(상대적으로 무거움) 계산을 생략해 속도 개선
    if tanimoto < prefilter_tanimoto:
        return {
            "tanimoto_chiral_morgan": tanimoto,
            "mcs_atoms": -1.0,
            "mcs_bonds": -1.0,
            "tox_heavy_atoms": float(tox.GetNumHeavyAtoms()),
            "non_heavy_atoms": float(non.GetNumHeavyAtoms()),
            "tox_bonds": float(tox.GetNumBonds()),
            "non_bonds": float(non.GetNumBonds()),
            "tox_ring_count": float(tox.GetRingInfo().NumRings()),
            "non_ring_count": float(non.GetRingInfo().NumRings()),
            "tox_rotb": float(AllChem.CalcNumRotatableBonds(tox)),
            "non_rotb": float(AllChem.CalcNumRotatableBonds(non)),
            "mcs_atom_ratio": 0.0,
            "mcs_bond_ratio": 0.0,
            "atom_diff_abs": float(abs(tox.GetNumHeavyAtoms() - non.GetNumHeavyAtoms())),
            "bond_diff_abs": float(abs(tox.GetNumBonds() - non.GetNumBonds())),
            "atom_diff_ratio": 1.0,
            "bond_diff_ratio": 1.0,
            "optimal_score": -1.0,
        }

    # 2) MCS 기반 local 차이 측정
    mcs = rdFMCS.FindMCS(
        [tox, non],
        timeout=mcs_timeout_sec,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        matchValences=False,
    )
    mcs_atoms = int(mcs.numAtoms or 0)
    mcs_bonds = int(mcs.numBonds or 0)

    tox_atoms = int(tox.GetNumHeavyAtoms())
    non_atoms = int(non.GetNumHeavyAtoms())
    tox_bonds = int(tox.GetNumBonds())
    non_bonds = int(non.GetNumBonds())
    tox_ring_count = int(tox.GetRingInfo().NumRings())
    non_ring_count = int(non.GetRingInfo().NumRings())
    tox_rotb = int(AllChem.CalcNumRotatableBonds(tox))
    non_rotb = int(AllChem.CalcNumRotatableBonds(non))

    max_atoms = max(tox_atoms, non_atoms, 1)
    max_bonds = max(tox_bonds, non_bonds, 1)

    mcs_atom_ratio = mcs_atoms / max_atoms
    mcs_bond_ratio = mcs_bonds / max_bonds

    atom_diff_abs = abs(tox_atoms - non_atoms)
    bond_diff_abs = abs(tox_bonds - non_bonds)
    atom_diff_ratio = atom_diff_abs / max_atoms
    bond_diff_ratio = bond_diff_abs / max_bonds

    # 3) 최종 점수: 유사도/공통부분은 +, 차이비율은 -
    optimal_score = (
        0.50 * tanimoto
        + 0.35 * mcs_atom_ratio
        + 0.15 * mcs_bond_ratio
        - 0.10 * atom_diff_ratio
        - 0.05 * bond_diff_ratio
    )

    return {
        "tanimoto_chiral_morgan": tanimoto,
        "mcs_atoms": float(mcs_atoms),
        "mcs_bonds": float(mcs_bonds),
        "tox_heavy_atoms": float(tox_atoms),
        "non_heavy_atoms": float(non_atoms),
        "tox_bonds": float(tox_bonds),
        "non_bonds": float(non_bonds),
        "tox_ring_count": float(tox_ring_count),
        "non_ring_count": float(non_ring_count),
        "tox_rotb": float(tox_rotb),
        "non_rotb": float(non_rotb),
        "mcs_atom_ratio": mcs_atom_ratio,
        "mcs_bond_ratio": mcs_bond_ratio,
        "atom_diff_abs": float(atom_diff_abs),
        "bond_diff_abs": float(bond_diff_abs),
        "atom_diff_ratio": atom_diff_ratio,
        "bond_diff_ratio": bond_diff_ratio,
        "optimal_score": optimal_score,
    }


def select_optimal_pairs(
    input_csv: Path,
    top_k: int = 10,
    min_tanimoto: float = 0.70,
    max_tanimoto: float = 0.90,
    min_atom_diff_abs: int = 1,
    require_single_diff_fragment: bool = True,
    min_fragment_size: int = 4,
    max_fragment_size: int = 20,
    max_heavy_atoms: Optional[int] = None,
    max_ring_count: Optional[int] = None,
    max_rotb: Optional[int] = None,
    ring_diff_abs_exact: Optional[int] = None,
    aromatic_ring_diff_abs_exact: Optional[int] = None,
    min_atom_diff_ratio: Optional[float] = None,
    max_atom_diff_ratio: Optional[float] = None,
    radius: int = 2,
    fp_size: int = 2048,
    mcs_timeout_sec: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_csv)

    tox_col = _first_existing_column(
        df,
        ["toxic_canonical_smiles", "toxic_safe_decoded_smiles", "toxic_smiles"],
    )
    non_col = _first_existing_column(
        df,
        ["nontoxic_canonical_smiles", "nontoxic_safe_decoded_smiles", "nontoxic_smiles"],
    )
    only_tox_frag_col = _first_existing_column(df, ["only_toxic_safe_fragments"])
    only_non_frag_col = _first_existing_column(df, ["only_nontoxic_safe_fragments"])

    rows = []
    for idx, row in df.iterrows():
        tox_s = str(row.get(tox_col, "") or "").strip()
        non_s = str(row.get(non_col, "") or "").strip()
        tox_frag = str(row.get(only_tox_frag_col, "") or "").strip()
        non_frag = str(row.get(only_non_frag_col, "") or "").strip()

        tox_frag_count = _count_safe_fragments(tox_frag)
        non_frag_count = _count_safe_fragments(non_frag)
        tox_frag_size = _fragment_size_from_safe_fragment(tox_frag) if tox_frag_count == 1 else None
        non_frag_size = _fragment_size_from_safe_fragment(non_frag) if non_frag_count == 1 else None

        if require_single_diff_fragment:
            if tox_frag_count != 1 or non_frag_count != 1:
                continue
            if tox_frag_size is None or non_frag_size is None:
                continue
            if not (min_fragment_size <= tox_frag_size <= max_fragment_size):
                continue
            if not (min_fragment_size <= non_frag_size <= max_fragment_size):
                continue

        metrics = _pair_metrics(
            tox_s,
            non_s,
            radius=radius,
            fp_size=fp_size,
            mcs_timeout_sec=mcs_timeout_sec,
            prefilter_tanimoto=min_tanimoto,
        )
        if metrics is None:
            continue
        if metrics["tanimoto_chiral_morgan"] < min_tanimoto:
            continue
        if metrics["tanimoto_chiral_morgan"] > max_tanimoto:
            continue
        if metrics["atom_diff_abs"] < float(min_atom_diff_abs):
            continue
        if max_heavy_atoms is not None:
            if metrics["tox_heavy_atoms"] > float(max_heavy_atoms):
                continue
            if metrics["non_heavy_atoms"] > float(max_heavy_atoms):
                continue
        if max_ring_count is not None:
            if metrics["tox_ring_count"] > float(max_ring_count):
                continue
            if metrics["non_ring_count"] > float(max_ring_count):
                continue
        if max_rotb is not None:
            if metrics["tox_rotb"] > float(max_rotb):
                continue
            if metrics["non_rotb"] > float(max_rotb):
                continue
        ring_diff_abs = abs(metrics["tox_ring_count"] - metrics["non_ring_count"])
        if ring_diff_abs_exact is not None and ring_diff_abs != float(ring_diff_abs_exact):
            continue
        if aromatic_ring_diff_abs_exact is not None:
            tox_arom = float(row.get("toxic_AromRingCount", 0.0) or 0.0)
            non_arom = float(row.get("nontoxic_AromRingCount", 0.0) or 0.0)
            arom_diff_abs = abs(tox_arom - non_arom)
            if arom_diff_abs != float(aromatic_ring_diff_abs_exact):
                continue
        if min_atom_diff_ratio is not None and metrics["atom_diff_ratio"] < float(min_atom_diff_ratio):
            continue
        if max_atom_diff_ratio is not None and metrics["atom_diff_ratio"] > float(max_atom_diff_ratio):
            continue

        out = {
            "row_index": int(idx),
            "dataset_name": row.get("dataset_name", ""),
            "endpoint": row.get("endpoint", ""),
            "toxic_smiles": tox_s,
            "nontoxic_smiles": non_s,
            "only_toxic_safe_fragment": tox_frag,
            "only_nontoxic_safe_fragment": non_frag,
            "only_toxic_fragment_count": float(tox_frag_count),
            "only_nontoxic_fragment_count": float(non_frag_count),
            "only_toxic_fragment_size": float(tox_frag_size) if tox_frag_size is not None else None,
            "only_nontoxic_fragment_size": float(non_frag_size) if non_frag_size is not None else None,
            "ring_diff_abs": float(ring_diff_abs),
        }
        out.update(metrics)
        rows.append(out)

    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored, scored

    # 상위 점수 우선 + 동점이면 더 local change(원자/결합 차이 적음) 우선
    scored = scored.sort_values(
        by=["optimal_score", "tanimoto_chiral_morgan", "atom_diff_abs", "bond_diff_abs"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    topk = scored.head(top_k).copy()
    topk.insert(0, "rank", range(1, len(topk) + 1))
    return scored, topk


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select top-K local-change optimal toxic/nontoxic pairs from merged_test.csv"
    )
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT, help="Input merged_test.csv path")
    ap.add_argument("--top-k", type=int, default=10, help="Number of optimal pairs to keep")
    ap.add_argument(
        "--min-tanimoto",
        type=float,
        default=0.70,
        help="Minimum Morgan(chiral) Tanimoto threshold before ranking",
    )
    ap.add_argument(
        "--max-tanimoto",
        type=float,
        default=0.90,
        help="Maximum Morgan(chiral) Tanimoto threshold before ranking",
    )
    ap.add_argument(
        "--min-atom-diff-abs",
        type=int,
        default=1,
        help="국소 변경이 실제 존재하도록 최소 heavy-atom 차이 수 (기본: 1)",
    )
    ap.add_argument(
        "--require-single-diff-fragment",
        action="store_true",
        default=True,
        help="only_toxic/only_nontoxic safe fragment가 각각 1개인 pair만 사용 (기본: True)",
    )
    ap.add_argument(
        "--min-fragment-size",
        type=int,
        default=4,
        help="차이나는 단일 fragment의 최소 heavy-atom 수 (기본: 4)",
    )
    ap.add_argument(
        "--max-fragment-size",
        type=int,
        default=20,
        help="차이나는 단일 fragment의 최대 heavy-atom 수 (기본: 20)",
    )
    ap.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=None,
        help="분자 복잡도 제한: toxic/nontoxic 각각 heavy-atom 최대값",
    )
    ap.add_argument(
        "--max-ring-count",
        type=int,
        default=None,
        help="분자 복잡도 제한: toxic/nontoxic 각각 ring count 최대값",
    )
    ap.add_argument(
        "--max-rotb",
        type=int,
        default=None,
        help="분자 복잡도 제한: toxic/nontoxic 각각 rotatable bonds 최대값",
    )
    ap.add_argument(
        "--ring-diff-abs-exact",
        type=int,
        default=None,
        help="toxic/nontoxic 전체 ring count 차이를 정확히 이 값으로 제한 (예: 1)",
    )
    ap.add_argument(
        "--aromatic-ring-diff-abs-exact",
        type=int,
        default=None,
        help="toxic/nontoxic aromatic ring count 차이를 정확히 이 값으로 제한 (예: 1)",
    )
    ap.add_argument(
        "--min-atom-diff-ratio",
        type=float,
        default=None,
        help="시각적 차이 최소화를 위한 atom_diff_ratio 하한",
    )
    ap.add_argument(
        "--max-atom-diff-ratio",
        type=float,
        default=None,
        help="시각적 차이 과대 방지를 위한 atom_diff_ratio 상한",
    )
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius")
    ap.add_argument("--fp-size", type=int, default=2048, help="Morgan fingerprint size")
    ap.add_argument("--mcs-timeout-sec", type=int, default=5, help="MCS timeout seconds per pair")
    ap.add_argument("--output-topk", type=Path, default=DEFAULT_OUTPUT_TOPK, help="Top-K output CSV path")
    ap.add_argument("--output-all", type=Path, default=DEFAULT_OUTPUT_ALL, help="Scored-all output CSV path")
    ap.add_argument(
        "--more-pairs",
        action="store_true",
        help=(
            "row858 스타일 후보를 더 넓게: top_k≥40, min_tanimoto=0.62, max_tanimoto=0.93 "
            "(다른 인자보다 우선 적용)"
        ),
    )
    args = ap.parse_args()

    top_k = args.top_k
    min_tanimoto = args.min_tanimoto
    max_tanimoto = args.max_tanimoto
    if args.more_pairs:
        top_k = max(top_k, 40)
        min_tanimoto = 0.62
        max_tanimoto = 0.93

    scored, topk = select_optimal_pairs(
        input_csv=args.input_csv,
        top_k=top_k,
        min_tanimoto=min_tanimoto,
        max_tanimoto=max_tanimoto,
        min_atom_diff_abs=args.min_atom_diff_abs,
        require_single_diff_fragment=args.require_single_diff_fragment,
        min_fragment_size=args.min_fragment_size,
        max_fragment_size=args.max_fragment_size,
        max_heavy_atoms=args.max_heavy_atoms,
        max_ring_count=args.max_ring_count,
        max_rotb=args.max_rotb,
        ring_diff_abs_exact=args.ring_diff_abs_exact,
        aromatic_ring_diff_abs_exact=args.aromatic_ring_diff_abs_exact,
        min_atom_diff_ratio=args.min_atom_diff_ratio,
        max_atom_diff_ratio=args.max_atom_diff_ratio,
        radius=args.radius,
        fp_size=args.fp_size,
        mcs_timeout_sec=args.mcs_timeout_sec,
    )

    args.output_all.parent.mkdir(parents=True, exist_ok=True)
    args.output_topk.parent.mkdir(parents=True, exist_ok=True)

    scored.to_csv(args.output_all, index=False)
    topk.to_csv(args.output_topk, index=False)

    print(f"Input: {args.input_csv}")
    if args.more_pairs:
        print("Mode: --more-pairs (top_k>=40, tanimoto in [0.62, 0.93])")
    print(f"Scored pairs: {len(scored)} -> {args.output_all}")
    print(f"Top-{len(topk)}: {len(topk)} -> {args.output_topk}")
    if not topk.empty:
        print("\nTop preview:")
        cols = ["rank", "row_index", "dataset_name", "endpoint", "tanimoto_chiral_morgan", "mcs_atom_ratio", "atom_diff_abs", "optimal_score"]
        show_cols = [c for c in cols if c in topk.columns]
        print(topk[show_cols].head(min(args.top_k, 10)).to_string(index=False))


if __name__ == "__main__":
    main()
