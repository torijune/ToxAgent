from __future__ import annotations

"""
새로 추가된 metabolism CYP 데이터(.tab)에 대해
MolecularACE 스타일 pairing(ECFP/Scaffold/SMILES) 전 과정을 수행하는 스크립트.

입력 형식 (예: cyp1a2_veith.tab)
--------------------------------
  Drug_ID    Drug                        Y
  6602638    "CCCC(=O)Nc1ccc(...)"       0 or 1

- Drug: SMILES 문자열
- Y   : 1 → toxic, 0 → nontoxic 로 간주

처리 개요
--------
1) metabolism_ver/data/*.tab 를 모두 탐색
2) 각 파일마다:
   - endpoint = 파일 stem (예: cyp1a2_veith)
   - dataset  = "metabolism"
   - Y==1 인 Drug → toxic_smiles 리스트
   - Y==0 인 Drug → nontoxic_smiles 리스트
3) molecular_ace_pairing.process_endpoint 를 호출하여
   - ECFP w/o chirality, w/ chirality
   - Bemis-Murcko scaffold ECFP
   - SMILES Levenshtein 기반 유사도를 계산하고
   - FG / Stereo / Isomer / All 스타일별 pair를 생성
   - sim_matrices/<endpoint>.npz 로 similarity 행렬 저장
4) 모든 endpoint 에 대해:
   - metabolism_ver/pairs.csv
   - metabolism_ver/mol_fg/pairs.csv
   - metabolism_ver/mol_stereo/pairs.csv
   - metabolism_ver/mol_isomer/pairs.csv
   - metabolism_ver/pair_counts_cyps.csv
   를 생성 (molecular_ace_pairing.py 출력 구조와 유사).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# molecular_ace_pairing.py는 molecularACE_ver 루트에 있으므로 상위 디렉터리를 sys.path에 추가
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # molecularACE_ver
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molecular_ace_pairing import process_endpoint


DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR

# 출력 경로 (molecular_ace_pairing.py와 동일한 구조를 metabolism_ver 아래에 복제)
PAIRS_CSV = OUT_DIR / "pairs.csv"
PAIR_COUNTS_CSV = OUT_DIR / "pair_counts_cyps.csv"
SIM_MATRICES_DIR = OUT_DIR / "sim_matrices"
MOL_FG_DIR = OUT_DIR / "mol_fg"
MOL_STEREO_DIR = OUT_DIR / "mol_stereo"
MOL_ISOMER_DIR = OUT_DIR / "mol_isomer"


def _load_cyp_tab(path: Path) -> Tuple[list[str], list[str]]:
    """
    하나의 CYP .tab 파일에서 toxic / nontoxic SMILES 리스트를 추출.

    - 컬럼명 가정: Drug (SMILES), Y (0/1).
    - Y == 1  → toxic
    - Y == 0  → nontoxic
    """
    df = pd.read_csv(path, sep="\t")
    if "Drug" not in df.columns or "Y" not in df.columns:
        raise ValueError(f"{path} must have columns 'Drug' and 'Y'. Found: {df.columns.tolist()}")

    # 결측/비문자열은 드랍
    df = df.dropna(subset=["Drug", "Y"])

    toxic_df = df[df["Y"] == 1]
    nontoxic_df = df[df["Y"] == 0]

    toxic_smiles = [str(s).strip().strip('"') for s in toxic_df["Drug"].tolist() if str(s).strip()]
    nontoxic_smiles = [str(s).strip().strip('"') for s in nontoxic_df["Drug"].tolist() if str(s).strip()]

    return toxic_smiles, nontoxic_smiles


def run_pairing_for_cyps(canonicalize_smiles: bool = True) -> None:
    """
    metabolism_ver/data/*.tab 에 대해 MolecularACE 스타일 pairing 수행.

    - dataset_name: "metabolism"
    - endpoint    : 파일 stem (예: cyp1a2_veith)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in (MOL_FG_DIR, MOL_STEREO_DIR, MOL_ISOMER_DIR, SIM_MATRICES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    tab_files = sorted(DATA_DIR.glob("*.tab"))
    if not tab_files:
        raise FileNotFoundError(f"No .tab files found under {DATA_DIR}")

    all_rows: List[Dict] = []
    fg_rows: List[Dict] = []
    stereo_rows: List[Dict] = []
    isomer_rows: List[Dict] = []
    count_rows: List[Dict] = []

    dataset_name = "metabolism"

    for tab_path in tab_files:
        endpoint = tab_path.stem
        print(f"[CYP] Processing {tab_path.name} (endpoint={endpoint})")
        toxic_smiles, nontoxic_smiles = _load_cyp_tab(tab_path)
        if not toxic_smiles or not nontoxic_smiles:
            print(
                f"  - Skip: no toxic or nontoxic molecules "
                f"(n_toxic={len(toxic_smiles)}, n_nontoxic={len(nontoxic_smiles)})"
            )
            continue

        sim_npz_path = SIM_MATRICES_DIR / f"{endpoint}.npz"
        (
            _ds,
            _ep,
            rows_all,
            rows_fg,
            rows_stereo,
            rows_isomer,
            n_pairs,
        ) = process_endpoint(
            dataset=dataset_name,
            endpoint=endpoint,
            toxic_smiles=toxic_smiles,
            nontoxic_smiles=nontoxic_smiles,
            save_sim_path=sim_npz_path,
            canonicalize_smiles=canonicalize_smiles,
        )

        count_rows.append(
            {
                "dataset": dataset_name,
                "endpoint": endpoint,
                "n_pairs": n_pairs,
                "n_toxic": len(toxic_smiles),
                "n_nontoxic": len(nontoxic_smiles),
            }
        )

        def _add(rows_acc: List[Dict], pair_list: list[tuple[str, str]]) -> None:
            for s_t, s_n in pair_list:
                rows_acc.append(
                    {
                        "dataset_name": dataset_name,
                        "endpoint": endpoint,
                        "toxic_smiles": s_t,
                        "nontoxic_smiles": s_n,
                    }
                )

        _add(all_rows, rows_all)
        _add(fg_rows, rows_fg)
        _add(stereo_rows, rows_stereo)
        _add(isomer_rows, rows_isomer)

    def _save_pairs(path: Path, rows: List[Dict], name: str) -> None:
        if rows:
            df = pd.DataFrame(rows).drop_duplicates()
        else:
            df = pd.DataFrame(columns=["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"])
        df.to_csv(path, index=False)
        print(f"[CYP] Saved {name}: {path} ({len(df):,} rows)")

    _save_pairs(PAIRS_CSV, all_rows, "all pairs")
    _save_pairs(MOL_FG_DIR / "pairs.csv", fg_rows, "FG pairs")
    _save_pairs(MOL_STEREO_DIR / "pairs.csv", stereo_rows, "Stereo pairs")
    _save_pairs(MOL_ISOMER_DIR / "pairs.csv", isomer_rows, "Isomer pairs")

    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(PAIR_COUNTS_CSV, index=False)
    print(f"[CYP] Saved pair counts: {PAIR_COUNTS_CSV} ({len(df_counts):,} endpoints)")
    if not df_counts.empty:
        print(f"[CYP] Total pairs (all): {int(df_counts['n_pairs'].sum()):,}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run MolecularACE-style pairing for metabolism CYP .tab datasets "
            "under metabolism_ver/data."
        )
    )
    ap.add_argument(
        "--no_canonicalize_smiles",
        action="store_true",
        help="SMILES canonicalization을 비활성화하고 원본 SMILES로 pairing을 수행합니다.",
    )
    args = ap.parse_args()
    run_pairing_for_cyps(canonicalize_smiles=not args.no_canonicalize_smiles)


if __name__ == "__main__":
    main()

