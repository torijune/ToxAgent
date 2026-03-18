from __future__ import annotations

"""
SIDER 데이터(molecularACE_ver/sider.csv)에 대해
MolecularACE 스타일 pairing(ECFP/Scaffold/SMILES) 전 과정을 수행하는 스크립트.

입력 형식 (sider.csv)
--------------------
  X,Task,Y
  C(CNCCNCCNCCN)N,Hepatobiliary disorders,1
  ...

- X    : SMILES 문자열
- Task : endpoint (SIDER category)
- Y    : 1 → toxic, 0 → nontoxic 로 간주

처리 개요
--------
1) sider.csv를 읽어 endpoint(Task)별로 그룹화
2) 각 endpoint마다:
   - dataset  = "sider"
   - endpoint = Task 문자열
   - Y==1 인 X → toxic_smiles 리스트
   - Y==0 인 X → nontoxic_smiles 리스트
3) molecular_ace_pairing.process_endpoint 를 호출하여
   - ECFP w/o chirality, w/ chirality
   - Bemis-Murcko scaffold ECFP
   - SMILES Levenshtein 기반 유사도를 계산하고
   - FG / Stereo / Isomer / All 스타일별 pair를 생성
   - sim_matrices/<endpoint>.npz 로 similarity 행렬 저장
4) 모든 endpoint 에 대해:
   - molecularACE_ver/pairs_sider.csv
   - molecularACE_ver/mol_fg_sider/pairs.csv
   - molecularACE_ver/mol_stereo_sider/pairs.csv
   - molecularACE_ver/mol_isomer_sider/pairs.csv
   - molecularACE_ver/pair_counts_sider.csv
   를 생성 (molecular_ace_pairing.py 출력 구조와 유사).
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # molecularACE_ver 루트
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molecular_ace_pairing import process_endpoint  # type: ignore


SIDER_CSV = SCRIPT_DIR / "sider.csv"

# 출력 경로 (herg/metabolism과 겹치지 않도록 *_sider suffix 사용)
OUT_DIR = SCRIPT_DIR
PAIRS_CSV = OUT_DIR / "pairs_sider.csv"
PAIR_COUNTS_CSV = OUT_DIR / "pair_counts_sider.csv"
SIM_MATRICES_DIR = OUT_DIR / "sim_matrices_sider"
MOL_FG_DIR = OUT_DIR / "mol_fg_sider"
MOL_STEREO_DIR = OUT_DIR / "mol_stereo_sider"
MOL_ISOMER_DIR = OUT_DIR / "mol_isomer_sider"


def _load_sider() -> pd.DataFrame:
    """
    sider.csv를 읽어 (smiles, endpoint, label) 형식으로 반환.
    """
    if not SIDER_CSV.exists():
        raise FileNotFoundError(f"SIDER CSV not found: {SIDER_CSV}")
    df = pd.read_csv(SIDER_CSV)
    for col in ["X", "Task", "Y"]:
        if col not in df.columns:
            raise ValueError(f"sider.csv must have columns X, Task, Y. Found: {list(df.columns)}")
    df = df.rename(columns={"X": "smiles", "Task": "endpoint", "Y": "label"})
    # 결측 제거
    df = df.dropna(subset=["smiles", "endpoint", "label"])
    return df


def run_pairing_for_sider(canonicalize_smiles: bool = True) -> None:
    """
    sider.csv 에 대해 MolecularACE 스타일 pairing 수행.

    - dataset_name: "sider"
    - endpoint    : Task (SIDER category)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for d in (MOL_FG_DIR, MOL_STEREO_DIR, MOL_ISOMER_DIR, SIM_MATRICES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = _load_sider()

    all_rows: List[Dict] = []
    fg_rows: List[Dict] = []
    stereo_rows: List[Dict] = []
    isomer_rows: List[Dict] = []
    count_rows: List[Dict] = []

    dataset_name = "sider"

    for endpoint, grp in df.groupby("endpoint", dropna=False):
        g = grp.copy()
        smiles_toxic = (
            g[g["label"] == 1]["smiles"].dropna().astype(str).str.strip().tolist()
        )
        smiles_nontoxic = (
            g[g["label"] == 0]["smiles"].dropna().astype(str).str.strip().tolist()
        )

        # 중복 제거
        smiles_toxic = sorted({s for s in smiles_toxic if s})
        smiles_nontoxic = sorted({s for s in smiles_nontoxic if s})

        if not smiles_toxic or not smiles_nontoxic:
            print(
                f"[SIDER] Skip endpoint={endpoint!r}: "
                f"n_toxic={len(smiles_toxic)}, n_nontoxic={len(smiles_nontoxic)}"
            )
            continue

        print(
            f"[SIDER] Pairing endpoint={endpoint!r}: "
            f"n_toxic={len(smiles_toxic)}, n_nontoxic={len(smiles_nontoxic)}"
        )

        sim_npz_path = SIM_MATRICES_DIR / f"{Path(str(endpoint)).stem}.npz"
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
            endpoint=str(endpoint),
            toxic_smiles=smiles_toxic,
            nontoxic_smiles=smiles_nontoxic,
            save_sim_path=sim_npz_path,
            canonicalize_smiles=canonicalize_smiles,
        )

        count_rows.append(
            {
                "dataset": dataset_name,
                "endpoint": endpoint,
                "n_pairs": n_pairs,
                "n_toxic": len(smiles_toxic),
                "n_nontoxic": len(smiles_nontoxic),
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
            df_out = pd.DataFrame(rows).drop_duplicates()
        else:
            df_out = pd.DataFrame(
                columns=["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"]
            )
        df_out.to_csv(path, index=False)
        print(f"[SIDER] Saved {name}: {path} ({len(df_out):,} rows)")

    _save_pairs(PAIRS_CSV, all_rows, "all pairs")
    _save_pairs(MOL_FG_DIR / "pairs.csv", fg_rows, "FG pairs")
    _save_pairs(MOL_STEREO_DIR / "pairs.csv", stereo_rows, "Stereo pairs")
    _save_pairs(MOL_ISOMER_DIR / "pairs.csv", isomer_rows, "Isomer pairs")

    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(PAIR_COUNTS_CSV, index=False)
    print(f"[SIDER] Saved pair counts: {PAIR_COUNTS_CSV} ({len(df_counts):,} endpoints)")
    if not df_counts.empty:
        print(f"[SIDER] Total pairs (all): {int(df_counts['n_pairs'].sum()):,}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run MolecularACE-style pairing for SIDER (molecularACE_ver/sider.csv)."
        )
    )
    ap.add_argument(
        "--no_canonicalize_smiles",
        action="store_true",
        help="SMILES canonicalization을 비활성화하고 원본 SMILES로 pairing을 수행합니다.",
    )
    args = ap.parse_args()
    run_pairing_for_sider(canonicalize_smiles=not args.no_canonicalize_smiles)


if __name__ == "__main__":
    main()

