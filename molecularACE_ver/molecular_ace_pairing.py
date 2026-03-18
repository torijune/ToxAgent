"""
MolecularACE 스타일별 pairing: FG / Stereo / Isomer 각각의 조건으로 쌍을 생성합니다.

- Substructure: ECFP4 (R=2, 1024 bits) Tanimoto
  - FG:   w/o chirality sim ≥ 0.9
  - Stereo/Isomer: w/ chirality sim ≥ 0.9
- Scaffold: Bemis-Murcko scaffold → ECFP → Tanimoto. FG/Stereo/Isomer 모두 ≥ 0.9
- SMILES: Levenshtein 정규화 유사도. FG/Stereo/Isomer 모두 ≥ 0.9

Pairing: 3개 유사도 중 하나라도 ≥ 0.9 이면 pairing (OR).
- FG:   (sub_wo ≥ 0.9 OR scaffold ≥ 0.9 OR smiles ≥ 0.9)
- Stereo: (sub_w_chiral ≥ 0.9 OR scaffold ≥ 0.9 OR smiles ≥ 0.9)
- Isomer: (sub_w_chiral ≥ 0.9 OR scaffold ≥ 0.9 OR smiles ≥ 0.9) AND 동일 분자식

ToxCast (toxcast_df)는 pairing 대상에서 제외.

출력:
  pairs.csv, mol_fg/pairs.csv, mol_stereo/pairs.csv, mol_isomer/pairs.csv, pair_counts.csv
  sim_matrices/<dataset>/<safe_ep>.npz  # substructure_sim, substructure_sim_w_chiral, scaffold_sim, smiles_sim
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# MolecularACE 스펙
ECFP_RADIUS = 2
ECFP_SIZE = 1024
SIM_THRESHOLD = 0.9
# 배치 크기: SMILES는 toxic 청크 × 전체 nontoxic 행렬 (메모리 고려)
CHUNK_TOXIC_SMILES = 200
# Substructure/Scaffold는 행 단위 BulkTanimoto (이미 배치)
CHUNK_TOXIC_FP = 256

BASE = Path(__file__).resolve().parent.parent  # detoxicity_model
SCAFFOLD_SIM_DIR = BASE / "scaffold_sim"
SUMMARY_CSV = SCAFFOLD_SIM_DIR / "summary.csv"
OUT_DIR = Path(__file__).resolve().parent
PAIRS_CSV = OUT_DIR / "pairs.csv"
PAIR_COUNTS_CSV = OUT_DIR / "pair_counts.csv"
SIM_MATRICES_DIR = OUT_DIR / "sim_matrices"
# 스타일별 substructure: FG=w/o chirality 0.9, Stereo/Isomer=w/ chirality 0.9. Scaffold/SMILES는 공통 0.9
MOL_FG_DIR = OUT_DIR / "mol_fg"
MOL_STEREO_DIR = OUT_DIR / "mol_stereo"
MOL_ISOMER_DIR = OUT_DIR / "mol_isomer"

def canonicalize_smiles_list(
    smiles_list: list[str],
    *,
    isomeric: bool = True,
    keep_invalid: bool = True,
) -> list[str]:
    """RDKit canonical SMILES로 정규화. 파싱 실패 시 원본 유지(keep_invalid=True) 또는 빈 문자열."""
    out: list[str] = []
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            out.append("" if not keep_invalid else (smi if isinstance(smi, str) else ""))
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                out.append(smi if keep_invalid else "")
                continue
            out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric))
        except Exception:
            out.append(smi if keep_invalid else "")
    return out


def build_ecfp4_list(smiles_list: list[str], include_chirality: bool = False) -> list:
    """ECFP4 (Radius 2, 1024 bits) fingerprint 리스트. 실패 시 None."""
    fpgen = AllChem.GetMorganGenerator(
        radius=ECFP_RADIUS, fpSize=ECFP_SIZE, includeChirality=include_chirality
    )
    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                out.append(fpgen.GetFingerprint(mol))
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out


def _mol_formula(smiles: str) -> str:
    """SMILES -> 분자식. 실패 시 빈 문자열."""
    if not smiles or not isinstance(smiles, str):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return rdMolDescriptors.CalcMolFormula(mol) or ""
    except Exception:
        return ""


def build_scaffold_fp_list(smiles_list: list[str]) -> list:
    """Bemis-Murcko scaffold 추출 후 ECFP (R=2, 1024). 실패 시 None."""
    fpgen = AllChem.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_SIZE, includeChirality=False)
    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                out.append(None)
                continue
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
                out.append(None)
                continue
            fp = fpgen.GetFingerprint(scaffold)
            out.append(fp)
        except Exception:
            out.append(None)
    return out


def _smiles_similarity_batch(
    toxic_chunk: list[str],
    nontoxic_list: list[str],
) -> np.ndarray:
    """toxic_chunk × nontoxic_list 에 대한 SMILES 유사도 행렬 (1 - lev/max_len). shape (len(toxic_chunk), len(nontoxic_list))."""
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        return np.full((len(toxic_chunk), len(nontoxic_list)), np.nan, dtype=np.float32)
    # 거리 행렬 (int)
    dist = process.cdist(
        toxic_chunk,
        nontoxic_list,
        scorer=Levenshtein.distance,
        dtype=np.int32,
        workers=1,
    )
    len_t = np.array([len(s) for s in toxic_chunk], dtype=np.float32)
    len_n = np.array([len(s) for s in nontoxic_list], dtype=np.float32)
    max_len = np.maximum(len_t[:, None], len_n[None, :])
    np.maximum(max_len, 1.0, out=max_len)
    sim = 1.0 - (dist.astype(np.float32) / max_len)
    return sim


def process_endpoint(
    dataset: str,
    endpoint: str,
    toxic_smiles: list[str],
    nontoxic_smiles: list[str],
    save_sim_path: Path | None = None,
    canonicalize_smiles: bool = True,
) -> tuple[str, str, list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]], int]:
    """
    스타일별 pairing: FG(w/o chirality 0.9), Stereo/Isomer(w/ chirality 0.9), Scaffold/SMILES 공통 0.9.
    Isomer는 동일 분자식 AND 조건 추가.
    Returns: (dataset, endpoint, rows_all, rows_fg, rows_stereo, rows_isomer, n_all).
    """
    n_t, n_n = len(toxic_smiles), len(nontoxic_smiles)
    empty: list[tuple[str, str]] = []
    if n_t == 0 or n_n == 0:
        return dataset, endpoint, empty, empty, empty, empty, 0

    # SMILES canonicalization (문자열 비교/중복 제거 안정화)
    if canonicalize_smiles:
        toxic_smiles = canonicalize_smiles_list(toxic_smiles, isomeric=True, keep_invalid=True)
        nontoxic_smiles = canonicalize_smiles_list(nontoxic_smiles, isomeric=True, keep_invalid=True)

    # 1) ECFP4 w/o chirality (FG용)
    fp_toxic_wo = build_ecfp4_list(toxic_smiles, include_chirality=False)
    fp_nontoxic_wo = build_ecfp4_list(nontoxic_smiles, include_chirality=False)
    valid_n_wo = [(j, fp) for j, fp in enumerate(fp_nontoxic_wo) if fp is not None]
    if not valid_n_wo:
        return dataset, endpoint, empty, empty, empty, empty, 0
    fp_nontoxic_wo_list = [f for _, f in valid_n_wo]

    # 2) ECFP4 w/ chirality (Stereo/Isomer용)
    fp_toxic_w = build_ecfp4_list(toxic_smiles, include_chirality=True)
    fp_nontoxic_w = build_ecfp4_list(nontoxic_smiles, include_chirality=True)
    valid_n_w = [(j, fp) for j, fp in enumerate(fp_nontoxic_w) if fp is not None]
    fp_nontoxic_w_list = [f for _, f in valid_n_w]

    # 3) Scaffold ECFP
    fp_toxic_scaffold = build_scaffold_fp_list(toxic_smiles)
    fp_nontoxic_scaffold = build_scaffold_fp_list(nontoxic_smiles)
    valid_n_scaffold = [(j, fp) for j, fp in enumerate(fp_nontoxic_scaffold) if fp is not None]
    fp_nontoxic_scaffold_list = [f for _, f in valid_n_scaffold]

    substructure_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)       # w/o chirality
    substructure_sim_w_chiral = np.full((n_t, n_n), np.nan, dtype=np.float32)
    scaffold_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)

    for i in range(n_t):
        if fp_toxic_wo[i] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp_toxic_wo[i], fp_nontoxic_wo_list)
            for k, sim in enumerate(sims):
                j = valid_n_wo[k][0]
                substructure_sim[i, j] = sim
        if fp_toxic_w[i] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp_toxic_w[i], fp_nontoxic_w_list)
            for k, sim in enumerate(sims):
                j = valid_n_w[k][0]
                substructure_sim_w_chiral[i, j] = sim
        if fp_toxic_scaffold[i] is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp_toxic_scaffold[i], fp_nontoxic_scaffold_list)
            for k, sim in enumerate(sims):
                j = valid_n_scaffold[k][0]
                scaffold_sim[i, j] = sim

    # 4) SMILES 유사도 행렬
    try:
        from rapidfuzz import process  # noqa: F401
        from rapidfuzz.distance import Levenshtein  # noqa: F401
        has_rapidfuzz = True
    except ImportError:
        has_rapidfuzz = False
    smiles_sim = np.full((n_t, n_n), np.nan, dtype=np.float32)
    if has_rapidfuzz and nontoxic_smiles:
        for start in range(0, n_t, CHUNK_TOXIC_SMILES):
            end = min(start + CHUNK_TOXIC_SMILES, n_t)
            sim_mat = _smiles_similarity_batch(toxic_smiles[start:end], nontoxic_smiles)
            smiles_sim[start:end, :] = sim_mat

    # 5) 분자식 (Isomer용)
    formula_t = np.array([_mol_formula(s) for s in toxic_smiles], dtype=object)
    formula_n = np.array([_mol_formula(s) for s in nontoxic_smiles], dtype=object)

    # 6) 스타일별 pairing
    sub_wo_ok = substructure_sim >= SIM_THRESHOLD
    sub_w_ok = substructure_sim_w_chiral >= SIM_THRESHOLD
    scaffold_ok = scaffold_sim >= SIM_THRESHOLD
    smiles_ok = np.isfinite(smiles_sim) & (smiles_sim >= SIM_THRESHOLD)

    # FG: (sub_wo ≥ 0.9 OR scaffold ≥ 0.9 OR smiles ≥ 0.9)
    pass_fg = sub_wo_ok | scaffold_ok | smiles_ok
    # Stereo: (sub_w ≥ 0.9 OR scaffold ≥ 0.9 OR smiles ≥ 0.9)
    pass_stereo = sub_w_ok | scaffold_ok | smiles_ok
    # Isomer: (sub_w ≥ 0.9 OR scaffold ≥ 0.9 OR smiles ≥ 0.9) AND same formula
    same_formula = (formula_t[:, None] == formula_n[None, :]) & (formula_t[:, None] != "")
    pass_isomer = (sub_w_ok | scaffold_ok | smiles_ok) & same_formula

    ij_fg = np.where(pass_fg)
    ij_stereo = np.where(pass_stereo)
    ij_isomer = np.where(pass_isomer)
    pairs_fg = set(zip(ij_fg[0].tolist(), ij_fg[1].tolist()))
    pairs_stereo = set(zip(ij_stereo[0].tolist(), ij_stereo[1].tolist()))
    pairs_isomer = set(zip(ij_isomer[0].tolist(), ij_isomer[1].tolist()))
    pairs_all = pairs_fg | pairs_stereo | pairs_isomer

    def to_rows(pair_set: set) -> list[tuple[str, str]]:
        return [(toxic_smiles[i], nontoxic_smiles[j]) for i, j in pair_set]

    rows_all = to_rows(pairs_all)
    rows_fg = to_rows(pairs_fg)
    rows_stereo = to_rows(pairs_stereo)
    rows_isomer = to_rows(pairs_isomer)

    if save_sim_path is not None:
        save_sim_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_sim_path,
            substructure_sim=substructure_sim,
            substructure_sim_w_chiral=substructure_sim_w_chiral,
            scaffold_sim=scaffold_sim,
            smiles_sim=smiles_sim,
            n_toxic=n_t,
            n_nontoxic=n_n,
        )

    return dataset, endpoint, rows_all, rows_fg, rows_stereo, rows_isomer, len(pairs_all)


def _load_and_process_one(args: tuple):
    """멀티프로세스 워커: (dataset, endpoint, path_toxic, path_nontoxic, path_sim_npz) -> process_endpoint 결과."""
    dataset, endpoint, path_toxic, path_nontoxic, path_sim_npz, canonicalize_smiles = args
    with open(path_toxic, "rb") as f:
        toxic_smiles = pickle.load(f)
    with open(path_nontoxic, "rb") as f:
        nontoxic_smiles = pickle.load(f)
    return process_endpoint(
        dataset,
        endpoint,
        toxic_smiles,
        nontoxic_smiles,
        save_sim_path=path_sim_npz,
        canonicalize_smiles=canonicalize_smiles,
    )


def main(n_workers: int | None = None, canonicalize_smiles: bool = True) -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Not found: {SUMMARY_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV)
    summary["_safe_ep"] = summary["path_npz"].apply(lambda p: Path(p).stem)

    SIM_MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    # ToxCast 제외 (pairing 대상에서 제외)
    EXCLUDE_DATASETS = {"toxcast_df"}
    summary = summary[~summary["dataset"].isin(EXCLUDE_DATASETS)]
    # 실행할 endpoint 인자 목록 (pkl 존재하는 것만) + sim matrix 저장 경로
    task_args: list[tuple[str, str, Path, Path, Path, bool]] = []
    for _, row in summary.iterrows():
        dataset, endpoint = row["dataset"], row["endpoint"]
        safe_ep = row["_safe_ep"]
        npz_dir = (BASE / Path(row["path_npz"])).parent
        path_toxic = npz_dir / f"{safe_ep}_toxic_smiles.pkl"
        path_nontoxic = npz_dir / f"{safe_ep}_nontoxic_smiles.pkl"
        if path_toxic.exists() and path_nontoxic.exists():
            path_sim_npz = SIM_MATRICES_DIR / dataset / f"{safe_ep}.npz"
            task_args.append((dataset, endpoint, path_toxic, path_nontoxic, path_sim_npz, canonicalize_smiles))

    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    all_rows: list[dict] = []
    fg_rows: list[dict] = []
    stereo_rows: list[dict] = []
    isomer_rows: list[dict] = []
    count_rows: list[dict] = []

    for d in (MOL_FG_DIR, MOL_STEREO_DIR, MOL_ISOMER_DIR):
        d.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_load_and_process_one, a): a for a in task_args}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Endpoints"):
            try:
                (dataset, endpoint, pairs_list, list_fg, list_stereo, list_isomer, n_pairs) = fut.result()
                count_rows.append({"dataset": dataset, "endpoint": endpoint, "n_pairs": n_pairs})
                def add_rows(rows: list[dict], lst: list[tuple[str, str]]) -> None:
                    for (s_t, s_n) in lst:
                        rows.append({
                            "dataset_name": dataset,
                            "endpoint": endpoint,
                            "toxic_smiles": s_t,
                            "nontoxic_smiles": s_n,
                        })
                add_rows(all_rows, pairs_list)
                add_rows(fg_rows, list_fg)
                add_rows(stereo_rows, list_stereo)
                add_rows(isomer_rows, list_isomer)
            except Exception as e:
                args = futures[fut]
                tqdm.write(f"Error {args[0]}/{args[1]}: {e}")

    def save_pairs(path: Path, rows: list[dict], name: str) -> None:
        if rows:
            df = pd.DataFrame(rows).drop_duplicates()
        else:
            df = pd.DataFrame(columns=["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"])
        df.to_csv(path, index=False)
        print(f"Saved: {path} ({len(df):,} rows) [{name}]")

    save_pairs(PAIRS_CSV, all_rows, "all")
    save_pairs(MOL_FG_DIR / "pairs.csv", fg_rows, "Mol_FG")
    save_pairs(MOL_STEREO_DIR / "pairs.csv", stereo_rows, "Mol_stereo")
    save_pairs(MOL_ISOMER_DIR / "pairs.csv", isomer_rows, "Mol_isomer")

    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(PAIR_COUNTS_CSV, index=False)
    print(f"Saved: {PAIR_COUNTS_CSV} ({len(df_counts):,} endpoints)")
    print(f"Total pairs (all): {df_counts['n_pairs'].sum():,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MolecularACE pairing (배치 유사도 + 멀티프로세스)")
    parser.add_argument("--workers", type=int, default=None, help="병렬 endpoint 수 (기본: CPU코어-1)")
    parser.add_argument(
        "--no_canonicalize_smiles",
        action="store_true",
        help="SMILES canonicalize를 끄고(원본 그대로) pairing 수행",
    )
    args = parser.parse_args()
    main(n_workers=args.workers, canonicalize_smiles=not args.no_canonicalize_smiles)
