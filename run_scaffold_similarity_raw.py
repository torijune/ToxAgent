"""
Raw_data 내 모든 CSV를 읽어, endpoint별로 독성-비독성 스캐폴드 유사도 행렬을 계산해 scaffold_sim/에 저장합니다.
- dili, toxcast는 사용하지 않고 제외합니다.
- Mol_FG의 DILIst/DICTrank/DIRIL raw CSV를 추가합니다.

저장 구조:
  scaffold_sim/
    summary.csv                    # endpoint별 통계 (n_toxic, n_nontoxic, mean, max, min, path)
    <dataset>/                    # 예: ames, clintox, dilist, dictrank, diril
      <endpoint_safe>.npz         # sim_matrix (float32)
      <endpoint_safe>_toxic_smiles.pkl
      <endpoint_safe>_nontoxic_smiles.pkl
"""
from pathlib import Path
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# Raw_data, scaffold_sim 경로 (스크립트 기준)
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "Raw_data"
OUT_DIR = BASE / "scaffold_sim"
MOL_FG_DIR = BASE / "Mol_FG"
# Mol_FG 디렉터리 내 raw CSV (dataset_name, 상대경로). dili 제거 후 DILIst/DICTrank/DIRIL 추가용.
MOL_FG_RAW = [
    ("dilist", "DILIst/dilist_raw.csv"),
    ("dictrank", "DICTrank/dictrank_raw.csv"),
    ("diril", "DIRIL/diril_raw.csv"),
]
# Raw_data에서 제외할 데이터셋 (toxcast는 사용 안 함)
RAW_EXCLUDE = {"dili", "toxcast_df", "toxcast"}

# SMILES 컬럼 후보 (순서대로 시도)
SMILES_COLS = ["X", "SMILES", "Drug", "smiles"]
LABEL_COL = "Y"
TASK_COL = "Task"
TOXIC_LABELS = {1, 1.0}
NONTOXIC_LABELS = {0, 0.0}

# 대용량 endpoint: 이 크기를 넘으면 무작위 샘플링하여 저장 (메모리/시간 방지)
MAX_TOXIC = 10000
MAX_NONTOXIC = 10000
RANDOM_SEED = 42


def _safe_filename(s: str) -> str:
    """파일명으로 쓸 수 있게 문자 치환."""
    return re.sub(r"[^\w\-.]", "_", s)[:200]


def _get_smiles_column(df: pd.DataFrame) -> str:
    for c in SMILES_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"No SMILES column in {df.columns.tolist()}")


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Raw_data CSV 로드. 컬럼 정규화."""
    df = pd.read_csv(path)
    smi_col = _get_smiles_column(df)
    # 표준 이름으로 복사해 두기 (내부적으로만 사용)
    df = df.rename(columns={smi_col: "_smiles"})
    df["_smiles"] = df["_smiles"].astype(str).str.strip()
    df = df[df["_smiles"].notna() & (df["_smiles"] != "")]
    if LABEL_COL not in df.columns:
        raise ValueError(f"No column '{LABEL_COL}' in {path.name}")
    if TASK_COL not in df.columns:
        df[TASK_COL] = path.stem
    return df


def load_molfg_raw(path: Path) -> tuple[list, list]:
    """Mol_FG 형식 CSV (SMILES, Target) 로드. 반환: (toxic_smiles, nontoxic_smiles)."""
    df = pd.read_csv(path)
    if "SMILES" not in df.columns or "Target" not in df.columns:
        raise ValueError(f"Mol_FG CSV needs SMILES and Target: {path.name}")
    df["_smiles"] = df["SMILES"].astype(str).str.strip()
    df = df[df["_smiles"].notna() & (df["_smiles"] != "")]
    toxic_mask = df["Target"].isin(TOXIC_LABELS)
    nontoxic_mask = df["Target"].isin(NONTOXIC_LABELS)
    toxic_smiles = df.loc[toxic_mask, "_smiles"].tolist()
    nontoxic_smiles = df.loc[nontoxic_mask, "_smiles"].tolist()
    return toxic_smiles, nontoxic_smiles


def run_one_endpoint(
    dataset_name: str,
    endpoint: str,
    toxic_smiles: list,
    nontoxic_smiles: list,
    scaffold_similarity_matrix_fn,
    generic: bool = False,
    radius: int = 2,
    fp_size: int = 1024,
    n_workers: int = None,
    use_sampling: bool = True,
):
    """한 endpoint에 대해 스캐폴드 유사도 행렬 계산 후 저장. 반환: (통계 dict, 저장 경로)."""
    toxic_smiles = list(dict.fromkeys(s for s in toxic_smiles if s))
    nontoxic_smiles = list(dict.fromkeys(s for s in nontoxic_smiles if s))
    if not toxic_smiles or not nontoxic_smiles:
        return None, None

    sampled = False
    if use_sampling and (len(toxic_smiles) > MAX_TOXIC or len(nontoxic_smiles) > MAX_NONTOXIC):
        rng = np.random.default_rng(RANDOM_SEED)
        if len(toxic_smiles) > MAX_TOXIC:
            toxic_smiles = rng.choice(toxic_smiles, size=MAX_TOXIC, replace=False).tolist()
            sampled = True
        if len(nontoxic_smiles) > MAX_NONTOXIC:
            nontoxic_smiles = rng.choice(nontoxic_smiles, size=MAX_NONTOXIC, replace=False).tolist()
            sampled = True

    kwargs = dict(generic=generic, radius=radius, fp_size=fp_size)
    if n_workers is not None and hasattr(scaffold_similarity_matrix_fn, "__code__"):
        sig = scaffold_similarity_matrix_fn.__code__.co_varnames
        if "n_workers" in sig:
            kwargs["n_workers"] = n_workers
    sim_matrix, valid_toxic, valid_nontoxic = scaffold_similarity_matrix_fn(
        toxic_smiles,
        nontoxic_smiles,
        **kwargs,
    )
    if sim_matrix.size == 0:
        return None, None

    safe_ep = _safe_filename(str(endpoint))
    out_sub = OUT_DIR / dataset_name
    out_sub.mkdir(parents=True, exist_ok=True)

    npz_path = out_sub / f"{safe_ep}.npz"
    np.savez_compressed(npz_path, sim_matrix=sim_matrix)
    with open(out_sub / f"{safe_ep}_toxic_smiles.pkl", "wb") as f:
        pickle.dump(valid_toxic, f)
    with open(out_sub / f"{safe_ep}_nontoxic_smiles.pkl", "wb") as f:
        pickle.dump(valid_nontoxic, f)

    stats = {
        "dataset": dataset_name,
        "endpoint": endpoint,
        "n_toxic": sim_matrix.shape[0],
        "n_nontoxic": sim_matrix.shape[1],
        "n_pairs": int(sim_matrix.size),
        "scaffold_sim_mean": float(np.nanmean(sim_matrix)),
        "scaffold_sim_max": float(np.nanmax(sim_matrix)),
        "scaffold_sim_min": float(np.nanmin(sim_matrix)),
        "path_npz": str(npz_path.relative_to(BASE)),
        "sampled": sampled,
    }
    return stats, npz_path


def main(n_workers=None):
    from scaffold_similarity import scaffold_similarity_matrix_parallel

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = [p for p in sorted(RAW_DIR.glob("*.csv")) if p.stem not in RAW_EXCLUDE]  # toxcast 미사용
    if not csv_files and not MOL_FG_RAW:
        print(f"No CSV files in {RAW_DIR} (dili/toxcast excluded) and no Mol_FG raw to add.")
        return

    all_stats = []
    for path in tqdm(csv_files, desc="Raw_data"):
        try:
            df = load_raw_csv(path)
        except Exception as e:
            print(f"Skip {path.name}: {e}")
            continue
        dataset_name = path.stem
        smiles_col = "_smiles"
        for endpoint in tqdm(
            df[TASK_COL].dropna().unique().tolist(),
            desc=dataset_name,
            leave=False,
        ):
            sub = df[df[TASK_COL] == endpoint]
            y = sub[LABEL_COL]
            toxic_mask = y.isin(TOXIC_LABELS)
            nontoxic_mask = y.isin(NONTOXIC_LABELS)
            toxic_smiles = sub.loc[toxic_mask, smiles_col].tolist()
            nontoxic_smiles = sub.loc[nontoxic_mask, smiles_col].tolist()
            stats, _ = run_one_endpoint(
                dataset_name,
                endpoint,
                toxic_smiles,
                nontoxic_smiles,
                scaffold_similarity_matrix_parallel,
                generic=False,
                radius=2,
                fp_size=1024,
                n_workers=n_workers,
            )
            if stats is not None:
                all_stats.append(stats)

    for dataset_name, rel_path in tqdm(MOL_FG_RAW, desc="Mol_FG (DILIst/DICTrank/DIRIL)"):
        path = MOL_FG_DIR / rel_path
        if not path.exists():
            print(f"Skip Mol_FG {dataset_name}: not found {path}")
            continue
        try:
            toxic_smiles, nontoxic_smiles = load_molfg_raw(path)
        except Exception as e:
            print(f"Skip Mol_FG {dataset_name}: {e}")
            continue
        if not toxic_smiles or not nontoxic_smiles:
            print(f"Skip Mol_FG {dataset_name}: no toxic or nontoxic")
            continue
        stats, _ = run_one_endpoint(
            dataset_name,
            dataset_name,
            toxic_smiles,
            nontoxic_smiles,
            scaffold_similarity_matrix_parallel,
            generic=False,
            radius=2,
            fp_size=1024,
            n_workers=n_workers,
        )
        if stats is not None:
            all_stats.append(stats)

    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_path = OUT_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path} ({len(summary_df)} endpoints)")
    print("Done.")


if __name__ == "__main__":
    main()
