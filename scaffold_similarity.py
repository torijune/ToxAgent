"""
Bemis-Murcko scaffold 기반 유사도 측정

같은 독성 endpoint 내 독성-비독성 약물 쌍에 대해 Bemis-Murcko 스캐폴드 유사도를 계산합니다.
RDKit의 MurckoScaffold를 사용하며, 스캐폴드 간 유사도는 Morgan fingerprint + Tanimoto으로 측정합니다.

대량 샘플 가속: scaffold_similarity_matrix_parallel() 사용 시 멀티프로세스로 스캐폴드/FP 생성 및
행렬 행 단위 유사도 계산을 병렬화합니다.
"""
from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
# RDKit 로그 억제 (Deprecation, Explicit valence 등 터미널 플러딩 방지)
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")
RDLogger.DisableLog("rdApp.error")


def _disable_rdkit_logging():
    """서브프로세스에서 호출: RDKit 로그 전부 끄기."""
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.info")
    RDLogger.DisableLog("rdApp.error")

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(it, **kwargs):
        return it

try:
    from concurrent.futures import ProcessPoolExecutor
    CONCURRENT_AVAILABLE = True
except ImportError:
    CONCURRENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# 스캐폴드 추출
# ---------------------------------------------------------------------------

def get_murcko_scaffold_smiles(
    smiles: str,
    generic: bool = False,
    include_chirality: bool = False,
) -> Optional[str]:
    """
    SMILES로부터 Bemis-Murcko 스캐폴드 SMILES 추출.

    Args:
        smiles: 분자 SMILES
        generic: True면 MakeScaffoldGeneric 적용 (원자 타입 -> C, 결합 -> 단일결합)
        include_chirality: 스캐폴드 SMILES에 입체정보 포함 여부

    Returns:
        스캐폴드 SMILES 또는 None (유효하지 않은 분자/스캐폴드 없음)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
            return None
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=include_chirality)
    except Exception:
        return None


def get_murcko_scaffold_mol(smiles: str, generic: bool = False):
    """
    SMILES로부터 Bemis-Murcko 스캐폴드 Mol 객체 반환.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
            return None
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        return scaffold
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 스캐폴드 유사도 (한 쌍)
# ---------------------------------------------------------------------------

def scaffold_tanimoto_similarity(
    smiles1: str,
    smiles2: str,
    generic: bool = False,
    radius: int = 2,
    fp_size: int = 1024,
) -> Optional[float]:
    """
    두 분자의 Bemis-Murcko 스캐폴드 간 Tanimoto 유사도 계산.
    스캐폴드에 대해 Morgan fingerprint를 만든 뒤 Tanimoto을 반환.

    Args:
        smiles1, smiles2: 두 분자 SMILES
        generic: 스캐폴드를 generic으로 할지 여부
        radius: Morgan fingerprint radius
        fp_size: Morgan fingerprint 비트 수

    Returns:
        0.0 ~ 1.0 유사도 또는 None (스캐폴드 추출 실패 시)
    """
    mol1 = get_murcko_scaffold_mol(smiles1, generic=generic)
    mol2 = get_murcko_scaffold_mol(smiles2, generic=generic)
    if mol1 is None or mol2 is None:
        return None
    try:
        fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
        fp1 = fpgen.GetFingerprint(mol1)
        fp2 = fpgen.GetFingerprint(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DataFrame: TN 쌍에 스캐폴드 유사도 컬럼 추가
# ---------------------------------------------------------------------------

def add_scaffold_similarity_to_pairs(
    df: pd.DataFrame,
    toxic_col: str = "Toxic_SMILES",
    nontoxic_col: str = "NonToxic_SMILES",
    generic: bool = False,
    radius: int = 2,
    fp_size: int = 1024,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    독성-비독성 쌍이 있는 DataFrame에 Bemis-Murcko 스캐폴드 유사도 컬럼 추가.

    Args:
        df: Toxic_SMILES, NonToxic_SMILES (또는 지정한 컬럼명) 포함 DataFrame
        toxic_col: 독성 분자 SMILES 컬럼명
        nontoxic_col: 비독성 분자 SMILES 컬럼명
        generic: 스캐폴드 generic 여부
        radius, fp_size: Morgan FP 파라미터
        verbose: 진행률 출력 여부

    Returns:
        'scaffold_similarity', 'scaffold_toxic', 'scaffold_nontoxic' 컬럼이 추가된 복사본
    """
    if toxic_col not in df.columns or nontoxic_col not in df.columns:
        raise ValueError(f"DataFrame must have columns '{toxic_col}' and '{nontoxic_col}'")

    out = df.copy()
    out["scaffold_toxic"] = None
    out["scaffold_nontoxic"] = None
    out["scaffold_similarity"] = np.nan

    it = out.iterrows()
    if verbose and TQDM_AVAILABLE:
        it = tqdm(list(it), desc="Scaffold similarity")

    for idx, row in it:
        s_tox = row[toxic_col]
        s_nt = row[nontoxic_col]
        if pd.isna(s_tox) or pd.isna(s_nt) or not s_tox or not s_nt:
            continue
        sc_tox = get_murcko_scaffold_smiles(s_tox, generic=generic)
        sc_nt = get_murcko_scaffold_smiles(s_nt, generic=generic)
        out.at[idx, "scaffold_toxic"] = sc_tox
        out.at[idx, "scaffold_nontoxic"] = sc_nt
        if sc_tox and sc_nt:
            sim = scaffold_tanimoto_similarity(s_tox, s_nt, generic=generic, radius=radius, fp_size=fp_size)
            if sim is not None:
                out.at[idx, "scaffold_similarity"] = sim

    return out


# ---------------------------------------------------------------------------
# 같은 endpoint 내: 독성 vs 비독성 스캐폴드 유사도 매트릭스
# ---------------------------------------------------------------------------

def scaffold_similarity_matrix(
    toxic_smiles_list: List[str],
    nontoxic_smiles_list: List[str],
    generic: bool = False,
    radius: int = 2,
    fp_size: int = 1024,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    독성 분자 리스트와 비독성 분자 리스트에 대해
    스캐폴드 간 Tanimoto 유사도 매트릭스 계산.

    Returns:
        sim_matrix: (len(toxic_smiles_list), len(nontoxic_smiles_list)) 유사도 행렬
        valid_toxic_smiles: 유효 스캐폴드가 있는 독성 SMILES 순서
        valid_nontoxic_smiles: 유효 스캐폴드가 있는 비독성 SMILES 순서
    """
    toxic_mols = []
    valid_toxic = []
    for s in toxic_smiles_list:
        m = get_murcko_scaffold_mol(s, generic=generic)
        if m is not None:
            toxic_mols.append(m)
            valid_toxic.append(s)

    nontoxic_mols = []
    valid_nontoxic = []
    for s in nontoxic_smiles_list:
        m = get_murcko_scaffold_mol(s, generic=generic)
        if m is not None:
            nontoxic_mols.append(m)
            valid_nontoxic.append(s)

    n_t, n_n = len(toxic_mols), len(nontoxic_mols)
    sim_matrix = np.zeros((n_t, n_n), dtype=np.float32)

    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    toxic_fps = [fpgen.GetFingerprint(m) for m in toxic_mols]
    nontoxic_fps = [fpgen.GetFingerprint(m) for m in nontoxic_mols]

    for i in range(n_t):
        sims = DataStructs.BulkTanimotoSimilarity(toxic_fps[i], nontoxic_fps)
        sim_matrix[i, :] = sims

    return sim_matrix, valid_toxic, valid_nontoxic


# ---------------------------------------------------------------------------
# 병렬화: 워커 함수 (모듈 최상단에 두어 pickle 가능하도록)
# ---------------------------------------------------------------------------

def _worker_build_scaffold_fps(
    smiles_chunk: List[str],
    generic: bool,
    radius: int,
    fp_size: int,
) -> Tuple[List[str], List]:
    """SMILES 청크에 대해 스캐폴드 Mol → FP 생성. (valid_smiles, fps) 반환."""
    _disable_rdkit_logging()
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    valid_smiles = []
    fps = []
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    for s in smiles_chunk:
        if not s or not isinstance(s, str):
            continue
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
                continue
            if generic:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            fp = fp_gen.GetFingerprint(scaffold)
            valid_smiles.append(s)
            fps.append(fp)
        except Exception:
            continue
    return valid_smiles, fps


def _worker_matrix_rows(
    toxic_fps_chunk: List,
    nontoxic_fps: List,
) -> np.ndarray:
    """toxic_fps 청크 × nontoxic_fps 전체에 대한 Tanimoto 행렬 (len(chunk), len(nontoxic_fps))."""
    from rdkit import DataStructs

    n_n = len(nontoxic_fps)
    out = np.zeros((len(toxic_fps_chunk), n_n), dtype=np.float32)
    for i, fp in enumerate(toxic_fps_chunk):
        sims = DataStructs.BulkTanimotoSimilarity(fp, nontoxic_fps)
        out[i, :] = sims
    return out


def scaffold_similarity_matrix_parallel(
    toxic_smiles_list: List[str],
    nontoxic_smiles_list: List[str],
    generic: bool = False,
    radius: int = 2,
    fp_size: int = 1024,
    n_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    독성 vs 비독성 스캐폴드 유사도 행렬을 멀티프로세스로 계산 (대량 샘플용).

    스캐폴드/FP 생성과 행렬의 행 단위 유사도 계산을 병렬화합니다.
    n_workers=None이면 CPU 코어 수 - 1 (최소 1). chunk_size=None이면 자동 분할.

    Returns:
        scaffold_similarity_matrix()와 동일: (sim_matrix, valid_toxic_smiles, valid_nontoxic_smiles)
    """
    import os
    if not CONCURRENT_AVAILABLE or n_workers == 0:
        return scaffold_similarity_matrix(
            toxic_smiles_list, nontoxic_smiles_list,
            generic=generic, radius=radius, fp_size=fp_size,
        )
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    toxic_smiles_list = [s for s in toxic_smiles_list if s and isinstance(s, str)]
    nontoxic_smiles_list = [s for s in nontoxic_smiles_list if s and isinstance(s, str)]
    if not toxic_smiles_list or not nontoxic_smiles_list:
        return np.zeros((0, 0), dtype=np.float32), [], []

    # 1) 비독성 스캐폴드 FP 전체 생성 (한 번만), 청크 순서 유지
    if chunk_size is None:
        chunk_size = max(1, (len(nontoxic_smiles_list) + n_workers - 1) // n_workers)
    nontoxic_chunks = [
        nontoxic_smiles_list[i : i + chunk_size]
        for i in range(0, len(nontoxic_smiles_list), chunk_size)
    ]
    valid_nontoxic = []
    nontoxic_fps = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_worker_build_scaffold_fps, c, generic, radius, fp_size) for c in nontoxic_chunks]
        for f in futures:
            vs, fps = f.result()
            valid_nontoxic.extend(vs)
            nontoxic_fps.extend(fps)
    n_n = len(nontoxic_fps)
    if n_n == 0:
        return np.zeros((0, 0), dtype=np.float32), [], []

    # 2) 독성 스캐폴드 FP 병렬 생성, 청크 순서 유지
    t_chunk_size = max(1, (len(toxic_smiles_list) + n_workers - 1) // n_workers)
    toxic_chunks = [
        toxic_smiles_list[i : i + t_chunk_size]
        for i in range(0, len(toxic_smiles_list), t_chunk_size)
    ]
    valid_toxic = []
    toxic_fps = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_worker_build_scaffold_fps, c, generic, radius, fp_size) for c in toxic_chunks]
        for f in futures:
            vs, fps = f.result()
            valid_toxic.extend(vs)
            toxic_fps.extend(fps)
    n_t = len(toxic_fps)
    if n_t == 0:
        return np.zeros((0, n_n), dtype=np.float32), [], valid_nontoxic

    # 3) 행렬: 행 청크별 병렬 (각 워커가 일부 toxic_fps 행만 계산), 순서 유지
    row_chunk_size = max(1, (n_t + n_workers - 1) // n_workers)
    row_chunks = [
        toxic_fps[i : i + row_chunk_size]
        for i in range(0, n_t, row_chunk_size)
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(_worker_matrix_rows, chunk, nontoxic_fps)
            for chunk in row_chunks
        ]
        parts = [f.result() for f in futures]
    sim_matrix = np.vstack(parts).astype(np.float32)
    return sim_matrix, valid_toxic, valid_nontoxic


def scaffold_similarity_per_endpoint(
    df: pd.DataFrame,
    smiles_column: str = "SMILES",
    label_column: str = "Y",
    endpoint_column: str = "Task",
    toxic_label: int = 1,
    non_toxic_label: int = 0,
    generic: bool = False,
    radius: int = 2,
    fp_size: int = 1024,
) -> pd.DataFrame:
    """
    endpoint별로 독성/비독성을 나눈 뒤, 같은 endpoint 내에서
    독성-비독성 스캐폴드 유사도 매트릭스의 통계(평균, 최대, 최소 등)를 계산.

    Args:
        df: SMILES, 라벨, endpoint(예: Task) 컬럼이 있는 DataFrame
        smiles_column: SMILES 컬럼명
        label_column: 라벨 컬럼명 (1=독성, 0=비독성 등)
        endpoint_column: endpoint 구분 컬럼명 (예: 'Task')
        toxic_label, non_toxic_label: 독성/비독성 라벨 값
        generic, radius, fp_size: 스캐폴드 유사도 파라미터

    Returns:
        endpoint별 행으로, scaffold_sim_mean, scaffold_sim_max, scaffold_sim_min,
        n_toxic, n_nontoxic, n_pairs 등 컬럼이 있는 DataFrame
    """
    required = [smiles_column, label_column, endpoint_column]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"DataFrame must have column '{c}'")

    rows = []
    endpoints = df[endpoint_column].dropna().unique().tolist()
    it = endpoints
    if TQDM_AVAILABLE:
        it = tqdm(it, desc="Scaffold similarity per endpoint")

    for ep in it:
        sub = df[df[endpoint_column] == ep]
        toxic_smiles = sub[sub[label_column] == toxic_label][smiles_column].dropna().unique().tolist()
        nontoxic_smiles = sub[sub[label_column] == non_toxic_label][smiles_column].dropna().unique().tolist()
        if not toxic_smiles or not nontoxic_smiles:
            rows.append({
                endpoint_column: ep,
                "n_toxic": len(toxic_smiles),
                "n_nontoxic": len(nontoxic_smiles),
                "scaffold_sim_mean": np.nan,
                "scaffold_sim_max": np.nan,
                "scaffold_sim_min": np.nan,
                "n_pairs": 0,
            })
            continue
        sim_matrix, _, _ = scaffold_similarity_matrix(
            toxic_smiles, nontoxic_smiles,
            generic=generic, radius=radius, fp_size=fp_size,
        )
        if sim_matrix.size == 0:
            rows.append({
                endpoint_column: ep,
                "n_toxic": len(toxic_smiles),
                "n_nontoxic": len(nontoxic_smiles),
                "scaffold_sim_mean": np.nan,
                "scaffold_sim_max": np.nan,
                "scaffold_sim_min": np.nan,
                "n_pairs": 0,
            })
            continue
        rows.append({
            endpoint_column: ep,
            "n_toxic": sim_matrix.shape[0],
            "n_nontoxic": sim_matrix.shape[1],
            "scaffold_sim_mean": float(np.nanmean(sim_matrix)),
            "scaffold_sim_max": float(np.nanmax(sim_matrix)),
            "scaffold_sim_min": float(np.nanmin(sim_matrix)),
            "n_pairs": int(sim_matrix.size),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 간단 사용 예
    s1 = "Cc1cc(Oc2nccc(CCC)c2)ccc1"
    s2 = "Cc1cc(Oc2ccccn2)ccc1"
    sc1 = get_murcko_scaffold_smiles(s1)
    sc2 = get_murcko_scaffold_smiles(s2)
    print("Scaffold 1:", sc1)
    print("Scaffold 2:", sc2)
    sim = scaffold_tanimoto_similarity(s1, s2)
    print("Scaffold Tanimoto similarity:", sim)

    # TN 쌍 DataFrame 예시
    pairs_df = pd.DataFrame({
        "Toxic_SMILES": [s1],
        "NonToxic_SMILES": [s2],
    })
    out_df = add_scaffold_similarity_to_pairs(pairs_df, verbose=False)
    print("\nPairs with scaffold similarity:")
    print(out_df[["Toxic_SMILES", "NonToxic_SMILES", "scaffold_toxic", "scaffold_nontoxic", "scaffold_similarity"]])
