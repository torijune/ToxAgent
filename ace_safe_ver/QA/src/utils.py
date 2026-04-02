"""
QA용 유틸: toxic_smiles 유사도 행렬 생성·저장 등
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_QA_SRC = Path(__file__).resolve().parent
_ACE = _QA_SRC.parent.parent.parent  # QA/src -> QA -> ace_safe_ver
if str(_ACE) not in sys.path:
    sys.path.insert(0, str(_ACE))
import ace_local  # noqa: E402

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from similarity_utils import build_full_similarity_matrix

DEFAULT_PAIRS_CSV = ace_local.DEFAULT_COMMOM_FRAGE_PAIRS_CSV
DEFAULT_SIM_OUT_DIR = ace_local.DEFAULT_TOXIC_SIM_OUT_DIR


def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return float("nan")
    return rdMolDescriptors.GetTanimotoSimilarity(mol1, mol2)


def build_toxic_toxic_sim_matrix(
    pairs_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
    radius: int = 2,
    fp_size: int = 1024,
    use_float32: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    commom_frage_pairs_with_smiles.csv에서 unique toxic_smiles를 추출한 뒤
    Tanimoto 유사도 행렬을 계산하고 저장합니다.

    Args:
        pairs_csv: pairs CSV 경로 (기본: ace_safe_ver/data/commom_frage_pairs_with_smiles.csv)
        out_dir: 행렬·SMILES 리스트 저장 디렉터리 (기본: ace_safe_ver/toxic_sim_matrix)
        radius: Morgan fingerprint radius
        fp_size: Fingerprint 크기
        use_float32: True면 float32 행렬로 저장

    Returns:
        sim_matrix: (N, N) 유사도 행렬
        toxic_smiles_list: 행/열 인덱스에 대응하는 unique toxic_smiles 리스트

    저장 파일:
        - {out_dir}/toxic_sim_matrix.npy  : numpy array
        - {out_dir}/toxic_smiles_list.json : SMILES 리스트 (순서 = 행/열 인덱스)
    """
    pairs_csv = Path(pairs_csv or DEFAULT_PAIRS_CSV)
    out_dir = Path(out_dir or DEFAULT_SIM_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pairs_csv)
    if "toxic_safe_decoded_smiles" not in df.columns:
        raise ValueError(f"CSV must have 'toxic_safe_decoded_smiles' column: {pairs_csv}")

    # unique toxic_smiles (순서 유지, 빈/NaN 제외)
    toxic_series = df["toxic_safe_decoded_smiles"].dropna().astype(str).str.strip()
    toxic_series = toxic_series[toxic_series != ""]
    unique_toxic = toxic_series.unique().tolist()

    sim_matrix, valid_smiles = build_full_similarity_matrix(
        unique_toxic,
        radius=radius,
        fpSize=fp_size,
        use_float32=use_float32,
    )

    # 저장
    np.save(out_dir / "toxic_safe_decoded_smiles_matrix.npy", sim_matrix)
    with open(out_dir / "toxic_safe_decoded_smiles_list.json", "w", encoding="utf-8") as f:
        json.dump(valid_smiles, f, ensure_ascii=False, indent=0)

    print(f"Saved sim_matrix shape {sim_matrix.shape} and {len(valid_smiles)} SMILES to {out_dir}")
    return sim_matrix, valid_smiles


def load_toxic_sim_matrix(
    out_dir: str | Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """저장된 toxic 유사도 행렬과 SMILES 리스트를 불러옵니다."""
    out_dir = Path(out_dir or DEFAULT_SIM_OUT_DIR)
    npy = out_dir / "toxic_safe_decoded_smiles_matrix.npy"
    if not npy.is_file():
        raise FileNotFoundError(
            f"ICL용 유사도 행렬이 없습니다: {npy}\n"
            "먼저 생성하세요 (QA와 동일한 pairs CSV 권장):\n"
            "  cd ace_safe_ver/QA/src && python utils.py\n"
            "또는 Python에서:\n"
            "  from utils import build_toxic_toxic_sim_matrix; "
            "build_toxic_toxic_sim_matrix(pairs_csv='.../merged_test.csv', out_dir='.../toxic_sim_matrix')\n"
            "또는 build_safe_qa.py 에 --prebuild-toxic-sim-matrix 옵션 사용."
        )
    sim_matrix = np.load(npy)
    with open(out_dir / "toxic_safe_decoded_smiles_list.json", "r", encoding="utf-8") as f:
        smiles_list = json.load(f)
    return sim_matrix, smiles_list


if __name__ == "__main__":
    build_toxic_toxic_sim_matrix()
