"""
분자 간 유사도 계산 유틸리티 함수들
build_dataset.py에서 추출한 유사도 관련 함수들

===========================
데이터 포맷 요구사항
===========================

1. DataFrame 기반 함수들:
   - generate_fingerprints(df): 'smiles' 컬럼 필요
   - get_similarity_df(df): 'smiles' 컬럼 필요
   - build_similarity_matrix_from_df(df, id_column, smiles_column): 임의의 컬럼명 지정 가능
   - find_similar_pairs_from_df(df, threshold): 'smiles' 컬럼 필요

2. 리스트 기반 함수들:
   - build_full_similarity_matrix(smiles_list): SMILES 문자열 리스트
   - get_similar_pairs(smiles_list, fp_list, threshold): SMILES 리스트 + Fingerprint 리스트
   - find_most_similar_molecules(target_smiles, smiles_list): SMILES 문자열 + SMILES 리스트

3. clintox_df 예시:
   컬럼: ['task', 'dataset_title', 'dataset_description', 'task_description', 'Drug_ID', 'Drug', 'Y']
   - 'Drug' 컬럼: SMILES 문자열
   - 'Drug_ID' 컬럼: 고유 ID
"""

from tqdm.auto import tqdm
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def canonicalize_smiles(smiles):
    """SMILES를 정규화"""
    try:
        if '.' in smiles:  # skip multi-component SMILES
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        return None


def generate_fingerprints(df, radius=2, fpSize=1024):
    """
    DataFrame에서 Morgan Fingerprint 생성
    
    Args:
        df: 'smiles' 컬럼이 있는 DataFrame
        radius: Morgan Fingerprint radius (기본값: 2)
        fpSize: Fingerprint 크기 (기본값: 1024)
    
    Returns:
        smi_list: SMILES 리스트
        fp_list: Fingerprint 리스트
    """
    assert 'smiles' in df.columns, "DataFrame must have 'smiles' column"
    
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
    smi_list = df['smiles'].tolist()
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
    fp_list = [fpgen.GetFingerprint(mol) for mol in mol_list]
    return smi_list, fp_list


def get_similarity_df(df, radius=2, fpSize=1024):
    """
    DataFrame의 모든 분자 쌍에 대한 Tanimoto 유사도 매트릭스 생성
    
    Args:
        df: 'smiles' 컬럼이 있는 DataFrame
        radius: Morgan Fingerprint radius
        fpSize: Fingerprint 크기
    
    Returns:
        similarity_df: Tanimoto 유사도 매트릭스 (DataFrame)
    """
    assert 'smiles' in df.columns, "DataFrame must have 'smiles' column"
    
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
    smi_list = df['smiles'].tolist()
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
    fp_list = [fpgen.GetFingerprint(mol) for mol in mol_list]
    
    similarity_df = pd.DataFrame(index=smi_list, columns=smi_list)
    
    for i in tqdm(range(len(smi_list) - 1), desc="Computing similarities"):
        target_smi = smi_list[i]
        s = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i + 1:])
        similarity_df.loc[target_smi, smi_list[i + 1:]] = s
    
    return similarity_df


def get_similar_pairs(smiles_list, fp_list, threshold=0.7):
    """
    Threshold 이상의 유사도를 가진 분자 쌍 찾기 (Generator)
    
    Args:
        smiles_list: SMILES 문자열 리스트
        fp_list: Fingerprint 리스트
        threshold: 유사도 임계값 (기본값: 0.7)
    
    Yields:
        (smiles1, smiles2): 유사한 분자 쌍
    """
    n = len(smiles_list)
    for i in tqdm(range(n - 1), desc="Finding similar pairs"):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i + 1:])
        for j, sim in enumerate(sims):
            if sim > threshold:
                yield (smiles_list[i], smiles_list[i + 1 + j], sim)


def build_full_similarity_matrix(smiles_list, radius=2, fpSize=1024, use_float32=True):
    """
    SMILES 리스트로부터 전체 Tanimoto 유사도 매트릭스 생성 (NumPy array)
    
    Args:
        smiles_list: SMILES 문자열 리스트
        radius: Morgan Fingerprint radius
        fpSize: Fingerprint 크기
        use_float32: float32 사용 여부 (메모리 절약)
    
    Returns:
        sim_matrix: NxN Tanimoto 유사도 매트릭스 (NumPy array)
        valid_smiles: 유효한 SMILES 리스트
    """
    # 1. Fingerprint 생성
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
    fps = []
    valid_smiles = []
    
    for smi in tqdm(smiles_list, desc="Generating fingerprints"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                fps.append(fp)
                valid_smiles.append(smi)
        except:
            continue
    
    # 2. 유사도 매트릭스 생성
    n = len(fps)
    dtype = np.float32 if use_float32 else np.float64
    sim_matrix = np.eye(n, dtype=dtype)
    
    # 3. 상삼각 행렬만 계산
    for i in tqdm(range(n - 1), desc="Computing Tanimoto similarities"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim_matrix[i, i + 1:] = np.array(sims, dtype=dtype)
    
    # 4. 대칭 복사
    sim_matrix = sim_matrix + sim_matrix.T - np.eye(n, dtype=dtype)
    
    return sim_matrix, valid_smiles


def find_most_similar_molecules(target_smiles, smiles_list, fp_list=None, top_k=10, radius=2, fpSize=1024):
    """
    특정 분자와 가장 유사한 분자들 찾기
    
    Args:
        target_smiles: 기준 분자 SMILES
        smiles_list: 비교할 분자들의 SMILES 리스트
        fp_list: 미리 계산된 Fingerprint 리스트 (선택)
        top_k: 반환할 상위 분자 개수
        radius: Fingerprint radius (fp_list가 None일 때만 사용)
        fpSize: Fingerprint 크기 (fp_list가 None일 때만 사용)
    
    Returns:
        similar_df: 유사한 분자들의 DataFrame (smiles, similarity)
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        raise ValueError(f"Invalid SMILES: {target_smiles}")
    
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
    target_fp = fpgen.GetFingerprint(target_mol)
    
    # Fingerprint가 없으면 생성
    if fp_list is None:
        fp_list = []
        for smi in tqdm(smiles_list, desc="Generating fingerprints"):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp_list.append(fpgen.GetFingerprint(mol))
            else:
                fp_list.append(None)
    
    # 유사도 계산
    similarities = []
    for i, (smi, fp) in enumerate(zip(smiles_list, fp_list)):
        if fp is not None and smi != target_smiles:
            sim = DataStructs.TanimotoSimilarity(target_fp, fp)
            similarities.append((smi, sim))
    
    # 정렬하고 상위 k개 반환
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_df = pd.DataFrame(similarities[:top_k], columns=['smiles', 'similarity'])
    
    return similar_df


def find_similar_pairs_from_df(df, smiles_column='smiles', id_column=None, threshold=0.7, radius=2, fpSize=1024):
    """
    DataFrame에서 유사도 threshold 이상인 분자 쌍을 찾기 (build_dataset.py 스타일)
    
    Args:
        df: DataFrame
        smiles_column: SMILES 컬럼 이름 (기본값: 'smiles')
        id_column: ID 컬럼 이름 (선택, 있으면 ID 포함하여 반환)
        threshold: 유사도 임계값 (기본값: 0.7)
        radius: Fingerprint radius
        fpSize: Fingerprint 크기
    
    Returns:
        pairs_df: 유사한 쌍의 DataFrame
                 컬럼: smiles_1, smiles_2, similarity (id_column이 있으면 id_1, id_2 추가)
    """
    assert smiles_column in df.columns, f"DataFrame must have '{smiles_column}' column"
    
    # Fingerprint 생성
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
    smiles_list = df[smiles_column].tolist()
    
    fps = []
    valid_indices = []
    for i, smi in enumerate(tqdm(smiles_list, desc="Generating fingerprints")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fps.append(fpgen.GetFingerprint(mol))
                valid_indices.append(i)
        except:
            continue
    
    # 유사 쌍 찾기
    pairs = []
    n = len(fps)
    for i in tqdm(range(n - 1), desc="Finding similar pairs"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, sim in enumerate(sims):
            if sim > threshold:
                idx1 = valid_indices[i]
                idx2 = valid_indices[i + 1 + j]
                
                pair_data = {
                    'smiles_1': smiles_list[idx1],
                    'smiles_2': smiles_list[idx2],
                    'similarity': sim
                }
                
                if id_column and id_column in df.columns:
                    ids = df[id_column].tolist()
                    pair_data['id_1'] = ids[idx1]
                    pair_data['id_2'] = ids[idx2]
                
                pairs.append(pair_data)
    
    pairs_df = pd.DataFrame(pairs)
    print(f"Found {len(pairs_df)} pairs with similarity > {threshold}")
    
    return pairs_df


def build_similarity_matrix_from_df(df, id_column, smiles_column, radius=2, fpSize=1024):
    """
    DataFrame으로부터 유사도 매트릭스 생성 (ID를 index로 사용, 중복 처리)
    
    Args:
        df: DataFrame
        id_column: ID 컬럼 이름 (예: 'Drug_ID')
        smiles_column: SMILES 컬럼 이름 (예: 'Drug')
        radius: Fingerprint radius
        fpSize: Fingerprint 크기
    
    Returns:
        sim_df: 유사도 매트릭스 (index/columns = ID)
    """
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)

    ids = df[id_column].tolist()
    smiles_list = df[smiles_column].tolist()

    # Fingerprint 생성 (모든 행에 대해, invalid도 포함)
    fps = []
    invalid_indices = []
    print(f"Processing {len(smiles_list)} molecules...")
    for i, smi in enumerate(tqdm(smiles_list, desc="Generating fingerprints")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                fps.append(fp)
            else:
                # Dummy fingerprint (zero-vector)
                arr = np.zeros(fpSize, dtype=np.uint8)
                fp = DataStructs.cDataStructs.CreateFromBitString("0"*fpSize)
                fps.append(fp)
                invalid_indices.append(i)
        except:
            arr = np.zeros(fpSize, dtype=np.uint8)
            fp = DataStructs.cDataStructs.CreateFromBitString("0"*fpSize)
            fps.append(fp)
            invalid_indices.append(i)

    n = len(fps)
    sim_matrix = np.eye(n, dtype=np.float32)

    # Precompute which are invalid for fast lookup
    invalid_set = set(invalid_indices)
    print(f"Detected {len(invalid_indices)} invalid molecules (replaced with dummy fingerprints).")
    print(f"Computing similarities for all {n} molecules (including invalids)...")
    for i in tqdm(range(n - 1), desc="Computing Tanimoto similarities"):
        # If i is invalid, set all similarities to 0
        if i in invalid_set:
            sim_matrix[i, i + 1:] = 0.0
            continue
        sims = []
        for j in range(i + 1, n):
            if j in invalid_set:
                sims.append(0.0)
            else:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sims.append(sim)
        sim_matrix[i, i + 1:] = np.array(sims, dtype=np.float32)

    # 대칭 복사
    sim_matrix = sim_matrix + sim_matrix.T - np.eye(n, dtype=np.float32)

    # DataFrame으로 변환 (ID 사용)
    valid_ids = list(ids)

    # 중복 ID 확인
    if len(valid_ids) != len(set(valid_ids)):
        print(f"⚠️ Warning: Found {len(valid_ids) - len(set(valid_ids))} duplicate IDs")
        print("Adding suffix to make IDs unique...")
        # 중복 처리: 같은 ID에 _1, _2 등 추가
        id_counts = {}
        unique_ids = []
        for id_val in valid_ids:
            if id_val in id_counts:
                id_counts[id_val] += 1
                unique_ids.append(f"{id_val}_{id_counts[id_val]}")
            else:
                id_counts[id_val] = 0
                unique_ids.append(id_val)
        valid_ids = unique_ids

    sim_df = pd.DataFrame(sim_matrix, index=valid_ids, columns=valid_ids)

    return sim_df


def build_full_similarity_matrix_stereo(df, smiles_list=None, smiles_column='smiles', id_column=None, radius=2, fpSize=1024, use_float32=True):
    """
    입체화학을 고려한 Tanimoto 유사도 매트릭스 생성
    
    Args:
        df: DataFrame (선택, df가 제공되면 smiles_column에서 SMILES 추출)
        smiles_list: SMILES 문자열 리스트 (df가 None일 때 사용)
        smiles_column: SMILES 컬럼 이름 (df가 제공될 때만 사용, 기본값: 'smiles')
        id_column: ID 컬럼 이름 (선택, 제공되면 DataFrame 반환, ID를 index/columns로 사용)
        radius: Morgan Fingerprint radius
        fpSize: Fingerprint 크기
        use_float32: float32 사용 여부 (메모리 절약)
    
    Returns:
        id_column이 제공된 경우:
            sim_df: 유사도 매트릭스 DataFrame (index/columns = ID)
        id_column이 없는 경우:
            sim_matrix: NxN Tanimoto 유사도 매트릭스 (NumPy array)
            valid_smiles: 유효한 SMILES 리스트
    """
    # DataFrame이 제공되면 SMILES 리스트 추출
    ids = None
    if df is not None:
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert smiles_column in df.columns, f"DataFrame must have '{smiles_column}' column"
        smiles_list = df[smiles_column].tolist()
        if id_column is not None:
            assert id_column in df.columns, f"DataFrame must have '{id_column}' column"
            ids = df[id_column].tolist()
    elif smiles_list is None:
        raise ValueError("Either df or smiles_list must be provided")
    
    # 1. 입체화학을 고려한 Fingerprint 생성
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize, includeChirality=True)
    
    # id_column이 제공되면 build_similarity_matrix_from_df_stereo와 동일한 방식 (invalid 포함)
    if id_column is not None and df is not None:
        fps = []
        invalid_indices = []
        print(f"Processing {len(smiles_list)} molecules (with chirality)...")
        for i, smi in enumerate(tqdm(smiles_list, desc="Generating fingerprints (with chirality)")):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = fpgen.GetFingerprint(mol)
                    fps.append(fp)
                else:
                    # Dummy fingerprint (zero-vector)
                    arr = np.zeros(fpSize, dtype=np.uint8)
                    fp = DataStructs.cDataStructs.CreateFromBitString("0"*fpSize)
                    fps.append(fp)
                    invalid_indices.append(i)
            except:
                arr = np.zeros(fpSize, dtype=np.uint8)
                fp = DataStructs.cDataStructs.CreateFromBitString("0"*fpSize)
                fps.append(fp)
                invalid_indices.append(i)
        
        n = len(fps)
        sim_matrix = np.eye(n, dtype=np.float32)
        
        # Precompute which are invalid for fast lookup
        invalid_set = set(invalid_indices)
        print(f"Detected {len(invalid_indices)} invalid molecules (replaced with dummy fingerprints).")
        print(f"Computing similarities for all {n} molecules (with chirality)...")
        for i in tqdm(range(n - 1), desc="Computing Tanimoto similarities (with chirality)"):
            # If i is invalid, set all similarities to 0
            if i in invalid_set:
                sim_matrix[i, i + 1:] = 0.0
                continue
            sims = []
            for j in range(i + 1, n):
                if j in invalid_set:
                    sims.append(0.0)
                else:
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sims.append(sim)
            sim_matrix[i, i + 1:] = np.array(sims, dtype=np.float32)
        
        # 대칭 복사
        sim_matrix = sim_matrix + sim_matrix.T - np.eye(n, dtype=np.float32)
        
        # DataFrame으로 변환 (ID 사용)
        valid_ids = list(ids)
        
        # 중복 ID 확인
        if len(valid_ids) != len(set(valid_ids)):
            print(f"⚠️ Warning: Found {len(valid_ids) - len(set(valid_ids))} duplicate IDs")
            print("Adding suffix to make IDs unique...")
            # 중복 처리: 같은 ID에 _1, _2 등 추가
            id_counts = {}
            unique_ids = []
            for id_val in valid_ids:
                if id_val in id_counts:
                    id_counts[id_val] += 1
                    unique_ids.append(f"{id_val}_{id_counts[id_val]}")
                else:
                    id_counts[id_val] = 0
                    unique_ids.append(id_val)
            valid_ids = unique_ids
        
        sim_df = pd.DataFrame(sim_matrix, index=valid_ids, columns=valid_ids)
        return sim_df
    
    # id_column이 없는 경우: 기존 방식 (valid만 포함)
    fps = []
    valid_smiles = []
    
    for smi in tqdm(smiles_list, desc="Generating fingerprints (with chirality)"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                fps.append(fp)
                valid_smiles.append(smi)
        except:
            continue
    
    # 2. 유사도 매트릭스 생성
    n = len(fps)
    dtype = np.float32 if use_float32 else np.float64
    sim_matrix = np.eye(n, dtype=dtype)
    
    # 3. 상삼각 행렬만 계산
    for i in tqdm(range(n - 1), desc="Computing Tanimoto similarities (with chirality)"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim_matrix[i, i + 1:] = np.array(sims, dtype=dtype)
    
    # 4. 대칭 복사
    sim_matrix = sim_matrix + sim_matrix.T - np.eye(n, dtype=dtype)
    
    return sim_matrix, valid_smiles


def find_most_similar_molecules_stereo(target_smiles, smiles_list, fp_list=None, top_k=10, radius=2, fpSize=1024):
    """
    입체화학을 고려하여 특정 분자와 가장 유사한 분자들 찾기
    
    Args:
        target_smiles: 기준 분자 SMILES
        smiles_list: 비교할 분자들의 SMILES 리스트
        fp_list: 미리 계산된 Fingerprint 리스트 (선택)
        top_k: 반환할 상위 분자 개수
        radius: Fingerprint radius (fp_list가 None일 때만 사용)
        fpSize: Fingerprint 크기 (fp_list가 None일 때만 사용)
    
    Returns:
        similar_df: 유사한 분자들의 DataFrame (smiles, similarity)
    """
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        raise ValueError(f"Invalid SMILES: {target_smiles}")
    
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize, includeChirality=True)
    target_fp = fpgen.GetFingerprint(target_mol)
    
    # Fingerprint가 없으면 생성
    if fp_list is None:
        fp_list = []
        for smi in tqdm(smiles_list, desc="Generating fingerprints (with chirality)"):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp_list.append(fpgen.GetFingerprint(mol))
            else:
                fp_list.append(None)
    
    # 유사도 계산
    similarities = []
    for i, (smi, fp) in enumerate(zip(smiles_list, fp_list)):
        if fp is not None and smi != target_smiles:
            sim = DataStructs.TanimotoSimilarity(target_fp, fp)
            similarities.append((smi, sim))
    
    # 정렬하고 상위 k개 반환
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_df = pd.DataFrame(similarities[:top_k], columns=['smiles', 'similarity'])
    
    return similar_df


def build_similarity_matrix_from_df_stereo(df, id_column, smiles_column, radius=2, fpSize=1024):
    """
    입체화학을 고려한 유사도 매트릭스 생성 (ID를 index로 사용, 중복 처리)
    
    Args:
        df: DataFrame
        id_column: ID 컬럼 이름 (예: 'Drug_ID')
        smiles_column: SMILES 컬럼 이름 (예: 'Drug')
        radius: Fingerprint radius
        fpSize: Fingerprint 크기
    
    Returns:
        sim_df: 유사도 매트릭스 (index/columns = ID)
    """
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize, includeChirality=True)

    ids = df[id_column].tolist()
    smiles_list = df[smiles_column].tolist()

    # Fingerprint 생성 (모든 행에 대해, invalid도 포함)
    fps = []
    invalid_indices = []
    print(f"Processing {len(smiles_list)} molecules (with chirality)...")
    for i, smi in enumerate(tqdm(smiles_list, desc="Generating fingerprints (with chirality)")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                fps.append(fp)
            else:
                # Dummy fingerprint (zero-vector)
                arr = np.zeros(fpSize, dtype=np.uint8)
                fp = DataStructs.cDataStructs.CreateFromBitString("0"*fpSize)
                fps.append(fp)
                invalid_indices.append(i)
        except:
            arr = np.zeros(fpSize, dtype=np.uint8)
            fp = DataStructs.cDataStructs.CreateFromBitString("0"*fpSize)
            fps.append(fp)
            invalid_indices.append(i)

    n = len(fps)
    sim_matrix = np.eye(n, dtype=np.float32)

    # Precompute which are invalid for fast lookup
    invalid_set = set(invalid_indices)
    print(f"Detected {len(invalid_indices)} invalid molecules (replaced with dummy fingerprints).")
    print(f"Computing similarities for all {n} molecules (with chirality)...")
    for i in tqdm(range(n - 1), desc="Computing Tanimoto similarities (with chirality)"):
        # If i is invalid, set all similarities to 0
        if i in invalid_set:
            sim_matrix[i, i + 1:] = 0.0
            continue
        sims = []
        for j in range(i + 1, n):
            if j in invalid_set:
                sims.append(0.0)
            else:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                sims.append(sim)
        sim_matrix[i, i + 1:] = np.array(sims, dtype=np.float32)

    # 대칭 복사
    sim_matrix = sim_matrix + sim_matrix.T - np.eye(n, dtype=np.float32)

    # DataFrame으로 변환 (ID 사용)
    valid_ids = list(ids)

    # 중복 ID 확인
    if len(valid_ids) != len(set(valid_ids)):
        print(f"⚠️ Warning: Found {len(valid_ids) - len(set(valid_ids))} duplicate IDs")
        print("Adding suffix to make IDs unique...")
        # 중복 처리: 같은 ID에 _1, _2 등 추가
        id_counts = {}
        unique_ids = []
        for id_val in valid_ids:
            if id_val in id_counts:
                id_counts[id_val] += 1
                unique_ids.append(f"{id_val}_{id_counts[id_val]}")
            else:
                id_counts[id_val] = 0
                unique_ids.append(id_val)
        valid_ids = unique_ids

    sim_df = pd.DataFrame(sim_matrix, index=valid_ids, columns=valid_ids)

    return sim_df


def build_similarity_matrix_from_df_simple_stereo(df, smiles_column='smiles', radius=2, fpSize=1024, use_float32=True):
    """
    DataFrame으로부터 입체화학을 고려한 유사도 매트릭스 생성 (SMILES를 index/columns로 사용, 간단한 버전)
    
    Args:
        df: DataFrame
        smiles_column: SMILES 컬럼 이름 (기본값: 'smiles')
        radius: Fingerprint radius
        fpSize: Fingerprint 크기
        use_float32: float32 사용 여부 (메모리 절약)
    
    Returns:
        sim_df: 유사도 매트릭스 DataFrame (index/columns = SMILES, 입체화학 고려)
    """
    assert smiles_column in df.columns, f"DataFrame must have '{smiles_column}' column"
    
    # SMILES 리스트 추출
    smiles_list = df[smiles_column].tolist()
    
    # 입체화학을 고려한 Fingerprint 생성
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize, includeChirality=True)
    fps = []
    valid_smiles = []
    valid_indices = []
    
    print(f"Processing {len(smiles_list)} molecules (with chirality)...")
    for i, smi in enumerate(tqdm(smiles_list, desc="Generating fingerprints (with chirality)")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                fps.append(fp)
                valid_smiles.append(smi)
                valid_indices.append(i)
        except:
            continue
    
    if not fps:
        print("Warning: No valid fingerprints generated. Returning empty matrix.")
        return pd.DataFrame()
    
    # 유사도 매트릭스 생성
    n = len(fps)
    dtype = np.float32 if use_float32 else np.float64
    sim_matrix = np.eye(n, dtype=dtype)
    
    print(f"Computing similarities for {n} valid molecules (with chirality)...")
    for i in tqdm(range(n - 1), desc="Computing Tanimoto similarities (with chirality)"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim_matrix[i, i + 1:] = np.array(sims, dtype=dtype)
    
    # 대칭 복사
    sim_matrix = sim_matrix + sim_matrix.T - np.eye(n, dtype=dtype)
    
    # DataFrame으로 변환 (SMILES를 index/columns로 사용)
    sim_df = pd.DataFrame(sim_matrix, index=valid_smiles, columns=valid_smiles)
    
    return sim_df


def build_cross_similarity_matrix_from_dfs(df1, id_column1, smiles_column1, 
                                           df2, id_column2, smiles_column2, 
                                           radius=2, fpSize=1024):
    """
    두 개의 DataFrame에 있는 분자들 간의 교차 유사도 매트릭스를 생성합니다.
    예: non_toxic_df와 toxic_df 간의 유사도.
    
    Args:
        df1: 첫 번째 DataFrame. id_column1과 smiles_column1을 포함해야 합니다.
        id_column1: 첫 번째 DataFrame의 ID 컬럼 이름.
        smiles_column1: 첫 번째 DataFrame의 SMILES 컬럼 이름.
        df2: 두 번째 DataFrame. id_column2와 smiles_column2를 포함해야 합니다.
        id_column2: 두 번째 DataFrame의 ID 컬럼 이름.
        smiles_column2: 두 번째 DataFrame의 SMILES 컬럼 이름.
        radius: Morgan Fingerprint radius (기본값: 2).
        fpSize: Fingerprint 크기 (기본값: 1024).
    
    Returns:
        cross_sim_df: df1의 ID를 인덱스로, df2의 ID를 컬럼으로 하는 교차 유사도 DataFrame.
                      유효하지 않은 SMILES는 NaN으로 처리됩니다.
    """
    print(f"Processing {len(df1)} molecules from df1 and {len(df2)} molecules from df2 for cross-similarity...")
    
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
    
    # df1 처리
    df1_ids = df1[id_column1].tolist()
    df1_smiles = df1[smiles_column1].tolist()
    
    df1_fps = []
    df1_valid_indices = []
    
    print("Generating fingerprints for df1...")
    for i, smi in enumerate(tqdm(df1_smiles, desc="df1 fingerprints")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                df1_fps.append(fp)
                df1_valid_indices.append(i)
        except:
            continue
    
    # df2 처리
    df2_ids = df2[id_column2].tolist()
    df2_smiles = df2[smiles_column2].tolist()
    
    df2_fps = []
    df2_valid_indices = []
    
    print("Generating fingerprints for df2...")
    for i, smi in enumerate(tqdm(df2_smiles, desc="df2 fingerprints")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                df2_fps.append(fp)
                df2_valid_indices.append(i)
        except:
            continue
    
    if not df1_fps or not df2_fps:
        print("Warning: No valid fingerprints generated for one or both DataFrames. Returning empty matrix.")
        return pd.DataFrame()
    
    # 유효한 ID들
    df1_valid_ids = [df1_ids[i] for i in df1_valid_indices]
    df2_valid_ids = [df2_ids[i] for i in df2_valid_indices]
    
    # 중복 ID 처리 (df1)
    if len(df1_valid_ids) != len(set(df1_valid_ids)):
        print(f"⚠️ Warning: Found {len(df1_valid_ids) - len(set(df1_valid_ids))} duplicate IDs in df1")
        id_counts = {}
        unique_df1_ids = []
        for id_val in df1_valid_ids:
            if id_val in id_counts:
                id_counts[id_val] += 1
                unique_df1_ids.append(f"{id_val}_{id_counts[id_val]}")
            else:
                id_counts[id_val] = 0
                unique_df1_ids.append(id_val)
        df1_valid_ids = unique_df1_ids
    
    # 중복 ID 처리 (df2)
    if len(df2_valid_ids) != len(set(df2_valid_ids)):
        print(f"⚠️ Warning: Found {len(df2_valid_ids) - len(set(df2_valid_ids))} duplicate IDs in df2")
        id_counts = {}
        unique_df2_ids = []
        for id_val in df2_valid_ids:
            if id_val in id_counts:
                id_counts[id_val] += 1
                unique_df2_ids.append(f"{id_val}_{id_counts[id_val]}")
            else:
                id_counts[id_val] = 0
                unique_df2_ids.append(id_val)
        df2_valid_ids = unique_df2_ids
    
    # 교차 유사도 매트릭스 생성
    n1, n2 = len(df1_fps), len(df2_fps)
    cross_sim_matrix = np.zeros((n1, n2), dtype=np.float32)
    
    print(f"Computing cross-similarity matrix ({n1} x {n2})...")
    for i in tqdm(range(n1), desc="Computing cross-similarities"):
        sims = DataStructs.BulkTanimotoSimilarity(df1_fps[i], df2_fps)
        cross_sim_matrix[i, :] = np.array(sims, dtype=np.float32)
    
    # DataFrame으로 변환
    cross_sim_df = pd.DataFrame(
        cross_sim_matrix,
        index=df1_valid_ids,
        columns=df2_valid_ids
    )
    
    print(f"✅ Cross-similarity matrix created: {cross_sim_df.shape}")
    return cross_sim_df


def build_cross_similarity_matrix_from_dfs_stereo(df1, id_column1, smiles_column1, 
                                                   df2, id_column2, smiles_column2, 
                                                   radius=2, fpSize=1024):
    """
    입체화학을 고려하여 두 개의 DataFrame에 있는 분자들 간의 교차 유사도 매트릭스를 생성합니다.
    예: non_toxic_df와 toxic_df 간의 유사도 (chirality 고려).
    
    Args:
        df1: 첫 번째 DataFrame. id_column1과 smiles_column1을 포함해야 합니다.
        id_column1: 첫 번째 DataFrame의 ID 컬럼 이름.
        smiles_column1: 첫 번째 DataFrame의 SMILES 컬럼 이름.
        df2: 두 번째 DataFrame. id_column2와 smiles_column2를 포함해야 합니다.
        id_column2: 두 번째 DataFrame의 ID 컬럼 이름.
        smiles_column2: 두 번째 DataFrame의 SMILES 컬럼 이름.
        radius: Morgan Fingerprint radius (기본값: 2).
        fpSize: Fingerprint 크기 (기본값: 1024).
    
    Returns:
        cross_sim_df: df1의 ID를 인덱스로, df2의 ID를 컬럼으로 하는 교차 유사도 DataFrame.
                      유효하지 않은 SMILES는 NaN으로 처리됩니다.
    """
    print(f"Processing {len(df1)} molecules from df1 and {len(df2)} molecules from df2 for cross-similarity (with chirality)...")
    
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize, includeChirality=True)
    
    # df1 처리
    df1_ids = df1[id_column1].tolist()
    df1_smiles = df1[smiles_column1].tolist()
    
    df1_fps = []
    df1_valid_indices = []
    
    print("Generating fingerprints for df1 (with chirality)...")
    for i, smi in enumerate(tqdm(df1_smiles, desc="df1 fingerprints (with chirality)")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                df1_fps.append(fp)
                df1_valid_indices.append(i)
        except:
            continue
    
    # df2 처리
    df2_ids = df2[id_column2].tolist()
    df2_smiles = df2[smiles_column2].tolist()
    
    df2_fps = []
    df2_valid_indices = []
    
    print("Generating fingerprints for df2 (with chirality)...")
    for i, smi in enumerate(tqdm(df2_smiles, desc="df2 fingerprints (with chirality)")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = fpgen.GetFingerprint(mol)
                df2_fps.append(fp)
                df2_valid_indices.append(i)
        except:
            continue
    
    if not df1_fps or not df2_fps:
        print("Warning: No valid fingerprints generated for one or both DataFrames. Returning empty matrix.")
        return pd.DataFrame()
    
    # 유효한 ID들
    df1_valid_ids = [df1_ids[i] for i in df1_valid_indices]
    df2_valid_ids = [df2_ids[i] for i in df2_valid_indices]
    
    # 중복 ID 처리 (df1)
    if len(df1_valid_ids) != len(set(df1_valid_ids)):
        print(f"⚠️ Warning: Found {len(df1_valid_ids) - len(set(df1_valid_ids))} duplicate IDs in df1")
        id_counts = {}
        unique_df1_ids = []
        for id_val in df1_valid_ids:
            if id_val in id_counts:
                id_counts[id_val] += 1
                unique_df1_ids.append(f"{id_val}_{id_counts[id_val]}")
            else:
                id_counts[id_val] = 0
                unique_df1_ids.append(id_val)
        df1_valid_ids = unique_df1_ids
    
    # 중복 ID 처리 (df2)
    if len(df2_valid_ids) != len(set(df2_valid_ids)):
        print(f"⚠️ Warning: Found {len(df2_valid_ids) - len(set(df2_valid_ids))} duplicate IDs in df2")
        id_counts = {}
        unique_df2_ids = []
        for id_val in df2_valid_ids:
            if id_val in id_counts:
                id_counts[id_val] += 1
                unique_df2_ids.append(f"{id_val}_{id_counts[id_val]}")
            else:
                id_counts[id_val] = 0
                unique_df2_ids.append(id_val)
        df2_valid_ids = unique_df2_ids
    
    # 교차 유사도 매트릭스 생성
    n1, n2 = len(df1_fps), len(df2_fps)
    cross_sim_matrix = np.zeros((n1, n2), dtype=np.float32)
    
    print(f"Computing cross-similarity matrix ({n1} x {n2}) (with chirality)...")
    for i in tqdm(range(n1), desc="Computing cross-similarities (with chirality)"):
        sims = DataStructs.BulkTanimotoSimilarity(df1_fps[i], df2_fps)
        cross_sim_matrix[i, :] = np.array(sims, dtype=np.float32)
    
    # DataFrame으로 변환
    cross_sim_df = pd.DataFrame(
        cross_sim_matrix,
        index=df1_valid_ids,
        columns=df2_valid_ids
    )
    
    print(f"✅ Cross-similarity matrix (with chirality) created: {cross_sim_df.shape}")
    return cross_sim_df


if __name__ == "__main__":
    # 기본 사용 예시
    print("=" * 60)
    print("Similarity utilities module loaded successfully!")
    print("=" * 60)
    
    # 테스트 데이터
    test_smiles = [
        'CCO',           # 에탄올
        'CCCO',          # 프로판올
        'c1ccccc1',      # 벤젠
        'c1ccccc1O',     # 페놀
    ]
    
    test_df = pd.DataFrame({'smiles': test_smiles})
    
    # 예시 1: Fingerprint 생성
    print("\n1. Generating fingerprints...")
    smi_list, fp_list = generate_fingerprints(test_df)
    print(f"Generated {len(fp_list)} fingerprints")
    
    # 예시 2: 유사 쌍 찾기
    print("\n2. Finding similar pairs (threshold=0.3)...")
    similar_pairs = list(get_similar_pairs(smi_list, fp_list, threshold=0.3))
    print(f"Found {len(similar_pairs)} similar pairs")
    for smi1, smi2, sim in similar_pairs:
        print(f"  {smi1} <-> {smi2}: {sim:.3f}")
    
    # 예시 3: 가장 유사한 분자 찾기
    print("\n3. Finding most similar molecules to CCO...")
    similar_df = find_most_similar_molecules('CCO', test_smiles[1:], top_k=3)
    print(similar_df)
    
    print("\n" + "=" * 60)
    print("📘 clintox_df 사용 가이드")
    print("=" * 60)
    print("""
clintox_df 데이터셋 구조:
    컬럼: ['task', 'dataset_title', 'dataset_description', 'task_description', 'Drug_ID', 'Drug', 'Y']
    - 'Drug': SMILES 문자열
    - 'Drug_ID': 고유 ID
    - 'Y': 독성 레이블

사용법 1: 전체 유사도 매트릭스 생성 (추천!)
-------------------------------------------------
from similarity_utils import build_similarity_matrix_from_df

sim_df = build_similarity_matrix_from_df(
    df=clintox_df,
    id_column='Drug_ID',      # ID 컬럼
    smiles_column='Drug',     # SMILES 컬럼
    radius=2,                 # Morgan FP radius
    fpSize=1024               # FP 크기
)

print(f"Shape: {sim_df.shape}")
sim_df.to_parquet('clintox_similarity.parquet')

# 특정 Drug의 유사도 확인
drug_id = 'Drug 0'
similar_drugs = sim_df.loc[drug_id].sort_values(ascending=False).head(10)
print(similar_drugs)


사용법 2: Threshold 이상인 쌍만 찾기 (메모리 효율적)
-------------------------------------------------
from similarity_utils import find_similar_pairs_from_df

pairs_df = find_similar_pairs_from_df(
    df=clintox_df,
    smiles_column='Drug',
    id_column='Drug_ID',
    threshold=0.7,
    radius=2,
    fpSize=1024
)

print(pairs_df.head())
# 결과: smiles_1, smiles_2, similarity, id_1, id_2

# 유사도 높은 순으로 정렬
pairs_df = pairs_df.sort_values('similarity', ascending=False)
pairs_df.to_csv('clintox_similar_pairs.csv', index=False)


사용법 3: 특정 분자와 유사한 분자 찾기
-------------------------------------------------
from similarity_utils import find_most_similar_molecules

target_smiles = clintox_df.iloc[0]['Drug']  # 첫 번째 분자
all_smiles = clintox_df['Drug'].tolist()

similar_df = find_most_similar_molecules(
    target_smiles=target_smiles,
    smiles_list=all_smiles,
    top_k=10,
    radius=2,
    fpSize=1024
)

print(similar_df)
# 결과: smiles, similarity (유사도 높은 순)


사용법 4: Fingerprint 재사용 (반복 계산 시 효율적)
-------------------------------------------------
from similarity_utils import generate_fingerprints, get_similar_pairs

# 1. DataFrame을 'smiles' 컬럼으로 변환
clintox_df_renamed = clintox_df.rename(columns={'Drug': 'smiles'})

# 2. Fingerprint 한 번만 생성
smi_list, fp_list = generate_fingerprints(clintox_df_renamed, radius=2, fpSize=1024)

# 3. 다양한 threshold로 쌍 찾기
for threshold in [0.5, 0.7, 0.9]:
    pairs = list(get_similar_pairs(smi_list, fp_list, threshold=threshold))
    print(f"Threshold {threshold}: {len(pairs)} pairs")


완전한 예제 코드:
-------------------------------------------------
import pandas as pd
from similarity_utils import build_similarity_matrix_from_df

# clintox_df 로드 (예시)
# clintox_df = pd.read_csv('clintox.csv')

# 전체 유사도 매트릭스 생성
sim_df = build_similarity_matrix_from_df(
    df=clintox_df,
    id_column='Drug_ID',
    smiles_column='Drug',
    radius=2,
    fpSize=1024
)

# 저장
sim_df.to_parquet('clintox_tanimoto_similarity.parquet')
print(f"✅ 유사도 매트릭스 저장 완료: {sim_df.shape}")

# 유사도 > 0.8인 쌍 필터링
high_sim_pairs = []
for i in range(len(sim_df)):
    for j in range(i+1, len(sim_df)):
        sim = sim_df.iloc[i, j]
        if sim > 0.8:
            high_sim_pairs.append({
                'drug_1': sim_df.index[i],
                'drug_2': sim_df.columns[j],
                'similarity': sim
            })

high_sim_df = pd.DataFrame(high_sim_pairs)
print(f"유사도 > 0.8인 쌍: {len(high_sim_df)}개")


사용법 5: 교차 유사도 매트릭스 (non_toxic vs toxic)
-------------------------------------------------
from similarity_utils import build_cross_similarity_matrix_from_dfs

# clintox_df를 toxic/non-toxic으로 분리
non_toxic_df = clintox_df[clintox_df['Y'] == 0].copy()
toxic_df = clintox_df[clintox_df['Y'] == 1].copy()

print(f"Non-toxic: {len(non_toxic_df)}개, Toxic: {len(toxic_df)}개")

# 교차 유사도 매트릭스 생성
cross_sim_df = build_cross_similarity_matrix_from_dfs(
    df1=non_toxic_df, id_column1='Drug_ID', smiles_column1='Drug',
    df2=toxic_df, id_column2='Drug_ID', smiles_column2='Drug',
    radius=2, fpSize=1024
)

print(f"Cross-similarity matrix shape: {cross_sim_df.shape}")
print("Non-toxic drugs (rows) vs Toxic drugs (columns)")

# 가장 유사한 쌍 찾기
max_similarity = cross_sim_df.max().max()
max_idx = cross_sim_df.stack().idxmax()
print(f"Highest similarity: {max_similarity:.4f} between {max_idx[0]} and {max_idx[1]}")

# 유사도 > 0.7인 쌍들
high_sim_pairs = []
for i in range(len(cross_sim_df)):
    for j in range(len(cross_sim_df.columns)):
        sim = cross_sim_df.iloc[i, j]
        if sim > 0.7:
            high_sim_pairs.append({
                'non_toxic_drug': cross_sim_df.index[i],
                'toxic_drug': cross_sim_df.columns[j],
                'similarity': sim
            })

high_sim_df = pd.DataFrame(high_sim_pairs)
print(f"Similarity > 0.7 pairs: {len(high_sim_df)}개")

# 결과 저장
cross_sim_df.to_parquet('clintox_cross_similarity.parquet')
high_sim_df.to_csv('clintox_high_similarity_pairs.csv', index=False)
print("✅ Cross-similarity results saved!")


사용법 6: DataFrame으로부터 간단한 유사도 매트릭스 생성 (SMILES 기반)
-------------------------------------------------
DataFrame을 넣으면 SMILES를 인덱스/컬럼으로 사용하는 유사도 매트릭스를 생성합니다.
ID 컬럼이 없어도 사용 가능한 간단한 버전입니다.

from similarity_utils import (
    build_similarity_matrix_from_df_simple,
    build_similarity_matrix_from_df_simple_stereo
)

# 예시 DataFrame
test_df = pd.DataFrame({
    'smiles': ['CCO', 'CCCO', 'c1ccccc1', 'c1ccccc1O'],
    'name': ['에탄올', '프로판올', '벤젠', '페놀']
})

# 1. 일반 유사도 매트릭스 (SMILES를 인덱스/컬럼으로 사용)
sim_df = build_similarity_matrix_from_df_simple(
    df=test_df,
    smiles_column='smiles',  # 기본값: 'smiles'
    radius=2,
    fpSize=1024
)

print(sim_df)
# 결과: SMILES를 인덱스/컬럼으로 하는 유사도 매트릭스
#        CCO      CCCO    c1ccccc1  c1ccccc1O
# CCO       1.0    0.85      0.12       0.15
# CCCO     0.85    1.0       0.10       0.13
# ...

# 특정 SMILES의 유사도 확인
target_smiles = 'CCO'
similar_smiles = sim_df.loc[target_smiles].sort_values(ascending=False)
print(f"\n{target_smiles}와 유사한 분자:")
print(similar_smiles.head())

# 2. 입체화학을 고려한 유사도 매트릭스
sim_df_stereo = build_similarity_matrix_from_df_simple_stereo(
    df=test_df,
    smiles_column='smiles',
    radius=2,
    fpSize=1024
)

# 유효하지 않은 SMILES는 자동으로 제외됩니다
print(f"유효한 분자 수: {len(sim_df_stereo)}")

# 유사도 > 0.7인 쌍 찾기
high_sim_pairs = []
for i in range(len(sim_df_stereo)):
    for j in range(i + 1, len(sim_df_stereo)):
        sim = sim_df_stereo.iloc[i, j]
        if sim > 0.7:
            high_sim_pairs.append({
                'smiles_1': sim_df_stereo.index[i],
                'smiles_2': sim_df_stereo.index[j],
                'similarity': sim
            })

high_sim_df = pd.DataFrame(high_sim_pairs)
print(f"\n유사도 > 0.7인 쌍: {len(high_sim_df)}개")
print(high_sim_df)

# 결과 저장
sim_df.to_parquet('similarity_matrix.parquet')
print("✅ 유사도 매트릭스 저장 완료!")


사용법 7: 입체화학을 고려한 유사도 계산 (includeChirality=True)
-------------------------------------------------
입체화학(R/S, E/Z configuration)까지 고려하여 유사도를 계산하려면 
_stereo suffix가 붙은 함수들을 사용하세요.

from similarity_utils import (
    generate_fingerprints_stereo,
    build_full_similarity_matrix_stereo,
    find_most_similar_molecules_stereo,
    build_similarity_matrix_from_df_stereo,
    build_similarity_matrix_from_df_simple_stereo,
    build_cross_similarity_matrix_from_dfs_stereo
)

# 1. DataFrame에서 입체화학을 고려한 fingerprint 생성
smi_list, fp_list = generate_fingerprints_stereo(df, radius=2, fpSize=1024)

# 2. 입체화학을 고려한 전체 유사도 매트릭스 (SMILES 리스트 사용)
sim_matrix, valid_smiles = build_full_similarity_matrix_stereo(
    smiles_list=['C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O'],  # L-Alanine vs D-Alanine
    radius=2, 
    fpSize=1024
)
print(f"Similarity between enantiomers: {sim_matrix[0, 1]:.4f}")

# 2-1. 입체화학을 고려한 전체 유사도 매트릭스 (DataFrame 사용, id_column 없음)
sim_matrix_df, valid_smiles_df = build_full_similarity_matrix_stereo(
    df=clintox_df,
    smiles_column='Drug',  # SMILES 컬럼 지정
    radius=2,
    fpSize=1024
)
print(f"DataFrame 기반 유사도 매트릭스 shape: {sim_matrix_df.shape}")

# 2-2. 입체화학을 고려한 전체 유사도 매트릭스 (DataFrame 사용, id_column 있음)
sim_df_stereo_id = build_full_similarity_matrix_stereo(
    df=clintox_df,
    smiles_column='Drug',
    id_column='Drug_ID',  # ID 컬럼 지정 → DataFrame 반환
    radius=2,
    fpSize=1024
)
print(f"ID 기반 유사도 매트릭스 shape: {sim_df_stereo_id.shape}")
print(f"Index/Columns: {sim_df_stereo_id.index.tolist()[:5]}...")  # 처음 5개 ID 확인

# 3. 특정 분자와 유사한 분자 찾기 (입체화학 고려)
target_smiles = 'C[C@H](N)C(=O)O'  # L-Alanine
similar_df = find_most_similar_molecules_stereo(
    target_smiles=target_smiles,
    smiles_list=all_smiles,
    top_k=10,
    radius=2,
    fpSize=1024
)

# 4. DataFrame으로부터 입체화학을 고려한 유사도 매트릭스 (ID 기반)
sim_df_stereo = build_similarity_matrix_from_df_stereo(
    df=clintox_df,
    id_column='Drug_ID',
    smiles_column='Drug',
    radius=2,
    fpSize=1024
)

# 4-1. DataFrame으로부터 입체화학을 고려한 유사도 매트릭스 (SMILES 기반, 간단한 버전)
sim_df_stereo_simple = build_similarity_matrix_from_df_simple_stereo(
    df=clintox_df,
    smiles_column='Drug',  # ID 컬럼 불필요
    radius=2,
    fpSize=1024
)

# 5. 교차 유사도 매트릭스 (입체화학 고려)
cross_sim_df_stereo = build_cross_similarity_matrix_from_dfs_stereo(
    df1=non_toxic_df, id_column1='Drug_ID', smiles_column1='Drug',
    df2=toxic_df, id_column2='Drug_ID', smiles_column2='Drug',
    radius=2, fpSize=1024
)

# 입체화학 고려 여부 비교 예시
print("\n입체화학 고려 여부에 따른 유사도 차이 (L-Alanine vs D-Alanine):")

# 기존 방식 (입체화학 미고려)
sim_matrix_normal, _ = build_full_similarity_matrix(
    ['C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O']
)
print(f"입체화학 미고려: {sim_matrix_normal[0, 1]:.4f}")

# 입체화학 고려 방식 (SMILES 리스트)
sim_matrix_stereo, _ = build_full_similarity_matrix_stereo(
    smiles_list=['C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O']
)
print(f"입체화학 고려 (리스트): {sim_matrix_stereo[0, 1]:.4f}")

# 입체화학 고려 방식 (DataFrame, id_column 없음)
test_df = pd.DataFrame({
    'smiles': ['C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O']
})
sim_matrix_stereo_df, _ = build_full_similarity_matrix_stereo(
    df=test_df,
    smiles_column='smiles'
)
print(f"입체화학 고려 (DataFrame, no ID): {sim_matrix_stereo_df[0, 1]:.4f}")

# 입체화학 고려 방식 (DataFrame, id_column 있음)
test_df_with_id = pd.DataFrame({
    'id': ['L-Ala', 'D-Ala'],
    'smiles': ['C[C@H](N)C(=O)O', 'C[C@@H](N)C(=O)O']
})
sim_df_stereo_with_id = build_full_similarity_matrix_stereo(
    df=test_df_with_id,
    smiles_column='smiles',
    id_column='id'  # ID 기반 DataFrame 반환
)
print(f"입체화학 고려 (DataFrame, with ID): {sim_df_stereo_with_id.loc['L-Ala', 'D-Ala']:.4f}")
print("→ 거울상 이성질체(enantiomer)는 입체화학을 고려하면 유사도가 낮아집니다!")

⚠️ 주의사항:
- includeChirality=True는 Morgan Fingerprint에만 적용됩니다
- SMILES에 입체화학 정보(@, @@, /, \)가 없으면 일반 함수와 동일한 결과
- 입체화학을 구분해야 하는 경우 (거울상 이성질체, 기하 이성질체 등)에만 사용 권장
- 계산 시간이 약간 더 소요될 수 있습니다
    """)