"""
toxic_safe_fragments_smiles, nontoxic_safe_fragments_smiles를 기준으로
전체 분자(toxic_safe_decoded_smiles / nontoxic_safe_decoded_smiles) 내 atom index를 추출하고,
only_toxic_frag, only_nontoxic_frag에도 동일 방식으로 atom index를 적용.
여러 fragment / 여러 매칭인 경우 각각의 atom index 집합을 모두 저장 (| 로 구분).
"""
import re
from pathlib import Path

import pandas as pd
import datamol as dm
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles_matched.csv"
OUTPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles_matched.csv"  # 같은 파일에 컬럼 추가

SEP_FRAG = "."
SEP_MATCH = "|"


def fragment_smiles_to_smarts(smiles_with_dummies: str) -> str:
    """[*], [*:2], [*:7] 등 부착점 더미를 *(RDKit SMARTS 임의 원자)로 치환"""
    if not smiles_with_dummies or pd.isna(smiles_with_dummies):
        return ""
    return re.sub(r"\[\*[^\]]*\]", "*", str(smiles_with_dummies).strip())


def get_all_matched_atom_indices(mol, query_smarts: str) -> list[list[int]]:
    """
    mol에서 query_smarts로 서브구조 매칭한 모든 결과의 atom index 리스트 반환.
    반환: [[0,1,2,3], [4,5,6,7], ...] (각 매칭마다 정렬된 원자 인덱스 리스트)
    """
    if mol is None or not query_smarts or pd.isna(query_smarts):
        return []
    try:
        q = dm.from_smarts(query_smarts)
        if q is None:
            return []
        matches = mol.GetSubstructMatches(q, uniquify=True)
        return [sorted(list(m)) for m in matches]
    except Exception:
        return []


def fragments_smiles_to_atom_indices_str(
    fragments_smiles_str: str,
    full_mol_smiles: str,
    sep_frag: str = SEP_FRAG,
    sep_match: str = SEP_MATCH,
) -> str:
    """
    fragments_smiles_str: 점으로 구분된 fragment SMILES (더미 포함).
    full_mol_smiles: 전체 분자 SMILES.
    각 fragment를 SMARTS로 바꾼 뒤 full_mol에 매칭하고, 모든 매칭의 atom index를 반환.
    형식: "0,1,2,3|4,5,6|7,8" (매칭 구간은 쉼표, 매칭/프래그먼트 구간은 |)
    """
    if not fragments_smiles_str or pd.isna(fragments_smiles_str) or not full_mol_smiles or pd.isna(full_mol_smiles):
        return ""
    fragments = [f.strip() for f in str(fragments_smiles_str).split(sep_frag) if f.strip()]
    if not fragments:
        return ""

    with dm.without_rdkit_log():
        mol = dm.to_mol(str(full_mol_smiles).strip())
    if mol is None:
        return ""

    all_parts = []
    for frag_smiles in fragments:
        smarts = fragment_smiles_to_smarts(frag_smiles)
        if not smarts:
            continue
        matches = get_all_matched_atom_indices(mol, smarts)
        for atom_list in matches:
            all_parts.append(",".join(map(str, atom_list)))
    return sep_match.join(all_parts)


def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required = [
        "toxic_safe_fragments_smiles",
        "nontoxic_safe_fragments_smiles",
        "only_toxic_safe_fragments_smiles",
        "only_nontoxic_safe_fragments_smiles",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # 1) toxic_safe_fragments_smiles 기준 전체 fragment atom indices
    print("Extracting toxic_safe_fragments_atom_indices...")
    toxic_frag_indices = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="toxic_frag"):
        s = fragments_smiles_to_atom_indices_str(
            row["toxic_safe_fragments_smiles"],
            row["toxic_safe_decoded_smiles"],
        )
        toxic_frag_indices.append(s)

    # 2) nontoxic_safe_fragments_smiles 기준
    print("Extracting nontoxic_safe_fragments_atom_indices...")
    nontoxic_frag_indices = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="nontoxic_frag"):
        s = fragments_smiles_to_atom_indices_str(
            row["nontoxic_safe_fragments_smiles"],
            row["nontoxic_safe_decoded_smiles"],
        )
        nontoxic_frag_indices.append(s)

    # 3) only_toxic_frag atom indices (toxic 분자 기준)
    print("Extracting only_toxic_frag_atom_indices...")
    only_toxic_indices = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="only_toxic"):
        s = fragments_smiles_to_atom_indices_str(
            row["only_toxic_safe_fragments_smiles"],
            row["toxic_safe_decoded_smiles"],
        )
        only_toxic_indices.append(s)

    # 4) only_nontoxic_frag atom indices (nontoxic 분자 기준)
    print("Extracting only_nontoxic_frag_atom_indices...")
    only_nontoxic_indices = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="only_nontoxic"):
        s = fragments_smiles_to_atom_indices_str(
            row["only_nontoxic_safe_fragments_smiles"],
            row["nontoxic_safe_decoded_smiles"],
        )
        only_nontoxic_indices.append(s)

    df = df.assign(
        toxic_safe_fragments_atom_indices=toxic_frag_indices,
        nontoxic_safe_fragments_atom_indices=nontoxic_frag_indices,
        only_toxic_frag_atom_indices=only_toxic_indices,
        only_nontoxic_frag_atom_indices=only_nontoxic_indices,
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")
    print("Added columns: toxic_safe_fragments_atom_indices, nontoxic_safe_fragments_atom_indices,")
    print("               only_toxic_frag_atom_indices, only_nontoxic_frag_atom_indices")


if __name__ == "__main__":
    main()
