"""
only_toxic_safe_fragments_smiles, only_nontoxic_safe_fragments_smiles를 SMARTS로 변환한 뒤
toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles에 서브구조 매칭하여
매칭된 부분의 SMILES를 only_toxic_frag_smiles, only_nontoxic_frag_smiles 컬럼으로 저장.
"""
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
import datamol as dm
from tqdm import tqdm

# paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles.csv"
OUTPUT_CSV = SCRIPT_DIR / "commom_frage_pairs_with_smiles_matched.csv"

SEP = "."


def fragment_smiles_to_smarts(smiles_with_dummies: str) -> str:
    """[*], [*:2], [*:7] 등 부착점 더미를 *(RDKit SMARTS 임의 원자)로 치환"""
    if not smiles_with_dummies or pd.isna(smiles_with_dummies):
        return ""
    return re.sub(r"\[\*[^\]]*\]", "*", str(smiles_with_dummies).strip())


def get_matched_substructure_smiles(mol, query_smarts: str) -> str:
    """
    mol에서 query_smarts로 서브구조 매칭 후, 첫 번째 매칭에 해당하는 부분분자 SMILES 반환.
    매칭 없으면 빈 문자열.
    """
    if mol is None or not query_smarts or pd.isna(query_smarts):
        return ""
    try:
        q = dm.from_smarts(query_smarts)
        if q is None:
            return ""
        matches = mol.GetSubstructMatches(q, uniquify=True)
        if not matches:
            return ""
        atom_idx = list(matches[0])
        return Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_idx,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return ""


def fragments_smiles_to_matched_smiles(
    fragments_smiles_str: str,
    full_mol_smiles: str,
    sep: str = SEP,
) -> str:
    """
    fragments_smiles_str: 점으로 구분된 fragment SMILES (더미 포함).
    full_mol_smiles: 전체 분자 SMILES.
    각 fragment를 SMARTS로 바꾼 뒤 full_mol에 매칭하고, 매칭된 부분의 SMILES를 이어서 반환.
    """
    if not fragments_smiles_str or pd.isna(fragments_smiles_str) or not full_mol_smiles or pd.isna(full_mol_smiles):
        return ""
    fragments = [f.strip() for f in str(fragments_smiles_str).split(sep) if f.strip()]
    if not fragments:
        return ""

    with dm.without_rdkit_log():
        mol = dm.to_mol(str(full_mol_smiles).strip())
    if mol is None:
        return ""

    matched_smiles_list = []
    for frag_smiles in fragments:
        smarts = fragment_smiles_to_smarts(frag_smiles)
        if not smarts:
            continue
        s = get_matched_substructure_smiles(mol, smarts)
        if s:
            matched_smiles_list.append(s)
    return sep.join(matched_smiles_list)


def main():
    print(f"Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required = [
        "only_toxic_safe_fragments_smiles",
        "only_nontoxic_safe_fragments_smiles",
        "toxic_safe_decoded_smiles",
        "nontoxic_safe_decoded_smiles",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    print("Building only_toxic_frag_smiles (only_toxic SMARTS → toxic_safe_decoded_smiles match)...")
    only_toxic_frag_smiles = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="only_toxic match"):
        s = fragments_smiles_to_matched_smiles(
            row["only_toxic_safe_fragments_smiles"],
            row["toxic_safe_decoded_smiles"],
            sep=SEP,
        )
        only_toxic_frag_smiles.append(s)

    print("Building only_nontoxic_frag_smiles (only_nontoxic SMARTS → nontoxic_safe_decoded_smiles match)...")
    only_nontoxic_frag_smiles = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="only_nontoxic match"):
        s = fragments_smiles_to_matched_smiles(
            row["only_nontoxic_safe_fragments_smiles"],
            row["nontoxic_safe_decoded_smiles"],
            sep=SEP,
        )
        only_nontoxic_frag_smiles.append(s)

    df = df.assign(
        only_toxic_frag_smiles=only_toxic_frag_smiles,
        only_nontoxic_frag_smiles=only_nontoxic_frag_smiles,
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} (added columns: only_toxic_frag_smiles, only_nontoxic_frag_smiles)")


if __name__ == "__main__":
    main()
