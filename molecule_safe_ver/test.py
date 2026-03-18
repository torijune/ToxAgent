import re
import datamol as dm
from rdkit import Chem

def fragment_smiles_to_smarts(smiles_with_dummies: str) -> str:
    """[*], [*:2], [*:7] 등 부착점 더미를 *(RDKit SMARTS 임의 원자)로 치환"""
    return re.sub(r"\[\*[^\]]*\]", "*", smiles_with_dummies)

def print_substructure_hits(mol, query, mol_name: str, query_name: str, max_hits: int = 5):
    """서브구조 매칭 True/False + 매칭된 부분의 SMILES들을 출력"""
    if mol is None or query is None:
        print(f"  {query_name} → {mol_name}: query/mol is None")
        return

    matches = mol.GetSubstructMatches(query, uniquify=True)
    print(f"  {query_name} → {mol_name}: {len(matches)} hit(s)")

    if not matches:
        return

    # 매칭된 각 hit에서 해당 원자들만 뽑아 fragment SMILES 생성
    for i, atom_idx_tuple in enumerate(matches[:max_hits], start=1):
        atom_idx = list(atom_idx_tuple)
        frag_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_idx,
            canonical=True,
            isomericSmiles=True
        )
        print(f"    - hit {i}: {frag_smiles}   (atoms={atom_idx})")

# 예시 fragment (only_toxic / only_nontoxic)
frag_toxic = "[CH2]([CH2][*:7])[*:2]"
frag_nontoxic = "[C](=[O])([CH2][CH2][*:7])[*:2]"

# SMARTS로 변환 적용 (string)
smarts_toxic = fragment_smiles_to_smarts(frag_toxic)
smarts_nontoxic = fragment_smiles_to_smarts(frag_nontoxic)

print("Fragment → SMARTS string (부착점 [*:n] → * 로 치환):")
print("  only_toxic:   ", smarts_toxic)
print("  only_nontoxic:", smarts_nontoxic)

# 전체 decode SMILES (한 분자)
toxic_full = "c1(CCOC)ccc(OCC(O)CNC(C)C)cc1"
nontoxic_full = "c1(CCC(=O)OC)ccc(OCC(O)CNC(C)C)cc1"

with dm.without_rdkit_log():
    mol_toxic = dm.to_mol(toxic_full)
    mol_nontoxic = dm.to_mol(nontoxic_full)

    # query mol 생성
    q_toxic = dm.from_smarts(smarts_toxic)
    q_nontoxic = dm.from_smarts(smarts_nontoxic)

print("\nQuery Mol → SMARTS (RDKit query로 파싱된 형태):")
print("  q_toxic SMARTS:   ", dm.to_smarts(q_toxic) if q_toxic else None)
print("  q_nontoxic SMARTS:", dm.to_smarts(q_nontoxic) if q_nontoxic else None)

print("\n서브구조 매칭 + 매칭된 부분 SMILES:")
print_substructure_hits(mol_toxic,    q_toxic,    "toxic_full",    "only_toxic SMARTS")
# print_substructure_hits(mol_nontoxic, q_toxic,    "nontoxic_full", "only_toxic SMARTS")
# print_substructure_hits(mol_toxic,    q_nontoxic, "toxic_full",    "only_nontoxic SMARTS")
print_substructure_hits(mol_nontoxic, q_nontoxic, "nontoxic_full", "only_nontoxic SMARTS")