"""
스캐폴드 유사도를 0.9가 아닌 **1**로 했을 때의 스타일별·조건별 샘플 수를 집계합니다.

sim_matrices + scaffold_sim pkl(분자식 계산용)을 사용하며, ToxCast 제외.
"""
from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

OUT_DIR = Path(__file__).resolve().parent
BASE = OUT_DIR.parent
SIM_MATRICES_DIR = OUT_DIR / "sim_matrices"
SCAFFOLD_SIM_DIR = BASE / "scaffold_sim"
SUMMARY_CSV = SCAFFOLD_SIM_DIR / "summary.csv"
SIM_THRESHOLD = 0.9
SCAFFOLD_EQ1_THRESHOLD = 1.0 - 1e-6  # scaffold 유사도 1


def _mol_formula(smiles: str) -> str:
    if not smiles or not isinstance(smiles, str):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return rdMolDescriptors.CalcMolFormula(mol) or ""
    except Exception:
        return ""


def main() -> None:
    if not SIM_MATRICES_DIR.exists():
        print(f"Not found: {SIM_MATRICES_DIR}")
        return
    if not SUMMARY_CSV.exists():
        print(f"Not found: {SUMMARY_CSV}")
        return

    summary = pd.read_csv(SUMMARY_CSV)
    summary["_safe_ep"] = summary["path_npz"].apply(lambda p: Path(p).stem)
    summary = summary[~summary["dataset"].isin({"toxcast_df"})]

    totals_cond = {"sub_wo": 0, "sub_w_chiral": 0, "scaffold_eq1": 0, "smiles": 0}
    totals_style = {"mol_fg": 0, "mol_stereo": 0, "mol_isomer": 0}

    for _, row in tqdm(summary.iterrows(), total=len(summary), desc="Endpoints"):
        dataset, endpoint = row["dataset"], row["endpoint"]
        safe_ep = row["_safe_ep"]
        npz_path = SIM_MATRICES_DIR / dataset / f"{safe_ep}.npz"
        npz_dir = (BASE / Path(row["path_npz"])).parent
        path_toxic = npz_dir / f"{safe_ep}_toxic_smiles.pkl"
        path_nontoxic = npz_dir / f"{safe_ep}_nontoxic_smiles.pkl"
        if not npz_path.exists() or not path_toxic.exists() or not path_nontoxic.exists():
            continue
        try:
            data = np.load(npz_path)
            sub_wo = data["substructure_sim"]
            sub_w = data["substructure_sim_w_chiral"]
            scaffold = data["scaffold_sim"]
            smiles = data["smiles_sim"]
        except Exception:
            continue
        with open(path_toxic, "rb") as f:
            toxic_smiles = pickle.load(f)
        with open(path_nontoxic, "rb") as f:
            nontoxic_smiles = pickle.load(f)
        formula_t = np.array([_mol_formula(s) for s in toxic_smiles], dtype=object)
        formula_n = np.array([_mol_formula(s) for s in nontoxic_smiles], dtype=object)
        same_formula = (formula_t[:, None] == formula_n[None, :]) & (formula_t[:, None] != "")

        sub_wo_ok = np.isfinite(sub_wo) & (sub_wo >= SIM_THRESHOLD)
        sub_w_ok = np.isfinite(sub_w) & (sub_w >= SIM_THRESHOLD)
        scaffold_ok = np.isfinite(scaffold) & (scaffold >= SCAFFOLD_EQ1_THRESHOLD)
        smiles_ok = np.isfinite(smiles) & (smiles >= SIM_THRESHOLD)

        totals_cond["sub_wo"] += int(np.sum(sub_wo_ok))
        totals_cond["sub_w_chiral"] += int(np.sum(sub_w_ok))
        totals_cond["scaffold_eq1"] += int(np.sum(scaffold_ok))
        totals_cond["smiles"] += int(np.sum(smiles_ok))

        pass_fg = sub_wo_ok | scaffold_ok | smiles_ok
        pass_stereo = sub_w_ok | scaffold_ok | smiles_ok
        pass_isomer = (sub_w_ok | scaffold_ok | smiles_ok) & same_formula

        totals_style["mol_fg"] += int(np.sum(pass_fg))
        totals_style["mol_stereo"] += int(np.sum(pass_stereo))
        totals_style["mol_isomer"] += int(np.sum(pass_isomer))

    print()
    print("스캐폴드 유사도 **1** 기준 (ToxCast 제외)")
    print()
    print("【스타일】")
    print("-" * 55)
    print(f"  {'스타일':<12}  {'경로':<42}  {'샘플수':>12}")
    print("-" * 55)
    print(f"  {'Mol_FG':<12}  {'molecularACE_ver/mol_fg/pairs.csv':<42}  {totals_style['mol_fg']:>12,}")
    print(f"  {'Mol_stereo':<12}  {'molecularACE_ver/mol_stereo/pairs.csv':<42}  {totals_style['mol_stereo']:>12,}")
    print(f"  {'Mol_isomer':<12}  {'molecularACE_ver/mol_isomer/pairs.csv':<42}  {totals_style['mol_isomer']:>12,}")
    print()
    print("【조건】")
    print("-" * 55)
    print(f"  {'조건':<45}  {'샘플수':>12}")
    print("-" * 55)
    print(f"  {'Substructure (w/o chirality) ≥ 0.9':<45}  {totals_cond['sub_wo']:>12,}")
    print(f"  {'Substructure (w/ chirality)  ≥ 0.9':<45}  {totals_cond['sub_w_chiral']:>12,}")
    print(f"  {'Scaffold ≥ 1 (스캐폴드 유사도 1)':<45}  {totals_cond['scaffold_eq1']:>12,}")
    print(f"  {'SMILES similarity ≥ 0.9':<45}  {totals_cond['smiles']:>12,}")
    print("-" * 55)


if __name__ == "__main__":
    main()
