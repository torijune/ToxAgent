"""
Pair CSV(tanimoto_512_0.7_pairs_all.csv)의 각 (toxic, nontoxic) 쌍에 대해
분자 물성(MW, cLogP, TPSA, HBD, HBA, Rotatable bonds, Ring count, Aromatic ring count)의
차이 절댓값 |nontoxic - toxic| 을 계산해 CSV에 컬럼으로 추가해 저장합니다.

참고: build_style_delta_matrices.py, extract_filtered_pairs_descriptors.py 와 동일한 descriptor 정의.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

BASE = Path(__file__).resolve().parent.parent  # ace_safe_ver
DEFAULT_TRAIN = BASE / "splits" / "scaffold_by_endpoint_unseen_ver" / "merged_train.csv"
DEFAULT_TEST = BASE / "splits" / "scaffold_by_endpoint_unseen_ver" / "merged_test.csv"

DESCRIPTOR_NAMES = ["MW", "logP", "TPSA", "HBD", "HBA", "RotB", "RingCount", "AromRingCount"]


def get_descriptors(smiles: str) -> dict | None:
    """단일 SMILES에 대해 RDKit 물성 dict 반환. 실패 시 None."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            "MW": Descriptors.ExactMolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "TPSA": MolSurf.TPSA(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotB": Lipinski.NumRotatableBonds(mol),
            "RingCount": Lipinski.RingCount(mol),
            "AromRingCount": rdMolDescriptors.CalcNumAromaticRings(mol),
        }
    except Exception:
        return None


def add_property_deltas(
    df: pd.DataFrame,
    toxic_col: str = "toxic_smiles",
    nontoxic_col: str = "nontoxic_smiles",
    cache: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    df에 toxic/nontoxic property 및 delta 컬럼을 추가.

    - toxic_<name>, nontoxic_<name>
    - delta_<name>      : toxic - nontoxic (signed)
    - delta_abs_<name>  : |toxic - nontoxic|
    """
    if toxic_col not in df.columns or nontoxic_col not in df.columns:
        raise ValueError(f"CSV must have '{toxic_col}' and '{nontoxic_col}'.")

    n_pairs = len(df)
    cache = cache if cache is not None else {}

    toxic_prop = {name: [float("nan")] * n_pairs for name in DESCRIPTOR_NAMES}
    nontoxic_prop = {name: [float("nan")] * n_pairs for name in DESCRIPTOR_NAMES}
    delta_signed = {name: [float("nan")] * n_pairs for name in DESCRIPTOR_NAMES}
    delta_abs = {name: [float("nan")] * n_pairs for name in DESCRIPTOR_NAMES}

    it = df.iterrows()
    if verbose:
        it = tqdm(it, total=n_pairs, desc="Property delta (toxic/nontoxic)")

    for pos, (_, row) in enumerate(it):
        smi_t = row[toxic_col]
        smi_n = row[nontoxic_col]

        if smi_t not in cache:
            cache[smi_t] = get_descriptors(smi_t)
        if smi_n not in cache:
            cache[smi_n] = get_descriptors(smi_n)

        desc_t = cache.get(smi_t)
        desc_n = cache.get(smi_n)
        if desc_t is None or desc_n is None:
            continue

        for name in DESCRIPTOR_NAMES:
            vt, vn = desc_t.get(name), desc_n.get(name)
            if vt is None or vn is None:
                continue
            if not (np.isfinite(vt) and np.isfinite(vn)):
                continue
            vt_f = float(vt)
            vn_f = float(vn)
            toxic_prop[name][pos] = vt_f
            nontoxic_prop[name][pos] = vn_f
            d = vt_f - vn_f
            delta_signed[name][pos] = d
            delta_abs[name][pos] = abs(d)

    for name in DESCRIPTOR_NAMES:
        df[f"toxic_{name}"] = toxic_prop[name]
        df[f"nontoxic_{name}"] = nontoxic_prop[name]
        df[f"delta_{name}"] = delta_signed[name]
        df[f"delta_abs_{name}"] = delta_abs[name]

    return df


def run_one(input_csv: Path, output_csv: Path, cache: dict | None = None, verbose: bool = True) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Not found: {input_csv}")
    df = pd.read_csv(input_csv)
    df = add_property_deltas(df, toxic_col="toxic_smiles", nontoxic_col="nontoxic_smiles", cache=cache, verbose=verbose)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    n_pairs = len(df)
    n_valid = df["delta_abs_MW"].notna().sum() if "delta_abs_MW" in df.columns else 0
    print(f"Saved: {output_csv}")
    print(f"  Pairs: {n_pairs:,}, valid(MW computed): {n_valid:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Compute RDKit molecular property deltas for toxic/nontoxic pairs (pair CSV).",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=None,
        help="단일 pair CSV 입력 경로. 지정 시 이 파일만 전체 계산 후 --output으로 저장.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="--input 사용 시 출력 CSV 경로. 미지정이면 <input>_property_delta.csv 로 저장.",
    )
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Input merged_train.csv path")
    ap.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Input merged_test.csv path")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=BASE / "splits" / "scaffold_by_endpoint_unseen_ver",
        help="Output directory (default: same split dir).",
    )
    ap.add_argument("--quiet", action="store_true", help="Less output")
    args = ap.parse_args()

    cache: dict = {}
    if args.input is not None:
        in_path = Path(args.input)
        if args.output is not None:
            out_path = Path(args.output)
        else:
            suffix = in_path.suffix if in_path.suffix else ".csv"
            out_path = in_path.with_name(in_path.stem + "_property_delta" + suffix)
        run_one(in_path, out_path, cache=cache, verbose=not args.quiet)
    else:
        out_train = args.out_dir / "merged_train_property_delta.csv"
        out_test = args.out_dir / "merged_test_property_delta.csv"
        run_one(args.train, out_train, cache=cache, verbose=not args.quiet)
        run_one(args.test, out_test, cache=cache, verbose=not args.quiet)
