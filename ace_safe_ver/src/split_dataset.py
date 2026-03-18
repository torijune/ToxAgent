"""
Endpoint-wise split for ACE SAFE pair dataset.

Input (default)
---------------
- pairs_safe_filtered_valid.csv (SAFE decode 검증 통과 데이터, ace_safe_ver 루트)

This script groups rows by (dataset_name, endpoint) and performs either a
Bemis–Murcko scaffold split or a UMAP-based split within each group.
By default: scaffold split, train:test 8:2 (valid 없음). UMAP split 시 시각화 저장.

Unseen-endpoint split mode (--use_unseen_split)
------------------------------------------------
- n_total <= unseen_threshold (default 89, Q1): 해당 endpoint 전체를 unseen endpoint test로 저장
  → out_dir/unseen_endpoint_test/<dataset>/<endpoint>/test.csv
- n_total > threshold: 9:1 scaffold split, test 최소 min_test_pairs(기본 30) 보장
  → out_dir/train_test/<dataset>/<endpoint>/train.csv, test.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
# ace_safe_ver 루트 디렉토리 (pairs_safe_filtered.csv 위치)
PROJECT_ROOT = SCRIPT_DIR.parent

# Local import (avoid package install requirement)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from spliter import ScaffoldSplitter, UMAPSplitter
except Exception as e:  # pragma: no cover
    raise ImportError(
        f"splitter import 실패: {e}. 경로를 확인하세요: {SCRIPT_DIR / 'spliter.py'}"
    )

# 기본: SAFE decode 검증 통과한 valid 데이터로 scaffold 8:2 split
DEFAULT_INPUT = PROJECT_ROOT / "pairs_safe_filtered_full_herg_metabolism_sider.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "splits" / "scaffold_by_endpoint"
# herg 통합본으로 새로 split 할 때
INPUT_HERG_MERGED = PROJECT_ROOT / "pairs_safe_filtered_herg_merged.csv"
OUT_DIR_HERG_MERGED = PROJECT_ROOT / "splits" / "scaffold_by_endpoint_herg_merged"


def _sanitize_for_path(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "unknown"
    # Replace path-hostile characters
    for ch in ["/", "\\", ":", ";", "|", "?", "*", "<", ">", "\"", "\n", "\t"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._ ") or "unknown"


def _choose_smiles_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"--smiles_col={requested!r} 컬럼이 CSV에 없습니다.")
        return requested
    # Prefer decoded/canonical-ish smiles if present
    for c in ["toxic_safe_decoded_smiles", "toxic_smiles"]:
        if c in df.columns:
            return c
    raise ValueError("SMILES 컬럼을 찾을 수 없습니다. (--smiles_col로 지정하세요)")


def _split_indices(
    smiles_list: list[str],
    frac_train: float,
    frac_valid: float,
    frac_test: float,
    seed: Optional[int],
    split_method: str = "scaffold",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_n_components: int = 2,
    umap_metric: str = "jaccard",
    umap_n_clusters: Optional[int] = None,
) -> Tuple[list[int], list[int], list[int]]:
    class _SimpleDataset:
        def __init__(self, ids: list[str]):
            self.ids = ids

        def __len__(self) -> int:
            return len(self.ids)

    ds = _SimpleDataset(smiles_list)

    split_method = str(split_method).strip().lower()
    if split_method == "scaffold":
        splitter = ScaffoldSplitter()
    elif split_method == "umap":
        splitter = UMAPSplitter(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            n_components=umap_n_components,
            metric=umap_metric,
            n_clusters=umap_n_clusters,
        )
    else:
        raise ValueError(
            f"지원하지 않는 split_method={split_method!r}. 'scaffold' 또는 'umap'을 사용하세요."
        )

    train_inds, valid_inds, test_inds = splitter.split(
        ds,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed,
    )
    return list(train_inds), list(valid_inds), list(test_inds)


def _save_umap_plot(
    smiles_list: list[str],
    split_labels: list[str],
    out_path: Path,
    seed: Optional[int],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "jaccard",
) -> None:
    """Save a 2D UMAP scatter plot colored by split labels."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        import umap
    except ModuleNotFoundError as exc:
        raise ImportError(
            "UMAP visualization requires RDKit, umap-learn, and matplotlib to be installed."
        ) from exc

    if seed is None:
        seed = 0

    valid_smiles = []
    valid_labels = []
    mols = []
    for smi, label in zip(smiles_list, split_labels):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(str(smi))
            valid_labels.append(str(label))

    if len(mols) == 0:
        return

    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols]
    fp_array = np.zeros((len(fps), 1024), dtype=np.float32)
    for i, fp in enumerate(fps):
        arr = np.zeros((1024,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array[i] = arr

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
        init="random",
    )
    embedding = reducer.fit_transform(fp_array)

    coords_df = pd.DataFrame(
        {
            "smiles": valid_smiles,
            "split": valid_labels,
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
        }
    )
    coords_path = out_path.with_suffix(".csv")
    coords_df.to_csv(coords_path, index=False)

    try:
        fig, ax = plt.subplots(figsize=(8, 6))

        color_map = {
            "train": "tab:blue",
            "valid": "tab:orange",
            "test": "tab:green",
            "unspecified": "tab:gray",
        }
        marker_map = {
            "train": "o",
            "valid": "s",
            "test": "^",
            "unspecified": "x",
        }

        unique_labels = []
        for label in ["train", "valid", "test", "unspecified"]:
            if label in valid_labels:
                unique_labels.append(label)
        for label in valid_labels:
            if label not in unique_labels:
                unique_labels.append(label)

        for label in unique_labels:
            idxs = [i for i, x in enumerate(valid_labels) if x == label]
            if not idxs:
                continue
            points = embedding[idxs]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=14,
                alpha=0.8,
                marker=marker_map.get(label, "o"),
                c=color_map.get(label, "tab:gray"),
                edgecolors="none",
            )

        ax.set_title("UMAP split visualization")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

        legend_text = " / ".join(unique_labels) if unique_labels else "no labels"
        fig.text(0.01, 0.01, f"splits: {legend_text}", fontsize=8)

        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except RecursionError:
        note_path = out_path.with_suffix(".txt")
        note_path.write_text(
            "Matplotlib recursion error occurred while saving the PNG. "
            "UMAP coordinates were still saved as CSV.",
            encoding="utf-8",
        )
        plt.close("all")
        return
    except Exception as exc:
        note_path = out_path.with_suffix(".txt")
        note_path.write_text(
            f"Failed to save UMAP PNG: {exc}\nUMAP coordinates were still saved as CSV.",
            encoding="utf-8",
        )
        plt.close("all")
        return


def run(
    input_csv: Path,
    out_dir: Path,
    smiles_col: Optional[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.0,
    frac_test: float = 0.2,
    seed: Optional[int] = None,
    no_valid: bool = True,
    split_method: str = "scaffold",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_n_components: int = 2,
    umap_metric: str = "jaccard",
    umap_n_clusters: Optional[int] = None,
    use_unseen_split: bool = False,
    unseen_threshold: int = 89,
    min_test_pairs: int = 30,
) -> None:
    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if no_valid and not use_unseen_split:
        frac_train = 0.8
        frac_valid = 0.0
        frac_test = 0.2

    df = pd.read_csv(input_csv)
    if "dataset_name" not in df.columns or "endpoint" not in df.columns:
        raise ValueError("CSV must have columns: dataset_name, endpoint")

    smiles_col = _choose_smiles_col(df, smiles_col)
    required = ["dataset_name", "endpoint", smiles_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Ensure string
    df[smiles_col] = df[smiles_col].fillna("").astype(str)

    groups = df.groupby(["dataset_name", "endpoint"], dropna=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_unseen_split:
        out_unseen = out_dir / "unseen_endpoint_test"
        out_train_test = out_dir / "train_test"
        out_unseen.mkdir(parents=True, exist_ok=True)
        out_train_test.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for (dataset_name, endpoint), g in groups:
        g = g.copy().reset_index(drop=True)
        # Filter empty SMILES rows
        mask = g[smiles_col].str.strip().astype(bool)
        g_valid = g.loc[mask].copy().reset_index(drop=True)
        n_total = len(g)
        n_valid = len(g_valid)

        if use_unseen_split:
            # 1) n_valid <= threshold → unseen endpoint test (전체를 test로 저장)
            # 2) n_valid > threshold → 9:1 scaffold split, test 최소 min_test_pairs
            if n_valid == 0:
                summary_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "endpoint": endpoint,
                        "smiles_col": smiles_col,
                        "split_method": split_method,
                        "split_type": "skip",
                        "n_total": n_total,
                        "n_valid_smiles": n_valid,
                        "n_train": 0,
                        "n_valid": 0,
                        "n_test": 0,
                        "note": "no valid smiles",
                    }
                )
                continue

            if n_valid <= unseen_threshold:
                # Unseen endpoint test: 전체를 test로 저장
                endpoint_dir = out_unseen / _sanitize_for_path(dataset_name) / _sanitize_for_path(endpoint)
                endpoint_dir.mkdir(parents=True, exist_ok=True)
                g_valid.to_csv(endpoint_dir / "test.csv", index=False)
                g_out = g_valid.copy()
                g_out["split"] = "test"
                g_out.to_csv(endpoint_dir / "all_with_split.csv", index=False)
                summary_rows.append(
                    {
                        "dataset_name": dataset_name,
                        "endpoint": endpoint,
                        "smiles_col": smiles_col,
                        "split_method": split_method,
                        "split_type": "unseen_endpoint_test",
                        "n_total": n_total,
                        "n_valid_smiles": n_valid,
                        "n_train": 0,
                        "n_valid": 0,
                        "n_test": n_valid,
                        "note": f"n_total<={unseen_threshold}; full test",
                    }
                )
                continue

            # 9:1 split, test 최소 min_test_pairs
            n_test = max(min_test_pairs, round(0.1 * n_valid))
            n_test = min(n_test, n_valid - 1)  # train 최소 1개
            n_train = n_valid - n_test
            ep_frac_train = n_train / n_valid
            ep_frac_test = n_test / n_valid
            ep_frac_valid = 0.0

            smiles_list = g_valid[smiles_col].astype(str).tolist()
            train_inds, valid_inds, test_inds = _split_indices(
                smiles_list=smiles_list,
                frac_train=ep_frac_train,
                frac_valid=ep_frac_valid,
                frac_test=ep_frac_test,
                seed=seed,
                split_method=split_method,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                umap_n_components=umap_n_components,
                umap_metric=umap_metric,
                umap_n_clusters=umap_n_clusters,
            )
            # Scaffold split은 그룹 단위라 test가 min_test_pairs 미만일 수 있음 → train에서 보충
            if len(test_inds) < min_test_pairs and len(train_inds) > min_test_pairs - len(test_inds):
                rng = random.Random(seed)
                need = min_test_pairs - len(test_inds)
                move = rng.sample(train_inds, need)
                train_inds = [i for i in train_inds if i not in move]
                test_inds = list(test_inds) + move

            endpoint_dir = out_train_test / _sanitize_for_path(dataset_name) / _sanitize_for_path(endpoint)
            endpoint_dir.mkdir(parents=True, exist_ok=True)
            df_train = g_valid.iloc[train_inds].copy()
            df_test = g_valid.iloc[test_inds].copy()
            df_train.to_csv(endpoint_dir / "train.csv", index=False)
            df_test.to_csv(endpoint_dir / "test.csv", index=False)
            df_valid = g_valid.iloc[valid_inds].copy()
            df_valid.head(0).to_csv(endpoint_dir / "valid.csv", index=False)

            g_out = g_valid.copy()
            g_out["split"] = ""
            g_out.loc[train_inds, "split"] = "train"
            g_out.loc[test_inds, "split"] = "test"
            g_out.to_csv(endpoint_dir / "all_with_split.csv", index=False)

            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "endpoint": endpoint,
                    "smiles_col": smiles_col,
                    "split_method": split_method,
                    "split_type": "train_test",
                    "n_total": n_total,
                    "n_valid_smiles": n_valid,
                    "n_train": len(df_train),
                    "n_valid": 0,
                    "n_test": len(df_test),
                    "note": f"9:1 scaffold, min_test={min_test_pairs}",
                }
            )
            continue

        # ----- 기존 로직 (use_unseen_split=False) -----
        dataset_dir = out_dir / _sanitize_for_path(dataset_name)
        endpoint_dir = dataset_dir / _sanitize_for_path(endpoint)
        endpoint_dir.mkdir(parents=True, exist_ok=True)

        if n_valid == 0:
            # Save empty splits for completeness
            (endpoint_dir / "train.csv").write_text("", encoding="utf-8")
            (endpoint_dir / "valid.csv").write_text("", encoding="utf-8")
            (endpoint_dir / "test.csv").write_text("", encoding="utf-8")
            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "endpoint": endpoint,
                    "smiles_col": smiles_col,
                    "split_method": split_method,
                    "n_total": n_total,
                    "n_valid_smiles": n_valid,
                    "n_train": 0,
                    "n_valid": 0,
                    "n_test": 0,
                    "note": "no valid smiles; train/test only",
                }
            )
            continue

        smiles_list = g_valid[smiles_col].astype(str).tolist()
        train_inds, valid_inds, test_inds = _split_indices(
            smiles_list=smiles_list,
            frac_train=frac_train,
            frac_valid=frac_valid,
            frac_test=frac_test,
            seed=seed,
            split_method=split_method,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_n_components=umap_n_components,
            umap_metric=umap_metric,
            umap_n_clusters=umap_n_clusters,
        )

        df_train = g_valid.iloc[train_inds].copy()
        df_valid = g_valid.iloc[valid_inds].copy()
        df_test = g_valid.iloc[test_inds].copy()

        df_train.to_csv(endpoint_dir / "train.csv", index=False)
        if no_valid:
            # valid 없이 train/test만 사용 시: 빈 valid.csv (헤더만) 호환용
            df_valid.head(0).to_csv(endpoint_dir / "valid.csv", index=False)
        else:
            df_valid.to_csv(endpoint_dir / "valid.csv", index=False)
        df_test.to_csv(endpoint_dir / "test.csv", index=False)

        # Also save a single file with split labels
        g_out = g_valid.copy()
        g_out["split"] = ""
        g_out.loc[train_inds, "split"] = "train"
        if not no_valid:
            g_out.loc[valid_inds, "split"] = "valid"
        g_out.loc[test_inds, "split"] = "test"
        g_out.to_csv(endpoint_dir / "all_with_split.csv", index=False)

        if split_method == "umap":
            split_labels = [""] * len(g_valid)
            for idx in train_inds:
                split_labels[idx] = "train"
            for idx in test_inds:
                split_labels[idx] = "test"
            if not no_valid:
                for idx in valid_inds:
                    split_labels[idx] = "valid"
            split_labels = [label if label else "unspecified" for label in split_labels]
            _save_umap_plot(
                smiles_list=smiles_list,
                split_labels=split_labels,
                out_path=endpoint_dir / "umap_split.png",
                seed=seed,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                n_components=umap_n_components,
                metric=umap_metric,
            )

        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "endpoint": endpoint,
                "smiles_col": smiles_col,
                "split_method": split_method,
                "n_total": n_total,
                "n_valid_smiles": n_valid,
                "n_train": len(df_train),
                "n_valid": len(df_valid),
                "n_test": len(df_test),
                "note": "",
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["dataset_name", "endpoint"], kind="stable"
    )
    summary_path = out_dir / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved split summary -> {summary_path}")
    print(f"Output dir -> {out_dir}")
    if use_unseen_split:
        print(f"  unseen_endpoint_test -> {out_dir / 'unseen_endpoint_test'}")
        print(f"  train_test (9:1)      -> {out_dir / 'train_test'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Endpoint-wise train/test split (8:2) for SAFE pair CSV using scaffold or UMAP splitter. Saves UMAP visualization for UMAP splits."
    )
    ap.add_argument(
        "--herg_merged",
        action="store_true",
        help="Use pairs_safe_filtered_herg_merged.csv and save to splits/scaffold_by_endpoint_herg_merged (new split with herg unified).",
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT}). Ignored if --herg_merged.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}). Ignored if --herg_merged.",
    )
    ap.add_argument(
        "--smiles_col",
        type=str,
        default=None,
        help="Which SMILES column to use for splitting (default: toxic_safe_decoded_smiles if exists else toxic_smiles).",
    )
    ap.add_argument(
        "--split_method",
        type=str,
        default="scaffold",
        choices=["scaffold", "umap"],
        help="Split method to use: scaffold or umap (default: scaffold).",
    )
    ap.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (used only when --split_method umap).",
    )
    ap.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist (used only when --split_method umap).",
    )
    ap.add_argument(
        "--umap_n_components",
        type=int,
        default=2,
        help="UMAP n_components (used only when --split_method umap).",
    )
    ap.add_argument(
        "--umap_metric",
        type=str,
        default="jaccard",
        help="UMAP metric (used only when --split_method umap).",
    )
    ap.add_argument(
        "--umap_n_clusters",
        type=int,
        default=None,
        help="KMeans cluster count in UMAP space (used only when --split_method umap). Default: heuristic.",
    )
    ap.add_argument(
        "--no_valid",
        action="store_true",
        default=True,
        help="Always use train/test only split. Validation split is disabled and the ratio is fixed to 8:2.",
    )
    ap.add_argument("--frac_train", type=float, default=0.8)
    ap.add_argument("--frac_valid", type=float, default=0.0)
    ap.add_argument("--frac_test", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use_unseen_split",
        action="store_true",
        help="n_total<=unseen_threshold인 endpoint는 전부 unseen_endpoint_test로 저장, 그 외는 9:1 scaffold split (test 최소 min_test_pairs). 출력: out_dir/unseen_endpoint_test, out_dir/train_test.",
    )
    ap.add_argument(
        "--unseen_threshold",
        type=int,
        default=89,
        help="Unseen endpoint로 쓸 n_total 상한 (default 89, Q1). --use_unseen_split 시 사용.",
    )
    ap.add_argument(
        "--min_test_pairs",
        type=int,
        default=30,
        help="9:1 split 시 test set 최소 pair 수 (default 30). --use_unseen_split 시 사용.",
    )
    args = ap.parse_args()

    if args.herg_merged:
        input_csv = INPUT_HERG_MERGED
        out_dir = OUT_DIR_HERG_MERGED
        if not input_csv.exists():
            raise FileNotFoundError(
                f"herg_merged input not found: {input_csv}. Run merge_herg_datasets.py first."
            )
    else:
        input_csv = args.input
        out_dir = args.out_dir

    run(
        input_csv=input_csv,
        out_dir=out_dir,
        smiles_col=args.smiles_col,
        frac_train=args.frac_train,
        frac_valid=args.frac_valid,
        frac_test=args.frac_test,
        seed=args.seed,
        no_valid=args.no_valid,
        split_method=args.split_method,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_n_components=args.umap_n_components,
        umap_metric=args.umap_metric,
        umap_n_clusters=args.umap_n_clusters,
        use_unseen_split=args.use_unseen_split,
        unseen_threshold=args.unseen_threshold,
        min_test_pairs=args.min_test_pairs,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
