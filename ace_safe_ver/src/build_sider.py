from __future__ import annotations

"""
SIDER 데이터(molecularACE_ver/sider.csv)를 대상으로
다음 과정을 한 번에 수행하는 스크립트.

1) MolecularACE 스타일 pairing (process_endpoint 이용)
   - dataset_name = "sider"
   - endpoint     = Task (SIDER category)
   - 각 endpoint별 (toxic_smiles, nontoxic_smiles) 목록을 만들고,
     molecular_ace_pairing.process_endpoint 로 similarity 기반 pair 생성

2) SMILES → SAFE 매핑 생성 (SIDER 전용)
   - SIDER pair에 등장하는 모든 SMILES에 대해
     canonical SMILES → SAFE 인코딩

3) SAFE 부착 + fragment 비교 + 필터링
   - ACE SAFE와 동일 파이프라인:
     pairs_to_safe → compare_safe → filter_safe_pairs

출력 (ace_safe_ver 루트):
  - pairs_safe_sider_pairs.csv            : SIDER pairing 결과 (SMILES 기준 pair)
  - pairs_safe_sider.csv                 : SIDER pair + toxic_safe, nontoxic_safe
  - pairs_safe_sider_compared.csv        : fragment 비교 결과
  - pairs_safe_sider_filtered.csv        : 필터 적용 후 최종 SIDER SAFE pair
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


# 경로 설정
ACE_SAFE_SRC = Path(__file__).resolve().parent
ACE_SAFE_DIR = ACE_SAFE_SRC.parent
REPO_ROOT = ACE_SAFE_DIR.parent  # .../ToxAgent
MOLECULAR_ACE_DIR = REPO_ROOT / "molecularACE_ver"

# molecular_ace_pairing import를 위해 molecularACE_ver를 sys.path에 추가
if str(MOLECULAR_ACE_DIR) not in sys.path:
    sys.path.insert(0, str(MOLECULAR_ACE_DIR))

from molecular_ace_pairing import process_endpoint  # type: ignore

# SAFE 인코더: ToxAgent 루트를 sys.path에 넣고 local safe 패키지 사용
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datamol as dm  # type: ignore
from safe.safe.converter import (  # type: ignore
    encode as safe_encode,
    SAFEEncodeError,
    SAFEFragmentationError,
)


# 입력/출력 경로
SIDER_CSV = MOLECULAR_ACE_DIR / "sider.csv"
OUT_PAIRS_SMILES = ACE_SAFE_DIR / "pairs_safe_sider_pairs.csv"
OUT_PAIRS_SAFE = ACE_SAFE_DIR / "pairs_safe_sider.csv"
OUT_COMPARED = ACE_SAFE_DIR / "pairs_safe_sider_compared.csv"
OUT_FILTERED = ACE_SAFE_DIR / "pairs_safe_sider_filtered.csv"

SEP = "."


def canonical_smiles(smiles: str) -> str | None:
    """SMILES를 canonical form으로 변환. 실패 시 None."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    with dm.without_rdkit_log():
        try:
            mol = dm.to_mol(str(smiles))
            if mol is None:
                return None
            return dm.standardize_smiles(dm.to_smiles(mol, canonical=True))
        except Exception:
            return None


def smiles_to_safe(smiles: str | None) -> str | None:
    """Canonical SMILES를 SAFE 문자열로 변환. 실패 시 None."""
    if smiles is None or not str(smiles).strip():
        return None
    with dm.without_rdkit_log():
        try:
            return safe_encode(str(smiles), canonical=True)
        except (SAFEEncodeError, SAFEFragmentationError, Exception):
            return None


def _load_sider() -> pd.DataFrame:
    if not SIDER_CSV.exists():
        raise FileNotFoundError(f"SIDER CSV not found: {SIDER_CSV}")
    df = pd.read_csv(SIDER_CSV)
    for col in ["X", "Task", "Y"]:
        if col not in df.columns:
            raise ValueError(f"SIDER CSV must have columns X, Task, Y. Found: {list(df.columns)}")
    # X = SMILES, Task = endpoint, Y = 1(toxic) / 0(nontoxic)
    df = df.rename(columns={"X": "smiles", "Task": "endpoint", "Y": "label"})
    return df


def build_sider_pairs() -> pd.DataFrame:
    """
    SIDER CSV를 읽어 endpoint(Task)별로 toxic/nontoxic SMILES를 나누고,
    molecular_ace_pairing.process_endpoint 로 similarity 기반 pairs 생성.

    Returns:
        DataFrame with columns: dataset_name, endpoint, toxic_smiles, nontoxic_smiles
    """
    df = _load_sider()

    all_rows: List[Dict[str, Any]] = []
    dataset_name = "sider"

    for endpoint, grp in df.groupby("endpoint", dropna=False):
        g = grp.copy()
        smiles_toxic = (
            g[g["label"] == 1]["smiles"].dropna().astype(str).str.strip().tolist()
        )
        smiles_nontoxic = (
            g[g["label"] == 0]["smiles"].dropna().astype(str).str.strip().tolist()
        )

        # 중복 제거
        smiles_toxic = sorted({s for s in smiles_toxic if s})
        smiles_nontoxic = sorted({s for s in smiles_nontoxic if s})

        if not smiles_toxic or not smiles_nontoxic:
            print(
                f"[SIDER] Skip endpoint={endpoint!r}: "
                f"n_toxic={len(smiles_toxic)}, n_nontoxic={len(smiles_nontoxic)}"
            )
            continue

        print(
            f"[SIDER] Pairing endpoint={endpoint!r}: "
            f"n_toxic={len(smiles_toxic)}, n_nontoxic={len(smiles_nontoxic)}"
        )

        (
            _ds,
            _ep,
            rows_all,
            _rows_fg,
            _rows_stereo,
            _rows_isomer,
            n_pairs,
        ) = process_endpoint(
            dataset=dataset_name,
            endpoint=str(endpoint),
            toxic_smiles=smiles_toxic,
            nontoxic_smiles=smiles_nontoxic,
            save_sim_path=None,
            canonicalize_smiles=True,
        )

        for s_t, s_n in rows_all:
            all_rows.append(
                {
                    "dataset_name": dataset_name,
                    "endpoint": endpoint,
                    "toxic_smiles": s_t,
                    "nontoxic_smiles": s_n,
                }
            )

        print(f"[SIDER]   -> pairs (all styles OR) = {n_pairs:,}")

    df_pairs = pd.DataFrame(all_rows)
    if df_pairs.empty:
        print("[SIDER] WARNING: no pairs generated.")
    else:
        OUT_PAIRS_SMILES.parent.mkdir(parents=True, exist_ok=True)
        df_pairs.to_csv(OUT_PAIRS_SMILES, index=False)
        print(f"[SIDER] Saved SMILES pairs -> {OUT_PAIRS_SMILES} ({len(df_pairs):,} rows)")
    return df_pairs


def build_safe_mapping_for_sider(df_pairs: pd.DataFrame) -> Tuple[dict[str, str], dict[str, str]]:
    """
    SIDER pairs에 등장하는 모든 SMILES에 대해 canonical + SAFE 인코딩하여
    (원본 SMILES → SAFE), (canonical_smiles → SAFE) 매핑을 만든다.
    """
    smiles_set = set(
        str(s).strip()
        for col in ["toxic_smiles", "nontoxic_smiles"]
        for s in df_pairs[col].dropna().astype(str).tolist()
        if str(s).strip()
    )
    smiles_list = sorted(smiles_set)
    print(f"[SIDER] Total unique SMILES for SAFE mapping: {len(smiles_list):,}")

    smiles_to_safe_map: Dict[str, str] = {}
    canon_to_safe_map: Dict[str, str] = {}

    for smi in tqdm(smiles_list, desc="SIDER SMILES → canonical → SAFE"):
        canon = canonical_smiles(smi)
        safe_str = smiles_to_safe(canon) if canon is not None else None

        if safe_str is None:
            continue
        smiles_to_safe_map[smi] = safe_str
        if canon is not None:
            canon_to_safe_map[canon] = safe_str

    print(
        f"[SIDER] SAFE mapping sizes: "
        f"smiles_to_safe={len(smiles_to_safe_map):,}, canon_to_safe={len(canon_to_safe_map):,}"
    )
    return smiles_to_safe_map, canon_to_safe_map


def lookup_safe_column(
    smiles_series: pd.Series,
    smiles_to_safe: dict[str, str],
    canon_to_safe: dict[str, str],
) -> list[str]:
    out: list[str] = []
    for s in smiles_series:
        s = str(s).strip() if pd.notna(s) else ""
        safe_str = smiles_to_safe.get(s)
        if safe_str is None and s:
            safe_str = canon_to_safe.get(s, "")
        out.append(safe_str if safe_str is not None else "")
    return out


def safe_to_fragments(safe_str: Any) -> set[str]:
    if pd.isna(safe_str) or not str(safe_str).strip():
        return set()
    return {s.strip() for s in str(safe_str).split(SEP) if s.strip()}


def compare_fragments(toxic_safe: Any, nontoxic_safe: Any) -> Tuple[set[str], set[str], set[str]]:
    t_set = safe_to_fragments(toxic_safe)
    n_set = safe_to_fragments(nontoxic_safe)
    common = t_set & n_set
    only_toxic = t_set - n_set
    only_nontoxic = n_set - t_set
    return common, only_toxic, only_nontoxic


def _has_any_fragment_ge(s: Any, min_length: int) -> bool:
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return False
    parts = [p.strip() for p in t.split(SEP) if p.strip()]
    return any(len(p) >= min_length for p in parts)


def run_sider_pipeline(max_frag_len: int = 28, max_frag_num: int = 4) -> None:
    # 1) Pairing (SMILES 기준)
    df_pairs = build_sider_pairs()
    if df_pairs.empty:
        print("[SIDER] No pairs to process; aborting.")
        return

    # 2) SAFE 매핑 생성
    smiles_to_safe_map, canon_to_safe_map = build_safe_mapping_for_sider(df_pairs)

    # 3) SAFE 부착
    print("[SIDER] Attaching toxic_safe / nontoxic_safe")
    df_safe = df_pairs.assign(
        toxic_safe=lookup_safe_column(
            df_pairs["toxic_smiles"], smiles_to_safe_map, canon_to_safe_map
        ),
        nontoxic_safe=lookup_safe_column(
            df_pairs["nontoxic_smiles"], smiles_to_safe_map, canon_to_safe_map
        ),
    )

    n_rows = len(df_safe)
    toxic_miss = (df_safe["toxic_safe"] == "").sum()
    nontoxic_miss = (df_safe["nontoxic_safe"] == "").sum()
    print(f"[SIDER] Rows with missing toxic_safe:   {toxic_miss} / {n_rows}")
    print(f"[SIDER] Rows with missing nontoxic_safe: {nontoxic_miss} / {n_rows}")

    OUT_PAIRS_SAFE.parent.mkdir(parents=True, exist_ok=True)
    df_safe.to_csv(OUT_PAIRS_SAFE, index=False)
    print(f"[SIDER] Saved pairs_safe_sider -> {OUT_PAIRS_SAFE}")

    # 4) fragment 비교
    print("[SIDER] Comparing SAFE fragments (common / only_toxic / only_nontoxic)")
    common_list: List[str] = []
    only_toxic_list: List[str] = []
    only_nontoxic_list: List[str] = []
    n_common_list: List[int] = []
    n_only_toxic_list: List[int] = []
    n_only_nontoxic_list: List[int] = []
    toxic_fragments_str_list: List[str] = []
    nontoxic_fragments_str_list: List[str] = []

    for _, row in df_safe.iterrows():
        toxic_safe = row.get("toxic_safe", "")
        nontoxic_safe = row.get("nontoxic_safe", "")
        common, only_toxic, only_nontoxic = compare_fragments(toxic_safe, nontoxic_safe)

        toxic_fragments_str_list.append(SEP.join(sorted(safe_to_fragments(toxic_safe))))
        nontoxic_fragments_str_list.append(SEP.join(sorted(safe_to_fragments(nontoxic_safe))))
        common_list.append(SEP.join(sorted(common)))
        only_toxic_list.append(SEP.join(sorted(only_toxic)))
        only_nontoxic_list.append(SEP.join(sorted(only_nontoxic)))
        n_common_list.append(len(common))
        n_only_toxic_list.append(len(only_toxic))
        n_only_nontoxic_list.append(len(only_nontoxic))

    df_comp = df_safe.assign(
        toxic_safe_fragments=toxic_fragments_str_list,
        nontoxic_safe_fragments=nontoxic_fragments_str_list,
        common_safe_fragments=common_list,
        only_toxic_safe_fragments=only_toxic_list,
        only_nontoxic_safe_fragments=only_nontoxic_list,
        n_common_safe=n_common_list,
        n_only_toxic_safe=n_only_toxic_list,
        n_only_nontoxic_safe=n_only_nontoxic_list,
    )
    df_comp.to_csv(OUT_COMPARED, index=False)
    print(f"[SIDER] Saved pairs_safe_sider_compared -> {OUT_COMPARED} ({len(df_comp):,} rows)")

    # 5) 필터 적용 (filter_safe_pairs.py 로직)
    print("[SIDER] Applying SAFE fragment filters (common!=0, diff!=0, length/num thresholds)")
    n_start = len(df_comp)

    # 5-1) n_common_safe != 0
    df_f = df_comp[df_comp["n_common_safe"].ne(0)].copy()
    n_after_common = len(df_f)
    print(
        f"  Filter 1 (n_common_safe != 0): {n_start} -> {n_after_common} "
        f"(-{n_start - n_after_common})"
    )

    # 5-2) not (both only == 0)
    mask_both_zero = (df_f["n_only_nontoxic_safe"] == 0) & (df_f["n_only_toxic_safe"] == 0)
    df_f = df_f[~mask_both_zero].copy()
    n_after_diff = len(df_f)
    print(
        f"  Filter 2 (has only_toxic or only_nontoxic): "
        f"{n_after_common} -> {n_after_diff} (-{n_after_common - n_after_diff})"
    )

    # 5-3) no fragment length >= max_frag_len
    mask_long = df_f.apply(
        lambda r: _has_any_fragment_ge(r.get("only_toxic_safe_fragments"), max_frag_len)
        or _has_any_fragment_ge(r.get("only_nontoxic_safe_fragments"), max_frag_len),
        axis=1,
    )
    df_f = df_f[~mask_long].copy()
    n_after_len = len(df_f)
    print(
        f"  Filter 3 (no fragment length >= {max_frag_len}): "
        f"{n_after_diff} -> {n_after_len} (-{n_after_diff - n_after_len})"
    )

    # 5-4) n_only_* <= max_frag_num
    mask_too_many = (df_f["n_only_toxic_safe"] > max_frag_num) | (
        df_f["n_only_nontoxic_safe"] > max_frag_num
    )
    df_f = df_f[~mask_too_many].copy()
    n_final = len(df_f)
    print(
        f"  Filter 4 (n_only_* <= {max_frag_num}): "
        f"{n_after_len} -> {n_final} (-{n_after_len - n_final})"
    )

    OUT_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(OUT_FILTERED, index=False)
    print(
        f"[SIDER] Saved pairs_safe_sider_filtered -> {OUT_FILTERED} "
        f"({n_final} rows, total removed {n_start - n_final})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run full SAFE pipeline for SIDER:\n"
            "  pairing (process_endpoint) -> SAFE mapping -> compare -> filter"
        )
    )
    ap.add_argument(
        "--max-frag-len",
        type=int,
        default=28,
        help="Drop pair if any fragment length >= this (default 28).",
    )
    ap.add_argument(
        "--max-frag-num",
        type=int,
        default=4,
        help=(
            "Keep only rows with n_only_toxic_safe <= this and "
            "n_only_nontoxic_safe <= this (default 4)."
        ),
    )
    args = ap.parse_args()

    run_sider_pipeline(max_frag_len=args.max_frag_len, max_frag_num=args.max_frag_num)


if __name__ == "__main__":
    main()

