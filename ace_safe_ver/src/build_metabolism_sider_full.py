from __future__ import annotations

"""
metabolism / sider MolecularACE pair CSV들을 입력으로 받아,
ACE SAFE 파이프라인을 "끝까지" 한 번에 수행해 최종 통합(valid only) CSV를 생성합니다.

입력:
  - data/metabolism_ver/pairs.csv
  - data/pairs_sider.csv

처리:
  1) 두 pairs에 등장하는 모든 SMILES에 대해 canonical → SAFE 매핑 생성
  2) toxic_safe / nontoxic_safe 부착
  3) compare_safe 로직(공통/전용 fragment, n_* 계산)
  4) filter_safe_pairs 로직(4개 필터) 적용
  5) smiles_to_safe_valid.validate_df 로 SAFE decode 검증 수행 → valid/invalid 분리
  6) metabolism_valid + sider_valid 를 concat → 최종 통합(full) valid CSV 저장

출력(기본, ace_safe_ver 루트):
  - pairs_safe_metabolism_sider_attached.csv
  - pairs_safe_metabolism_sider_compared.csv
  - pairs_safe_metabolism_sider_filtered.csv
  - pairs_safe_metabolism_sider_filtered_valid.csv
  - pairs_safe_metabolism_sider_filtered_invalid.csv
  - pairs_safe_metabolism_sider_final_valid_merged.csv
  - smiles_to_safe_metabolism_sider.csv  (생성한 매핑)
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ACE_SAFE_DIR = SCRIPT_DIR.parent
if str(ACE_SAFE_DIR) not in sys.path:
    sys.path.insert(0, str(ACE_SAFE_DIR))
import ace_local  # noqa: E402

DEFAULT_METAB_PAIRS = ace_local.DEFAULT_METABOLISM_PAIRS_CSV
DEFAULT_SIDER_PAIRS = ace_local.DEFAULT_PAIRS_SIDER_CSV

DEFAULT_OUT_MAPPING = ACE_SAFE_DIR / "smiles_to_safe_metabolism_sider.csv"
DEFAULT_OUT_ATTACHED = ACE_SAFE_DIR / "pairs_safe_metabolism_sider_attached.csv"
DEFAULT_OUT_COMPARED = ACE_SAFE_DIR / "pairs_safe_metabolism_sider_compared.csv"
DEFAULT_OUT_FILTERED = ACE_SAFE_DIR / "pairs_safe_metabolism_sider_filtered.csv"
DEFAULT_OUT_VALID = ACE_SAFE_DIR / "pairs_safe_metabolism_sider_filtered_valid.csv"
DEFAULT_OUT_INVALID = ACE_SAFE_DIR / "pairs_safe_metabolism_sider_filtered_invalid.csv"
DEFAULT_OUT_FINAL = ACE_SAFE_DIR / "pairs_safe_metabolism_sider_final_valid_merged.csv"

SEP = "."


def _import_safe_encoder():
    """번들 third_party/safe 인코더."""
    ace_local.ensure_safe_pkg_path()
    from safe.safe.converter import encode as _encode, SAFEEncodeError, SAFEFragmentationError

    return _encode, SAFEEncodeError, SAFEFragmentationError


def _import_validate_df():
    """
    ace_safe_ver/src/smiles_to_safe_valid.py의 validate_df를 재사용.
    """
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from smiles_to_safe_valid import validate_df  # type: ignore

    return validate_df


def _canonical_smiles(smiles: str) -> str | None:
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    import datamol as dm  # local env dependency

    with dm.without_rdkit_log():
        try:
            mol = dm.to_mol(str(smiles))
            if mol is None:
                return None
            return dm.standardize_smiles(dm.to_smiles(mol, canonical=True))
        except Exception:
            return None


def _encode_safe(canon_smiles: str | None) -> str | None:
    if canon_smiles is None or not str(canon_smiles).strip():
        return None
    safe_encode, SAFEEncodeError, SAFEFragmentationError = _import_safe_encoder()
    try:
        return safe_encode(str(canon_smiles), canonical=True)
    except (SAFEEncodeError, SAFEFragmentationError, Exception):
        return None


def _escape_csv(s: str | None) -> str:
    if s is None:
        return ""
    s = str(s)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def _build_mapping(
    df_pairs: pd.DataFrame,
    out_csv: Path,
) -> Tuple[dict[str, str], dict[str, str]]:
    """
    df_pairs(toxic_smiles/nontoxic_smiles)에 등장하는 모든 SMILES에 대해
    canonical → SAFE 를 인코딩해 매핑 dict + CSV를 생성.
    """
    for c in ["toxic_smiles", "nontoxic_smiles"]:
        if c not in df_pairs.columns:
            raise ValueError(f"pairs df missing column: {c}")

    smiles_set = set(
        str(s).strip()
        for col in ["toxic_smiles", "nontoxic_smiles"]
        for s in df_pairs[col].dropna().astype(str).tolist()
        if str(s).strip()
    )
    smiles_list = sorted(smiles_set)
    print(f"[full] unique SMILES for mapping: {len(smiles_list):,}")

    smiles_to_safe: Dict[str, str] = {}
    canon_to_safe: Dict[str, str] = {}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("smiles,canonical_smiles,safe\n")
        for smi in tqdm(smiles_list, desc="SMILES → canonical → SAFE"):
            canon = _canonical_smiles(smi)
            safe_str = _encode_safe(canon)
            if safe_str is not None:
                smiles_to_safe[smi] = safe_str
                if canon is not None:
                    canon_to_safe[canon] = safe_str
            f.write(f"{_escape_csv(smi)},{_escape_csv(canon)},{_escape_csv(safe_str)}\n")

    print(
        f"[full] mapping sizes: smiles_to_safe={len(smiles_to_safe):,}, canon_to_safe={len(canon_to_safe):,}"
    )
    print(f"[full] saved mapping -> {out_csv}")
    return smiles_to_safe, canon_to_safe


def _lookup_safe(smiles: str, smiles_to_safe: dict[str, str], canon_to_safe: dict[str, str]) -> str:
    s = (smiles or "").strip()
    if not s:
        return ""
    v = smiles_to_safe.get(s)
    if v is not None:
        return v
    return canon_to_safe.get(s, "")


def _safe_to_fragments(safe_str: Any) -> set[str]:
    if pd.isna(safe_str) or not str(safe_str).strip():
        return set()
    return {s.strip() for s in str(safe_str).split(SEP) if s.strip()}


def _compare_fragments(toxic_safe: Any, nontoxic_safe: Any) -> tuple[set[str], set[str], set[str]]:
    t_set = _safe_to_fragments(toxic_safe)
    n_set = _safe_to_fragments(nontoxic_safe)
    return t_set & n_set, t_set - n_set, n_set - t_set


def _has_any_fragment_ge(s: Any, min_length: int) -> bool:
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "nan":
        return False
    parts = [p.strip() for p in t.split(SEP) if p.strip()]
    return any(len(p) >= min_length for p in parts)


def run(
    metabolism_pairs: Path,
    sider_pairs: Path,
    out_mapping: Path,
    out_attached: Path,
    out_compared: Path,
    out_filtered: Path,
    out_valid: Path,
    out_invalid: Path,
    out_final: Path,
    max_frag_len: int = 28,
    max_frag_num: int = 4,
) -> None:
    # 0) load pairs
    for p in [metabolism_pairs, sider_pairs]:
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

    df_m = pd.read_csv(metabolism_pairs)
    df_s = pd.read_csv(sider_pairs)
    print(f"[full] loaded metabolism pairs: {len(df_m):,}")
    print(f"[full] loaded sider pairs:      {len(df_s):,}")

    # ensure core columns
    for name, df in [("metabolism", df_m), ("sider", df_s)]:
        for c in ["dataset_name", "endpoint", "toxic_smiles", "nontoxic_smiles"]:
            if c not in df.columns:
                raise ValueError(f"{name} pairs missing column {c!r}. columns={list(df.columns)}")

    df_pairs = pd.concat([df_m, df_s], ignore_index=True)

    # 1) build mapping
    smiles_to_safe, canon_to_safe = _build_mapping(df_pairs, out_csv=out_mapping)

    # 2) attach SAFE
    df_att = df_pairs.copy()
    df_att["toxic_smiles"] = df_att["toxic_smiles"].fillna("").astype(str)
    df_att["nontoxic_smiles"] = df_att["nontoxic_smiles"].fillna("").astype(str)
    df_att["toxic_safe"] = [
        _lookup_safe(s, smiles_to_safe, canon_to_safe) for s in df_att["toxic_smiles"].tolist()
    ]
    df_att["nontoxic_safe"] = [
        _lookup_safe(s, smiles_to_safe, canon_to_safe) for s in df_att["nontoxic_smiles"].tolist()
    ]
    out_attached.parent.mkdir(parents=True, exist_ok=True)
    df_att.to_csv(out_attached, index=False)
    print(f"[full] saved attached -> {out_attached} (rows={len(df_att):,})")

    # 3) compare SAFE fragments
    common_list: list[str] = []
    only_toxic_list: list[str] = []
    only_nontoxic_list: list[str] = []
    n_common_list: list[int] = []
    n_only_toxic_list: list[int] = []
    n_only_nontoxic_list: list[int] = []
    toxic_frags_list: list[str] = []
    nontoxic_frags_list: list[str] = []

    for _, row in df_att.iterrows():
        t_safe = row.get("toxic_safe", "")
        n_safe = row.get("nontoxic_safe", "")
        common, only_t, only_n = _compare_fragments(t_safe, n_safe)

        toxic_frags_list.append(SEP.join(sorted(_safe_to_fragments(t_safe))))
        nontoxic_frags_list.append(SEP.join(sorted(_safe_to_fragments(n_safe))))
        common_list.append(SEP.join(sorted(common)))
        only_toxic_list.append(SEP.join(sorted(only_t)))
        only_nontoxic_list.append(SEP.join(sorted(only_n)))
        n_common_list.append(len(common))
        n_only_toxic_list.append(len(only_t))
        n_only_nontoxic_list.append(len(only_n))

    df_comp = df_att.assign(
        toxic_safe_fragments=toxic_frags_list,
        nontoxic_safe_fragments=nontoxic_frags_list,
        common_safe_fragments=common_list,
        only_toxic_safe_fragments=only_toxic_list,
        only_nontoxic_safe_fragments=only_nontoxic_list,
        n_common_safe=n_common_list,
        n_only_toxic_safe=n_only_toxic_list,
        n_only_nontoxic_safe=n_only_nontoxic_list,
    )
    df_comp.to_csv(out_compared, index=False)
    print(f"[full] saved compared -> {out_compared} (rows={len(df_comp):,})")

    # 4) filter (same as filter_safe_pairs.py)
    n_start = len(df_comp)
    df_f = df_comp[df_comp["n_common_safe"].ne(0)].copy()
    df_f = df_f[~((df_f["n_only_nontoxic_safe"] == 0) & (df_f["n_only_toxic_safe"] == 0))].copy()
    mask_long = df_f.apply(
        lambda r: _has_any_fragment_ge(r.get("only_toxic_safe_fragments"), max_frag_len)
        or _has_any_fragment_ge(r.get("only_nontoxic_safe_fragments"), max_frag_len),
        axis=1,
    )
    df_f = df_f[~mask_long].copy()
    df_f = df_f[
        (df_f["n_only_toxic_safe"] <= max_frag_num) & (df_f["n_only_nontoxic_safe"] <= max_frag_num)
    ].copy()
    out_filtered.parent.mkdir(parents=True, exist_ok=True)
    df_f.to_csv(out_filtered, index=False)
    print(f"[full] saved filtered -> {out_filtered} (rows={len(df_f):,}, removed={n_start-len(df_f):,})")

    # 5) validate SAFE decode (valid/invalid split)
    validate_df = _import_validate_df()
    validated = validate_df(df_f)
    valid_df = validated[validated["safe_decode_all_ok"]].copy()
    invalid_df = validated[~validated["safe_decode_all_ok"]].copy()
    valid_df.to_csv(out_valid, index=False)
    invalid_df.to_csv(out_invalid, index=False)
    print(f"[full] saved validated valid   -> {out_valid} (rows={len(valid_df):,})")
    print(f"[full] saved validated invalid -> {out_invalid} (rows={len(invalid_df):,})")

    # 6) final merged (valid only) — 여기서는 valid_df 자체가 이미 metabolism+sider 통합임
    valid_df.to_csv(out_final, index=False)
    print(f"[full] saved FINAL valid merged -> {out_final} (rows={len(valid_df):,})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run full ACE SAFE pipeline for metabolism+pairs.csv and pairs_sider.csv."
    )
    ap.add_argument("--metabolism_pairs", type=Path, default=DEFAULT_METAB_PAIRS)
    ap.add_argument("--sider_pairs", type=Path, default=DEFAULT_SIDER_PAIRS)
    ap.add_argument("--out_mapping", type=Path, default=DEFAULT_OUT_MAPPING)
    ap.add_argument("--out_attached", type=Path, default=DEFAULT_OUT_ATTACHED)
    ap.add_argument("--out_compared", type=Path, default=DEFAULT_OUT_COMPARED)
    ap.add_argument("--out_filtered", type=Path, default=DEFAULT_OUT_FILTERED)
    ap.add_argument("--out_valid", type=Path, default=DEFAULT_OUT_VALID)
    ap.add_argument("--out_invalid", type=Path, default=DEFAULT_OUT_INVALID)
    ap.add_argument("--out_final", type=Path, default=DEFAULT_OUT_FINAL)
    ap.add_argument("--max-frag-len", type=int, default=28)
    ap.add_argument("--max-frag-num", type=int, default=4)
    args = ap.parse_args()

    run(
        metabolism_pairs=args.metabolism_pairs.expanduser().resolve(),
        sider_pairs=args.sider_pairs.expanduser().resolve(),
        out_mapping=args.out_mapping.expanduser().resolve(),
        out_attached=args.out_attached.expanduser().resolve(),
        out_compared=args.out_compared.expanduser().resolve(),
        out_filtered=args.out_filtered.expanduser().resolve(),
        out_valid=args.out_valid.expanduser().resolve(),
        out_invalid=args.out_invalid.expanduser().resolve(),
        out_final=args.out_final.expanduser().resolve(),
        max_frag_len=args.max_frag_len,
        max_frag_num=args.max_frag_num,
    )


if __name__ == "__main__":
    main()

