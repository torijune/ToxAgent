from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


SCRIPT_DIR = Path(__file__).resolve().parent
ACE_SAFE_VER_DIR = SCRIPT_DIR.parent
if str(ACE_SAFE_VER_DIR) not in sys.path:
    sys.path.insert(0, str(ACE_SAFE_VER_DIR))

DEFAULT_SPLIT_DIR = ACE_SAFE_VER_DIR / "splits" / "scaffold_by_endpoint"
# 기본 입력: pairs_safe_filtered.csv (valid/invalid 분리 출력)
DEFAULT_RAW_CSV = ACE_SAFE_VER_DIR / "pairs_safe_sider_filtered.csv"


def _try_import_safe_decoder():
    """번들 third_party/safe 디코더."""
    try:
        import ace_local

        ace_local.ensure_safe_pkg_path()
        from safe.safe.converter import decode as _decode

        return _decode
    except Exception:
        return None


SAFE_DECODE = _try_import_safe_decoder()


def _canon_smiles(s: str) -> str:
    """RDKit canonical SMILES (실패 시 빈 문자열)."""
    if Chem is None:
        return (s or "").strip()
    s = (s or "").strip()
    if not s:
        return ""
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def _decode_safe(safe_str: str) -> str:
    """SAFE -> SMILES 디코딩. 실패 시 빈 문자열."""
    safe_str = (safe_str or "").strip()
    if not safe_str or SAFE_DECODE is None:
        return ""
    try:
        decoded = SAFE_DECODE(safe_str)
        return (decoded or "").strip()
    except Exception:
        return ""


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"toxic_smiles", "nontoxic_smiles", "toxic_safe", "nontoxic_safe"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    out = df.copy()
    out["toxic_safe_decoded_smiles"] = out["toxic_safe"].astype(str).map(_decode_safe)
    out["nontoxic_safe_decoded_smiles"] = out["nontoxic_safe"].astype(str).map(_decode_safe)

    tox_canon = out["toxic_smiles"].astype(str).map(_canon_smiles)
    nontox_canon = out["nontoxic_smiles"].astype(str).map(_canon_smiles)
    tox_dec_canon = out["toxic_safe_decoded_smiles"].astype(str).map(_canon_smiles)
    nontox_dec_canon = out["nontoxic_safe_decoded_smiles"].astype(str).map(_canon_smiles)

    out["toxic_safe_decode_ok"] = tox_dec_canon.astype(bool)
    out["nontoxic_safe_decode_ok"] = nontox_dec_canon.astype(bool)
    out["toxic_safe_decoded_matches"] = (tox_canon != "") & (tox_canon == tox_dec_canon)
    out["nontoxic_safe_decoded_matches"] = (nontox_canon != "") & (nontox_canon == nontox_dec_canon)
    out["safe_decode_all_ok"] = out["toxic_safe_decoded_matches"] & out["nontoxic_safe_decoded_matches"]
    return out


def _default_inputs() -> list[Path]:
    """기본 입력: ace_safe_ver/pairs_safe_filtered.csv → *_valid.csv, *_invalid.csv 출력."""
    return [DEFAULT_RAW_CSV]


def _resolve_inputs(paths: list[str] | None) -> list[Path]:
    if not paths:
        return [p for p in _default_inputs() if p.exists()]
    out: list[Path] = []
    for p in paths:
        out.append(Path(p).expanduser().resolve())
    return out


def run(inputs: list[Path], out_dir: Optional[Path], write_invalid: bool) -> None:
    if SAFE_DECODE is None:
        raise ImportError(
            "SAFE decoder import 실패: `from safe.safe.converter import decode`가 필요합니다. "
            "환경(venv)에서 safe 패키지 설치/경로를 확인하세요."
        )
    if Chem is None:
        raise ImportError("RDKit import 실패: rdkit이 설치된 환경에서 실행하세요.")

    if not inputs:
        raise FileNotFoundError(
            f"입력 CSV를 찾을 수 없습니다. 기본: {DEFAULT_RAW_CSV} 또는 --inputs 로 경로를 지정하세요."
        )

    base_dir = out_dir.expanduser().resolve() if out_dir is not None else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for in_path in inputs:
        in_path = in_path.expanduser().resolve()
        if not in_path.exists():
            print(f"[SKIP] not found: {in_path}", file=sys.stderr)
            continue

        df = pd.read_csv(in_path)
        validated = validate_df(df)

        out_parent = base_dir if base_dir is not None else in_path.parent
        stem = in_path.stem
        valid_path = out_parent / f"{stem}_valid.csv"
        invalid_path = out_parent / f"{stem}_invalid.csv"

        valid_df = validated[validated["safe_decode_all_ok"]].copy()
        invalid_df = validated[~validated["safe_decode_all_ok"]].copy()

        valid_df.to_csv(valid_path, index=False)
        invalid_df.to_csv(invalid_path, index=False)

        n = len(validated)
        n_ok = len(valid_df)
        n_fail = len(invalid_df)
        print(f"[OK] Input  : {in_path}")
        print(f"[OK] Valid  : {valid_path} ({n_ok:,} rows)")
        print(f"[OK] Invalid: {invalid_path} ({n_fail:,} rows)")
        print(f"[STAT] total={n:,}  valid={n_ok:,}  invalid={n_fail:,}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Validate that SAFE decoding returns the original SMILES for merged scaffold splits.\n"
            "Adds columns: toxic_safe_decoded_smiles, nontoxic_safe_decoded_smiles, and match flags."
        )
    )
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help=f"Input CSV path(s). Default: {DEFAULT_RAW_CSV}",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory. Default: same as input → <stem>_valid.csv, <stem>_invalid.csv",
    )
    ap.add_argument(
        "--write_invalid",
        action="store_true",
        help="Deprecated; valid/invalid 분리 출력은 항상 수행됩니다.",
    )
    args = ap.parse_args()

    inputs = _resolve_inputs(args.inputs)
    run(inputs=inputs, out_dir=args.out_dir, write_invalid=args.write_invalid)


if __name__ == "__main__":  # pragma: no cover
    main()

