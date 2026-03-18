"""
Evaluation metrics for SAFE QA tasks.

각 task의 answer 포맷(예: {\"answer\": \"...\"})에 맞춰서
gold / LLM answer를 받아 metric을 계산한다.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional, Tuple

import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

try:
    from rdkit.Chem.Fingerprints import FingerprintMols
except ImportError:
    FingerprintMols = None
try:
    from rdkit.Chem import MACCSkeys
except ImportError:
    MACCSkeys = None

_QA_SRC = Path(__file__).resolve().parent
_PROJECT_ROOT = _QA_SRC.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    # SAFE decoder (SAFE -> SMILES)
    from safe.safe.converter import decode as safe_decode
except Exception:  # pragma: no cover - optional dependency
    safe_decode = None

_DATA_PAIRS_CSV = _PROJECT_ROOT / "molecule_safe_ver" / "commom_frage_pairs_with_smiles.csv"
_DATA_SMILES_TO_SAFE_CSV = _PROJECT_ROOT / "molecule_safe_ver" / "smiles_to_safe.csv"
_PAIRS_DF: Optional[pd.DataFrame] = None
_SMILES_TO_SAFE_DF: Optional[pd.DataFrame] = None

# task별 고정 메트릭 키: 동일 task/variant면 모델과 관계없이 항상 같은 키로 평가 결과를 내기 위함
# Task 1, 3: single/multi_step 공통. multi_step에서는 fragment set 기준 Precision/Recall/F1 사용.
TASK_METRIC_KEYS = {
    1: [
        "fragment_EM",
        "fragment_BLEU1",
        "fragment_Precision",
        "fragment_Recall",
        "fragment_F1",
        "molecule_EM",
        "molecule_morganFT",
        "molecule_validity",
    ],
    2: [
        "EM",
        "BLEU1",
        "validity",
        "levenshtein_dist",
        "levenshtein_norm",
        "molecule_EM",
        "molecule_morganFT",
        "molecule_validity",
    ],
    3: [
        "fragment_EM",
        "fragment_BLEU1",
        "fragment_Precision",
        "fragment_Recall",
        "fragment_F1",
    ],
    4: [
        "exact_match",
        "bleu",
        "levenshtein",
        "rdk_fts",
        "maccs_fts",
        "morgan_fts",
        "validity",
    ],
}


def _load_pairs_df() -> Optional[pd.DataFrame]:
    global _PAIRS_DF
    if _PAIRS_DF is None:
        try:
            _PAIRS_DF = pd.read_csv(_DATA_PAIRS_CSV)
        except Exception:
            _PAIRS_DF = None
    return _PAIRS_DF


def _extract_answer(ans: Any) -> str:
    """지원되는 answer 포맷(dict 또는 str)에서 실제 문자열만 추출."""
    if ans is None:
        return ""
    if isinstance(ans, dict):
        return str(ans.get("answer", "")).strip()
    return str(ans).strip()


def _tokenize_safe_fragments(s: str) -> list[str]:
    """
    SAFE string을 토큰 시퀀스로 변환.
    기본적으로 '.' 기준으로 fragment를 자르고, 공백/빈 토큰은 제거.
    single_step: 1개, multi_step: 2개 이상.
    """
    s = (s or "").strip()
    if not s:
        return []
    # 공백 제거 후 '.' 기준 분할
    return [tok for tok in s.replace(" ", "").split(".") if tok]


def _fragment_set_precision_recall_f1(gold: str, pred: str) -> Tuple[float, float, float]:
    """
    Gold/Pred SAFE fragment 문자열을 '.' 기준 fragment set으로 비교하여
    Precision, Recall, F1을 계산. (multi_step 평가용, single_step에서도 동일 공식 적용)

    - gold/pred를 _tokenize_safe_fragments로 리스트로 만든 뒤 set으로 변환.
    - TP = |gold_set ∩ pred_set|
    - Precision = TP / |pred_set| (pred 비어 있으면 0)
    - Recall = TP / |gold_set| (gold 비어 있으면 1로 간주)
    - F1 = 2*P*R/(P+R), 단 P+R=0이면 0.
    - 둘 다 비어 있으면 P=R=F1=1.0.
    """
    gold_toks = _tokenize_safe_fragments(gold)
    pred_toks = _tokenize_safe_fragments(pred)
    gold_set = set(gold_toks)
    pred_set = set(pred_toks)

    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set:
        return 0.0 if pred_set else 1.0, 1.0, (0.0 if pred_set else 1.0)
    if not pred_set:
        return 0.0, 0.0, 0.0

    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _tokenize_char_ngrams(s: str, n: int = 4) -> list[str]:
    """문자열을 문자 n-gram 리스트로 변환 (부분 문자열 겹침 반영용)."""
    s = (s or "").strip().replace(" ", "")
    if not s or n < 1:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def _bleu1_safe_fragments(gold: str, pred: str, use_char_ngrams: bool = True, ngram_n: int = 4) -> float:
    """
    SAFE fragment 문자열에 대한 BLEU-1.

    - use_char_ngrams=True (기본): 문자 n-gram 기준으로 겹치는 부분 문자열을 반영.
      단일 fragment여도 gold/pred가 비슷하면 0이 아닌 점수를 준다.
    - use_char_ngrams=False: '.' 기준 fragment 토큰 정확 일치만 (기존 방식).
    """
    if use_char_ngrams:
        gold_tokens = _tokenize_char_ngrams(gold, ngram_n)
        pred_tokens = _tokenize_char_ngrams(pred, ngram_n)
    else:
        gold_tokens = _tokenize_safe_fragments(gold)
        pred_tokens = _tokenize_safe_fragments(pred)

    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)

    overlap = 0
    for t, c in pred_counts.items():
        overlap += min(c, gold_counts.get(t, 0))

    precision = overlap / max(len(pred_tokens), 1)
    return float(precision)


def _safe_to_smiles_validity(safe_str: str) -> float:
    """
    SAFE string을 decode하여 RDKit SMILES validity를 체크.

    - safe_decode가 없거나, decode 실패, MolFromSmiles 실패 시 0.0
    - 정상적으로 SMILES로 decode되고 RDKit Mol이 생성되면 1.0
    """
    safe_str = (safe_str or "").strip()
    if not safe_str or safe_decode is None:
        return 0.0
    try:
        # SAFE decode -> SMILES
        decoded_smiles = safe_decode(safe_str)
    except Exception:
        return 0.0

    if not decoded_smiles:
        return 0.0

    try:
        mol = Chem.MolFromSmiles(str(decoded_smiles))
    except Exception:
        return 0.0

    return 1.0 if mol is not None else 0.0


def _decode_safe_to_smiles(safe_str: str) -> Optional[str]:
    """SAFE string을 SMILES로 decode (실패 시 None)."""
    safe_str = (safe_str or "").strip()
    if not safe_str or safe_decode is None:
        return None
    try:
        decoded_smiles = safe_decode(safe_str)
    except Exception:
        return None
    decoded_smiles = (decoded_smiles or "").strip()
    return decoded_smiles or None


def _mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _morgan_tanimoto(smiles1: str, smiles2: str, radius: int = 2, nbits: int = 2048) -> Optional[float]:
    """두 SMILES 간 Morgan fingerprint Tanimoto similarity (실패 시 None)."""
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None


def _rdkit_tanimoto(smiles1: str, smiles2: str) -> float:
    """두 SMILES 간 RDKit fingerprint Tanimoto similarity (실패 시 0.0)."""
    if FingerprintMols is None:
        return 0.0
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _maccs_tanimoto(smiles1: str, smiles2: str) -> float:
    """두 SMILES 간 MACCS fingerprint Tanimoto similarity (실패 시 0.0)."""
    if MACCSkeys is None:
        return 0.0
    mol1 = _mol_from_smiles(smiles1)
    mol2 = _mol_from_smiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return 0.0


def _join_safe_fragments(*parts: str) -> str:
    """여러 SAFE fragment 문자열을 '.'로 안전하게 이어붙인다."""
    toks = []
    for p in parts:
        p = (p or "").strip()
        if p:
            toks.append(p)
    return ".".join(toks)


def _get_row_by_id(row_id: Optional[int]) -> Optional[pd.Series]:
    """commom_frage_pairs_with_smiles.csv에서 row_id (index)에 해당하는 row를 가져온다."""
    if row_id is None:
        return None
    df = _load_pairs_df()
    if df is None:
        return None
    if row_id < 0 or row_id >= len(df):
        return None
    return df.iloc[int(row_id)]


def _load_smiles_to_safe_df() -> Optional[pd.DataFrame]:
    global _SMILES_TO_SAFE_DF
    if _SMILES_TO_SAFE_DF is None:
        try:
            _SMILES_TO_SAFE_DF = pd.read_csv(_DATA_SMILES_TO_SAFE_CSV)
        except Exception:
            _SMILES_TO_SAFE_DF = None
    return _SMILES_TO_SAFE_DF


def _get_row_smiles_to_safe_by_id(row_id: Optional[int]) -> Optional[pd.Series]:
    """smiles_to_safe.csv에서 row_id (index)에 해당하는 row를 가져온다 (Task2 참조 SMILES용)."""
    if row_id is None:
        return None
    df = _load_smiles_to_safe_df()
    if df is None:
        return None
    if row_id < 0 or row_id >= len(df):
        return None
    return df.iloc[int(row_id)]


def task1_toxic_safe_to_nontoxic_safe_eval(
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int] = None,
) -> Tuple[
    float, float, float, float, float,
    Optional[float], Optional[float], Optional[float],
]:
    """
    Task 1: toxic_safe_to_nontoxic_safe 평가.
    single_step/multi_step 공통: answer는 단일 fragment 또는 dot-separated fragments.

    Metrics
    -------
    - fragment_EM: gold와의 Exact Match (전체 문자열 일치, 1.0/0.0)
    - fragment_BLEU1: fragment/문자 n-gram BLEU-1
    - fragment_Precision, fragment_Recall, fragment_F1: fragment set 기준 (multi_step에서 유의미)
    - molecule_EM, molecule_morganFT, molecule_validity: row_id 있을 때 분자 수준 평가
    """
    gold = _extract_answer(gold_answer)
    pred = _extract_answer(llm_answer)

    fragment_EM = 1.0 if gold and (gold == pred) else 0.0
    fragment_BLEU1 = _bleu1_safe_fragments(gold, pred)
    fragment_Precision, fragment_Recall, fragment_F1 = _fragment_set_precision_recall_f1(gold, pred)

    molecule_EM: Optional[float] = None
    molecule_morganFT: Optional[float] = None
    molecule_validity: Optional[float] = None

    row = _get_row_by_id(row_id)
    if row is not None and pred:
        # common_safe_fragments + pred_only_nontoxic_safe_fragments 로 full SAFE 구성
        common_safe = str(row.get("common_safe_fragments", "") or "").strip()
        pred_full_safe = _join_safe_fragments(common_safe, pred)
        gold_smiles = str(row.get("nontoxic_safe_decoded_smiles", "") or "").strip()

        # 전체 SAFE를 decode한 SMILES와 nontoxic_safe_decoded_smiles를 둘 다 canonical 후 평가
        pred_smiles = _decode_safe_to_smiles(pred_full_safe)
        if pred_smiles is not None:
            mol_pred = _mol_from_smiles(pred_smiles)
            molecule_validity = 1.0 if mol_pred is not None else 0.0
            mol_gold = _mol_from_smiles(gold_smiles)
            if mol_pred is not None and mol_gold is not None:
                can_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                can_gold = Chem.MolToSmiles(mol_gold, canonical=True)
                molecule_EM = 1.0 if can_pred == can_gold else 0.0
                molecule_morganFT = _morgan_tanimoto(can_pred, can_gold)

    return (
        fragment_EM,
        fragment_BLEU1,
        fragment_Precision,
        fragment_Recall,
        fragment_F1,
        molecule_EM,
        molecule_morganFT,
        molecule_validity,
    )


def _levenshtein(a: str, b: str) -> int:
    """단순 Levenshtein 거리 (편집 거리)."""
    a = a or ""
    b = b or ""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[lb]


def task2_smiles_to_safe_eval(
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int] = None,
) -> Tuple[
    float, float, Optional[float], Optional[float], Optional[float],
    Optional[float], Optional[float], Optional[float],
]:
    """
    Task 2: smiles_to_safe 평가.

    목적: 예측 SAFE를 decode한 SMILES가 참조(입력) SMILES와 동일한 분자가 되도록.

    Metrics
    -------
    - EM: gold SAFE와의 Exact Match (1.0/0.0)
    - BLEU1: SAFE 문자열 문자 n-gram BLEU-1
    - validity: pred SAFE → SMILES → RDKit Mol 생성 성공 여부 (1.0/0.0)
    - levenshtein_dist, levenshtein_norm: SAFE 문자열 편집거리
    - molecule_EM, molecule_morganFT, molecule_validity: row_id 있을 때,
      pred SAFE decode → canonical SMILES vs 참조 canonical SMILES 비교
    """
    gold = _extract_answer(gold_answer)
    pred = _extract_answer(llm_answer)

    EM = 1.0 if gold and (gold == pred) else 0.0
    BLEU1 = _bleu1_safe_fragments(gold, pred)

    validity = _safe_to_smiles_validity(pred)
    dist = float(_levenshtein(gold, pred))
    max_len = float(max(len(gold or ""), len(pred or ""), 1))
    lev_norm = 1.0 - dist / max_len

    molecule_EM: Optional[float] = None
    molecule_morganFT: Optional[float] = None
    molecule_validity: Optional[float] = None

    # 전체 SAFE를 decode한 SMILES와 참조(입력) SMILES를 둘 다 canonical 후 평가
    row = _get_row_smiles_to_safe_by_id(row_id)
    if row is not None and pred:
        ref_smiles = str(row.get("canonical_smiles", "") or row.get("smiles", "") or "").strip()
        pred_smiles = _decode_safe_to_smiles(pred)
        if pred_smiles is not None and ref_smiles:
            mol_pred = _mol_from_smiles(pred_smiles)
            molecule_validity = 1.0 if mol_pred is not None else 0.0
            mol_ref = _mol_from_smiles(ref_smiles)
            if mol_pred is not None and mol_ref is not None:
                can_pred = Chem.MolToSmiles(mol_pred, canonical=True)
                can_ref = Chem.MolToSmiles(mol_ref, canonical=True)
                molecule_EM = 1.0 if can_pred == can_ref else 0.0
                molecule_morganFT = _morgan_tanimoto(can_pred, can_ref)

    return EM, BLEU1, validity, dist, lev_norm, molecule_EM, molecule_morganFT, molecule_validity


def task3_toxic_fragment_identification_eval(
    gold_answer: Any,
    llm_answer: Any,
) -> Tuple[float, float, float, float, float]:
    """
    Task 3: toxic_fragment_identification 평가.
    single_step/multi_step 공통: answer는 단일 fragment 또는 dot-separated fragments.

    Metrics
    -------
    - fragment_EM: gold only_toxic_safe_fragments와의 Exact Match (1.0/0.0)
    - fragment_BLEU1: fragment/문자 n-gram BLEU-1
    - fragment_Precision, fragment_Recall, fragment_F1: fragment set 기준 (multi_step에서 유의미)
    """
    gold = _extract_answer(gold_answer)
    pred = _extract_answer(llm_answer)

    fragment_EM = 1.0 if gold and (gold == pred) else 0.0
    fragment_BLEU1 = _bleu1_safe_fragments(gold, pred)
    fragment_Precision, fragment_Recall, fragment_F1 = _fragment_set_precision_recall_f1(gold, pred)
    return fragment_EM, fragment_BLEU1, fragment_Precision, fragment_Recall, fragment_F1


def task4_safe_to_nontoxic_smiles_eval(
    gold_answer: Any,
    llm_answer: Any,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Task 4: safe_to_nontoxic_smiles 평가 (SMILES 생성).
    answer는 단일 SMILES 문자열 (nontoxic_safe_decoded_smiles).

    Table 5 스타일 메트릭:
    - exact_match (EXACT↑): canonical gold == canonical pred
    - bleu (BLEU↑): 문자 4-gram BLEU-1
    - levenshtein (LEVENSHTEIN↓): 편집 거리 (숫자 그대로 반환, 낮을수록 좋음)
    - rdk_fts (RDK FTS↑), maccs_fts (MACCS FTS↑), morgan_fts (MORGAN FTS↑): Tanimoto similarity
    - validity (VALIDITY↑): 예측 SMILES가 유효한 분자면 1.0
    """
    gold_s = ( _extract_answer(gold_answer) or "" ).strip()
    pred_s = ( _extract_answer(llm_answer) or "" ).strip()

    validity = 1.0 if pred_s and _mol_from_smiles(pred_s) is not None else 0.0

    can_gold: Optional[str] = None
    can_pred: Optional[str] = None
    if gold_s:
        mol_g = _mol_from_smiles(gold_s)
        if mol_g is not None:
            can_gold = Chem.MolToSmiles(mol_g, canonical=True)
    if pred_s:
        mol_p = _mol_from_smiles(pred_s)
        if mol_p is not None:
            can_pred = Chem.MolToSmiles(mol_p, canonical=True)

    exact_match = 1.0 if (can_gold and can_pred and can_gold == can_pred) else 0.0
    bleu = _bleu1_safe_fragments(gold_s, pred_s, use_char_ngrams=True, ngram_n=4)
    levenshtein = float(_levenshtein(gold_s, pred_s))

    rdk_fts = 0.0
    maccs_fts = 0.0
    morgan_fts = 0.0
    if can_gold and can_pred:
        rdk_fts = _rdkit_tanimoto(can_gold, can_pred)
        maccs_fts = _maccs_tanimoto(can_gold, can_pred)
        m = _morgan_tanimoto(can_gold, can_pred)
        morgan_fts = m if m is not None else 0.0

    return exact_match, bleu, levenshtein, rdk_fts, maccs_fts, morgan_fts, validity
