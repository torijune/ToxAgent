#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFE QA에 대해 GPT(OpenAI) 또는 Gemini(google-genai) inference 및 evaluation 수행.
build_safe_qa.py와 동일한 인자(--split, --task, --variant, --molecule_repr, --step)로 동일한 QA 데이터 경로 사용.

Gemini: 모델명이 gemini 로 시작하면 Google Gen AI SDK 사용.
  pip install google-genai
  환경변수: GOOGLE_API_KEY 또는 GEMINI_API_KEY (또는 Vertex 등 SDK 기본 인증)
  예: --model gemini-3.1-pro / gemini-3-flash / gemini-flash-lite

출력 디렉터리 구조 (build_qa와 동일한 트리):
  out_dir / <split> / <task> / [<molecule_repr>] / <step> /
    results/          <- 샘플별 결과 (predictions_<model>.jsonl)
    evaluation/       <- 총 evaluation 요약 (evaluation_summary_<model>.json)
"""

import os
import sys
import json
import time
import base64
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from threading import Lock

from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

try:
    from google import genai as google_genai  # type: ignore
    _GENAI_AVAILABLE = True
except ImportError:
    google_genai = None  # type: ignore
    _GENAI_AVAILABLE = False

try:
    import anthropic  # type: ignore
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False

# QA 디렉터리 (LLMs의 상위)
_LLM_DIR = Path(__file__).resolve().parent
_QA_DIR = _LLM_DIR.parent
_QA_SRC = _QA_DIR / "src"

# eval_metric import (QA/src)
import sys
if str(_QA_SRC) not in sys.path:
    sys.path.insert(0, str(_QA_SRC))
try:
    from eval_metric import (
        task1_toxic_fragment_identification_eval,
        task2_nontoxic_fragment_generation_eval,
        task3_nontoxic_smiles_generation_eval,
        task3_nontoxic_safe_generation_eval,
        task3_stepwise_cot_nontoxic_smiles_generation_eval,
        task3_stepwise_cot_nontoxic_safe_generation_eval,
        subtask1_safe_to_smiles_eval,
        subtask2_smiles_to_safe_eval,
        TASK_METRIC_KEYS,
    )
except ImportError:
    task1_toxic_fragment_identification_eval = None
    task2_nontoxic_fragment_generation_eval = None
    task3_nontoxic_smiles_generation_eval = None
    task3_nontoxic_safe_generation_eval = None
    task3_stepwise_cot_nontoxic_smiles_generation_eval = None
    task3_stepwise_cot_nontoxic_safe_generation_eval = None
    subtask1_safe_to_smiles_eval = None
    subtask2_smiles_to_safe_eval = None
    TASK_METRIC_KEYS = {}

from ace_root_find import resolve_ace_safe_ver_root  # noqa: E402

_ACE = resolve_ace_safe_ver_root(__file__)

DEFAULT_QA_DIR = _QA_DIR
DEFAULT_DATA_PATH = _QA_DIR / "test" / "task1_toxic_fragment_identification" / "both_repre" / "single_step" / "task1_toxic_fragment_identification_qa.jsonl"
DEFAULT_ENV_PATH = _ACE / ".env"

REPRE_CHOICES = ["only_safe", "only_smiles", "both_repre"]
REPRE_CHOICES_WITH_ALL = REPRE_CHOICES + ["all"]


def _is_gemini_model(model: str) -> bool:
    return str(model).strip().lower().startswith("gemini")

def _is_claude_model(model: str) -> bool:
    return str(model).strip().lower().startswith("claude")

def _gemini_api_model(model: str) -> str:
    """
    사용자가 흔히 쓰는 짧은 모델명(gemini-3-flash 등)을
    google-genai SDK에서 실제로 동작하는 API 모델명으로 매핑.
    """
    m = str(model).strip()
    ml = m.lower()
    mapping = {
        # Gemini 3 Flash 계열 별칭
        "gemini-3-flash": "gemini-3-flash-preview",
        "gemini-flash-lite": "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
    }
    # 이미 preview/최신 suffix가 있으면 그대로 사용
    return mapping.get(ml, m)


def _gemini_response_text(response: Any) -> str:
    """google-genai 응답에서 텍스트 추출."""
    if response is None:
        return ""
    t = getattr(response, "text", None)
    if t is not None and str(t).strip():
        return str(t)
    try:
        cands = getattr(response, "candidates", None) or []
        if cands:
            parts = getattr(cands[0].content, "parts", None) or []
            if parts and getattr(parts[0], "text", None):
                return str(parts[0].text)
    except Exception:
        pass
    return str(response or "")


def _normalize_step(step: str) -> str:
    if step in ("single", "single_step"):
        return "single_step"
    if step in ("multi", "multi_step"):
        return "multi_step"
    return step


def _data_path_for(
    task: str,
    variant: str,
    step: str,
    split: str = "test",
    repres: str = "both_repre",
) -> Path:
    qa_base = _QA_DIR / split
    if task == "subtask1":
        return qa_base / "subtask1_safe_to_smiles" / "subtask1_safe_to_smiles_qa.jsonl"
    if task == "subtask2":
        return qa_base / "subtask2_smiles_to_safe" / "subtask2_smiles_to_safe_qa.jsonl"
    step_norm = _normalize_step(step)
    if task == "task1":
        base = qa_base / "task1_toxic_fragment_identification" / repres / step_norm
        fname = "task1_toxic_fragment_identification_qa.jsonl" if variant == "base" else f"task1_toxic_fragment_identification_qa_{variant}.jsonl"
        return base / fname
    if task == "task2":
        base = qa_base / "task2_nontoxic_fragment_generation" / repres / step_norm
        fname = "task2_nontoxic_fragment_generation_qa.jsonl" if variant == "base" else f"task2_nontoxic_fragment_generation_qa_{variant}.jsonl"
        return base / fname
    # task3
    if task == "task3_instruction":
        base = qa_base / "task3_instruction_nontoxic_smiles_generation" / repres / step_norm
        fname = "task3_instruction_nontoxic_smiles_generation_qa.jsonl"
        return base / fname
    if task == "task3_nontoxic_safe_generation":
        base = qa_base / "task3_nontoxic_safe_generation" / repres / step_norm
        fname = "task3_nontoxic_safe_generation_qa.jsonl" if variant == "base" else f"task3_nontoxic_safe_generation_qa_{variant}.jsonl"
        return base / fname
    if task == "task3_stepwise_cot":
        base = qa_base / "task3_stepwise_cot_nontoxic_smiles_generation" / repres / step_norm
        fname = (
            "task3_stepwise_cot_nontoxic_smiles_generation_qa.jsonl"
            if variant == "base"
            else f"task3_stepwise_cot_nontoxic_smiles_generation_qa_{variant}.jsonl"
        )
        return base / fname
    if task == "task3_stepwise_cot_safe_generation":
        base = qa_base / "task3_stepwise_cot_nontoxic_safe_generation" / repres / step_norm
        fname = (
            "task3_stepwise_cot_nontoxic_safe_generation_qa.jsonl"
            if variant == "base"
            else f"task3_stepwise_cot_nontoxic_safe_generation_qa_{variant}.jsonl"
        )
        return base / fname
    base = qa_base / "task3_nontoxic_smiles_generation" / repres / step_norm
    fname = "task3_nontoxic_smiles_generation_qa.jsonl" if variant == "base" else f"task3_nontoxic_smiles_generation_qa_{variant}.jsonl"
    return base / fname


JSON_SCHEMA = {
    "name": "only_nontoxic_safe_fragments",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}

JSON_SCHEMA_STEPWISE_COT = {
    "name": "task3_stepwise_cot_output",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "step1_only_toxic_safe_fragments": {"type": "string"},
            "step1_reasoning": {"type": "string"},
            "step2_only_nontoxic_safe_fragments": {"type": "string"},
            "step2_reasoning": {"type": "string"},
            "step3_reasoning": {"type": "string"},
        },
        # OpenAI strict schema requires `required` to include every key in `properties`.
        "required": [
            "answer",
            "step1_only_toxic_safe_fragments",
            "step1_reasoning",
            "step2_only_nontoxic_safe_fragments",
            "step2_reasoning",
            "step3_reasoning",
        ],
        # OpenAI json_schema response_format requires additionalProperties to be present and false at the root.
        "additionalProperties": False,
    },
}


def _common_system_instruction() -> str:
    return (
        "You are a molecular toxicity reasoning assistant specialized in SAFE and SMILES representations.\n"
        "Follow the task instruction exactly and return ONLY the requested JSON object.\n"
        "Do not add explanations, markdown, code fences, or prose outside the JSON.\n"
        "Do not add extra keys unless explicitly required.\n"
    )


def _system_instruction_for_task(task: str) -> str:
    base = _common_system_instruction()

    if task == "task1":
        return (
            base
            + "Your task is to identify the fragment(s) in the toxic molecule that are most likely associated with toxicity.\n"
            + "Return the toxic-only SAFE fragment string exactly.\n"
            + "If there are multiple fragments, return them as a dot-separated SAFE string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )

    if task == "task2":
        return (
            base
            + "Your task is to generate the non-toxic replacement fragment(s) corresponding to the toxic fragment(s).\n"
            + "Return the non-toxic-only SAFE fragment string exactly.\n"
            + "If there are multiple fragments, return them as a dot-separated SAFE string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )

    if task == "task3_nontoxic_safe_generation":
        return (
            base
            + "Your task is to generate the resulting full non-toxic molecule in SAFE representation.\n"
            + "Return the complete non-toxic SAFE string for the whole molecule.\n"
            + "If there are multiple fragments, return them as a dot-separated SAFE string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )

    if task == "task3" or task == "task3_instruction":
        return (
            base
            + "Your task is to generate the final non-toxic molecule as a single SMILES string.\n"
            + "Preserve the original molecular characteristics as much as possible while reducing toxicity.\n"
            + "Return only the final non-toxic molecule SMILES string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )

    if task == "task3_stepwise_cot":
        return (
            "You are a molecular toxicity reasoning assistant specialized in SAFE and SMILES representations.\n"
            "Solve the task through explicit intermediate reasoning steps.\n"
            "Return ONLY a single JSON object.\n"
            "Do not add markdown, code fences, or any text outside the JSON.\n"
            'The JSON must include "answer" as the final non-toxic molecule SMILES string.\n'
            "Also include the required step1/step2 fragment fields and reasoning fields exactly as instructed in the prompt.\n"
        )

    if task == "task3_stepwise_cot_safe_generation":
        return (
            "You are a molecular toxicity reasoning assistant specialized in SAFE and SMILES representations.\n"
            "Solve the task through explicit intermediate reasoning steps.\n"
            "Return ONLY a single JSON object.\n"
            "Do not add markdown, code fences, or any text outside the JSON.\n"
            'The JSON must include "answer" as the final full non-toxic SAFE string for the whole molecule.\n'
            "Also include the required step1/step2 fragment fields and reasoning fields exactly as instructed in the prompt.\n"
        )

    if task == "subtask1":
        return (
            base
            + "Your task is to reconstruct the molecule from the given SAFE representation.\n"
            + "Return the reconstructed molecule as a single SMILES string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )

    if task == "subtask2":
        return (
            base
            + "Your task is to convert the given molecule from SMILES into SAFE representation.\n"
            + "Return the SAFE representation string exactly.\n"
            + "If there are multiple fragments, return them as a dot-separated SAFE string.\n"
            + 'Output schema: {"answer": "..."}\n'
        )

    return base


def read_jsonl(
    path: str | Path,
    *,
    skip_bad_lines: bool = False,
    allow_missing: bool = False,
) -> List[Dict[str, Any]]:
    """
    JSONL 읽기.

    - QA 입력 등: skip_bad_lines=False → 한 줄이라도 깨지면 예외 (기존 동작).
    - predictions_*.jsonl 등: skip_bad_lines=True → 깨진 줄은 건너뛰고 경고만 출력.
    - allow_missing=True 이며 파일이 없으면 [] (요약 단계에서 크래시 방지; 경고 출력).
    """
    p = Path(path)
    if not p.is_file():
        if allow_missing:
            print(f"[WARN] JSONL 없음(빈 결과로 처리): {p.resolve()}", file=sys.stderr)
            return []
        raise FileNotFoundError(f"JSONL not found: {p.resolve()}")

    rows: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_bad_lines:
                    print(
                        f"[WARN] JSONL 파싱 실패, 줄 스킵: {p.resolve()} line {lineno}: {e}",
                        file=sys.stderr,
                    )
                    continue
                raise RuntimeError(
                    f"JSONL 파싱 실패: {p.resolve()} line {lineno}: {e}"
                ) from e
    return rows


def _load_done_ids(predictions_path: Path) -> set:
    """이미 저장된 예측 파일에서 row id 집합 반환 (이어하기용)."""
    done: set = set()
    if not predictions_path.exists():
        return done
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                i = obj.get("id")
                # If a previous run recorded an API/schema error, allow re-try by NOT marking it done.
                raw = str(obj.get("raw", "") or "")
                pred = obj.get("pred", None)
                is_error = raw.startswith("ERROR:")
                is_empty_pred = pred is None or pred == "" or pred == {}
                if i is not None and (not is_error) and (not is_empty_pred):
                    done.add(i)
            except (json.JSONDecodeError, TypeError):
                continue
    return done


def extract_gold(row: Dict[str, Any]) -> str:
    a = row.get("answer", "")
    if isinstance(a, dict):
        return str(a.get("answer", "")).strip()
    return str(a).strip()


def extract_question(row: Dict[str, Any]) -> str:
    return str(row.get("question", ""))


def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(text[l:r + 1])
            except json.JSONDecodeError:
                return None
        return None


def call_model(
    client: OpenAI,
    model: str,
    question: str,
    system_instruction: str,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    max_retries: int = 3,
    sleep_s: float = 0.5,
    json_schema: Optional[dict] = None,
) -> Tuple[Optional[Any], str]:
    last_err = None
    if image_bytes:
        mime = image_mime_type or "image/svg+xml"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        # chat.completions에서 vision 입력은 content를 list로 구성
        user_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    else:
        user_content = question

    messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": user_content}]
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": (json_schema or JSON_SCHEMA),
                    },
                )
            except TypeError as te:
                if "response_format" in str(te):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                else:
                    raise

            raw = (resp.choices[0].message.content if resp.choices else "") or ""
            obj = parse_model_json(raw)
            if obj and "answer" in obj:
                return obj, raw

            return (raw.strip() if raw else None), raw

        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))

    return None, f"ERROR: {last_err}"


def call_gemini(
    client: Any,
    model: str,
    question: str,
    system_instruction: str,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    max_retries: int = 3,
    sleep_s: float = 0.5,
    response_schema: Optional[dict] = None,
) -> Tuple[Optional[Any], str]:
    last_err = None

    def _sanitize_schema(obj: Any) -> Any:
        """google-genai response_schema에서 추가 옵션이 거부될 수 있어 키를 제거."""
        if isinstance(obj, dict):
            sanitized: Dict[str, Any] = {}
            for k, v in obj.items():
                lk = str(k)
                # 서버가 additional_properties 를 모르면 거부됨
                if lk in ("additionalProperties", "additional_properties"):
                    continue
                sanitized[k] = _sanitize_schema(v)
            return sanitized
        if isinstance(obj, list):
            return [_sanitize_schema(x) for x in obj]
        return obj

    for attempt in range(max_retries):
        try:
            from google.genai import types

            sanitized_schema = _sanitize_schema(response_schema) if response_schema else None
            if not sanitized_schema:
                sanitized_schema = {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=sanitized_schema,
            )

            if image_bytes:
                mime = image_mime_type or "image/svg+xml"
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=question),
                            types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        ],
                    )
                ]
            else:
                contents = question

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            raw = _gemini_response_text(response)
            obj = parse_model_json(raw)
            if obj and "answer" in obj:
                return obj, raw
            return (raw.strip() if raw else None), raw

        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))

    return None, f"ERROR: {last_err}"


def _supports_images_for_model(model: str) -> bool:
    m = str(model).strip().lower()
    # Gemini 계열은 기본적으로 vision input을 parts로 받을 수 있음
    if _is_gemini_model(m):
        return True
    # OpenAI vision 계열: GPT-4o / GPT-4.1 / GPT-5 family 등
    if (
        m.startswith("gpt-4o")
        or m.startswith("gpt-4.1")
        or m.startswith("gpt-5")
        or "vision" in m
    ):
        return True
    return False


def _effective_image_mime_type_for_model(model: str, cli_image_mime_type: str) -> Optional[str]:
    """
    모델별 MIME 타입을 결정합니다.
    - Gemini는 SVG 유지
    - OpenAI vision 계열(gpt-4o / gpt-4.1 / gpt-5)은 PNG 사용
    """
    if _is_gemini_model(model):
        return "image/svg+xml"
    if _is_claude_model(model):
        return None
    if _supports_images_for_model(model):
        # OpenAI 쪽은 svg를 거부하는 케이스가 있어 png로 통일
        return "image/png"
    return None


_IMAGE_CACHE_LOCK = Lock()


def _get_or_create_toxic_image_bytes(
    *,
    question: str,
    cache_dir: Path,
    cache_key: str,
    image_mime_type: str,
    mol_size: int,
    image_highlight_mode: Optional[str],
) -> Optional[bytes]:
    """
    add_image.py로 toxic 2D 이미지 생성 후 bytes로 캐시합니다.
    - 기본 출력은 SVG라서 image_mime_type='image/svg+xml'을 권장
    """
    if not cache_dir:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    if image_mime_type == "image/svg+xml":
        ext = ".svg"
    elif image_mime_type == "image/png":
        ext = ".png"
    else:
        ext = ".bin"
    cache_path = cache_dir / f"{cache_key}{ext}"
    if cache_path.exists():
        try:
            return cache_path.read_bytes()
        except Exception:
            pass

    with _IMAGE_CACHE_LOCK:
        # lock 잡은 뒤 한 번 더 확인(경합 방지)
        if cache_path.exists():
            try:
                return cache_path.read_bytes()
            except Exception:
                pass

        from add_image import render_toxic_molecule_image_from_question

        img_obj = render_toxic_molecule_image_from_question(
            question,
            mol_size=(mol_size, mol_size),
            highlight_mode=image_highlight_mode,
            use_svg=(image_mime_type == "image/svg+xml"),
        )
        if isinstance(img_obj, str):
            data = img_obj.encode("utf-8")
        elif isinstance(img_obj, bytes):
            data = img_obj
        else:
            # svg 이외 포맷의 bytes/pil 이미지 등을 받지 못하면 캐시 불가
            return None

        try:
            cache_path.write_bytes(data)
        except Exception:
            pass
        return data


def call_claude(
    client: Any,
    model: str,
    question: str,
    system_instruction: str,
    max_retries: int = 3,
    sleep_s: float = 0.5,
) -> Tuple[Optional[Any], str]:
    """Anthropic Claude 호출. JSON 형태로 답하도록 prompt(시스템 지시) 사용."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                system=system_instruction,
                messages=[{"role": "user", "content": question}],
            )
            # anthropic SDK는 content 블록 리스트 형태인 경우가 많음
            texts: List[str] = []
            content = getattr(resp, "content", None) or []
            if isinstance(content, list):
                for b in content:
                    t = getattr(b, "text", None)
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
            raw = "\n".join(texts).strip() if texts else str(resp)
            obj = parse_model_json(raw)
            if obj and "answer" in obj:
                return obj, raw
            return (raw.strip() if raw else None), raw
        except Exception as e:
            last_err = e
            time.sleep(sleep_s * (attempt + 1))
    return None, f"ERROR: {last_err}"


def _call_model_for_row(
    openai_client: Optional[OpenAI],
    gemini_client: Optional[Any],
    claude_client: Optional[Any],
    model: str,
    row: dict,
    system_instruction: str,
    max_retries: int,
    sleep_s: float,
    json_schema: Optional[dict] = None,
    with_image: bool = False,
    image_cache_dir: Optional[Path] = None,
    image_mime_type: str = "image/svg+xml",
    image_size: int = 300,
    image_highlight_mode: Optional[str] = None,
) -> Tuple[dict, Optional[Any], str]:
    """한 행에 대해 OpenAI / Gemini / Claude 호출. (row, pred, raw) 반환. 배치 병렬용."""
    q = extract_question(row)
    image_bytes: Optional[bytes] = None
    effective_mime_type: Optional[str] = None
    if with_image:
        if not _supports_images_for_model(model):
            return (row, None, f"ERROR: --with_image was requested, but model does not support image input in this runner: {model}")
        if image_cache_dir is None:
            return (row, None, "ERROR: --with_image was requested, but image_cache_dir is not set")

        row_id = row.get("source_index", row.get("id", "row"))
        effective_mime_type = _effective_image_mime_type_for_model(model, image_mime_type)
        if effective_mime_type is None:
            return (row, None, f"ERROR: Model does not support image mime: {model}")

        # 이미지 생성 옵션(highlight/mime)까지 캐시 키에 포함해서,
        # 옵션이 달라졌을 때 이전 캐시가 섞이는 문제를 방지합니다.
        safe_highlight = image_highlight_mode or "none"
        key_raw = (
            f"{row_id}|"
            f"{hashlib.sha256(q.encode('utf-8', errors='ignore')).hexdigest()}|"
            f"img_highlight={safe_highlight}|mime={effective_mime_type}|img_size={image_size}"
        )
        key = hashlib.sha256(key_raw.encode("utf-8", errors="ignore")).hexdigest()[:32]
        image_bytes = _get_or_create_toxic_image_bytes(
            question=q,
            cache_dir=image_cache_dir,
            cache_key=str(key),
            image_mime_type=effective_mime_type,
            mol_size=image_size,
            image_highlight_mode=image_highlight_mode,
        )
        if not image_bytes:
            return (row, None, f"ERROR: Failed to render/load image for --with_image: model={model}, row_id={row_id}")

        # 캐시 키 문자열에 cli mime가 들어갈 수 있으니, request는 effective mime로 고정
        image_mime_type = effective_mime_type
    if _is_gemini_model(model):
        if gemini_client is None:
            return (row, None, "ERROR: Gemini client not initialized (install google-genai, set GOOGLE_API_KEY)")
        api_model = _gemini_api_model(model)
        pred, raw = call_gemini(
            client=gemini_client,
            model=api_model,
            question=q,
            system_instruction=system_instruction,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            max_retries=max_retries,
            sleep_s=sleep_s,
            response_schema=(json_schema["schema"] if json_schema else JSON_SCHEMA["schema"]),
        )
    elif _is_claude_model(model):
        if claude_client is None:
            return (row, None, "ERROR: Claude client not initialized (install anthropic, set ANTHROPIC_API_KEY)")
        pred, raw = call_claude(
            client=claude_client,
            model=model,
            question=q,
            system_instruction=system_instruction,
            max_retries=max_retries,
            sleep_s=sleep_s,
        )
    else:
        if openai_client is None:
            return (row, None, "ERROR: OpenAI client not initialized (set OPENAI_API_KEY)")
        pred, raw = call_model(
            client=openai_client,
            model=model,
            question=q,
            system_instruction=system_instruction,
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            max_retries=max_retries,
            sleep_s=sleep_s,
            json_schema=json_schema,
        )
    return (row, pred, raw)


def normalize_answer(ans: Any) -> str:
    if isinstance(ans, dict):
        return str(ans.get("answer", "") or "").strip()
    return str(ans or "").strip()


def _get_metrics_for_task(
    task: str,
    gold_answer: Any,
    llm_answer: Any,
    row_id: Optional[int],
) -> Dict[str, Any]:
    if task == "task1" and task1_toxic_fragment_identification_eval is not None:
        (
            fragment_EM,
            fragment_BLEU1,
            fragment_Precision,
            fragment_Recall,
            fragment_F1,
        ) = task1_toxic_fragment_identification_eval(gold_answer, llm_answer)
        return {
            "fragment_EM": fragment_EM,
            "fragment_BLEU1": fragment_BLEU1,
            "fragment_Precision": fragment_Precision,
            "fragment_Recall": fragment_Recall,
            "fragment_F1": fragment_F1,
        }

    if task == "task2" and task2_nontoxic_fragment_generation_eval is not None:
        (
            fragment_EM,
            fragment_BLEU1,
            fragment_Precision,
            fragment_Recall,
            fragment_F1,
            molecule_EM,
            molecule_morganFT,
            molecule_validity,
        ) = task2_nontoxic_fragment_generation_eval(gold_answer, llm_answer, row_id=row_id)
        return {
            "fragment_EM": fragment_EM,
            "fragment_BLEU1": fragment_BLEU1,
            "fragment_Precision": fragment_Precision,
            "fragment_Recall": fragment_Recall,
            "fragment_F1": fragment_F1,
            "molecule_EM": molecule_EM,
            "molecule_morganFT": molecule_morganFT,
            "molecule_validity": molecule_validity,
        }

    if task == "task3_nontoxic_safe_generation" and task3_nontoxic_safe_generation_eval is not None:
        (
            safe_EM,
            exact_match,
            bleu,
            levenshtein,
            rdk_fts,
            maccs_fts,
            morgan_fts,
            validity,
        ) = task3_nontoxic_safe_generation_eval(gold_answer, llm_answer)
        return {
            "safe_EM": safe_EM,
            "exact_match": exact_match,
            "bleu": bleu,
            "levenshtein": levenshtein,
            "rdk_fts": rdk_fts,
            "maccs_fts": maccs_fts,
            "morgan_fts": morgan_fts,
            "validity": validity,
        }

    if task in ("task3", "task3_instruction") and task3_nontoxic_smiles_generation_eval is not None:
        (
            exact_match,
            bleu,
            levenshtein,
            rdk_fts,
            maccs_fts,
            morgan_fts,
            validity,
        ) = task3_nontoxic_smiles_generation_eval(gold_answer, llm_answer)
        return {
            "exact_match": exact_match,
            "bleu": bleu,
            "levenshtein": levenshtein,
            "rdk_fts": rdk_fts,
            "maccs_fts": maccs_fts,
            "morgan_fts": morgan_fts,
            "validity": validity,
        }

    if task == "task3_stepwise_cot" and task3_stepwise_cot_nontoxic_smiles_generation_eval is not None:
        return task3_stepwise_cot_nontoxic_smiles_generation_eval(
            gold_answer=gold_answer,
            llm_answer=llm_answer,
            row_id=row_id,
        )

    if task == "task3_stepwise_cot_safe_generation" and task3_stepwise_cot_nontoxic_safe_generation_eval is not None:
        return task3_stepwise_cot_nontoxic_safe_generation_eval(
            gold_answer=gold_answer,
            llm_answer=llm_answer,
            row_id=row_id,
        )

    if task == "subtask1" and subtask1_safe_to_smiles_eval is not None:
        (
            exact_match,
            bleu,
            levenshtein,
            rdk_fts,
            maccs_fts,
            morgan_fts,
            validity,
        ) = subtask1_safe_to_smiles_eval(gold_answer, llm_answer)
        return {
            "exact_match": exact_match,
            "bleu": bleu,
            "levenshtein": levenshtein,
            "rdk_fts": rdk_fts,
            "maccs_fts": maccs_fts,
            "morgan_fts": morgan_fts,
            "validity": validity,
        }

    if task == "subtask2" and subtask2_smiles_to_safe_eval is not None:
        (
            EM,
            BLEU1,
            validity,
            lev_dist,
            lev_norm,
            molecule_EM,
            molecule_morganFT,
            molecule_validity,
        ) = subtask2_smiles_to_safe_eval(gold_answer, llm_answer, row_id=row_id)
        return {
            "EM": EM,
            "BLEU1": BLEU1,
            "validity": validity,
            "levenshtein_dist": lev_dist,
            "levenshtein_norm": lev_norm,
            "molecule_EM": molecule_EM,
            "molecule_morganFT": molecule_morganFT,
            "molecule_validity": molecule_validity,
        }

    return {}


def run_eval(
    data_path: str | Path,
    models: List[str],
    num_samples: int,
    out_dir: str | Path,
    sleep_s: float,
    variant: str = "base",
    task: str = "task1",
    step: str = "single_step",
    run_idx: Optional[int] = None,
    split: str = "test",
    repres: str = "both_repre",
    batch_size: int = 10,
    reset: bool = False,
    with_image: bool = False,
    image_cache_dir: Optional[str] = None,
    image_mime_type: str = "image/svg+xml",
    image_size: int = 300,
    image_highlight_mode: str = "none",
):
    os.makedirs(out_dir, exist_ok=True)
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    # 항상 하이라이트 없는 상태로만 이미지 생성/주입합니다.
    # (CLI로 lasso를 켜더라도 무시)
    image_highlight_mode_opt: Optional[str] = None
    image_cache_dir_path: Optional[Path] = None
    if with_image:
        image_cache_dir_path = Path(image_cache_dir) if image_cache_dir else (out_dir / "images_cache")
    step_norm = _normalize_step(step)
    print(f"Task: {task} | Variant: {variant} | Step: {step_norm} | Split: {split} | Repre: {repres} | Data: {data_path} | Samples: {num_samples or 'all'} | batch_size: {batch_size}")

    rows = read_jsonl(str(data_path))
    if num_samples and num_samples > 0:
        rows = rows[:num_samples]

    needs_gemini = any(_is_gemini_model(m) for m in models)
    needs_claude = any(_is_claude_model(m) for m in models)
    needs_openai = any((not _is_gemini_model(m)) and (not _is_claude_model(m)) for m in models)

    openai_client: Optional[OpenAI] = None
    if needs_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI 모델을 사용 중인데 OPENAI_API_KEY가 없습니다. .env에 OPENAI_API_KEY=...를 넣어주세요.")
        openai_client = OpenAI(api_key=api_key)

    gemini_client: Optional[Any] = None
    if needs_gemini:
        if not _GENAI_AVAILABLE or google_genai is None:
            raise RuntimeError(
                "Gemini 모델을 사용하려면 google-genai 패키지가 필요합니다: pip install google-genai"
            )
        g_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if g_key:
            gemini_client = google_genai.Client(api_key=g_key)
        else:
            # GOOGLE_GENAI_API_KEY 등 SDK가 읽는 기본 환경변수에 의존
            gemini_client = google_genai.Client()

    claude_client: Optional[Any] = None
    if needs_claude:
        if not _ANTHROPIC_AVAILABLE or anthropic is None:
            raise RuntimeError(
                "Claude 모델을 사용하려면 anthropic 패키지가 필요합니다: pip install anthropic"
            )
        a_key = os.environ.get("ANTHROPIC_API_KEY")
        if not a_key:
            raise RuntimeError("Claude 모델을 사용하려면 ANTHROPIC_API_KEY가 필요합니다.")
        claude_client = anthropic.Anthropic(api_key=a_key)

    system_instruction = _system_instruction_for_task(task)

    name_parts = []
    if variant != "base":
        name_parts.append(variant)
    if run_idx is not None:
        name_parts.append(f"run{run_idx}")

    model_suffix = "_".join(name_parts)
    out_name_template = (
        f"predictions_{{model}}.jsonl"
        if not model_suffix
        else f"predictions_{{model}}_{model_suffix}.jsonl"
    )

    for model in models:
        safe_model = model.replace("/", "_")

        # build_safe_qa와 동일한 디렉터리 구조: out_dir / split / task / [repre] / [step]
        task_out_dir = out_dir / split / task
        if task not in ("subtask1", "subtask2"):
            task_out_dir = task_out_dir / repres / step_norm
        task_out_dir.mkdir(parents=True, exist_ok=True)
        results_dir = task_out_dir / "results"
        evaluation_dir = task_out_dir / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        # 샘플별 결과: results/predictions_<model>.jsonl (절대경로로 고정해 cwd/상대경로 혼선 방지)
        task_out_path = (results_dir / out_name_template.format(model=safe_model)).resolve()

        # 이어하기: 이미 저장된 id는 건너뛰기 (reset이면 항상 처음부터 재수행)
        if reset:
            done_ids = set()
            rows_to_do = rows
        else:
            done_ids = _load_done_ids(task_out_path)
            rows_to_do = [r for r in rows if r.get("id") not in done_ids]
        if done_ids:
            print(f"  이어하기: {len(done_ids)}개 이미 완료, {len(rows_to_do)}개 남음")

        mode = "w" if reset else ("a" if task_out_path.exists() and done_ids else "w")
        if reset and task_out_path.exists():
            print(f"  reset: 기존 {task_out_path.name} 덮어쓰기")
        with open(task_out_path, mode, encoding="utf-8") as wf:
            for batch_start in tqdm(range(0, len(rows_to_do), batch_size), desc=f"[{model}] {variant}", total=(len(rows_to_do) + batch_size - 1) // max(batch_size, 1)):
                batch = rows_to_do[batch_start : batch_start + batch_size]
                # 완료되는 대로 순서 유지하며 즉시 저장
                results_by_idx: List[Optional[Tuple[dict, Optional[str], str]]] = [None] * len(batch)
                next_to_write = 0
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    future_to_idx = {
                        executor.submit(
                            _call_model_for_row,
                            openai_client,
                            gemini_client,
                            claude_client,
                            model,
                            row,
                            system_instruction,
                            3,
                            sleep_s,
                            JSON_SCHEMA_STEPWISE_COT
                            if task in ("task3_stepwise_cot", "task3_stepwise_cot_safe_generation")
                            else None,
                            with_image=with_image,
                            image_cache_dir=image_cache_dir_path,
                            image_mime_type=image_mime_type,
                            image_size=image_size,
                            image_highlight_mode=image_highlight_mode_opt,
                        ): i
                        for i, row in enumerate(batch)
                    }
                    for future in as_completed(future_to_idx):
                        i = future_to_idx[future]
                        row, pred, raw = future.result()
                        results_by_idx[i] = (row, pred, raw)
                        while next_to_write < len(batch) and results_by_idx[next_to_write] is not None:
                            row, pred, raw = results_by_idx[next_to_write]
                            gold = extract_gold(row)
                            pred_norm = normalize_answer(pred)
                            gold_norm = normalize_answer(gold)
                            is_correct = int(pred_norm == gold_norm)
                            gold_answer = row.get("answer", gold)
                            llm_answer = pred if isinstance(pred, dict) else {"answer": pred or ""}
                            row_id = row.get("source_index", row.get("id", None))
                            metrics = _get_metrics_for_task(task, gold_answer, llm_answer, row_id=row_id)
                            out_row = {
                                "model": model,
                                "id": row.get("id", None),
                                "dataset_name": row.get("dataset_name", ""),
                                "endpoint": row.get("endpoint", ""),
                                "source_index": row.get("source_index", None),
                                "gold": gold,
                                "pred": pred,
                                "correct": is_correct,
                                "raw": raw,
                            }
                            out_row.update(metrics)
                            wf.write(json.dumps(out_row, ensure_ascii=False, default=str) + "\n")
                            wf.flush()
                            next_to_write += 1
                if sleep_s > 0:
                    time.sleep(sleep_s)

        if rows_to_do and not task_out_path.is_file():
            print(
                f"[WARN] inference는 실행됐으나 예측 파일이 없음: {task_out_path}",
                file=sys.stderr,
            )

        # 전체 파일 기준으로 요약 재계산 (이어하기 포함)
        # 부분 쓰기/이전 손상 줄로 인한 JSONDecodeError 방지: 깨진 줄은 스킵
        all_lines = read_jsonl(
            task_out_path,
            skip_bad_lines=True,
            allow_missing=True,
        )
        if rows_to_do and not all_lines and task_out_path.is_file():
            print(
                f"[WARN] 예측 파일은 있으나 유효한 JSON 줄이 없음: {task_out_path}",
                file=sys.stderr,
            )
        correct = sum(int(line.get("correct", 0)) for line in all_lines)
        total = len(all_lines)
        metric_sums: Dict[str, float] = {}
        task_keys = TASK_METRIC_KEYS.get(task, [])
        for line in all_lines:
            for k in task_keys:
                v = line.get(k)
                if isinstance(v, (int, float)):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
        acc = correct / max(total, 1)
        metric_means = {}
        for k in task_keys:
            if k in metric_sums:
                metric_means[k] = metric_sums[k] / max(total, 1)
            else:
                metric_means[k] = None

        # 총 evaluation 결과: evaluation/evaluation_summary_<model>.json
        summary = {
            "task": task,
            "variant": variant,
            "step": step_norm,
            "split": split,
            "repre": repres,
            "run": run_idx,
            "model": model,
            "total": total,
            "correct": correct,
            "accuracy": acc,
            "metrics_mean": metric_means,
        }

        summary_name_parts = [f"evaluation_summary_{safe_model}"]
        if model_suffix:
            summary_name_parts.append(model_suffix)
        summary_name = "_".join(summary_name_parts) + ".json"
        summary_path = evaluation_dir / summary_name
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(summary, sf, ensure_ascii=False, indent=2)

        print(f"\n=== {model} ===")
        print(f"total={total}, correct={correct}, acc={acc:.4f}")
        print(f"  results (per-sample) -> {task_out_path}")
        print(f"  evaluation (summary) -> {summary_path}\n")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run GPT inference & evaluation on SAFE QA. "
            "Arguments aligned with build_safe_qa.py (--split, --task, --variant, --molecule_repr, --step) so the same QA dataset is used."
        ),
    )
    ap.add_argument(
        "--env",
        type=str,
        default=str(DEFAULT_ENV_PATH),
        help="Path to .env (OPENAI_API_KEY, GOOGLE_API_KEY 또는 GEMINI_API_KEY)",
    )
    ap.add_argument("--data", type=str, default=None, help="Path to QA jsonl (overrides --split/--task/--variant/--molecule_repr/--step)")
    ap.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Split: train or test. QA/<split>/... 와 동일. 기본: test (build_safe_qa와 동일)",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=[
            "task1",
            "task2",
            "task3",
            "task3_nontoxic_safe_generation",
            "task3_instruction",
            "task3_stepwise_cot",
            "task3_stepwise_cot_safe_generation",
            "subtask1",
            "subtask2",
            "all",
        ],
        default="task1",
        help=(
            "task1, task2, task3, task3_nontoxic_safe_generation, task3_instruction, task3_stepwise_cot, "
            "task3_stepwise_cot_safe_generation, "
            "subtask1, subtask2, all. "
            "all은 위 메인 태스크만 순회(subtask1/2 제외). subtask는 --task subtask1 또는 subtask2로 실행. 기본: task1"
        ),
    )
    ap.add_argument(
        "--variant",
        type=str,
        choices=["base", "icl1", "icl2", "icl4", "all"],
        default="base",
        help="QA variant: base, icl1, icl2, icl4, all (build_safe_qa와 동일). 기본: base",
    )
    ap.add_argument(
        "--molecule_repr",
        type=str,
        dest="repre",
        choices=REPRE_CHOICES_WITH_ALL,
        default="both_repre",
        help="Molecule representation: only_safe, only_smiles, both_repre, all. all이면 모든 representation을 자동으로 inference하고 각 경로에 저장. 기본: both_repre",
    )
    ap.add_argument(
        "--step",
        type=str,
        choices=["single", "multi", "all", "single_step", "multi_step"],
        default="single",
        help="Step: single, multi, all (task1/2/3만). 기본: single",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="단일 모델명 (e.g. gpt-4o, gpt-5.2, gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-3.1-flash-lite-preview). --with_image 사용 시 gpt-4o / gpt-5 계열 / gemini 계열은 이미지까지 함께 전송합니다. 지정 시 --models 무시.",
    )
    ap.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="쉼표 구분 모델명. 기본: gpt-4o-mini,gpt-4o",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="상위 N개 샘플만 inference (0=전체). 기본: 0",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./safe_qa_outputs",
        help="출력 루트. 구조: out_dir/<split>/<task>/[<molecule_repr>/]<step>/results/ 및 evaluation/",
    )
    ap.add_argument("--sleep_s", type=float, default=0.2, help="API 호출 간 sleep(초)")
    ap.add_argument(
        "--with_image",
        action="store_true",
        help="질문에 포함된 toxic molecule의 2D 이미지를 add_image.py로 생성해서 멀티모달 모델 요청에 포함합니다. 현재 gpt-4o / gpt-5 계열 / gemini 계열에서 함께 전송되도록 설정됩니다.",
    )
    ap.add_argument(
        "--image_cache_dir",
        type=str,
        default=None,
        help="--with_image 캐시 저장 경로. 기본은 out_dir/images_cache 입니다.",
    )
    ap.add_argument(
        "--image_mime_type",
        type=str,
        default="image/svg+xml",
        choices=["image/svg+xml"],
        help="현재는 SVG 기반만 지원(추론 payload에 넣기 위해).",
    )
    ap.add_argument(
        "--image_size",
        type=int,
        default=300,
        help="2D 이미지 렌더링 크기(가로=세로, px).",
    )
    ap.add_argument(
        "--image_highlight_mode",
        type=str,
        default="none",
        choices=["none", "lasso"],
        help="이미지에 SAFE fragment 하이라이트 포함 여부. 기본은 none(순수 molecule 2D).",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="한 번에 병렬로 호출할 샘플 수 (배치 크기). 기본: 10",
    )
    ap.add_argument(
        "--run",
        type=int,
        default=None,
        help="실험 run 인덱스 (파일명에 run<N> 추가)",
    )
    ap.add_argument(
        "--reset",
        action="store_true",
        help="이미 결과가 있어도 처음부터 다시 수행합니다. predictions/evaluation 출력이 덮어써집니다.",
    )
    args = ap.parse_args()

    load_dotenv(args.env, override=True)

    # --with_image를 켠 경우, 텍스트-only 결과와 디렉토리를 분리합니다.
    # 기본값 ./safe_qa_outputs -> ./safe_qa_outputs_image
    effective_out_dir = Path(args.out_dir)
    if args.with_image and not str(effective_out_dir).endswith("_image"):
        effective_out_dir = effective_out_dir.with_name(effective_out_dir.name + "_image")

    if args.model:
        models = [args.model.strip()]
    else:
        # --models가 nargs='+'로 들어오므로 (공백으로도 여러 토큰 입력 가능),
        # 토큰들을 다시 ','로 합친 뒤 ',' 분리 처리.
        if not args.models:
            models_spec = "gpt-4o-mini,gpt-4o"
        else:
            models_spec = ",".join(list(args.models))
        models = [m.strip() for m in models_spec.split(",") if m.strip()]

    step_choices = ["single_step", "multi_step"]
    if args.step in ("single", "single_step"):
        steps = ["single_step"]
    elif args.step in ("multi", "multi_step"):
        steps = ["multi_step"]
    else:
        steps = step_choices

    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        p = str(data_path)
        if "subtask1" in p:
            task = "subtask1"
        elif "subtask2" in p:
            task = "subtask2"
        elif "task1" in p and "task3" not in p:
            task = "task1"
        elif "task2" in p:
            task = "task2"
        elif "task3_nontoxic_safe_generation" in p:
            task = "task3_nontoxic_safe_generation"
        elif "task3_instruction" in p or "task3_Instruction" in p:
            task = "task3_instruction"
        elif "task3_stepwise_cot_nontoxic_safe_generation" in p:
            task = "task3_stepwise_cot_safe_generation"
        elif "task3_stepwise_cot" in p:
            task = "task3_stepwise_cot"
        elif "task3" in p:
            task = "task3"
        else:
            task = "task1"

        step_for_eval = "single_step"
        if task not in ("subtask1", "subtask2") and "multi_step" in p:
            step_for_eval = "multi_step"
        elif task not in ("subtask1", "subtask2") and args.step in ("multi", "multi_step"):
            step_for_eval = "multi_step"

        run_eval(
            data_path=data_path,
            models=models,
            num_samples=args.num_samples,
            out_dir=effective_out_dir,
            sleep_s=args.sleep_s,
            variant=args.variant if args.variant != "all" else "base",
            task=task,
            step=step_for_eval,
            run_idx=args.run,
            split=args.split,
            repres=args.repre,
            batch_size=args.batch_size,
            reset=args.reset,
            with_image=args.with_image,
            image_cache_dir=args.image_cache_dir,
            image_mime_type=args.image_mime_type,
            image_size=args.image_size,
            image_highlight_mode=args.image_highlight_mode,
        )
        return

    # --task all: 메인 QA 태스크만 (subtask1/2는 별도 --task로만 실행)
    _MAIN_TASKS_FOR_ALL = [
        "task1",
        "task2",
        "task3",
        "task3_nontoxic_safe_generation",
        "task3_instruction",
        "task3_stepwise_cot",
        "task3_stepwise_cot_safe_generation",
    ]
    tasks = _MAIN_TASKS_FOR_ALL if args.task == "all" else [args.task]
    variants = ["base", "icl1", "icl2", "icl4"] if args.variant == "all" else [args.variant]
    split = args.split
    repres_list = REPRE_CHOICES if args.repre == "all" else [args.repre]

    runs: List[Tuple[Path, str, str, str, str]] = []  # (data_path, task, variant, step, repres)
    for repres in repres_list:
        for task in tasks:
            # subtask1/2는 data_path 및 출력 디렉터리 구조에 repres가 없음 → 중복 실행 방지
            if task in ("subtask1", "subtask2"):
                if repres != repres_list[0]:
                    continue
                path = _data_path_for(task, variants[0], "single_step", split=split, repres=repres)
                if path.exists():
                    runs.append((path, task, variants[0], "", repres))
                else:
                    print(f"Skip (not found): {path}")
                continue

            for variant in variants:
                for step in steps:
                    path = _data_path_for(task, variant, step, split=split, repres=repres)
                    if path.exists():
                        runs.append((path, task, variant, step, repres))
                    else:
                        print(f"Skip (not found): {path}")

    if not runs:
        raise FileNotFoundError("No QA data files found for the given --split/--task/--variant/--repre/--step.")

    for data_path, task, variant, step, repres in runs:
        run_eval(
            data_path=data_path,
            models=models,
            num_samples=args.num_samples,
            out_dir=effective_out_dir,
            sleep_s=args.sleep_s,
            variant=variant,
            task=task,
            step=step or "single_step",
            run_idx=args.run,
            split=split,
            repres=repres,
            batch_size=args.batch_size,
            reset=args.reset,
            with_image=args.with_image,
            image_cache_dir=args.image_cache_dir,
            image_mime_type=args.image_mime_type,
            image_size=args.image_size,
            image_highlight_mode=args.image_highlight_mode,
        )


if __name__ == "__main__":
    main()