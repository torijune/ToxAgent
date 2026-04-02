"""
Question 텍스트에 포함된 toxic molecule의 SAFE 표현을 추출한 뒤,
`safe/safe/viz.py`의 2D 렌더링 유틸(`to_image`)을 이용해 분자 이미지를 생성합니다.

주요 사용처:
- QA jsonl의 `question` 필드에서 "Full molecule representation (toxic): ... SAFE = '...'"
  형태를 파싱하여 SAFE 2D 이미지(약식)를 얻습니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import io
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

# 번들 third_party/safe: `safe.safe.viz` import
_QA_SRC = Path(__file__).resolve().parent
_ACE = _QA_SRC.parent.parent.parent
if str(_ACE) not in sys.path:
    sys.path.insert(0, str(_ACE))
import ace_local  # noqa: E402

ace_local.ensure_safe_pkg_path()

try:
    import datamol as dm
except Exception:  # pragma: no cover
    dm = None



DEFAULT_MOL_SIZE: Tuple[int, int] = (300, 300)

_TOXIC_BLOCK_RE = re.compile(
    r"Full molecule representation \(toxic\):(?P<block>.*?)(?:Full molecule representation \(|$)",
    flags=re.DOTALL,
)
_SAFE_EQ_RE = re.compile(r"SAFE\s*=\s*'([^']*)'", flags=re.DOTALL)
_SMILES_EQ_RE = re.compile(r"SMILES\s*=\s*'([^']*)'", flags=re.DOTALL)

RenderedImage = Union[str, bytes, "Image.Image"]


def _extract_toxic_block(question: str) -> str:
    """question에서 toxic molecule 관련 블록만 우선적으로 잘라냅니다."""
    q = question or ""
    m = _TOXIC_BLOCK_RE.search(q)
    if m:
        block = (m.group("block") or "").strip()
        if block:
            return block
    return q


def _normalize_render_output(img: object, *, use_svg: bool) -> RenderedImage:
    """
    렌더링 결과 타입을 모델 전송용으로 정규화합니다.
    - use_svg=True  -> 반드시 SVG 문자열 또는 UTF-8 bytes
    - use_svg=False -> 반드시 PNG bytes
    """
    if use_svg:
        if isinstance(img, str):
            return img
        if isinstance(img, bytes):
            try:
                decoded = img.decode("utf-8")
            except UnicodeDecodeError as e:
                raise TypeError(
                    "Expected SVG bytes for use_svg=True, but got non-UTF8 bytes."
                ) from e
            if "<svg" not in decoded.lower():
                raise TypeError(
                    "Expected SVG content for use_svg=True, but bytes do not look like SVG."
                )
            return decoded
        if Image is not None and isinstance(img, Image.Image):
            raise TypeError(
                "Renderer returned PIL Image although use_svg=True. "
                "Expected SVG string/bytes for Gemini-compatible input."
            )
        raise TypeError(f"Unexpected render type for SVG mode: {type(img)}")

    # PNG mode
    if isinstance(img, bytes):
        if img.startswith(b"\x89PNG\r\n\x1a\n"):
            return img
        raise TypeError(
            "Expected PNG bytes for use_svg=False, but bytes do not look like PNG."
        )
    if Image is not None and isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"Unexpected render type for PNG mode: {type(img)}")


def extract_toxic_safe_from_question(question: str) -> Optional[str]:
    """
    QA question에서 toxic molecule의 SAFE 값을 추출합니다.

    기대 포맷(qa_template.py):
      Full molecule representation (toxic): ... SAFE = '...'
    """
    q = _extract_toxic_block(question)
    m = _SAFE_EQ_RE.search(q)
    if not m:
        return None
    s = (m.group(1) or "").strip()
    return s or None


def extract_toxic_smiles_from_question(question: str) -> Optional[str]:
    """SAFE 파싱이 실패한 경우를 대비해 toxic molecule의 SMILES를 추출합니다."""
    q = _extract_toxic_block(question)
    m = _SMILES_EQ_RE.search(q)
    if not m:
        return None
    s = (m.group(1) or "").strip()
    return s or None


def render_toxic_molecule_image_from_safe(
    toxic_safe: str,
    *,
    mol_size: Tuple[int, int] = DEFAULT_MOL_SIZE,
    highlight_mode: Optional[str] = None,
    use_svg: bool = True,
) -> RenderedImage:
    """
    `toxic_safe`(SAFE 문자열) -> 2D 이미지(PIL)로 렌더링합니다.

    - highlight_mode: `safe/safe/viz.py`의 to_image과 동일한 옵션 사용
    - use_svg=False: PNG처럼 PIL 이미지를 얻기 위함
    """
    toxic_safe = (toxic_safe or "").strip()
    if not toxic_safe:
        raise ValueError("toxic_safe is empty; cannot render molecule image.")

    try:
        from safe.safe.viz import to_image as safe_to_image
    except Exception as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Failed to import `safe.safe.viz.to_image`. "
            "This likely means `datamol/rdkit` dependencies are missing in the current environment."
        ) from e

    img = safe_to_image(
        toxic_safe,
        fragments=None,
        legend=None,
        mol_size=mol_size,
        use_svg=use_svg,
        highlight_mode=highlight_mode,
    )

    return _normalize_render_output(img, use_svg=use_svg)


def render_toxic_molecule_image_from_question(
    question: str,
    *,
    mol_size: Tuple[int, int] = DEFAULT_MOL_SIZE,
    highlight_mode: Optional[str] = None,
    use_svg: bool = True,
) -> RenderedImage:
    """
    Question 텍스트에서 toxic molecule의 SAFE를 추출해 2D 이미지를 생성합니다.
    SAFE 추출이 안 되면 SMILES로 fallback 렌더링을 시도합니다.
    """
    toxic_safe = extract_toxic_safe_from_question(question)
    if toxic_safe:
        return render_toxic_molecule_image_from_safe(
            toxic_safe,
            mol_size=mol_size,
            highlight_mode=highlight_mode,
            use_svg=use_svg,
        )

    toxic_smiles = extract_toxic_smiles_from_question(question)
    if toxic_smiles and dm is not None:
        mol = dm.to_mol(toxic_smiles, remove_hs=False)
        if mol is None:
            raise ValueError("Failed to parse toxic_smiles into RDKit molecule.")
        img = dm.viz.to_image(mol, mol_size=mol_size, use_svg=use_svg)
        return _normalize_render_output(img, use_svg=use_svg)

    raise ValueError("Could not extract toxic SAFE/SMILES from question.")


def save_toxic_molecule_image_from_question(
    question: str,
    out_path: str | Path,
    *,
    mol_size: Tuple[int, int] = DEFAULT_MOL_SIZE,
    highlight_mode: Optional[str] = None,
    use_svg: bool = True,
) -> Path:
    """question -> normalized SVG text or PNG bytes -> out_path 저장."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = render_toxic_molecule_image_from_question(
        question,
        mol_size=mol_size,
        highlight_mode=highlight_mode,
        use_svg=use_svg,
    )

    if use_svg:
        out_path = out_path.with_suffix(".svg")
        if isinstance(img, bytes):
            out_path.write_bytes(img)
            return out_path
        if isinstance(img, str):
            out_path.write_text(img, encoding="utf-8")
            return out_path
        raise TypeError(f"Expected SVG str/bytes, got: {type(img)}")

    out_path = out_path.with_suffix(".png")
    if isinstance(img, bytes):
        out_path.write_bytes(img)
        return out_path
    raise TypeError(f"Expected PNG bytes, got: {type(img)}")


def _load_jsonl_line(path: Path, target_index: int) -> dict:
    """jsonl에서 target_index번째 라인(0-based) 레코드 1개를 로드합니다."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == target_index:
                return json.loads(line)
    raise IndexError(f"target_index={target_index} is out of range for {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SAFE 기반 toxic molecule 2D 이미지 생성 데모")
    ap.add_argument(
        "--input-jsonl",
        type=str,
        default=str(
            _QA_SRC.parent
            / "test"
            / "task3_instruction_nontoxic_smiles_generation"
            / "both_repre"
            / "multi_step"
            / "task3_instruction_nontoxic_smiles_generation_qa.jsonl"
        ),
        help="Demo로 사용할 QA jsonl 파일",
    )
    ap.add_argument("--index", type=int, default=0, help="jsonl에서 가져올 레코드 인덱스(0-based)")
    ap.add_argument("--out-dir", type=str, default=str(_QA_SRC / "demo_outputs"), help="이미지 출력 디렉터리")
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    record = _load_jsonl_line(input_path, args.index)
    question = record.get("question") or ""
    if not question.strip():
        raise ValueError("Record has empty 'question' field.")

    out_path = Path(args.out_dir) / f"toxic_molecule_demo_{args.index}.svg"
    saved_path = save_toxic_molecule_image_from_question(
        question,
        out_path,
        mol_size=DEFAULT_MOL_SIZE,
        highlight_mode=None,
        use_svg=True,
    )
    print(f"Saved demo image to: {saved_path}")


if __name__ == "__main__":
    main()