"""ace_safe_ver 루트 탐색 (ace_local.py 위치 기준)."""
from __future__ import annotations

from pathlib import Path


def resolve_ace_safe_ver_root(caller_file: str) -> Path:
    """임의의 스크립트 경로에서 상위로 올라가며 ace_local.py가 있는 폴더를 반환."""
    p = Path(caller_file).resolve().parent
    while p != p.parent:
        if (p / "ace_local.py").is_file():
            return p
        p = p.parent
    raise RuntimeError(
        "ace_safe_ver 루트를 찾을 수 없습니다(ace_local.py 없음). "
        "ace_safe_ver 폴더 구조를 유지한 채 실행하세요."
    )
