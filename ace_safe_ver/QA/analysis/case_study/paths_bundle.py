"""case_study 스크립트용: ace_safe_ver 루트 탐색 및 번들 SAFE 패키지 로드."""
from __future__ import annotations

import sys
from pathlib import Path


def get_ace_safe_ver_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "ace_local.py").is_file():
            return p
        p = p.parent
    raise RuntimeError(
        "ace_local.py 을 찾을 수 없습니다. ace_safe_ver 트리 안에서 스크립트를 실행하세요."
    )


def setup_bundle_paths(*, with_safe_pkg: bool = False) -> Path:
    """ace_safe_ver 루트를 sys.path에 넣고, 필요 시 third_party/safe 등록."""
    root = get_ace_safe_ver_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if with_safe_pkg:
        import ace_local

        ace_local.ensure_safe_pkg_path()
    return root
