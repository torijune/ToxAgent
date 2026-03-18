"""
MCQA용 선택지 생성: 정답 1개 + distractors (n_choices - 1)개를 샘플링하여
A/B/C/D 순서로 섞고, 정답 인덱스를 반환.
"""
from __future__ import annotations

import random
from typing import List, Optional


def get_choices(
    correct_answer: str,
    candidate_pool: List[str],
    n_choices: int = 4,
    seed: Optional[int] = None,
) -> tuple[List[str], int]:
    """
    정답 문자열과 후보 풀에서 n_choices개 선택지를 만든다.
    정답은 반드시 포함하고, 나머지는 풀에서 중복/정답 제외 후 샘플링.
    순서는 랜덤으로 섞아 정답 위치를 숨긴다.

    Args:
        correct_answer: 정답 문자열 (공백 strip 후 사용).
        candidate_pool: 다른 샘플들의 정답 문자열 리스트 (같은 task 컬럼).
        n_choices: 선택지 개수 (기본 4 = A/B/C/D).
        seed: 재현용 랜덤 시드.

    Returns:
        (options, correct_index): options는 길이 n_choices 리스트, correct_index는 0-based.
        후보가 부족하면 정답만 반복해서 채울 수 있으나, 가능하면 정답 제외 유니크 풀 사용.
    """
    correct = (correct_answer or "").strip()
    # 정답과 동일한 것 제외, 공백 제거 후 유니크 (순서 유지)
    seen: set[str] = set()
    pool: List[str] = []
    for s in candidate_pool:
        t = (s or "").strip()
        if not t or t == correct or t in seen:
            continue
        seen.add(t)
        pool.append(t)

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # 정답 + distractors (n_choices - 1)개. 부족하면 풀에서 중복 허용해서 뽑기
    need = n_choices - 1
    if len(pool) >= need:
        distractors = rng.sample(pool, need)
    else:
        distractors = list(pool)
        while len(distractors) < need:
            distractors.append(rng.choice(pool) if pool else correct)

    options = [correct] + distractors
    rng.shuffle(options)
    correct_index = options.index(correct)
    return options, correct_index
