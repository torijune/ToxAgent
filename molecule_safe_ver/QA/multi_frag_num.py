import json
from pathlib import Path
from collections import Counter

TASK_FILES = {
    "Task1": "/Users/jang-wonjun/Desktop/DMISLab/ToxAgent/molecule_safe_ver/QA/task1_safe_to_nontoxic/multi_step/task1_safe_qa.jsonl",
    "Task3": "/Users/jang-wonjun/Desktop/DMISLab/ToxAgent/molecule_safe_ver/QA/task3_toxic_fragment_identification/multi_step/task3_safe_qa.jsonl",
}

def extract_answer(item):
    ans = item.get("answer", "")
    if isinstance(ans, dict):
        return str(ans.get("answer", "")).strip()
    return str(ans).strip()

def count_fragments(answer_str):
    if not answer_str:
        return 0
    return len([frag for frag in answer_str.replace(" ", "").split(".") if frag])

def summarize_jsonl(task_name, file_path):
    file_path = Path(file_path)
    fragment_counts = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            answer_str = extract_answer(item)
            fragment_counts.append(count_fragments(answer_str))

    print(f"\n===== {task_name} =====")

    if not fragment_counts:
        print("데이터가 없습니다.")
        return

    dist = Counter(fragment_counts)
    avg_count = sum(fragment_counts) / len(fragment_counts)

    print(f"총 샘플 수: {len(fragment_counts)}")
    print(f"평균 fragment 개수: {avg_count:.2f}")
    print(f"최소 fragment 개수: {min(fragment_counts)}")
    print(f"최대 fragment 개수: {max(fragment_counts)}")
    print("fragment 개수 분포:")
    for k in sorted(dist):
        print(f"  {k}개: {dist[k]}개 샘플")

for task_name, file_path in TASK_FILES.items():
    summarize_jsonl(task_name, file_path)