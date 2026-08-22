"""6개 의료 데이터셋에서 정책-독립 Qwen 검수 파일럿 100건을 만든다."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).resolve().parents[1] / "data/teacher/medical_train_input.jsonl"))
    parser.add_argument("--output", default="data/medical_pilot_100.jsonl")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows(Path(args.input)):
        # Prompt에 들어갈 수 있는 필드만 보존한다. 기존 규칙 labels/task label은 의도적으로 버린다.
        groups[row["dataset_name"]].append({key: row[key] for key in ("id", "text", "words", "dataset_name", "source", "source_split")})
    names = sorted(groups)
    if not names:
        raise SystemExit("입력 행이 없습니다")
    rng = random.Random(args.seed)
    base, extra = divmod(args.n, len(names))
    sampled = []
    for index, name in enumerate(names):
        take = base + (index < extra)
        if len(groups[name]) < take:
            raise SystemExit(f"{name}: {take}개를 뽑기에는 행이 부족합니다")
        sampled.extend(rng.sample(groups[name], take))
    rng.shuffle(sampled)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "rows": len(sampled), "by_dataset": {k: sum(r["dataset_name"] == k for r in sampled) for k in names}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
