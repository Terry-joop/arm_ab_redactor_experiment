"""Qwen 의료 teacher 100건을 사람이 빠르게 검수할 수 있는 정적 HTML로 만든다."""
from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_html(words: list[str], labels: list[int], types: list[str]) -> str:
    parts = []
    for word, label, kind in zip(words, labels, types):
        safe = html.escape(word)
        if label:
            parts.append(f'<mark title="{html.escape(kind)}">{safe}<small>{html.escape(kind)}</small></mark>')
        else:
            parts.append(safe)
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/medical_pilot_100_qwen_labeled.jsonl")
    parser.add_argument("--rejected", default="data/medical_pilot_100_qwen_labeled_rejected.jsonl")
    parser.add_argument("--output", default="artifacts/medical_pilot_100_review.html")
    args = parser.parse_args()
    accepted, rejected = read(Path(args.input)), read(Path(args.rejected))
    counts = Counter(row.get("dataset_name", "unknown") for row in accepted)
    cards = []
    for index, row in enumerate(accepted, start=1):
        spans = "; ".join(f'{item["category"]}: {item["phrase"]}' for item in row.get("teacher_spans", [])) or "(없음)"
        cards.append(f'''<article class="card" data-dataset="{html.escape(row['dataset_name'])}">
<header><b>#{index} · {html.escape(row['dataset_name'])}</b><code>{html.escape(row['id'])}</code></header>
<p class="text">{token_html(row['words'], row['labels'], row['types'])}</p>
<p><b>Qwen span</b>: {html.escape(spans)}</p>
<details><summary>원문 출력</summary><pre>{html.escape(row.get('raw_response', ''))}</pre></details>
<p class="review">검수: <label><input type="radio" name="r{index}"> 맞음</label>
<label><input type="radio" name="r{index}"> 과잉</label>
<label><input type="radio" name="r{index}"> 누락</label>
<label><input type="radio" name="r{index}"> 애매</label></p></article>''')
    selector = '<option value="all">전체</option>' + ''.join(f'<option value="{html.escape(k)}">{html.escape(k)} ({v})</option>' for k, v in sorted(counts.items()))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Qwen 의료 Teacher · 100건 검수</title>
<style>body{{font-family:system-ui,sans-serif;margin:32px;max-width:1100px;background:#f7f9fb;color:#17212b}}h1{{margin-bottom:4px}}.note{{color:#536273}}select{{padding:7px;margin:18px 0}}.card{{background:#fff;border:1px solid #dce4eb;border-radius:10px;padding:16px;margin:12px 0}}header{{display:flex;justify-content:space-between;gap:16px}}code{{color:#607080}}.text{{line-height:2.15;font-size:16px}}mark{{background:#ffe3b0;padding:2px 3px;border-radius:3px;margin:1px;white-space:nowrap}}mark small{{font-size:10px;color:#854d00;margin-left:3px}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:10px}}.review{{border-top:1px solid #edf0f2;padding-top:10px}}label{{margin-right:12px}}</style>
<h1>Qwen3-32B 의료 Teacher · 100건 파일럿 검수</h1>
<p class="note">정책: 의료 법 범주 8종만. 노란 토큰은 Qwen이 가리라고 한 span이다. 이 화면의 라디오 선택은 브라우저 안에서만 바뀌므로, 최종 판정은 별도 JSONL로 옮긴다.</p>
<p><b>정렬 성공</b> {len(accepted)}건 · <b>정렬 거절</b> {len(rejected)}건 · 원문 words만 teacher에 제공, 기존 규칙 라벨은 제공하지 않음.</p>
<label>데이터셋 <select id="dataset">{selector}</select></label>
{''.join(cards) if cards else '<p>아직 라벨 결과가 없습니다.</p>'}
<script>document.querySelector('#dataset').onchange=e=>document.querySelectorAll('.card').forEach(c=>c.hidden=e.target.value!='all'&&c.dataset.dataset!=e.target.value)</script>
</html>'''
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(out), "accepted": len(accepted), "rejected": len(rejected), "by_dataset": counts}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
