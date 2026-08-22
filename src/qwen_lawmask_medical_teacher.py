"""Arm A/B 공용 Qwen3-32B 의료 teacher: 원문 span을 8 법 범주로 라벨한다.

teacher 출력에는 원문을 복사한 phrase만 허용한다. 모든 phrase는 원래 words로 exact-align
되며, 맞지 않는 행은 학습/평가에서 제외하고 rejected JSONL에 기록한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


POLICY_VERSION = "lawmask-medical-8cat-v1"
MODEL_ID = "Qwen/Qwen3-32B"
CATEGORIES = (
    "disease", "symptom", "medication", "medical treatment",
    "mental health condition", "substance use", "injury", "reproductive health",
)
SYSTEM_PROMPT = """You are the medical second layer of a privacy-redaction system.
Use only these eight categories: disease, symptom, medication, medical treatment,
mental health condition, substance use, injury, reproductive health.

For each phrase in the supplied original WORDS that reveals one of those categories,
return its category and the shortest meaningful contiguous phrase. Do not redact a
generic opinion, ordinary verb, pronoun, punctuation, generic healthcare context,
or a number/date/name merely because it is present. Do not use any external rule
labels: decide from the sentence itself.

Reply exactly one line:
REDACT: category :: exact phrase || category :: exact phrase
Each exact phrase must be copied from a contiguous part of WORDS. If there is no
medical sensitive span, reply exactly: REDACT: NONE."""
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()


def disable_incompatible_audio_import() -> None:
    import transformers.utils
    transformers.utils.is_torchaudio_available = lambda: False


def normalize(token: str) -> str:
    return token.casefold()


# `medical_train_input.jsonl`의 words를 만든 tokenizer와 정확히 같은 규약이다.
# 하이픈 결합어(beta-blocker, PPI-dependent)를 `beta - blocker`로 쪼개면 Qwen이
# 원문을 정확히 복사했어도 정렬 실패가 되므로, 여기서 별도 간이 토크나이저를 쓰면 안 된다.
WORD_RE = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)


def phrase_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def occurrences(words: list[str], phrase: list[str]) -> list[int]:
    target = [normalize(value) for value in phrase]
    source = [normalize(value) for value in words]
    return [start for start in range(len(source) - len(target) + 1) if source[start:start + len(target)] == target]


def parse_and_align(response: str, words: list[str]) -> tuple[list[int], list[str], list[dict]]:
    match = re.search(r"(?mi)^REDACT:\s*(.*)$", response)
    if not match:
        raise ValueError("REDACT line 없음")
    body = match.group(1).strip()
    if body.upper() in {"", "NONE", "NO", "N/A"}:
        return [0] * len(words), ["O"] * len(words), []
    labels, types, accepted = [0] * len(words), ["O"] * len(words), []
    for part in body.split("||"):
        if "::" not in part:
            raise ValueError(f"category :: phrase 형식이 아님: {part!r}")
        category, phrase = (piece.strip() for piece in part.split("::", 1))
        category = category.casefold()
        if category not in CATEGORIES:
            raise ValueError(f"허용하지 않은 범주: {category!r}")
        tokens = phrase_tokens(phrase.strip(" '`\""))
        starts = occurrences(words, tokens)
        if not tokens or not starts:
            raise ValueError(f"원문 words에 exact-align 불가: {phrase!r}")
        for start in starts:
            for index in range(start, start + len(tokens)):
                labels[index] = 1
                types[index] = category
        accepted.append({"category": category, "phrase": " ".join(tokens), "occurrences": len(starts)})
    return labels, types, accepted


def make_prompt(tokenizer, row: dict) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"id": row["id"], "words": row["words"]}, ensure_ascii=False)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "models/Qwen3-32B"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source, output = load_jsonl(Path(args.input)), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output.exists():
        output.unlink()
    done = {row["id"] for row in load_jsonl(output)} if output.exists() else set()
    pending = [row for row in source if row["id"] not in done]
    print(f"policy={POLICY_VERSION} input={len(source)} done={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        return
    disable_incompatible_audio_import()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
    rejected = output.with_name(output.stem + "_rejected.jsonl")
    accepted = failed = 0
    for offset in range(0, len(pending), args.batch_size):
        batch_rows = pending[offset:offset + args.batch_size]
        prompts = [make_prompt(tokenizer, row) for row in batch_rows]
        batch = tokenizer(prompts, return_tensors="pt", padding=True)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**batch, do_sample=False, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
        answers = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        for row, answer in zip(batch_rows, answers):
            try:
                labels, types, spans = parse_and_align(answer, row["words"])
            except ValueError as error:
                failed += 1
                append(rejected, {"id": row["id"], "error": str(error), "response": answer, "teacher_model": MODEL_ID, "policy_version": POLICY_VERSION})
                continue
            out = dict(row)
            out.update({"labels": labels, "types": types, "teacher_spans": spans, "raw_response": answer,
                        "annotation_source": "qwen3-32b-lawmask-medical-8cat-v1", "teacher_model": MODEL_ID,
                        "policy_version": POLICY_VERSION, "prompt_sha256": PROMPT_SHA256})
            append(output, out)
            accepted += 1
        print(f"progress {min(offset + len(batch_rows), len(pending))}/{len(pending)} accepted={accepted} rejected={failed}", flush=True)
    print(f"DONE accepted={accepted} rejected={failed} output={output}", flush=True)


if __name__ == "__main__":
    main()
