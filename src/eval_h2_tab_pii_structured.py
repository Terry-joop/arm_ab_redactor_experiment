#!/usr/bin/env python3
"""H2 TAB PII evaluation for the Arm-A structured-output Qwen variant.

The TAB ECHR train split is an immutable *hold-out* harness here.  It is never
used to create Qwen labels or to train ELECTRA.  All candidates share the
existing deterministic regex layer; this mirrors Redactformer's
``exam_pii_l2.py`` protocol while making Qwen return an index list aligned to
+the supplied ``words`` list, rather than free-text phrases.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REDACTFORMER = Path("/home/jovyan/Redactformer")
sys.path.insert(0, str(REDACTFORMER / "scripts" / "dataset_builders"))
import _pii_rules as pii  # noqa: E402

TAB = Path("/tmp/tab_train.json")
QWEN_DIR = EXPERIMENT / "models" / "Qwen3-32B"
OUT = EXPERIMENT / "results" / "h2_tab_pii_arm_a_structured.json"
WORD_RE = re.compile(r"\S+")
MASKABLE = {"DIRECT", "QUASI"}
CHUNK_WORDS = 120

PII_CATEGORIES = (
    "person", "organization", "location", "facility", "nationality group",
    "date", "address", "phone number", "email", "url", "identifier",
    "case number", "money", "percentage",
)

SYSTEM_PROMPT = """You are the second layer of a privacy redactor for English legal text.
The input has an indexed_words list. Each item is exactly "index: word", with indices
starting at zero. Return exactly one JSON object and no other text:
{\"indices\":[2,5,...]}
indices must contain only the zero-based indices of words that should be redacted; use []
when no word should be redacted. Select only words that are part of a personal identifier
or sensitive quasi-identifier: a person's name, organization, location/facility,
nationality/religion/political group, date, address, phone, email, URL,
case/identification number, money, or percentage. Do not select ordinary legal vocabulary,
generic roles without a named identity, or common words. Do not rewrite, quote, summarize,
or reorder words. Do not emit an explanation."""


def chars(spans):
    return {position for start, end in spans for position in range(start, end)}


def gold_sets(doc):
    """TAB's union-of-annotators DIRECT/QUASI gold and all annotated entities."""
    votes, any_entity = {}, set()
    for annotation in (doc.get("annotations") or {}).values():
        for entity in annotation.get("entity_mentions") or []:
            start, end = entity.get("start_offset"), entity.get("end_offset")
            if start is None or end is None:
                continue
            span = set(range(start, end))
            any_entity |= span
            if entity.get("identifier_type") in MASKABLE:
                for position in span:
                    votes[position] = votes.get(position, 0) + 1
    return set(votes), any_entity


class EmptyDoc:
    __slots__ = ("text", "ents")

    def __init__(self, text):
        self.text, self.ents = text, []


def regex_prediction(text):
    return chars(pii.pii_spans(EmptyDoc(text), strict=False))


def score(predictions, golds, all_entities):
    tp = fp = fn = fp_entity = fp_none = 0
    for predicted, gold, entities in zip(predictions, golds, all_entities):
        tp += len(predicted & gold)
        fn += len(gold - predicted)
        excess = predicted - gold
        fp += len(excess)
        fp_entity += len(excess & entities)
        fp_none += len(excess - entities)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "P": round(precision, 4), "R": round(recall, 4),
        "F1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
        "tp_chars": tp, "fp_chars": fp, "fn_chars": fn,
        "fp_entity_pct": round(100 * fp_entity / max(fp, 1), 1),
        "fp_true_pct": round(100 * fp_none / max(fp, 1), 1),
        "gold_chars": tp + fn,
    }


def make_chunks(docs):
    """All arms read the same word-bounded chunks and retain global offsets."""
    out = []
    for doc_index, doc in enumerate(docs):
        text = doc.get("text") or ""
        matches = list(WORD_RE.finditer(text))
        for first in range(0, len(matches), CHUNK_WORDS):
            last = min(first + CHUNK_WORDS, len(matches))
            start = 0 if first == 0 else matches[first].start()
            end = len(text) if last == len(matches) else matches[last].start()
            chunk = text[start:end]
            words = [match.group() for match in WORD_RE.finditer(chunk)]
            offsets = [(match.start(), match.end()) for match in WORD_RE.finditer(chunk)]
            if words:
                out.append((doc_index, start, words, offsets))
    return out


def parse_indices(response, expected_length):
    """Accept only a well-formed, in-range word-index array."""
    start = response.find("{")
    if start < 0:
        raise ValueError("JSON object 없음")
    try:
        value, _ = json.JSONDecoder().raw_decode(response[start:])
    except json.JSONDecodeError as error:
        raise ValueError("JSON decode 실패") from error
    indices = value.get("indices") if isinstance(value, dict) else None
    if not isinstance(indices, list):
        raise ValueError("indices 배열 없음")
    if any(type(index) is not int or not 0 <= index < expected_length for index in indices):
        raise ValueError("indices 범위 또는 타입 오류")
    if len(set(indices)) != len(indices):
        raise ValueError("indices 중복")
    return indices


def qwen_prompt(tokenizer, words):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(
            {"indexed_words": [f"{index}: {word}" for index, word in enumerate(words)]},
            ensure_ascii=False,
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def qwen_predictions(docs, chunks, batch_size):
    # This local import keeps non-Qwen baseline runs light.
    from qwen_lawmask_medical_teacher import disable_incompatible_audio_import
    disable_incompatible_audio_import()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_DIR, dtype=torch.bfloat16, device_map="auto", local_files_only=True
    ).eval()
    predicted, errors = [set() for _ in docs], {}
    for begin in range(0, len(chunks), batch_size):
        batch = chunks[begin:begin + batch_size]
        prompts = [qwen_prompt(tokenizer, words) for _doc, _base, words, _offsets in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=128,
                                       pad_token_id=tokenizer.eos_token_id)
        responses = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
        for (doc, base, words, offsets), response in zip(batch, responses):
            try:
                indices = parse_indices(response, len(words))
            except ValueError as error:
                errors[str(error)] = errors.get(str(error), 0) + 1
                continue  # Fail closed: no unverifiable L2 span.
            for index in indices:
                start, end = offsets[index]
                predicted[doc].update(range(base + start, base + end))
        done = begin + len(batch)
        if done % 100 < batch_size or done == len(chunks):
            print(f"Arm A structured {done}/{len(chunks)} chunks; errors={sum(errors.values())}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predicted, errors


def current_web_md_predictions(docs):
    import spacy
    nlp = spacy.load("en_core_web_md", disable=["parser", "tagger", "attribute_ruler", "lemmatizer"])
    return [chars(pii.pii_spans(nlp(doc.get("text") or ""), strict=False)) for doc in docs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1014, help="TAB documents; prefix only for a smoke run")
    parser.add_argument("--arms", nargs="+", choices=["l1", "current", "a"], default=["l1", "current", "a"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if not TAB.exists():
        raise FileNotFoundError(f"TAB hold-out missing: {TAB}")
    docs = json.loads(TAB.read_text(encoding="utf-8"))[:args.n]
    if args.n == 1014 and len(docs) != 1014:
        raise RuntimeError(f"Expected TAB 1,014 documents, got {len(docs)}")
    gold, all_entities = zip(*(gold_sets(doc) for doc in docs))
    base = [regex_prediction(doc.get("text") or "") for doc in docs]
    chunks = make_chunks(docs)
    result = {
        "harness": "H2 TAB PII structured Arm-A evaluation",
        "scope": "TAB ECHR is hold-out evaluation only; it is not Teacher or Student training data.",
        "n_documents": len(docs), "n_chunks": len(chunks), "chunk_words": CHUNK_WORDS,
        "common_l1": "existing deterministic regex branch of _pii_rules.pii_spans(strict=False)",
        "qwen_model": "Qwen/Qwen3-32B", "qwen_output": "validated zero-based word-index JSON",
        "arms": {},
    }
    if "l1" in args.arms:
        result["arms"]["L1 regex only"] = score(base, gold, all_entities)
    if "current" in args.arms:
        result["arms"]["L1 regex union current web_md"] = score(current_web_md_predictions(docs), gold, all_entities)
    if "a" in args.arms:
        extra, errors = qwen_predictions(docs, chunks, args.batch_size)
        value = score([left | right for left, right in zip(base, extra)], gold, all_entities)
        value["json_validation_errors"] = sum(errors.values())
        value["json_validation_error_breakdown"] = errors
        result["arms"]["L1 regex union Arm A Qwen3-32B structured"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
