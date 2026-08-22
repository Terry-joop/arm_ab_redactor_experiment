#!/usr/bin/env python3
"""Fair H3 MedMentions evaluation: every L2 sees the same <=120-word chunks.

The previous whole-abstract prototype was intentionally superseded: ELECTRA is
limited to 128 subwords while Qwen could read an entire abstract.  This runner
splits each source abstract at word boundaries, remaps all local character spans
to the original document, and scores document-level unions against the same
external UMLS gold.  No raw source text or LLM answer is emitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REDACTFORMER = Path("/home/jovyan/Redactformer")
TOKEN_PROBE = Path("/home/jovyan/token_redaction_probe")
MM_PATH = ROOT / "data/benchmarks_medmentions_pubtator.txt"
MODEL_DIR = ROOT / "artifacts/electra_small_seed42"
QWEN_DIR = ROOT / "models/Qwen3-32B"
OUT = ROOT / "results/h3_medmentions_arm_ab_chunked.json"
sys.path[:0] = [str(REDACTFORMER / "scripts/audit"), str(REDACTFORMER / "scripts/dataset_builders"),
                str(TOKEN_PROBE / "src"), str(ROOT / "src")]

from medmentions_eval import load_pubtator  # noqa: E402
import _lawmask_l1 as l1  # noqa: E402
from common import WORD_RE  # noqa: E402
from train import RedactionModel  # noqa: E402
from qwen_lawmask_medical_teacher import CATEGORIES, disable_incompatible_audio_import, make_prompt, parse_and_align  # noqa: E402

CORE_TUI = {"T047", "T048", "T191", "T046", "T184", "T037", "T019", "T121", "T195", "T200", "T061", "T060", "T059", "T034"}
MED_LABELS = list(CATEGORIES)
CHUNK_WORDS = 120
STUDENT_THRESHOLD = 0.51
GLINER_THRESHOLD = 0.5


def char_set(spans):
    return {position for start, end in spans for position in range(start, end)}


def make_chunks(docs):
    """Return (doc_index, global_char_offset, chunk_text) with full coverage."""
    chunks = []
    for doc_index, (text, _mentions) in enumerate(docs):
        matches = list(WORD_RE.finditer(text))
        if not matches:
            continue
        for start_index in range(0, len(matches), CHUNK_WORDS):
            end_index = min(start_index + CHUNK_WORDS, len(matches))
            start = 0 if start_index == 0 else matches[start_index].start()
            end = len(text) if end_index == len(matches) else matches[end_index].start()
            chunks.append((doc_index, start, text[start:end]))
    return chunks


def gold_sets(docs):
    gold, other = [], []
    for _text, mentions in docs:
        target, rest = set(), set()
        for start, end, tuis in mentions:
            (target if tuis & CORE_TUI else rest).update(range(start, end))
        gold.append(target); other.append(rest)
    return gold, other


def score(predictions, gold, other):
    tp = fp = fn = fp_other = fp_none = 0
    for prediction, target, non_target_gold in zip(predictions, gold, other):
        tp += len(prediction & target); fn += len(target - prediction)
        excess = prediction - target
        fp += len(excess); fp_other += len(excess & non_target_gold); fp_none += len(excess - non_target_gold)
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {"P": round(precision, 4), "R": round(recall, 4),
            "F1": round(2 * precision * recall / max(precision + recall, 1e-12), 4),
            "tp_chars": tp, "fp_chars": fp, "fn_chars": fn,
            "fp_other_gold_pct": round(100 * fp_other / max(fp, 1), 1),
            "fp_no_gold_pct": round(100 * fp_none / max(fp, 1), 1)}


def words_offsets(text):
    matches = list(WORD_RE.finditer(text))
    return [match.group() for match in matches], [(match.start(), match.end()) for match in matches]


def l1_pred(docs, dictionary):
    return [char_set((start, end) for start, end, _kind in l1.l1_spans(text, dictionary)) for text, _ in docs]


def union(l1_sets, l2_sets):
    return [left | right for left, right in zip(l1_sets, l2_sets)]


def student_pred(docs, chunks):
    from transformers import AutoTokenizer
    config = json.loads((MODEL_DIR / "experiment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval(); predicted = [set() for _ in docs]
    with torch.inference_mode():
        for number, (doc, base, text) in enumerate(chunks, 1):
            words, offsets = words_offsets(text)
            encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
            probabilities = model(**{key: value.to(device) for key, value in encoded.items()}).softmax(-1)[0, :, 1].cpu().tolist()
            selected, previous = set(), None
            for token_index, word_index in enumerate(encoded.word_ids(0)):
                if word_index is not None and word_index != previous and probabilities[token_index] >= STUDENT_THRESHOLD:
                    selected.add(word_index)
                previous = word_index
            predicted[doc] |= {base + position for word_index in selected for position in range(*offsets[word_index])}
            if number % 200 == 0: print(f"  Arm B {number}/{len(chunks)} chunks", flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return predicted


def gliner_pred(docs, chunks):
    from gliner import GLiNER
    model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5").to("cuda" if torch.cuda.is_available() else "cpu")
    predicted = [set() for _ in docs]
    for start in range(0, len(chunks), 24):
        part = chunks[start:start + 24]
        output = model.batch_predict_entities([text for _doc, _base, text in part], MED_LABELS,
                                              flat_ner=True, threshold=GLINER_THRESHOLD)
        for (doc, base, _text), entities in zip(part, output):
            predicted[doc] |= {base + position for entity in entities for position in range(entity["start"], entity["end"])}
        print(f"  current GLiNER {min(start + 24, len(chunks))}/{len(chunks)} chunks", flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return predicted


def arm_a_pred(docs, chunks, batch_size):
    disable_incompatible_audio_import()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, local_files_only=True); tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(QWEN_DIR, dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
    predicted, errors = [set() for _ in docs], 0
    for start in range(0, len(chunks), batch_size):
        part = chunks[start:start + batch_size]; rows = []
        for index, (doc, base, text) in enumerate(part, start):
            words, offsets = words_offsets(text)
            rows.append({"id": f"mm-{doc}-{index}", "words": words, "offsets": offsets, "doc": doc, "base": base})
        encoded = tokenizer([make_prompt(tokenizer, row) for row in rows], return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=96, pad_token_id=tokenizer.eos_token_id)
        answers = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for row, answer in zip(rows, answers):
            try:
                labels, _types, _spans = parse_and_align(answer, row["words"])
                predicted[row["doc"]] |= {row["base"] + position for word_index, label in enumerate(labels) if label for position in range(*row["offsets"][word_index])}
            except ValueError:
                errors += 1
        if (start + len(part)) % 100 < batch_size or start + len(part) == len(chunks):
            print(f"  Arm A {start + len(part)}/{len(chunks)} chunks · parse errors={errors}", flush=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return predicted, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--arms", nargs="+", choices=["l1", "current", "a", "b"], default=["l1", "current", "a", "b"])
    parser.add_argument("--qwen-batch-size", type=int, default=8)
    args = parser.parse_args()
    docs = load_pubtator(str(MM_PATH), args.n)
    if len(docs) != args.n: raise RuntimeError(f"Expected {args.n} docs, got {len(docs)}")
    chunks = make_chunks(docs); dictionary, fingerprint = l1.load_dictionaries(); gold, other = gold_sets(docs); base = l1_pred(docs, dictionary)
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    result = {"harness": "H3 MedMentions 600 external gold", "gold": "character-level core 14 UMLS TUI; not direct sensitivity gold",
              "n_documents": len(docs), "n_chunks": len(chunks), "chunk_words": CHUNK_WORDS,
              "l1_version": l1.LAWMASK_VERSION, "l1_dictionary_fingerprint": fingerprint,
              "student_threshold": STUDENT_THRESHOLD, "gliner_threshold": GLINER_THRESHOLD, "arms": dict(previous.get("arms", {}))}
    if "l1" in args.arms: result["arms"]["L1 only"] = score(base, gold, other)
    if "current" in args.arms: result["arms"]["L1 union current GLiNER"] = score(union(base, gliner_pred(docs, chunks)), gold, other)
    if "b" in args.arms: result["arms"]["L1 union Arm B ELECTRA-small"] = score(union(base, student_pred(docs, chunks)), gold, other)
    if "a" in args.arms:
        predicted, errors = arm_a_pred(docs, chunks, args.qwen_batch_size)
        value = score(union(base, predicted), gold, other); value["llm_parse_errors"] = errors
        result["arms"]["L1 union Arm A Qwen3-32B"] = value
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
