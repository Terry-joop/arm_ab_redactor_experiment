#!/usr/bin/env python3
"""H3: MedMentions 600 external-gold evaluation for the current L1 and Arm A/B.

The benchmark's gold labels are UMLS semantic types, not a direct "sensitive"
label.  We therefore report character-level P/R/F1 against the 14 registered
medical TUI types, and decompose false positives into other UMLS mentions and
text that is not a gold mention.  Every arm uses the same current deterministic
L1; only L2 is replaced.

Outputs contain aggregate counts only.  PubMed source text and teacher outputs
remain under ignored ``data/`` / ``artifacts/`` paths and are never published.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REDACTFORMER = Path("/home/jovyan/Redactformer")
TOKEN_PROBE = Path("/home/jovyan/token_redaction_probe")
MM_PATH = ROOT / "data/benchmarks_medmentions_pubtator.txt"
MODEL_DIR = ROOT / "artifacts/electra_small_seed42"
QWEN_DIR = ROOT / "models/Qwen3-32B"
OUT = ROOT / "results/h3_medmentions_arm_ab.json"

sys.path[:0] = [
    str(REDACTFORMER / "scripts/audit"),
    str(REDACTFORMER / "scripts/dataset_builders"),
    str(TOKEN_PROBE / "src"),
    str(ROOT / "src"),
]

from medmentions_eval import load_pubtator  # noqa: E402
import _lawmask_l1 as l1  # noqa: E402
from common import WORD_RE  # noqa: E402
from train import RedactionModel  # noqa: E402
from qwen_lawmask_medical_teacher import (  # noqa: E402
    CATEGORIES, disable_incompatible_audio_import, make_prompt, parse_and_align,
)


CORE_TUI = {
    "T047", "T048", "T191", "T046", "T184", "T037", "T019", "T121",
    "T195", "T200", "T061", "T060", "T059", "T034",
}
MED_LABELS = list(CATEGORIES)
STUDENT_THRESHOLD = 0.51  # validation F2 optimum, selected before H3
GLINER_THRESHOLD = 0.5    # current production L2 setting


def charset(spans):
    return {char for start, end in spans for char in range(start, end)}


def gold_sets(docs):
    gold, other = [], []
    for _text, mentions in docs:
        target, rest = set(), set()
        for start, end, tuis in mentions:
            (target if tuis & CORE_TUI else rest).update(range(start, end))
        gold.append(target)
        other.append(rest)
    return gold, other


def score(predictions, gold, other):
    tp = fp = fn = fp_other = fp_none = 0
    for predicted, target, nonsensitive_gold in zip(predictions, gold, other):
        tp += len(predicted & target)
        fn += len(target - predicted)
        excess = predicted - target
        fp += len(excess)
        fp_other += len(excess & nonsensitive_gold)
        fp_none += len(excess - nonsensitive_gold)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "P": round(precision, 4), "R": round(recall, 4), "F1": round(f1, 4),
        "tp_chars": tp, "fp_chars": fp, "fn_chars": fn,
        "fp_other_gold_pct": round(100 * fp_other / max(fp, 1), 1),
        "fp_no_gold_pct": round(100 * fp_none / max(fp, 1), 1),
    }


def words_with_offsets(text):
    matches = list(WORD_RE.finditer(text))
    return [match.group() for match in matches], [(match.start(), match.end()) for match in matches]


def l1_predictions(docs, dictionaries):
    return [charset((start, end) for start, end, _category in l1.l1_spans(text, dictionaries))
            for text, _mentions in docs]


def student_predictions(docs):
    from transformers import AutoTokenizer

    config = json.loads((MODEL_DIR / "experiment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for index, (text, _mentions) in enumerate(docs, 1):
            words, offsets = words_with_offsets(text)
            encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                                truncation=True, max_length=128)
            word_ids = encoded.word_ids(0)
            logits = model(**{key: value.to(device) for key, value in encoded.items()})
            probabilities = logits.softmax(-1)[0, :, 1].cpu().tolist()
            selected, previous = set(), None
            for token_index, word_id in enumerate(word_ids):
                if word_id is not None and word_id != previous and probabilities[token_index] >= STUDENT_THRESHOLD:
                    selected.add(word_id)
                previous = word_id
            predictions.append(charset(offsets[word_id] for word_id in selected))
            if index % 100 == 0:
                print(f"  Arm B ELECTRA {index}/{len(docs)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def gliner_predictions(docs):
    from gliner import GLiNER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5").to(device)
    texts = [text for text, _mentions in docs]
    predictions = []
    for start in range(0, len(texts), 24):
        batch = texts[start:start + 24]
        outputs = model.batch_predict_entities(batch, MED_LABELS, flat_ner=True,
                                               threshold=GLINER_THRESHOLD)
        predictions.extend(charset((entity["start"], entity["end"]) for entity in entities)
                           for entities in outputs)
        print(f"  current GLiNER {min(start + 24, len(texts))}/{len(texts)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def arm_a_predictions(docs, batch_size):
    disable_incompatible_audio_import()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_DIR, dtype=torch.bfloat16, device_map="auto", local_files_only=True
    ).eval()
    predictions, parse_errors = [], 0
    rows = []
    for index, (text, _mentions) in enumerate(docs):
        words, offsets = words_with_offsets(text)
        rows.append({"id": f"medmentions-{index}", "words": words, "offsets": offsets})
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        prompts = [make_prompt(tokenizer, row) for row in batch_rows]
        batch = tokenizer(prompts, return_tensors="pt", padding=True)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**batch, do_sample=False, max_new_tokens=256,
                                       pad_token_id=tokenizer.eos_token_id)
        answers = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        for row, answer in zip(batch_rows, answers):
            try:
                labels, _types, _spans = parse_and_align(answer, row["words"])
                predictions.append(charset(row["offsets"][i] for i, label in enumerate(labels) if label))
            except ValueError:
                # An unalignable generative response is deployed as no L2 hit, not silently repaired.
                parse_errors += 1
                predictions.append(set())
        print(f"  Arm A Qwen {min(start + batch_size, len(rows))}/{len(rows)} · parse errors={parse_errors}",
              flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, parse_errors


def union(left, right):
    return [a | b for a, b in zip(left, right)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--arms", nargs="+", choices=["l1", "current", "a", "b"],
                        default=["l1", "current", "a", "b"])
    parser.add_argument("--qwen-batch-size", type=int, default=4)
    args = parser.parse_args()

    docs = load_pubtator(str(MM_PATH), args.n)
    if len(docs) != args.n:
        raise RuntimeError(f"MedMentions {args.n} documents required, got {len(docs)}")
    dictionaries, fingerprint = l1.load_dictionaries()
    gold, other = gold_sets(docs)
    base = l1_predictions(docs, dictionaries)
    base_result = {}
    if OUT.exists():
        try:
            base_result = json.loads(OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A partial file must never be treated as an evaluation result.
            base_result = {}
    results = {
        "harness": "H3 MedMentions 600 external gold",
        "gold": "character-level core 14 UMLS TUI; not a direct sensitivity gold label",
        "n_documents": len(docs), "l1_version": l1.LAWMASK_VERSION,
        "l1_dictionary_fingerprint": fingerprint, "student_threshold": STUDENT_THRESHOLD,
        "gliner_threshold": GLINER_THRESHOLD,
        # A one-arm rerun preserves previous completed arms; Qwen runs alone.
        "arms": dict(base_result.get("arms", {})),
    }
    if "l1" in args.arms:
        results["arms"]["L1 only"] = score(base, gold, other)
    if "current" in args.arms:
        results["arms"]["L1 union current GLiNER"] = score(union(base, gliner_predictions(docs)), gold, other)
    if "a" in args.arms:
        prediction, errors = arm_a_predictions(docs, args.qwen_batch_size)
        value = score(union(base, prediction), gold, other)
        value["llm_parse_errors"] = errors
        results["arms"]["L1 union Arm A Qwen3-32B"] = value
    if "b" in args.arms:
        results["arms"]["L1 union Arm B ELECTRA-small"] = score(union(base, student_predictions(docs)), gold, other)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
