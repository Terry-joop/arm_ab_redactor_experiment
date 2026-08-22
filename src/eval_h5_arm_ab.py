#!/usr/bin/env python3
"""H5 fixed-regression evaluation for L1, current L2, Arm A, and Arm B.

This script imports the senior-maintained regression sentences without changing
them.  It reports aggregate must-mask and must-not-mask violations only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REDACTFORMER = Path("/home/jovyan/Redactformer")
TOKEN_PROBE = Path("/home/jovyan/token_redaction_probe")
MODEL_DIR = ROOT / "artifacts/electra_small_seed42"
QWEN_DIR = ROOT / "models/Qwen3-32B"
OUT = ROOT / "results/h5_arm_ab.json"
sys.path[:0] = [str(REDACTFORMER / "scripts/dataset_builders"), str(TOKEN_PROBE / "src"), str(ROOT / "src")]

import _lawmask_l1 as l1  # noqa: E402
from common import WORD_RE  # noqa: E402
from train import RedactionModel  # noqa: E402
from qwen_lawmask_medical_teacher import CATEGORIES, disable_incompatible_audio_import, make_prompt, parse_and_align  # noqa: E402

MED_LABELS = list(CATEGORIES)
STUDENT_THRESHOLD = 0.51
GLINER_THRESHOLD = 0.5


def cases():
    """Read the canonical synthetic cases from Redactformer without executing it."""
    source = REDACTFORMER / "scripts/lawmask/test_leak_regression.py"
    namespace = {"__name__": "h5_cases", "__file__": str(source)}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    return namespace["CASES"]


def chars(spans):
    return {p for start, end in spans for p in range(start, end)}


def covered(text, predicted, substring):
    start = text.find(substring)
    if start < 0:
        raise ValueError(f"Invalid H5 case: {substring!r}")
    return all(i in predicted for i in range(start, start + len(substring)) if not text[i].isspace())


def student_spans(text, tokenizer, model, device):
    matches = list(WORD_RE.finditer(text))
    words = [m.group() for m in matches]
    offsets = [(m.start(), m.end()) for m in matches]
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
    ids = encoded.word_ids(0)
    logits = model(**{key: value.to(device) for key, value in encoded.items()})
    probabilities = logits.softmax(-1)[0, :, 1].detach().cpu().tolist()
    selected, previous = set(), None
    for index, word_id in enumerate(ids):
        if word_id is not None and word_id != previous and probabilities[index] >= STUDENT_THRESHOLD:
            selected.add(word_id)
        previous = word_id
    return chars(offsets[i] for i in selected)


def evaluate(name, predictions, all_cases):
    must_total = not_total = miss = false = 0
    missed, over = [], []
    for case_index, ((text, required, forbidden), predicted) in enumerate(zip(all_cases, predictions), 1):
        for piece in required:
            must_total += 1
            if not covered(text, predicted, piece):
                miss += 1
                missed.append({"case": case_index, "target": piece})
        for piece in forbidden:
            not_total += 1
            if covered(text, predicted, piece):
                false += 1
                over.append({"case": case_index, "target": piece})
    return {"name": name, "must_mask_total": must_total, "must_mask_missed": miss,
            "must_not_mask_total": not_total, "must_not_mask_violations": false,
            "pass": miss == 0 and false == 0, "missed": missed, "overmasked": over}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", choices=["l1", "current", "a", "b"],
                        default=["l1", "current", "a", "b"])
    args = parser.parse_args()
    all_cases = cases()
    texts = [text for text, _must, _not in all_cases]
    dic, fingerprint = l1.load_dictionaries()
    base = [chars((s, e) for s, e, _c in l1.l1_spans(text, dic)) for text in texts]
    result = {"harness": "H5 canonical L1 regression", "l1_version": l1.LAWMASK_VERSION,
              "l1_dictionary_fingerprint": fingerprint, "n_sentences": len(texts), "arms": {}}
    if "l1" in args.arms:
        result["arms"]["L1 only"] = evaluate("L1 only", base, all_cases)
    if "b" in args.arms:
        from transformers import AutoTokenizer
        config = json.loads((MODEL_DIR / "experiment.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
        model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        with torch.inference_mode():
            extra = [student_spans(text, tokenizer, model, device) for text in texts]
        result["arms"]["L1 union Arm B ELECTRA-small"] = evaluate(
            "L1 union Arm B ELECTRA-small", [a | b for a, b in zip(base, extra)], all_cases)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if "current" in args.arms:
        from gliner import GLiNER
        model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5").to("cuda" if torch.cuda.is_available() else "cpu")
        output = model.batch_predict_entities(texts, MED_LABELS, flat_ner=True, threshold=GLINER_THRESHOLD)
        extra = [chars((e["start"], e["end"]) for e in entities) for entities in output]
        result["arms"]["L1 union current GLiNER"] = evaluate(
            "L1 union current GLiNER", [a | b for a, b in zip(base, extra)], all_cases)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    if "a" in args.arms:
        disable_incompatible_audio_import()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, local_files_only=True); tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(QWEN_DIR, dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
        extra, errors = [], 0
        for start in range(0, len(texts), 4):
            rows = []
            for index, text in enumerate(texts[start:start + 4], start):
                matches = list(WORD_RE.finditer(text))
                rows.append({"id": f"h5-{index}", "words": [m.group() for m in matches],
                             "offsets": [(m.start(), m.end()) for m in matches]})
            encoded = tokenizer([make_prompt(tokenizer, row) for row in rows], return_tensors="pt", padding=True)
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
            answers = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            for row, answer in zip(rows, answers):
                try:
                    labels, _types, _spans = parse_and_align(answer, row["words"])
                    extra.append(chars(row["offsets"][i] for i, value in enumerate(labels) if value))
                except ValueError:
                    errors += 1; extra.append(set())
        value = evaluate("L1 union Arm A Qwen3-32B", [a | b for a, b in zip(base, extra)], all_cases)
        value["llm_parse_errors"] = errors
        result["arms"]["L1 union Arm A Qwen3-32B"] = value
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
