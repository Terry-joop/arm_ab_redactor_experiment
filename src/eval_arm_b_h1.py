"""Evaluate the trained Arm B ELECTRA student on the shared H1 sense probe.

H1 marks one target word in each context.  This evaluates whether the student
marks that target at the deployment threshold; it is not teacher-agreement.
"""
import json
import sys
from pathlib import Path

import torch

PROBE_ROOT = Path("/home/jovyan/Redactformer")
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STUDENT_SOURCE_ROOT = Path("/home/jovyan/token_redaction_probe")
sys.path.insert(0, str(STUDENT_SOURCE_ROOT / "src"))

from common import word_tokenize  # noqa: E402
from train import RedactionModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_DIR = EXPERIMENT_ROOT / "artifacts/electra_small_seed42"
OUT = EXPERIMENT_ROOT / "artifacts/arm_b_h1_electra.json"
THRESHOLD = 0.51  # selected on the Qwen-labelled validation split by F2


def score(items, flags):
    health = [f for item, f in zip(items, flags) if item["sense"] == "HEALTH"]
    nonhealth = [f for item, f in zip(items, flags) if item["sense"] == "NONHEALTH"]
    recall = sum(health) / len(health)
    false = sum(nonhealth) / len(nonhealth)
    return {"recall": round(recall, 3), "false": round(false, 3),
            "bal": round((recall + 1 - false) / 2, 3)}


def main():
    config = json.loads((MODEL_DIR / "experiment.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    model.load_state_dict(torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    items = json.loads((PROBE_ROOT / "docs/evidence/2026-08-20/probe_sense_labels.json").read_text())
    items = [item for item in items if not item["sense"].startswith("UNK")]
    flags, missed_target = [], 0
    with torch.no_grad():
        for item in items:
            words = word_tokenize(item["ctx"])
            enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
            word_ids = enc.word_ids(0)
            logits = model(**{key: value.to(device) for key, value in enc.items()})
            probabilities = logits.softmax(-1)[0, :, 1].cpu().tolist()
            target = item["word"].casefold()
            hits = [probabilities[token_index] >= THRESHOLD for token_index, word_id in enumerate(word_ids)
                    if word_id is not None and words[word_id].casefold() == target]
            if not hits:
                missed_target += 1
            flags.append(int(any(hits)))
    result = {"student": "ELECTRA-small Arm B", "threshold": THRESHOLD,
              "n": len(items), "target_not_tokenized": missed_target, **score(items, flags)}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
