# -*- coding: utf-8 -*-
"""lawmask-1 "1층+LLM" 팔 시험 — 문맥 구분 시험지 420문장 (2026-08-20).

질문(사용자): 2층을 NER 이 아니라 LLM 으로 하면? — 보장은 어차피 1층이 지므로
2층=LLM 은 구조적으로 배제되지 않는다. 같은 시험지로 직접 잰다.

★순환 주의: 이 시험지의 라벨은 Qwen3-4B 가 만들었다(label_probe_senses.py).
  같은 모델을 후보로 재면 자기일치로 부풀므로 **다른 모델**로 잰다:
    · Qwen3-30B-A3B-Instruct (MoE, 다른 아키텍처·세대)
    · Qwen2.5-7B-Instruct (다른 세대, 경량 배포 시나리오)
  그래도 같은 Qwen 계열이라 잔여 상관은 남는다 — 결과 해석에 명시.
  (참고 상한으로 Qwen3-4B 자신도 찍되 '순환' 표기.)

채점: exam_medical_l2.py 와 동일 — HEALTH 플래그율(재현) vs NONHEALTH 플래그율(오탐)·균형.
출력: docs/evidence/2026-08-20/exam_llm_l2.json
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
import transformers.utils
transformers.utils.is_torchaudio_available = lambda: False
from transformers import AutoModelForCausalLM, AutoTokenizer

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REDACTFORMER_ROOT = Path("/home/jovyan/Redactformer")
items = json.loads((REDACTFORMER_ROOT / "docs/evidence/2026-08-20/probe_sense_labels.json").read_text())
items = [e for e in items if not e['sense'].startswith('UNK')]
OUT = EXPERIMENT_ROOT / "artifacts/arm_a_h1_qwen32b.json"

SYS = ("You are the second layer of a privacy masking system. Criterion (GDPR Art.9 health "
       "data): decide whether a specific word, AS USED in the given sentence, reveals a "
       "person's health condition, symptom, medication, treatment, mental health, substance "
       "use, or reproductive/sexual health. Answer with exactly one token: YES or NO.")


def score(flags):
    H = [f for e, f in zip(items, flags) if e['sense'] == 'HEALTH']
    N = [f for e, f in zip(items, flags) if e['sense'] == 'NONHEALTH']
    r = sum(H) / len(H); fp = sum(N) / len(N)
    return {'recall': round(r, 3), 'false': round(fp, 3), 'bal': round((r + 1 - fp) / 2, 3)}


results = {}
for tag, mid in [("Qwen3-32B Arm A", str(EXPERIMENT_ROOT / "models/Qwen3-32B"))]:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                               device_map='auto').eval()
    dev = next(mdl.parameters()).device
    print(f'{tag} 로드 {time.time()-t0:.0f}s · device={dev}', flush=True)
    flags = []
    B = 32
    for i in range(0, len(items), B):
        batch = items[i:i + B]
        prompts = [tok.apply_chat_template(
            [{'role': 'system', 'content': SYS},
             {'role': 'user', 'content': f"Sentence: \"{e['ctx'][:300]}\"\n"
                                         f"Word: \"{e['word']}\"\nYES or NO:"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for e in batch]
        enc = tok(prompts, return_tensors='pt', padding=True, padding_side='left').to(dev)
        with torch.no_grad():
            g = mdl.generate(**enc, max_new_tokens=3, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for row in g:
            ans = tok.decode(row[enc['input_ids'].shape[1]:], skip_special_tokens=True)
            flags.append(int('YES' in ans.upper()))
    results[tag] = score(flags)
    print(f"{tag:26s} 건강플래그 {results[tag]['recall']:.2f}  오탐 {results[tag]['false']:.2f}"
          f"  균형 {results[tag]['bal']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    del mdl
    torch.cuda.empty_cache()

json.dump(results, open(OUT, 'w'), ensure_ascii=False, indent=1)
print('저장:', OUT, flush=True)
