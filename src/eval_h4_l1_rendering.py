#!/usr/bin/env python3
"""H4 integration regression: verify current L1 spans survive no token rendering.

H4 is deliberately not a quality contest between L2 arms.  Since every final
arm is L1 union L2, this checks the shared deployment boundary: character L1
spans -> whole-word mask decisions -> visible rendered text -> fresh L1 scan.
Only aggregate residual counts are written; held-out medical text is not saved.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REDACTFORMER = Path('/home/jovyan/Redactformer')
TOKEN_PROBE = Path('/home/jovyan/token_redaction_probe')
TEST = ROOT / 'data/student/qwen32b_lawmask8_seed42/test.jsonl'
OUT = ROOT / 'results/h4_l1_rendering_heldout.json'
sys.path[:0] = [str(REDACTFORMER / 'scripts/dataset_builders'), str(TOKEN_PROBE / 'src')]
import _lawmask_l1 as l1  # noqa: E402
from common import WORD_RE  # noqa: E402


def scan_visible(text, dictionary):
    counts = Counter()
    for _s, _e, category in l1.l1_spans(text, dictionary):
        counts[category] += 1
    return counts


def main():
    dictionary, fingerprint = l1.load_dictionaries()
    totals, rows, alignment_errors = defaultdict(Counter), Counter(), Counter()
    with TEST.open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line); group = row['dataset_name']; text = row['text']
            matches = list(WORD_RE.finditer(text)); words = [m.group() for m in matches]
            if words != row['words']:
                alignment_errors[group] += 1
                continue
            l1_spans = [(start, end) for start, end, _category in l1.l1_spans(text, dictionary)]
            # A deployment token is masked if it overlaps any L1 character span.
            # Keep every original character boundary; rejoining tokens with spaces would
            # create artificial normalisations (e.g. punctuation-attached acronyms).
            rendered = list(text)
            for match in matches:
                if any(match.start() < end and start < match.end() for start, end in l1_spans):
                    for position in range(match.start(), match.end()):
                        rendered[position] = "␅"
            visible = "".join(rendered)
            totals[group] += scan_visible(visible, dictionary)
            rows[group] += 1
    result = {
        'harness': 'H4 shared L1 token-rendering regression on Arm-B-held-out rows',
        'scope': 'L1 only versus L1 union L2 has the same L1 safety result; this is not an L2 quality metric.',
        'l1_version': l1.LAWMASK_VERSION, 'l1_dictionary_fingerprint': fingerprint,
        'datasets': {group: {'rows': rows[group], 'tokenization_mismatch_rows': alignment_errors[group],
                             'residual_l1_patterns': sum(totals[group].values()),
                             'residual_by_category': dict(totals[group])}
                     for group in sorted(set(rows) | set(alignment_errors))},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
