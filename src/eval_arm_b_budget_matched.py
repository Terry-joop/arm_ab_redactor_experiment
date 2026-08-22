#!/usr/bin/env python3
"""Evaluate the pre-registered Arm-B equal-mask-budget operating point (th=0.80).

Threshold 0.80 was selected on the Qwen-labelled validation split to match the
teacher mask budget, not selected on MedMentions or H5.  These wrappers keep
that secondary operating point separate from the F2-optimised th=0.51 result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import eval_h3_medmentions_chunked as h3
import eval_h5_arm_ab as h5


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    h3.STUDENT_THRESHOLD = 0.80
    h3.OUT = root / "results/h3_medmentions_arm_b_budget_matched.json"
    sys.argv = ["eval_h3_medmentions_chunked.py", "--arms", "l1", "b"]
    h3.main()

    h5.STUDENT_THRESHOLD = 0.80
    h5.OUT = root / "results/h5_arm_b_budget_matched.json"
    sys.argv = ["eval_h5_arm_ab.py", "--arms", "l1", "b"]
    h5.main()


if __name__ == "__main__":
    main()
