"""A/B 공통 1층의 실행 지문과 H5(L1-only) 결과를 JSON으로 고정한다.

Usage:
  /opt/conda/bin/python3 src/snapshot_l1.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "dataset_builders"))
import _lawmask_l1 as l1  # noqa: E402


def main() -> None:
    dictionaries, floor_sha = l1.load_dictionaries()
    test = ROOT / "scripts" / "lawmask" / "test_leak_regression.py"
    proc = subprocess.run([sys.executable, str(test)], cwd=ROOT, text=True,
                          capture_output=True, check=False)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": "shared_l1_baseline",
        "l1_definition": l1.definition_dict(floor_sha),
        "h5_l1_only": {
            "exit_code": proc.returncode,
            "passed": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        },
    }
    out = ROOT / "experiments" / "arm_ab_20260820" / "artifacts" / "l1_baseline.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
