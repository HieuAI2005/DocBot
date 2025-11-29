#!/usr/bin/env bash
set -euo pipefail

# Compare model answers with ground-truth CSV and print accuracy stats.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PRED_CSV="${1:-$PROJECT_ROOT/outputs/answers.csv}"
GT_CSV="${2:-$PROJECT_ROOT/inputs/answer_grouth.csv}"

if [[ ! -f "$PRED_CSV" ]]; then
  echo "Prediction CSV not found: $PRED_CSV" >&2
  exit 1
fi

if [[ ! -f "$GT_CSV" ]]; then
  echo "Ground-truth CSV not found: $GT_CSV" >&2
  exit 1
fi

python - "$PRED_CSV" "$GT_CSV" <<'PYCODE'
import sys, csv, pathlib

pred_path = pathlib.Path(sys.argv[1])
gt_path = pathlib.Path(sys.argv[2])

with pred_path.open(encoding="utf-8") as f:
    pred_rows = list(csv.DictReader(f))
with gt_path.open(encoding="utf-8") as f:
    gt_rows = list(csv.DictReader(f))

total = min(len(pred_rows), len(gt_rows))
if len(pred_rows) != len(gt_rows):
    print(f"[WARN] Row count mismatch: pred={len(pred_rows)} gt={len(gt_rows)} -> comparing first {total}")

def norm(ans: str):
    if ans is None:
        return ()
    ans = ans.strip().strip('"')
    if not ans:
        return ()
    parts = [a.strip().upper() for a in ans.split(",") if a.strip()]
    return tuple(sorted(parts))

correct = 0
mistakes = []

for idx in range(total):
    p = pred_rows[idx]
    g = gt_rows[idx]
    p_ans = norm(p.get("answers", ""))
    g_ans = norm(g.get("answers", ""))

    if p_ans == g_ans and p.get("num_correct") == g.get("num_correct"):
        correct += 1
    else:
        mistakes.append({
            "index": idx + 1,
            "pred": (p.get("num_correct"), p.get("answers")),
            "gt": (g.get("num_correct"), g.get("answers")),
        })

accuracy = correct / total if total else 0.0

print("=" * 50)
print(f"Compared rows   : {total}")
print(f"Correct matches : {correct}")
print(f"Accuracy        : {accuracy:.2%}")
print("=" * 50)

if mistakes:
    print(f"First 10 mismatches (idx, pred, gt):")
    for m in mistakes[:10]:
        print(f"- #{m['index']:>3}: pred={m['pred']} | gt={m['gt']}")
else:
    print("Perfect match!")
PYCODE

