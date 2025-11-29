#!/usr/bin/env python3
"""
Phân loại lỗi giữa outputs/answers.csv và inputs/answer_grouth.csv.
"""

import csv
from pathlib import Path
from collections import Counter


def _norm(ans: str):
    if ans is None:
        return ()   
    ans = ans.strip().strip('"')
    if not ans:
        return ()
    parts = [a.strip().upper() for a in ans.split(",") if a.strip()]
    return tuple(sorted(parts))


def analyze():
    project_root = Path(__file__).parent.parent
    pred_csv = project_root / "outputs" / "answers.csv"
    gt_csv = project_root / "inputs" / "answer_grouth.csv"

    if not pred_csv.exists() or not gt_csv.exists():
        raise SystemExit("Missing outputs/answers.csv or inputs/answer_grouth.csv")

    with pred_csv.open(encoding="utf-8") as f:
        pred_rows = list(csv.DictReader(f))
    with gt_csv.open(encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    total = min(len(pred_rows), len(gt_rows))
    print(f"Comparing first {total} rows")

    stats = Counter()
    samples = {"blank": [], "single": [], "multi": []}

    for idx in range(total):
        p = pred_rows[idx]
        g = gt_rows[idx]
        p_ans = _norm(p.get("answers", ""))
        g_ans = _norm(g.get("answers", ""))
        p_num = int(p.get("num_correct") or 0)
        g_num = int(g.get("num_correct") or 0)

        if p_ans == g_ans and p_num == g_num:
            stats["correct"] += 1
            continue

        if not p_ans:
            stats["blank"] += 1
            if len(samples["blank"]) < 10:
                samples["blank"].append((idx+1, p_ans, g_ans))
            continue

        if len(g_ans) >= 2:
            stats["multi_miss"] += 1
            if len(samples["multi"]) < 10:
                samples["multi"].append((idx+1, p_ans, g_ans))
        else:
            stats["single_miss"] += 1
            if len(samples["single"]) < 10:
                samples["single"].append((idx+1, p_ans, g_ans))

    print("\n=== SUMMARY ===")
    print(f"Correct        : {stats['correct']}")
    print(f"Blank answers  : {stats['blank']}")
    print(f"Single misses  : {stats['single_miss']}")
    print(f"Multi misses   : {stats['multi_miss']}")

    for label, rows in samples.items():
        if not rows:
            continue
        print(f"\n--- {label.upper()} SAMPLES ---")
        for idx, p_ans, g_ans in rows:
            print(f"# {idx}: pred={p_ans or '∅'} | gt={g_ans or '∅'}")


if __name__ == "__main__":
    analyze()

