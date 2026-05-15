# src/chat_model/datasets/inspect_data.py
"""Inspect preprocessed medical splits and print a quality report.

Checks for:
  - Length distribution across train/val/test
  - Leftover noise (branding, HTML, URLs, MCQ language, etc.)
  - Answer word count distribution
  - Random sample examples

Also saves a JSON report to data/reports/inspection.json.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from chat_model import config

NOISE_PATTERNS = {
    "branding_leftover"  : re.compile(r"healthcaremagic|chatdoctor",                     re.IGNORECASE),
    "broken_greeting"    : re.compile(r"welcome to a doctor|posting on a doctor",         re.IGNORECASE),
    "html_tag"           : re.compile(r"<[a-z]+[\s>]",                                    re.IGNORECASE),
    "raw_url"            : re.compile(r"https?://"),
    "mcq_option"         : re.compile(r"\bthe correct answer is option [ABCD]\b",         re.IGNORECASE),
    "third_person_patient": re.compile(r"\bthe patient should\b|\bpatients should\b",     re.IGNORECASE),
    "passive_recommend"  : re.compile(r"\bit is recommended\b|\bit is advised\b",         re.IGNORECASE),
}


def _load_split(name: str) -> list:
    path = Path(config.SPLITS_DIR) / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run preprocess() first.")
    data = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _avg(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def inspect_data(n_samples: int = 5) -> None:
    """Run full data inspection and print report."""
    random.seed(42)
    print("Data inspection report")
    print("=" * 48)

    splits   = {name: _load_split(name) for name in ("train", "val", "test")}
    all_data = [row for split in splits.values() for row in split]
    report   = {"splits": {}, "noise": {}, "total_examples": len(all_data)}

    # Per-split stats
    for name, rows in splits.items():
        q_lens = [len(r[0]["content"]) for r in rows]
        a_lens = [len(r[1]["content"]) for r in rows]
        print(f"\n{name.upper()} split ({len(rows):,} examples)")
        print(f"  Questions avg={_avg(q_lens):>6.0f}  min={min(q_lens):>5}  max={max(q_lens):>5}")
        print(f"  Answers   avg={_avg(a_lens):>6.0f}  min={min(a_lens):>5}  max={max(a_lens):>5}")
        report["splits"][name] = {
            "examples": len(rows),
            "question_chars": {"avg": round(_avg(q_lens), 1), "min": min(q_lens), "max": max(q_lens)},
            "answer_chars":   {"avg": round(_avg(a_lens), 1), "min": min(a_lens), "max": max(a_lens)},
        }

    # Noise check
    print(f"\nNoise check ({len(all_data):,} total examples)")
    noisy = False
    for label, pattern in NOISE_PATTERNS.items():
        hits = [c[1]["content"] for c in all_data if pattern.search(c[1]["content"])]
        report["noise"][label] = {"count": len(hits), "example": hits[0][:250] if hits else None}
        if hits:
            noisy = True
            print(f"  WARN {label:<25} {len(hits):>6,}")
            print(f"       Example: {hits[0][:200].replace(chr(10), ' ')}")
        else:
            print(f"  OK   {label:<25} clean")

    # Word count distribution
    print("\nAnswer word count distribution")
    buckets: Counter = Counter()
    for conv in all_data:
        words = len(conv[1]["content"].split())
        if words < 20:        buckets["<20"]     += 1
        elif words < 50:      buckets["20-50"]   += 1
        elif words < 100:     buckets["50-100"]  += 1
        elif words < 150:     buckets["100-150"] += 1
        else:                 buckets["150+"]    += 1
    total = len(all_data)
    for bucket in ["<20", "20-50", "50-100", "100-150", "150+"]:
        count = buckets[bucket]
        bar   = "█" * int(count / total * 40)
        print(f"  {bucket:<10} {count:>7,}  {count/total*100:>5.1f}%  {bar}")

    # Random samples
    print(f"\nRandom samples ({n_samples})")
    for i, conv in enumerate(random.sample(all_data, min(n_samples, len(all_data))), 1):
        print(f"\n  [{i}] Q: {conv[0]['content'][:120]}")
        print(f"      A: {conv[1]['content'][:250]}")

    # Save report
    report_path = Path(config.DATA_DIR) / "reports" / "inspection.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSummary")
    print(f"  Total examples : {len(all_data):,}")
    print(f"  Noise issues   : {'yes — see above' if noisy else 'none found ✓'}")
    print(f"  Report saved   : {report_path}")


if __name__ == "__main__":
    inspect_data()
