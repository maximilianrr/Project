# src/chat_model/tokenizing/prepare_data.py
"""Build the shared tokenizer corpus from both training stages.

The tokenizer must see BOTH Stage 1 (oasst2) and Stage 2 (medical) data
so it has full vocabulary coverage for both training stages.

Sources written to tokenizer_text.txt:
  1. oasst2 pretraining pairs  (general English conversation)
  2. Medical train + val splits (medical vocabulary — test never used)
  3. WTND book text             (plain-language medical prose)
"""

from __future__ import annotations

import glob
import json
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

from chat_model import config


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _write_chunks(out, text: str) -> tuple[int, int]:
    """Write text in DOC_CAP-sized chunks. Returns (lines, chars) written."""
    text = _normalise(text)
    if not text:
        return 0, 0
    lines = chars = 0
    for i in range(0, len(text), config.DOC_CAP):
        chunk = text[i:i + config.DOC_CAP].strip()
        if chunk:
            out.write(chunk + "\n")
            lines += 1
            chars += len(chunk)
    return lines, chars


def _add_oasst2(out, max_chars: int | None) -> tuple[int, int]:
    shards = sorted(glob.glob(str(Path(config.PRETRAIN_PARQUET_DIR) / "oasst2_*.parquet")))
    if not shards:
        print(f"  oasst2: NOT FOUND in {config.PRETRAIN_PARQUET_DIR}")
        print("          Run: download_data() first")
        return 0, 0

    print(f"  oasst2: reading {len(shards)} shard(s)...")
    total_lines = total_chars = 0

    for shard_path in shards:
        table     = pq.read_table(shard_path, columns=["question", "answer"])
        questions = table.column("question").to_pylist()
        answers   = table.column("answer").to_pylist()

        for question, answer in zip(questions, answers):
            for text in (question, answer):
                lines, chars = _write_chunks(out, text)
                total_lines += lines
                total_chars += chars
                if max_chars and total_chars >= max_chars:
                    print(f"    Reached oasst2 char cap ({max_chars:,})")
                    return total_lines, total_chars

    return total_lines, total_chars


def _add_medical_splits(out) -> tuple[int, int]:
    total_lines = total_chars = 0
    for split in ("train", "val"):
        path = Path(config.SPLITS_DIR) / f"{split}.jsonl"
        if not path.exists():
            print(f"  Medical {split}: missing — run preprocess() first")
            continue
        split_lines = 0
        with path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    conv = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for msg in conv:
                    lines, chars = _write_chunks(out, msg.get("content", ""))
                    split_lines  += lines
                    total_lines  += lines
                    total_chars  += chars
        print(f"  Medical {split:<5}: {split_lines:>8,} lines")
    return total_lines, total_chars


def _add_wtnd(out) -> tuple[int, int]:
    path = Path(config.WTND_CLEAN)
    if not path.exists():
        print(f"  WTND: missing {path} — run download_wtnd() first")
        return 0, 0
    total_lines = total_chars = 0
    for paragraph in re.split(r"\n{2,}", path.read_text(encoding="utf-8")):
        if len(paragraph.strip()) >= 30:
            lines, chars = _write_chunks(out, paragraph)
            total_lines += lines
            total_chars += chars
    print(f"  WTND book  : {total_lines:>8,} lines")
    return total_lines, total_chars


def prepare_tokenizer_data(pretrain_char_cap: int | None = None) -> None:
    """Build tokenizer_text.txt from oasst2 + medical splits + WTND."""
    cap      = pretrain_char_cap if pretrain_char_cap is not None else config.OWT_TOKENIZER_CHARS
    out_path = Path(config.TOKENIZER_TEXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nBuilding shared tokenizer corpus")
    print("=" * 48)
    print(f"Output : {out_path}")
    print(f"Doc cap: {config.DOC_CAP:,} chars/line\n")

    total_lines = total_chars = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        lines, chars = _add_oasst2(out, cap)
        print(f"    oasst2 wrote {lines:,} lines ({chars / 1e6:.1f} MB)")
        total_lines += lines
        total_chars += chars

        lines, chars = _add_medical_splits(out)
        total_lines += lines
        total_chars += chars

        lines, chars = _add_wtnd(out)
        total_lines += lines
        total_chars += chars

    print(f"\nTotal lines : {total_lines:,}")
    print(f"Total chars : {total_chars / 1e6:.1f} MB")
    print(f"Saved to    : {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    prepare_tokenizer_data()
