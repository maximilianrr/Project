# src/chat_model/tokenizing/prepare_data.py
"""Build the shared tokenizer corpus for the 3-stage training pipeline.

The tokenizer is trained once and shared across all stages, so it must see
representative text from general pretraining, medical continued pretraining,
and chat fine-tuning.

Sources written to tokenizer_text.txt:
  1. climbmix raw documents       (Stage 1 general text)
  2. oasst2 Q/A pairs             (Stage 3 conversational vocabulary)
  3. PubMed title+abstract pairs  (Stage 2 medical/scientific vocabulary)
  4. Stage 3 chat splits          (medical/patient-facing chat vocabulary)
  5. WTND book text               (plain-language medical prose)
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


def _add_climbmix(out, max_chars: int | None) -> tuple[int, int]:
    """Add climbmix raw text documents — primary Stage 1 vocabulary source."""
    shards = sorted(glob.glob(str(Path(config.CLIMBMIX_PARQUET_DIR) / "climbmix_*.parquet")))
    if not shards:
        print(f"  climbmix: NOT FOUND in {config.CLIMBMIX_PARQUET_DIR}")
        print("            Run: download_climbmix() first")
        return 0, 0

    print(f"  climbmix: reading {len(shards)} shard(s)...")
    total_lines = total_chars = 0

    for shard_path in shards:
        table = pq.read_table(shard_path, columns=["text"])
        texts = table.column("text").to_pylist()

        for text in texts:
            lines, chars = _write_chunks(out, text)
            total_lines += lines
            total_chars += chars
            if max_chars and total_chars >= max_chars:
                print(f"    Reached climbmix char cap ({max_chars:,}) at {Path(shard_path).name}")
                return total_lines, total_chars

    return total_lines, total_chars


def _add_oasst2(out, max_chars: int | None) -> tuple[int, int]:
    """Add oasst2 Q/A pairs — general English conversational vocabulary."""
    shards = sorted(glob.glob(str(Path(config.PRETRAIN_PARQUET_DIR) / "oasst2_*.parquet")))
    if not shards:
        print(f"  oasst2: NOT FOUND in {config.PRETRAIN_PARQUET_DIR}")
        print("          Run: download_pretrain_data() first")
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
                    print(f"    Reached oasst2 char cap ({max_chars:,}) at {Path(shard_path).name}")
                    return total_lines, total_chars

    return total_lines, total_chars


def _add_pubmed(out, max_chars: int | None) -> tuple[int, int]:
    """Add PubMed abstracts — medical scientific vocabulary for Stage 2 coverage."""
    shards = sorted(glob.glob(str(Path(config.PUBMED_PARQUET_DIR) / "pubmed_*.parquet")))
    if not shards:
        print(f"  PubMed: NOT FOUND in {config.PUBMED_PARQUET_DIR}")
        print("          Run: download_pubmed() first (or with --max-abstracts for a sample)")
        return 0, 0

    print(f"  PubMed: reading {len(shards)} shard(s)...")
    total_lines = total_chars = 0

    for shard_path in shards:
        table     = pq.read_table(shard_path, columns=["title", "abstract"])
        titles    = table.column("title").to_pylist()
        abstracts = table.column("abstract").to_pylist()

        for title, abstract in zip(titles, abstracts):
            combined = f"{title}\n{abstract}" if title else abstract
            lines, chars = _write_chunks(out, combined)
            total_lines += lines
            total_chars += chars
            if max_chars and total_chars >= max_chars:
                print(f"    Reached PubMed char cap ({max_chars:,}) at {Path(shard_path).name}")
                return total_lines, total_chars

    return total_lines, total_chars


def _add_stage3_splits(out) -> tuple[int, int]:
    """Add Stage 3 fine-tuning splits - conversational medical vocabulary."""
    total_lines = total_chars = 0

    for split in ("train", "val"):
        path = Path(config.STAGE3_SPLITS_DIR) / f"{split}.jsonl"
        if not path.exists():
            print(f"  Stage 3 {split}: missing - run preprocess_stage3() first")
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
                    split_lines += lines
                    total_lines += lines
                    total_chars += chars

        print(f"  Stage 3 {split:<5}: {split_lines:>8,} lines")

    return total_lines, total_chars


def _add_wtnd(out) -> tuple[int, int]:
    """Add WTND plain-language medical prose."""
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


def prepare_tokenizer_data(
    climbmix_char_cap: int | None = None,
    oasst2_char_cap:   int | None = None,
    pubmed_char_cap:   int | None = None,
) -> None:
    """Build tokenizer_text.txt from all sources.

    Source order:
    1. climbmix        - Stage 1 general raw text
    2. oasst2          - Stage 3 conversational Q/A text
    3. PubMed          - Stage 2 medical scientific text
    4. Stage 3 splits  - medical chat fine-tuning text
    5. WTND book       - plain-language medical prose
    """
    climbmix_cap = climbmix_char_cap if climbmix_char_cap is not None else config.CLIMBMIX_TOKENIZER_CHARS
    oasst2_cap   = oasst2_char_cap   if oasst2_char_cap   is not None else config.OWT_TOKENIZER_CHARS
    pubmed_cap   = pubmed_char_cap   if pubmed_char_cap   is not None else config.OWT_TOKENIZER_CHARS
    out_path     = Path(config.TOKENIZER_TEXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nBuilding shared tokenizer corpus")
    print("=" * 48)
    print(f"Output        : {out_path}")
    print(f"Doc cap       : {config.DOC_CAP:,} chars/line")
    print(f"climbmix cap  : {climbmix_cap:,} chars" if climbmix_cap else "climbmix cap  : none")
    print(f"oasst2 cap    : {oasst2_cap:,} chars"   if oasst2_cap   else "oasst2 cap    : none")
    print(f"PubMed cap    : {pubmed_cap:,} chars\n"  if pubmed_cap   else "PubMed cap    : none\n")

    total_lines = total_chars = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out:

        lines, chars = _add_climbmix(out, climbmix_cap)
        print(f"    climbmix wrote {lines:,} lines ({chars / 1e6:.1f} MB)")
        total_lines += lines
        total_chars += chars

        lines, chars = _add_oasst2(out, oasst2_cap)
        print(f"    oasst2 wrote   {lines:,} lines ({chars / 1e6:.1f} MB)")
        total_lines += lines
        total_chars += chars

        lines, chars = _add_pubmed(out, pubmed_cap)
        print(f"    PubMed wrote   {lines:,} lines ({chars / 1e6:.1f} MB)")
        total_lines += lines
        total_chars += chars

        lines, chars = _add_stage3_splits(out)
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
    import argparse
    parser = argparse.ArgumentParser(description="Build shared tokenizer corpus.")
    parser.add_argument("--climbmix-char-cap", type=int, default=None,
                        help="Max chars from climbmix. Defaults to config.CLIMBMIX_TOKENIZER_CHARS.")
    parser.add_argument("--oasst2-char-cap",   type=int, default=None,
                        help="Max chars from oasst2. Defaults to config.OWT_TOKENIZER_CHARS.")
    parser.add_argument("--pubmed-char-cap",   type=int, default=None,
                        help="Max chars from PubMed. Defaults to config.OWT_TOKENIZER_CHARS.")
    args = parser.parse_args()
    prepare_tokenizer_data(
        climbmix_char_cap=args.climbmix_char_cap,
        oasst2_char_cap=args.oasst2_char_cap,
        pubmed_char_cap=args.pubmed_char_cap,
    )