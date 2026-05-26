# src/chat_model/tokenizing/convert_to_parquet.py
"""Optional parquet export for the 3-stage pipeline.

The current training entry point, scripts/train_3stage.py, does not require
these parquet files. It reads the text/JSONL artifacts directly:

  Stage 1: data/pretrain_splits/climbmix_docs.txt
  Stage 2: data/stage2_splits/stage2_docs.txt
  Stage 3: data/stage3_splits/train.jsonl, val.jsonl, test.jsonl

Use this script only when you want portable parquet shards for inspection,
archiving, or a future nanochat-style data path.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from chat_model import config


TEXT_SCHEMA = pa.schema([pa.field("text", pa.string())])


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _write_text_shards(rows: list[str], out_dir: Path, prefix: str) -> tuple[int, int]:
    """Write rows to snappy parquet shards. Returns (row_count, char_count)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_chars = 0

    for shard_idx, start in enumerate(range(0, len(rows), config.SHARD_SIZE)):
        shard = rows[start:start + config.SHARD_SIZE]
        out_path = out_dir / f"{prefix}_{shard_idx:05d}.parquet"
        table = pa.table({"text": shard}, schema=TEXT_SCHEMA)
        pq.write_table(table, out_path, compression="snappy")

        shard_chars = sum(len(row) for row in shard)
        total_chars += shard_chars
        print(
            f"  {out_path.name:<36} {len(shard):>8,} rows "
            f"{shard_chars / 1e6:>8.1f} MB"
        )

    return len(rows), total_chars


def _read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = _normalise(line)
            if row:
                rows.append(row)
    return rows


def _conversation_to_json(raw: str) -> str | None:
    try:
        conv = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(conv, list) or len(conv) < 2:
        return None

    for msg in conv:
        if not isinstance(msg, dict):
            return None
        if msg.get("role") not in {"system", "user", "assistant"}:
            return None
        if not isinstance(msg.get("content"), str) or not msg["content"].strip():
            return None

    return json.dumps(conv, ensure_ascii=False)


def convert_tokenizer_corpus() -> tuple[int, int]:
    """Convert data/tokenizer_text.txt to data/parquet/tokenizer_*.parquet."""
    in_path = Path(config.TOKENIZER_TEXT)
    rows = _read_text_lines(in_path)

    out_dir = Path(config.PARQUET_DIR) / "tokenizer"
    print(f"\nTokenizer corpus: {in_path}")
    print(f"Saving to: {out_dir}")
    return _write_text_shards(rows, out_dir, "tokenizer")


def convert_stage1_text() -> tuple[int, int]:
    """Convert Stage 1 climbmix text docs to parquet."""
    in_path = Path(config.PRETRAIN_SPLITS_DIR) / "climbmix_docs.txt"
    rows = _read_text_lines(in_path)

    out_dir = Path(config.PARQUET_DIR) / "stage1"
    print(f"\nStage 1 text: {in_path}")
    print(f"Saving to: {out_dir}")
    return _write_text_shards(rows, out_dir, "stage1")


def convert_stage2_text() -> tuple[int, int]:
    """Convert Stage 2 PubMed + climbmix docs to parquet."""
    in_path = Path(config.STAGE2_SPLITS_DIR) / "stage2_docs.txt"
    rows = _read_text_lines(in_path)

    out_dir = Path(config.PARQUET_DIR) / "stage2"
    print(f"\nStage 2 text: {in_path}")
    print(f"Saving to: {out_dir}")
    return _write_text_shards(rows, out_dir, "stage2")


def convert_stage3_splits() -> tuple[int, int]:
    """Convert Stage 3 JSONL conversation splits to parquet text rows."""
    out_dir = Path(config.PARQUET_DIR) / "stage3"
    total_rows = 0
    total_chars = 0

    print(f"\nStage 3 conversations: {config.STAGE3_SPLITS_DIR}")
    print(f"Saving to: {out_dir}")

    for split in ("train", "val", "test"):
        in_path = Path(config.STAGE3_SPLITS_DIR) / f"{split}.jsonl"
        if not in_path.exists():
            print(f"  {split}: missing {in_path} - skipped")
            continue

        rows = []
        with in_path.open(encoding="utf-8") as handle:
            for raw in handle:
                row = _conversation_to_json(raw.strip())
                if row:
                    rows.append(row)

        if not rows:
            print(f"  {split}: no valid conversations - skipped")
            continue

        row_count, char_count = _write_text_shards(rows, out_dir, split)
        total_rows += row_count
        total_chars += char_count

    return total_rows, total_chars


def _print_summary(results: list[tuple[str, int, int]]) -> None:
    print("\nSummary")
    print("=" * 60)
    total_rows = 0
    total_chars = 0

    for name, rows, chars in results:
        total_rows += rows
        total_chars += chars
        print(f"  {name:<18} {rows:>10,} rows  {chars / 1e6:>8.1f} MB")

    print("-" * 60)
    print(f"  {'total':<18} {total_rows:>10,} rows  {total_chars / 1e6:>8.1f} MB")
    print(f"\nParquet root: {Path(config.PARQUET_DIR)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optionally export 3-stage pipeline artifacts to parquet."
    )
    parser.add_argument("--tokenizer", action="store_true", help="Convert tokenizer_text.txt.")
    parser.add_argument("--stage1", action="store_true", help="Convert Stage 1 text docs.")
    parser.add_argument("--stage2", action="store_true", help="Convert Stage 2 text docs.")
    parser.add_argument("--stage3", action="store_true", help="Convert Stage 3 JSONL splits.")
    parser.add_argument("--all", action="store_true", help="Convert all available artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_all = args.all or not any(
        [args.tokenizer, args.stage1, args.stage2, args.stage3]
    )

    results: list[tuple[str, int, int]] = []

    print("\nOptional parquet export")
    print("=" * 60)
    print("Note: scripts/train_3stage.py does not require these parquet files.")

    if run_all or args.tokenizer:
        results.append(("tokenizer", *convert_tokenizer_corpus()))
    if run_all or args.stage1:
        results.append(("stage1", *convert_stage1_text()))
    if run_all or args.stage2:
        results.append(("stage2", *convert_stage2_text()))
    if run_all or args.stage3:
        results.append(("stage3", *convert_stage3_splits()))

    _print_summary(results)


def convert_to_parquet() -> None:
    """Backward-compatible entry point expected by chat_model.tokenizing.__init__."""
    main()


if __name__ == "__main__":
    main()
