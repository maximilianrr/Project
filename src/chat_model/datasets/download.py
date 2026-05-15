# src/chat_model/datasets/download.py
"""Download all raw data sources for both training stages.

Stage 1 (general pretraining): OpenAssistant oasst2
Stage 2 (medical fine-tuning): MedDialog, MedQuAD, WTND, MEDIQA-Chat (optional)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from chat_model import config


# ── Stage 1 — oasst2 ──────────────────────────────────────────────────────────

_MULTI_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{4,}")
_URL       = re.compile(r"https?://\S+|www\.\S+")

LANG        = "en"
MIN_Q_CHARS = 20
MIN_A_CHARS = 40
MAX_Q_CHARS = 2000
MAX_A_CHARS = 3000


def _clean_oasst(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _URL.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n\n", text)
    return text.strip()


def download_pretrain_data() -> None:
    """
    Downloads OpenAssistant oasst2 and saves as parquet shards for Stage 1.
    Reconstructs Q/A pairs from the message tree using parent_id.
    English only, high-quality messages only.
    """
    out_dir = Path(config.PRETRAIN_PARQUET_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    existing = list(out_dir.glob("oasst2_*.parquet"))
    if existing:
        print(f"  oasst2: found {len(existing)} existing shard(s) — skipping download.")
        return

    print("Downloading OpenAssistant/oasst2...")
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    print(f"  Total messages: {len(ds):,}")

    # Build message index for parent lookup
    msg_index = {row["message_id"]: row for row in ds}

    pairs = []
    skipped = {"lang": 0, "quality": 0, "deleted": 0, "length": 0, "no_parent": 0}

    for msg in ds:
        if msg.get("role") != "assistant":
            continue
        if msg.get("lang") != LANG:
            skipped["lang"] += 1
            continue
        if msg.get("deleted", False):
            skipped["deleted"] += 1
            continue
        if not msg.get("review_result", True):
            skipped["quality"] += 1
            continue

        parent_id = msg.get("parent_id")
        parent    = msg_index.get(parent_id) if parent_id else None
        if parent is None or parent.get("role") != "prompter":
            skipped["no_parent"] += 1
            continue
        if parent.get("lang") != LANG or parent.get("deleted", False):
            skipped["lang"] += 1
            continue

        question = _clean_oasst(parent.get("text") or "")
        answer   = _clean_oasst(msg.get("text") or "")

        if not (MIN_Q_CHARS <= len(question) <= MAX_Q_CHARS):
            skipped["length"] += 1
            continue
        if not (MIN_A_CHARS <= len(answer) <= MAX_A_CHARS):
            skipped["length"] += 1
            continue

        pairs.append({"question": question, "answer": answer})

    print(f"  Pairs extracted: {len(pairs):,}")
    for label, count in skipped.items():
        if count:
            print(f"  Skipped {label}: {count:,}")

    schema = pa.schema([pa.field("question", pa.string()), pa.field("answer", pa.string())])
    shards = [pairs[i:i + config.SHARD_SIZE] for i in range(0, len(pairs), config.SHARD_SIZE)]

    for i, shard in enumerate(shards):
        out_path = out_dir / f"oasst2_{i:05d}.parquet"
        table    = pa.table(
            {"question": [p["question"] for p in shard],
             "answer":   [p["answer"]   for p in shard]},
            schema=schema,
        )
        pq.write_table(table, out_path, compression="snappy")
        print(f"  oasst2_{i:05d}.parquet  {len(shard):,} pairs")

    print(f"  Done. {len(pairs):,} oasst2 pairs saved to {out_dir}/")


# ── Stage 2 — medical sources ─────────────────────────────────────────────────

def download_meddialog() -> None:
    path = Path(config.MEDDIALOG_DIR)
    if path.exists():
        print(f"  MedDialog: already exists — skipping.")
        return
    print("  Downloading MedDialog (ChatDoctor-HealthCareMagic-100k)...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
    ds.save_to_disk(str(path))
    print(f"  Done. {len(ds):,} examples.")


def download_medquad() -> None:
    path = Path(config.MEDQUAD_DIR)
    if path.exists():
        print(f"  MedQuAD: already exists — skipping.")
        return
    if not shutil.which("git"):
        raise RuntimeError("git is required to clone MedQuAD.")
    print("  Cloning MedQuAD...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/abachaa/MedQuAD.git", str(path)],
        check=True,
    )
    print(f"  Done.")


def download_mediqa() -> None:
    path = Path(config.MEDIQA_DIR)
    if path.exists():
        print(f"  MEDIQA-Chat: already exists — skipping.")
        return
    try:
        print("  Downloading MEDIQA-Chat (optional)...")
        ds = load_dataset("chiusers/mediqa-chat-2023", split="train")
        ds.save_to_disk(str(path))
        print(f"  Done. {len(ds):,} examples.")
    except Exception as exc:
        print(f"  MEDIQA-Chat unavailable — skipping. ({exc})")


def download_wtnd() -> None:
    """Downloads the WTND PDF, extracts and cleans text for tokenizer vocabulary."""
    pdf_path = Path(config.WTND_PDF)
    out_path = Path(config.WTND_CLEAN)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        print("  Downloading Where There Is No Doctor PDF...")
        url = (
            "https://ia601902.us.archive.org/24/items/"
            "WhereThereIsNoDoctor-English-DavidWerner/"
            "14.DavidWerner-WhereThereIsNoDoctor.pdf"
        )
        urllib.request.urlretrieve(url, str(pdf_path))
        print(f"  Saved to {pdf_path}")
    else:
        print(f"  WTND PDF: already exists — skipping download.")

    print("  Extracting and cleaning WTND text...")
    with fitz.open(str(pdf_path)) as doc:
        raw_text = "\n".join(page.get_text() for page in doc)
        print(f"  Extracted {len(raw_text):,} chars from {len(doc)} pages")

    # Clean: remove short lines, page numbers, low-alpha lines, fix hyphenation
    clean_lines = []
    pending = ""
    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or len(line) < 25:
            continue
        if re.fullmatch(r"[\d\s.\-]+", line):
            continue
        if sum(c.isalpha() for c in line) / max(len(line), 1) < 0.60:
            continue
        if pending:
            line = pending + line
            pending = ""
        if line.endswith("-"):
            pending = line[:-1]
            continue
        clean_lines.append(line)

    clean_text = re.sub(r"\n{3,}", "\n\n", "\n".join(clean_lines)).strip()
    out_path.write_text(clean_text, encoding="utf-8")
    print(f"  Saved {clean_text.count(chr(10)) + 1:,} lines to {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def download_data() -> None:
    """Downloads all data sources for Stage 1 and Stage 2."""
    Path(config.RAW_DIR).mkdir(parents=True, exist_ok=True)

    print("\nStage 1 — Pretraining data")
    print("=" * 48)
    download_pretrain_data()

    print("\nStage 2 — Medical fine-tuning data")
    print("=" * 48)
    download_meddialog()
    download_medquad()
    download_mediqa()
    download_wtnd()

    print("\nAll data downloaded.")


if __name__ == "__main__":
    download_data()
