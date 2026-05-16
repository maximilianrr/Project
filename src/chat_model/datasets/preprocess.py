# src/chat_model/datasets/preprocess.py
"""Clean, tone-normalise, upsample, deduplicate, and split medical fine-tuning data.

Sources:
  - MedDialog (ChatDoctor-HealthCareMagic) — gold, real patient/doctor conversations
  - MedQuAD   — NIH factual medical Q&A
  - MEDIQA-Chat — clinical dialogues (optional)

Excluded:
  - MedMCQA  — exam language, not patient language
  - PubMedQA — research paper titles as questions

Key design decisions for NanoChat:
  - Tight length filters: only examples the model can actually learn from
  - MedDialog upsampled 3x: real conversation dominates training signal
  - Tone normalisation: every answer sounds like the same doctor voice
  - Generic non-answer filter: drops useless filler responses
  - Hard cap at MAX_EXAMPLES: quality over quantity for small models
"""

from __future__ import annotations

import glob
import json
import random
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk
import pyarrow.parquet as pq

from chat_model import config

random.seed(config.RANDOM_SEED)

# ── Noise patterns ─────────────────────────────────────────────────────────────

_EMAIL     = re.compile(r"\S+@\S+")
_HTML_TAG  = re.compile(r"<[^>]{1,100}>")
_URL       = re.compile(r"https?://\S+|www\.\S+")
_BRACKET   = re.compile(r"\[.*?\]|\(fig\.?[\s\d]+\)", re.IGNORECASE)
_MULTI_WS  = re.compile(r"[ \t]+")
_MULTI_NL  = re.compile(r"\n{3,}")
_ENDS_PUNC = re.compile(r"[.!?]$")

# ── Branding / greeting / closing strip ───────────────────────────────────────

_BRAND_NAMES = re.compile(
    r"chat\s*doctor|healthcaremagic|health\s*care\s*magic", re.IGNORECASE
)
_FUSED_WORD = re.compile(r"\b([A-Z][a-z]+)([A-Z])")
_GREETING_LINE = re.compile(
    r"^(hi|hello|dear|welcome|thanks?\s+for\s+(posting|using|contacting|writing|reaching|choosing)|"
    r"good\s+(morning|afternoon|evening)|thank\s+you\s+for|howell\s+come|i\s+appreciate\s+you)[^.!?]*[.!?]\s*",
    re.IGNORECASE,
)
_CLOSING_FILLER = re.compile(
    r"("
    r"hope\s+(this\s+)?(helps?|answers?|clears?|resolves?)[^.!?]*[.!?]\s*"
    r"|i\s+wish\s+you[^.!?]*[.!?]\s*"
    r"|wish(ing)?\s+you[^.!?]*[.!?]\s*"
    r"|wishing\s+you\s+(a\s+)?(speedy\s+)?recovery[^.!?]*[.!?]\s*"
    r"|feel\s+free\s+to\s+(ask|consult)[^.!?]*[.!?]\s*"
    r"|i\s+(will\s+be|am)\s+happy\s+to\s+(help|answer)[^.!?]*[.!?]\s*"
    r"|let\s+me\s+know\s+if\s+(you\s+have\s+)?any\s+(other\s+)?questions?[^.!?]*[.!?]\s*"
    r"|if\s+you\s+have\s+any\s+(further|more|other)\s+(queries|questions|doubts)[^.!?]*[.!?]\s*"
    r"|regards[.!,]?\s*"
    r"|take\s+care[.!]?\s*"
    r"|thanks?\s+for\s+(using|choosing|writing|consulting)[^.!?]*[.!?]\s*"
    r"|thank\s+you\s+for\s+your\s+(query|question|trust)[^.!?]*[.!?]\s*"
    r"|\b[A-Z][a-z]{2,15}\.\s*$"
    r")",
    re.IGNORECASE,
)
_GENERIC_NONANSWER = re.compile(
    r"^(based on (the|your) history|"
    r"i (have )?(gone through|read) your (history|query|question)|"
    r"i (can |do )?understand your (concern|anxiety|worry)|"
    r"consult (a |your )?(doctor|physician|specialist))[^.!?]*[.!?]\s*$",
    re.IGNORECASE,
)

# ── Tone normalisation ────────────────────────────────────────────────────────

_TONE_REPLACEMENTS = [
    (r"\bthe patient should\b",          "you should",               re.IGNORECASE),
    (r"\bthe patient (can|may|must)\b",  r"you \1",                  re.IGNORECASE),
    (r"\bpatients should\b",             "you should",               re.IGNORECASE),
    (r"\bpatients (can|may|must)\b",     r"you \1",                  re.IGNORECASE),
    (r"\bin patients\b",                 "in people",                re.IGNORECASE),
    (r"\bthe patient\b",                 "you",                      re.IGNORECASE),
    (r"\bpatients\b",                    "people",                   re.IGNORECASE),
    (r"\bit is recommended( that)?\b",   "I'd recommend",            re.IGNORECASE),
    (r"\bit is advised( that)?\b",       "I'd advise",               re.IGNORECASE),
    (r"\bone should\b",                  "you should",               re.IGNORECASE),
    (r"\bmay be administered\b",         "can be given",             re.IGNORECASE),
    (r"\bis indicated\b",                "is the right approach",    re.IGNORECASE),
    (r"\bcontraindicated\b",             "not recommended",          re.IGNORECASE),
    (r"\bthe study (found|showed|demonstrated)\b", "research shows", re.IGNORECASE),
    (r"\bstatistically significant\b",   "meaningful",               re.IGNORECASE),
    (r"\betiology\b",                    "cause",                    re.IGNORECASE),
    (r"\bpathophysiology\b",             "how this condition works",  re.IGNORECASE),
    (r"\bpresents with\b",               "has symptoms of",          re.IGNORECASE),
    (r"\bpresentation\b",                "symptoms",                 re.IGNORECASE),
    (r"\bcomorbid(ity|ities)?\b",        "other health conditions",  re.IGNORECASE),
    (r"\bprognosis\b",                   "outlook",                  re.IGNORECASE),
    (r"\bprophylaxis\b",                 "prevention",               re.IGNORECASE),
    (r"\badminister(ed|ing)?\b",         "give",                     re.IGNORECASE),
]
_TONE_PATTERNS = [
    (re.compile(pat, flags), repl)
    for pat, repl, flags in _TONE_REPLACEMENTS
]

# ── Safety examples injected into every training run ──────────────────────────

_SAFETY_EXAMPLES = [
    (
        "I have crushing chest pain and shortness of breath. What should I do?",
        "Chest pain with shortness of breath can be an emergency. Please call emergency services now or go to the nearest emergency department. Do not try to manage this at home.",
    ),
    (
        "Can you diagnose this rash from my description?",
        "I cannot diagnose you from a message alone. A clinician needs to examine the rash and ask about timing, exposures, medicines, fever, pain, and other symptoms. Seek urgent care if it spreads quickly, is painful, involves the eyes or mouth, or comes with fever or trouble breathing.",
    ),
    (
        "Should I stop my prescription medicine because I feel better?",
        "Do not stop a prescribed medicine without speaking with your clinician or pharmacist, especially antibiotics, heart medicines, blood pressure medicines, steroids, seizure medicines, or mental health medicines. They can tell you whether it is safe to stop or taper.",
    ),
]


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _strip_greeting_and_closing(text: str) -> str:
    text = _BRAND_NAMES.sub("", text)
    text = _FUSED_WORD.sub(lambda m: m.group(1) + " " + m.group(2), text)
    for _ in range(3):
        text = _GREETING_LINE.sub("", text)
    for _ in range(3):
        text = _CLOSING_FILLER.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^[,\s]+", "", text)
    return text.strip()


def _is_generic_nonanswer(text: str) -> bool:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return not sentences or all(_GENERIC_NONANSWER.match(s) for s in sentences)


def _apply_tone(text: str) -> str:
    for pattern, replacement in _TONE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _fix_punctuation(text: str) -> str:
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])\s{2,}", r"\1 ", text)
    return text


def _capitalise_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(p[:1].upper() + p[1:] if p else p for p in parts)


def _ensure_sentence_end(text: str) -> str:
    text = text.rstrip()
    return text + "." if text and not _ENDS_PUNC.search(text) else text


def clean(text: str, is_answer: bool = False) -> str:
    """Full cleaning pipeline for questions and answers."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _EMAIL.sub("", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub("", text)
    text = _BRACKET.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text).strip()
    if is_answer:
        text = _strip_greeting_and_closing(text)
        text = _apply_tone(text)
    text = _fix_punctuation(text)
    text = _capitalise_sentences(text)
    text = _ensure_sentence_end(text)
    return text.strip()


def _is_valid(question: str, answer: str) -> bool:
    if not (config.MIN_Q_CHARS <= len(question) <= config.MAX_Q_CHARS):
        return False
    if not (config.MIN_A_CHARS <= len(answer) <= config.MAX_A_CHARS):
        return False
    if answer.lower().strip() == question.lower().strip():
        return False
    if len(answer.split()) < 10:
        return False
    if _is_generic_nonanswer(answer):
        return False
    return True


def _to_conversation(question: str, answer: str) -> list[dict]:
    return [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_pretrain_parquet(parquet_dir: str) -> list:
    """Load OpenAssistant pretraining pairs from parquet shards into conversations."""

    shard_paths = sorted(Path(parquet_dir).glob("oasst2_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No oasst2 parquet shards found in {parquet_dir}. Run download_data() first."
        )

    conversations = []
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["question", "answer"])
        questions = table.column("question").to_pylist()
        answers = table.column("answer").to_pylist()

        for question, answer in zip(questions, answers):
            q = clean(question or "", is_answer=False)
            a = clean(answer or "", is_answer=True)
            if len(q) < 20 or len(a) < 40:
                continue
            if q.lower().strip() == a.lower().strip():
                continue
            conversations.append(_to_conversation(q, a))

    print(f"  oasst2: {len(conversations):>7,} examples from {len(shard_paths)} shard(s)")
    return conversations

def _load_meddialog(path: str) -> list:
    if not Path(path).exists():
        raise FileNotFoundError(f"MedDialog not found at {path}. Run download_data() first.")
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        row = cast(dict[str, Any], item)
        q = clean((row.get("input") or ""), is_answer=False)
        a = clean((row.get("output") or ""), is_answer=True)
        if _is_valid(q, a):
            pairs.append(_to_conversation(q, a))
    upsampled = pairs.copy()
    for _ in range(max(config.MEDDIALOG_UPSAMPLE - 1, 0)):
        sample = pairs.copy()
        random.shuffle(sample)
        upsampled.extend(sample)
    print(f"  MedDialog: {len(pairs):>7,} unique → {len(upsampled):>7,} after {config.MEDDIALOG_UPSAMPLE}x upsample")
    return upsampled


def _load_medquad(path: str) -> list:
    if not Path(path).exists():
        print(f"  MedQuAD: missing {path} — skipped")
        return []
    pairs = []
    for xml_file in glob.glob(str(Path(path) / "**" / "*.xml"), recursive=True):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue
        for qa in root.findall(".//QAPair"):
            q_el = qa.find("Question")
            a_el = qa.find("Answer")
            if a_el is None or not (a_el.text or "").strip():
                continue
            q_text = q_el.text if q_el is not None and q_el.text is not None else ""
            q = clean(q_text, is_answer=False)
            a = clean(a_el.text or "", is_answer=True)
            if _is_valid(q, a):
                pairs.append(_to_conversation(q, a))
    print(f"  MedQuAD:   {len(pairs):>7,} examples")
    return pairs


def _load_mediqa(path: str) -> list:
    if not Path(path).exists():
        print("  MEDIQA:    not found — skipped (optional)")
        return []
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        row = cast(dict[str, Any], item)
        dialogue = row.get("dialogue") or ""
        summary  = clean((row.get("note") or row.get("summary") or ""), is_answer=True)
        if not dialogue or not summary:
            continue
        q = clean("Based on this conversation:\n" + dialogue, is_answer=False)
        if _is_valid(q, summary):
            pairs.append(_to_conversation(q, summary))
    print(f"  MEDIQA:    {len(pairs):>7,} examples")
    return pairs


def _deduplicate(data: list) -> list:
    counts: dict = {}
    unique = []
    for conv in data:
        key = (conv[0]["content"].lower().strip(), conv[1]["content"].lower().strip())
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= config.MEDDIALOG_UPSAMPLE:
            unique.append(conv)
    return unique


def _save_splits(data: list, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    random.shuffle(data)

    if config.MAX_EXAMPLES and len(data) > config.MAX_EXAMPLES:
        print(f"  Capping {len(data):,} → {config.MAX_EXAMPLES:,} examples")
        data = data[:config.MAX_EXAMPLES]

    n       = len(data)
    train   = data[:int(n * config.TRAIN_RATIO)]
    val     = data[int(n * config.TRAIN_RATIO):int(n * (config.TRAIN_RATIO + config.VAL_RATIO))]
    test    = data[int(n * (config.TRAIN_RATIO + config.VAL_RATIO)):]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = Path(out_dir) / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  {name:<5}: {len(split):>7,} examples → {out_path}")


def preprocess_pretrain() -> None:
    """Preprocess OpenAssistant parquet shards into pretraining train/val/test JSONL splits."""

    print("\nPreprocessing pretraining data")
    print("=" * 48)

    data = _load_pretrain_parquet(config.PRETRAIN_PARQUET_DIR)
    _save_splits(data, config.PRETRAIN_SPLITS_DIR)

    total = sum(
        sum(1 for _ in open(Path(config.PRETRAIN_SPLITS_DIR) / f"{s}.jsonl", encoding="utf-8"))
        for s in ("train", "val", "test")
    )
    print(f"\nTotal examples : {total:,}")
    print("\nDone.")


# ── Entry point ───────────────────────────────────────────────────────────────

def start_preprocess() -> None:
    """Preprocess all medical sources into train/val/test JSONL splits."""
    print("\nPreprocessing medical fine-tuning data")
    print("=" * 48)

    meddialog = _load_meddialog(config.MEDDIALOG_DIR)
    others    = _load_medquad(config.MEDQUAD_DIR) + _load_mediqa(config.MEDIQA_DIR)
    data      = meddialog + others

    # Add hardcoded safety examples
    for q, a in _SAFETY_EXAMPLES:
        data.append(_to_conversation(clean(q), clean(a, is_answer=True)))
    print(f"  Safety:    {len(_SAFETY_EXAMPLES):>7,} examples")

    before = len(data)
    data   = _deduplicate(data)
    print(f"\nAfter dedup: {before:,} → {len(data):,} ({before - len(data):,} removed)")

    print("\nSaving splits")
    _save_splits(data, config.SPLITS_DIR)

    total = sum(
        sum(1 for _ in open(Path(config.SPLITS_DIR) / f"{s}.jsonl", encoding="utf-8"))
        for s in ("train", "val", "test")
    )
    print(f"\nTotal examples : {total:,}")
    print(f"MedDialog weight: {config.MEDDIALOG_UPSAMPLE}x upsampled")
    print("\nDone.")


if __name__ == "__main__":
    start_preprocess()
