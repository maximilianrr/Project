# src/chat_model/datasets/preprocess.py
"""Clean, tone-normalise, upsample, deduplicate, and split training data.

Stage 1 — Pretraining sources (English only, no medical bias):
  - karpathy/climbmix-400b-shuffle  — raw general English documents (ChunkTextDataset)
  - OpenAssistant/oasst2            — general English Q/A conversations (ChunkChatDataset)

Stage 2 — Medical fine-tuning sources:
  - MedDialog (ChatDoctor-HealthCareMagic) — real patient/doctor conversations
  - MedQuAD   — NIH factual medical Q&A
  - MEDIQA-Chat — clinical dialogues (optional)
  - PubMed abstracts — dense medical vocabulary (text documents, not Q/A)

Excluded from Stage 1:
  - PubMed  — research prose biases general English; move to Stage 2 only
  - MedMCQA  — exam language, not patient language
  - PubMedQA — research paper titles as questions
"""

from __future__ import annotations

import glob
import itertools
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

# ── Unsafe content filter ─────────────────────────────────────────────────────

_UNSAFE = re.compile(r"""
    \b(take\s+\d+\s*(mg|ml|mcg|g|tablet|pill|drop|dose|unit)s?\b
    | \d+\s*(mg|ml|mcg|g)\s+of\b | dose\s+of\s+\d+ | \d+\s*x\s*\d+\s*(mg|ml)
    | twice\s+daily\s+\d+ | \d+\s+times?\s+(a\s+)?(day|daily|week)
    | safe\s+to\s+take | is\s+completely\s+safe | no\s+(risk|side\s*effects?|danger)
    | perfectly\s+safe | absolutely\s+safe | will\s+not\s+cause\s+any\s+(harm|damage|problem)
    | you\s+can\s+safely\s+(take|use|mix|combine)
    | (safe\s+to|okay\s+to|can)\s+(mix|combine|take)\s+with\s+alcohol
    | (safe|ok|okay)\s+with\s+alcohol | no\s+interaction\s+with | does\s+not\s+interact
    | (definitely|certainly|absolutely)\s+(not\s+)?(serious|cancer|benign|safe|normal)
    | (this\s+is\s+)?definitely\s+just | nothing\s+to\s+worry\s+about
    | (there\s+is|there's)\s+no\s+cause\s+for\s+(concern|alarm|worry)
    | guaranteed\s+to\s+(work|help|cure))\b""", re.IGNORECASE | re.VERBOSE)

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

# ── Epistemic hedging ─────────────────────────────────────────────────────────

_OVERCONFIDENT = re.compile(
    r"^(You have\b|You are\b|You definitely\b|You certainly\b|You should definitely\b"
    r"|The cause is\b|The diagnosis is\b|The answer is\b)", re.IGNORECASE
)
_ALREADY_HEDGED = re.compile(
    r"^(it('s| is) (possible|worth|important)|this (could|may|might)|one possibility"
    r"|based on|i (cannot|can't|am not able)|without (examining|seeing|knowing)"
    r"|please (consult|see|speak))", re.IGNORECASE
)
_HAS_REFERRAL = re.compile(
    r"(consult|see|speak\s+with|talk\s+to|visit|call)\s+(a|your)?\s*"
    r"(doctor|physician|clinician|specialist|pharmacist|provider|emergency)", re.IGNORECASE
)

_HEDGES = [
    "Based on what you've described, it's possible that ",
    "This could suggest ",
    "One possibility is that ",
    "It may be worth considering that ",
]
_REFERRALS = [
    " A clinician should evaluate this to be certain.",
    " Please consult your doctor or pharmacist before making any changes.",
    " This is general information — your own clinician knows your full history.",
    " If symptoms are severe or worsening, seek medical attention promptly.",
]

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

# ── Safety examples ───────────────────────────────────────────────────────────

_SAFETY_EXAMPLES: list[tuple[str, str]] = [
    ("I have crushing chest pain and shortness of breath. What should I do?",
     "Chest pain with shortness of breath can be life-threatening. Call emergency services immediately. Do not drive yourself or wait."),
    ("I took an entire bottle of sleeping pills. I feel dizzy.",
     "This is a medical emergency. Call emergency services now. In the US call Poison Control at 1-800-222-1222."),
    ("One side of my face is drooping and I can't lift my arm properly.",
     "These are stroke warning signs. Call emergency services immediately — time is critical."),
    ("I'm having trouble breathing and my lips are turning blue.",
     "Difficulty breathing with blue lips is a medical emergency. Call emergency services now."),
    ("I have severe abdominal pain that came on suddenly and isn't going away.",
     "Sudden severe abdominal pain can indicate appendicitis or another emergency. Go to an emergency department now."),
    ("My child has a seizure and won't wake up properly.",
     "A seizure followed by unresponsiveness is an emergency. Call emergency services and keep your child on their side."),
    ("I think I'm having an allergic reaction — my throat feels tight.",
     "A tight throat can progress to anaphylaxis. Use an EpiPen if available and call emergency services immediately."),
    ("I have severe chest pressure radiating to my left arm and jaw.",
     "This is a classic heart attack warning sign. Call emergency services immediately."),
    ("Can you diagnose this rash from my description?",
     "I cannot diagnose from a message alone. Please see a clinician, urgently if the rash spreads quickly or comes with fever."),
    ("Based on my symptoms, do I have diabetes?",
     "Diagnosis requires blood tests and clinical assessment. Please speak with your doctor."),
    ("Do you think this mole is skin cancer?",
     "I cannot evaluate a mole from a text description. Any mole that changes in size, shape, or colour should be seen by a doctor."),
    ("I've been having headaches every day for two weeks. Is this a brain tumour?",
     "I'm not able to diagnose your headaches. If they've continued for two weeks, please make an appointment with your doctor."),
    ("My urine is dark. Could this be kidney failure?",
     "Dark urine has many causes. I cannot determine the cause remotely — please see your doctor."),
    ("I've had a cough for six weeks. Is it lung cancer?",
     "A cough lasting six weeks should be assessed by a doctor. I cannot diagnose you remotely."),
    ("Should I stop my prescription medicine because I feel better?",
     "Please do not stop a prescribed medicine without speaking to your clinician first."),
    ("Can I double my dose if I missed yesterday's tablet?",
     "Do not double your dose without checking with your pharmacist or prescriber first."),
    ("I want to stop my antidepressants. Can I just stop taking them?",
     "Stopping antidepressants abruptly can cause discontinuation symptoms. Please speak with your prescriber before stopping."),
    ("My blood pressure tablets are making me feel tired. Should I halve the dose?",
     "Please do not adjust your blood pressure medication on your own. Contact your prescriber."),
    ("I feel tired all the time. What is wrong with me?",
     "Persistent tiredness has many causes. Please see your doctor for a proper assessment."),
    ("I've been feeling sad and hopeless for months. Is this depression?",
     "What you're describing deserves proper support. Please speak with a doctor or mental health professional."),
    ("Is it normal to feel my heart skipping beats?",
     "Occasional skipped beats are common, but frequent palpitations with dizziness should be assessed by a doctor."),
    ("I've had back pain for three months. What is causing it?",
     "Three months of back pain warrants a doctor's assessment. I cannot tell you the cause remotely."),
    ("Can I take ibuprofen with my blood pressure tablets?",
     "Ibuprofen can reduce blood pressure treatment effectiveness. Please ask your pharmacist before combining them."),
    ("Is it safe to take paracetamol and ibuprofen together?",
     "Whether this is appropriate depends on your history. Please confirm with your pharmacist."),
    ("Can I drink alcohol while on antibiotics?",
     "It depends on the antibiotic — with metronidazole alcohol causes a serious reaction. Check the leaflet or ask your pharmacist."),
    ("I'm on warfarin. Can I take aspirin for a headache?",
     "Do not take aspirin with warfarin without checking with your doctor — it significantly increases bleeding risk."),
    ("I've been having thoughts of hurting myself. What should I do?",
     "Please reach out now — call a crisis line (UK: 116 123, US: 988), go to your nearest emergency department, or call someone you trust."),
    ("I feel like there's no point in living anymore.",
     "Please talk to someone right now. UK: Samaritans 116 123. US: call or text 988. You deserve support."),
    ("Can you prescribe medication for my anxiety?",
     "I cannot prescribe medication. Please speak with your GP or a psychiatrist."),
    ("Can you order a blood test for me?",
     "I cannot order blood tests. Please see your doctor."),
    ("What exact dose of ibuprofen should I take for my pain?",
     "I cannot advise on specific doses. Please follow the packaging instructions or ask your pharmacist."),
    ("My doctor said I need surgery. Is that really necessary?",
     "I cannot evaluate whether surgery is necessary. It's reasonable to ask your doctor to explain or seek a second opinion."),
    ("Can you read my X-ray results?",
     "I cannot interpret imaging. Please discuss your results with the ordering doctor."),
    ("Is my BMI healthy?",
     "BMI has well-known limitations. Please speak with your GP who can assess your weight in full context."),
    ("I found a lump in my breast. Is it cancer?",
     "I cannot determine this from a description. Any new lump should be assessed by a doctor promptly."),
    ("My child has a fever of 40°C. What medicine should I give?",
     "A 40°C fever needs careful assessment. Follow packaging instructions for paracetamol or ibuprofen, and seek care if your child is very unwell."),
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


def _hedge(answer: str) -> str:
    if _ALREADY_HEDGED.match(answer.strip()):
        return answer
    if _OVERCONFIDENT.match(answer.strip()):
        text = random.choice(_HEDGES) + answer[0].lower() + answer[1:]
        if not _HAS_REFERRAL.search(text):
            text = text.rstrip(" .") + "." + random.choice(_REFERRALS)
        return text
    return answer


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
        text = _hedge(text)
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
    if _UNSAFE.search(answer):
        return False
    return True


def _to_conversation(question: str, answer: str) -> list[dict]:
    return [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(data: list) -> list:
    try:
        from datasketch import MinHash, MinHashLSH
        lsh = MinHashLSH(threshold=0.7, num_perm=128)
        kept = []
        for i, conv in enumerate(data):
            text = re.sub(r"\s+", " ", (conv[0]["content"] + " " + conv[1]["content"]).lower())
            grams: dict[str, int] = {}
            for j in range(len(text) - 3):
                g = text[j:j + 4]
                grams[g] = grams.get(g, 0) + 1
            mh = MinHash(num_perm=128)
            for g in sorted(grams, key=grams.__getitem__, reverse=True)[:128]:
                mh.update(g.encode())
            if not lsh.query(mh):
                lsh.insert(f"i{i}", mh)
                kept.append(conv)
        return kept
    except ImportError:
        print("  datasketch not found — falling back to exact dedup (pip install datasketch)")
        seen: set[str] = set()
        kept = []
        for conv in data:
            key = conv[0]["content"].strip().lower()
            if key not in seen:
                seen.add(key)
                kept.append(conv)
        return kept


# ── Stage 1 loaders ───────────────────────────────────────────────────────────

def _load_climbmix_documents(parquet_dir: str) -> list[str]:
    """
    Load raw text documents from climbmix parquet shards.
    Returns a list of strings (one per document) for use with ChunkTextDataset.
    """
    shard_paths = sorted(Path(parquet_dir).glob("climbmix_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No climbmix parquet shards found in {parquet_dir}. Run download_data() first."
        )
    documents = []
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["text"])
        for text in table.column("text").to_pylist():
            doc = (text or "").strip()
            if doc:
                documents.append(doc)

    print(f"  climbmix: {len(documents):>7,} documents from {len(shard_paths)} shard(s)")
    return documents


def _load_oasst2_conversations(parquet_dir: str) -> list:
    """
    Load oasst2 Q/A pairs from parquet shards into conversation dicts.
    Returns a list of conversations for use with ChunkChatDataset.
    """
    shard_paths = sorted(Path(parquet_dir).glob("oasst2_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No oasst2 parquet shards found in {parquet_dir}. Run download_data() first."
        )
    conversations = []
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["question", "answer"])
        for question, answer in zip(
            table.column("question").to_pylist(),
            table.column("answer").to_pylist(),
        ):
            q = clean(question or "", is_answer=False)
            a = clean(answer or "", is_answer=True)
            if _is_valid(q, a):
                conversations.append(_to_conversation(q, a))

    print(f"  oasst2:   {len(conversations):>7,} conversations from {len(shard_paths)} shard(s)")
    return conversations


# ── Stage 2 loaders ───────────────────────────────────────────────────────────

def _load_meddialog(path: str) -> list:
    if not Path(path).exists():
        raise FileNotFoundError(f"MedDialog not found at {path}. Run download_data() first.")
    ds = load_from_disk(path)
    pairs, unsafe = [], 0
    for item in ds:
        row = cast(dict[str, Any], item)
        raw_a = row.get("output") or ""
        if _UNSAFE.search(raw_a):
            unsafe += 1
            continue
        q = clean(row.get("input") or "", is_answer=False)
        a = clean(raw_a, is_answer=True)
        if _is_valid(q, a):
            pairs.append(_to_conversation(q, a))

    upsample = max(getattr(config, "MEDDIALOG_UPSAMPLE", 1), 1)
    extra    = int(len(pairs) * (upsample - 1.0))
    upsampled = pairs + list(itertools.islice(itertools.cycle(pairs), extra)) if extra > 0 else pairs
    print(f"  MedDialog: {len(pairs):>7,} unique ({unsafe:,} unsafe) → {len(upsampled):>7,} after {upsample}x upsample")
    return upsampled


def _load_medquad(path: str) -> list:
    if not Path(path).exists():
        print(f"  MedQuAD: missing {path} — skipped")
        return []
    pairs, unsafe = [], 0
    for xml_file in glob.glob(str(Path(path) / "**" / "*.xml"), recursive=True):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue
        for qa in root.findall(".//QAPair"):
            a_el = qa.find("Answer")
            if a_el is None or not (a_el.text or "").strip():
                continue
            if _UNSAFE.search(a_el.text):
                unsafe += 1
                continue
            q_el = qa.find("Question")
            q_text = q_el.text if q_el is not None and q_el.text is not None else ""
            q = clean(q_text, is_answer=False)
            a = clean(a_el.text or "", is_answer=True)
            if _is_valid(q, a):
                pairs.append(_to_conversation(q, a))
    print(f"  MedQuAD:   {len(pairs):>7,} examples ({unsafe:,} unsafe)")
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
        summary  = clean(row.get("note") or row.get("summary") or "", is_answer=True)
        if not dialogue or not summary:
            continue
        q = clean("Based on this conversation:\n" + dialogue, is_answer=False)
        if _is_valid(q, summary):
            pairs.append(_to_conversation(q, summary))
    print(f"  MEDIQA:    {len(pairs):>7,} examples")
    return pairs


def _load_pubmed_documents(parquet_dir: str) -> list[str]:
    """
    Load PubMed title+abstract pairs as raw text documents for Stage 2.
    Returns a list of strings for use with ChunkTextDataset alongside
    the conversational fine-tuning data.
    """
    shard_paths = sorted(Path(parquet_dir).glob("pubmed_*.parquet"))
    if not shard_paths:
        print(f"  PubMed: NOT FOUND in {parquet_dir} — skipped (run download_data() first)")
        return []

    documents = []
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["title", "abstract"])
        for title, abstract in zip(
            table.column("title").to_pylist(),
            table.column("abstract").to_pylist(),
        ):
            title    = (title or "").strip()
            abstract = (abstract or "").strip()
            if abstract:
                doc = f"{title}\n{abstract}" if title else abstract
                documents.append(doc)

    print(f"  PubMed:    {len(documents):>7,} documents from {len(shard_paths)} shard(s)")
    return documents


# ── Save splits ───────────────────────────────────────────────────────────────

def _save_splits(data: list, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    random.shuffle(data)

    cap = getattr(config, "MAX_EXAMPLES", None)
    if cap and len(data) > cap:
        print(f"  Capping {len(data):,} → {cap:,} examples")
        data = data[:cap]

    n     = len(data)
    train = data[:int(n * config.TRAIN_RATIO)]
    val   = data[int(n * config.TRAIN_RATIO):int(n * (config.TRAIN_RATIO + config.VAL_RATIO))]
    test  = data[int(n * (config.TRAIN_RATIO + config.VAL_RATIO)):]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = Path(out_dir) / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in split)
        print(f"  {name:<5}: {len(split):>7,} examples → {out_path}")


def _save_text_documents(documents: list[str], out_dir: str, filename: str = "pretrain_docs.txt") -> None:
    """Save raw text documents to a plain text file, one document per line (newlines escaped)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / filename
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for doc in documents:
            # Replace internal newlines with a space so one line = one document
            f.write(doc.replace("\n", " ") + "\n")
    print(f"  Saved {len(documents):,} documents → {out_path}")


# ── Entry points ──────────────────────────────────────────────────────────────

def preprocess_pretrain() -> None:
    """
    Preprocess Stage 1 pretraining data:
      - climbmix documents  → saved as raw text list (for ChunkTextDataset)
      - oasst2 conversations → saved as JSONL splits (for ChunkChatDataset)
    """
    print("\nPreprocessing Stage 1 pretraining data")
    print("=" * 48)

    # Raw text documents (climbmix) → plain text file
    print("\n[climbmix — raw text documents]")
    climbmix_docs = _load_climbmix_documents(config.CLIMBMIX_PARQUET_DIR)
    _save_text_documents(climbmix_docs, config.PRETRAIN_SPLITS_DIR, filename="climbmix_docs.txt")

    # Conversational data (oasst2) → JSONL splits
    print("\n[oasst2 — conversational Q/A]")
    oasst2_convs = _load_oasst2_conversations(config.PRETRAIN_PARQUET_DIR)
    _save_splits(oasst2_convs, config.PRETRAIN_SPLITS_DIR)

    total_docs  = len(climbmix_docs)
    total_convs = sum(
        sum(1 for _ in open(Path(config.PRETRAIN_SPLITS_DIR) / f"{s}.jsonl", encoding="utf-8"))
        for s in ("train", "val", "test")
    )
    print(f"\nTotal climbmix documents : {total_docs:,}")
    print(f"Total oasst2 examples    : {total_convs:,}")
    print("\nDone.")


def start_preprocess() -> None:
    """
    Preprocess Stage 2 medical fine-tuning data:
      - MedDialog, MedQuAD, MEDIQA-Chat conversational Q/A → JSONL splits (ChunkChatDataset)
      - PubMed abstracts are available as raw documents for supplementary vocab exposure
    """
    print("\nPreprocessing Stage 2 medical fine-tuning data")
    print("=" * 48)

    data = (
        _load_meddialog(config.MEDDIALOG_DIR)
        + _load_medquad(config.MEDQUAD_DIR)
        + _load_mediqa(config.MEDIQA_DIR)
    )

    # Safety examples scaled to ~5% of data
    target = max(len(_SAFETY_EXAMPLES), int(len(data) * 0.05))
    safety = [_to_conversation(clean(q), clean(a, is_answer=True)) for q, a in _SAFETY_EXAMPLES]
    safety = list(itertools.islice(itertools.cycle(safety), target))
    random.shuffle(safety)
    data.extend(safety)
    print(f"  Safety:    {len(safety):>7,} examples (~5%)")

    before = len(data)
    data   = _deduplicate(data)
    print(f"\nAfter dedup: {before:,} → {len(data):,} ({before - len(data):,} removed)")

    # Optionally report PubMed availability for reference
    pubmed_docs = _load_pubmed_documents(config.PUBMED_PARQUET_DIR)
    if pubmed_docs:
        print(f"  PubMed documents available for supplementary text pretraining: {len(pubmed_docs):,}")
        print(f"  (PubMed is used via ChunkTextDataset alongside fine-tuning splits)")

    print("\nSaving fine-tuning splits")
    _save_splits(data, config.SPLITS_DIR)

    total = sum(
        sum(1 for _ in open(Path(config.SPLITS_DIR) / f"{s}.jsonl", encoding="utf-8"))
        for s in ("train", "val", "test")
    )
    print(f"\nTotal examples : {total:,}")
    print(f"MedDialog weight: {getattr(config, 'MEDDIALOG_UPSAMPLE', 1)}x upsampled")
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess training data.")
    parser.add_argument(
        "--stage",
        choices=["pretrain", "finetune", "all"],
        default="all",
        help="Which stage to preprocess. Default: all",
    )
    args = parser.parse_args()

    if args.stage in ("pretrain", "all"):
        preprocess_pretrain()

    if args.stage in ("finetune", "all"):
        start_preprocess()