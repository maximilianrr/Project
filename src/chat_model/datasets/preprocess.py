# src/chat_model/datasets/preprocess.py
from __future__ import annotations

import glob
import itertools
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

from datasets import load_from_disk
import pyarrow.parquet as pq

from chat_model import config

random.seed(config.RANDOM_SEED)

_EMAIL     = re.compile(r"\S+@\S+")
_HTML_TAG  = re.compile(r"<[^>]{1,100}>")
_URL       = re.compile(r"https?://\S+|www\.\S+")
_BRACKET   = re.compile(r"\[.*?\]|\(fig\.?[\s\d]+\)", re.IGNORECASE)
_MULTI_WS  = re.compile(r"[ \t]+")
_MULTI_NL  = re.compile(r"\n{3,}")
_ENDS_PUNC = re.compile(r"[.!?]$")

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

_BRAND_NAMES = re.compile(r"chat\s*doctor|healthcaremagic|health\s*care\s*magic", re.IGNORECASE)
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
    ("I've been having thoughts of hurting myself. What should I do?",
     "Please reach out now — call a crisis line (UK: 116 123, US: 988), go to your nearest emergency department, or call someone you trust."),
    ("I feel like there's no point in living anymore.",
     "Please talk to someone right now. UK: Samaritans 116 123. US: call or text 988. You deserve support."),
]


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
        print("  datasketch not found — falling back to exact dedup")
        seen: set[str] = set()
        kept = []
        for conv in data:
            key = conv[0]["content"].strip().lower()
            if key not in seen:
                seen.add(key)
                kept.append(conv)
        return kept


# Raw document loaders

def _load_climbmix_documents(parquet_dir: str) -> list[str]:
    shard_paths = sorted(Path(parquet_dir).glob("climbmix_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No climbmix parquet shards found in {parquet_dir}.")
    documents = []
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=["text"])
        for text in table.column("text").to_pylist():
            doc = (text or "").strip()
            if doc:
                documents.append(doc)
    print(f"  climbmix: {len(documents):>7,} documents from {len(shard_paths)} shard(s)")
    return documents


def _load_pubmed_documents(parquet_dir: str) -> list[str]:
    shard_paths = sorted(Path(parquet_dir).glob("pubmed_*.parquet"))
    if not shard_paths:
        print(f"  PubMed: NOT FOUND in {parquet_dir} — skipped")
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


def _load_oasst2_conversations(parquet_dir: str) -> list:
    shard_paths = sorted(Path(parquet_dir).glob("oasst2_*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No oasst2 parquet shards found in {parquet_dir}.")
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


def _load_meddialog(path: str) -> list:
    if not Path(path).exists():
        raise FileNotFoundError(f"MedDialog not found at {path}. Run download_meddialog() first.")
    from datasets import load_from_disk
    from typing import cast, Any
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


def _load_emergency_cases(path: str) -> list:
    p = Path(path)
    if not p.exists():
        print(f"  Emergency cases: NOT FOUND at {path} — skipped")
        return []
    with p.open(encoding="utf-8") as f:
        raw = json.load(f)
    pairs = []
    # Support list-of-dicts with "prompt"/"response" or "question"/"answer" keys
    for item in raw:
        q = clean(item.get("prompt") or item.get("question") or "", is_answer=False)
        a = clean(item.get("response") or item.get("answer") or "", is_answer=True)
        if q and a:
            pairs.append(_to_conversation(q, a))
    print(f"  Emergency: {len(pairs):>7,} cases loaded from {path}")
    return pairs


# ── Save helpers ──────────────────────────────────────────────────────────────

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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / filename
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for doc in documents:
            f.write(doc.replace("\n", " ") + "\n")
    print(f"  Saved {len(documents):,} documents → {out_path}")


# ── Stage entry points ────────────────────────────────────────────────────────

def preprocess_stage1() -> None:
    """
    Stage 1: climbmix-400 only → raw text documents for ChunkTextDataset.
    Lots of climbmix, no other data mixed in.
    """
    print("\nPreprocessing Stage 1: climbmix pretraining")
    print("=" * 48)
    climbmix_docs = _load_climbmix_documents(config.CLIMBMIX_PARQUET_DIR)
    _save_text_documents(climbmix_docs, config.PRETRAIN_SPLITS_DIR, filename="climbmix_docs.txt")
    print(f"\nTotal climbmix documents: {len(climbmix_docs):,}")
    print("Done.")


def preprocess_stage2() -> None:
    """
    Stage 2: lots of PubMed text + ~10% climbmix mixed in.
    Saved as raw text docs (both sources) into stage2_splits/.
    No freezing during training, lower LR than Stage 1.
    """
    print("\nPreprocessing Stage 2: PubMed + 10% climbmix")
    print("=" * 48)

    pubmed_docs = _load_pubmed_documents(config.PUBMED_PARQUET_DIR)
    if not pubmed_docs:
        raise FileNotFoundError("PubMed documents required for Stage 2. Run download_pubmed() first.")

    climbmix_docs = _load_climbmix_documents(config.CLIMBMIX_PARQUET_DIR)

    # Take 10% of climbmix relative to pubmed volume
    climbmix_target = int(len(pubmed_docs) * config.STAGE2_CLIMBMIX_RATIO)
    climbmix_sample = random.sample(climbmix_docs, min(climbmix_target, len(climbmix_docs)))
    print(f"  Mixing {len(climbmix_sample):,} climbmix docs ({config.STAGE2_CLIMBMIX_RATIO*100:.0f}% of PubMed volume)")

    combined = pubmed_docs + climbmix_sample
    random.shuffle(combined)

    out_dir = Path(config.STAGE2_SPLITS_DIR)
    _save_text_documents(combined, str(out_dir), filename="stage2_docs.txt")
    print(f"\nTotal Stage 2 documents: {len(combined):,}  (PubMed: {len(pubmed_docs):,} + climbmix: {len(climbmix_sample):,})")
    print("Done.")


def preprocess_stage3() -> None:
    """
    Stage 3: oasst2 + MedQuAD + emergency_test_cases.json → chatbot fine-tuning.
    With freezing and much lower LR.
    """
    print("\nPreprocessing Stage 3: chatbot fine-tuning")
    data = _load_oasst2_conversations(config.OASST2_PARQUET_DIR)
    data += _load_meddialog(config.MEDDIALOG_DIR)
    data += _load_medquad(config.MEDQUAD_DIR)
    data += _load_emergency_cases(config.EMERGENCY_CASES_PATH)

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

    print("\nSaving Stage 3 splits")
    _save_splits(data, config.STAGE3_SPLITS_DIR)
    print("Done.")


# Legacy aliases so existing code that calls these still works
def preprocess_pretrain() -> None:
    preprocess_stage1()


def start_preprocess() -> None:
    preprocess_stage3()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for the 3-stage training pipeline.")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all", "pretrain", "medical", "finetune"],
        default="all",
        help=(
            "Which stage to preprocess: "
            "1/pretrain = climbmix text pretraining, "
            "2/medical = PubMed + climbmix continued pretraining, "
            "3/finetune = chat fine-tuning data, "
            "all = all stages."
        ),
    )
    args = parser.parse_args()
    stage = args.stage
    if stage in ("1", "pretrain", "all"):
        preprocess_stage1()

    if stage in ("2", "medical", "all"):
        preprocess_stage2()

    if stage in ("3", "finetune", "all"):
        preprocess_stage3()