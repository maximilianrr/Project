# scripts/preprocess.py
# Cleans, tone-normalises, upsamples, deduplicates, and splits all data sources
# into nanochat conversation format — optimised for a nano-scale model.
#
# Sources used:
#   - MedDialog (ChatDoctor-HealthCareMagic) — gold, real patient/doctor conversations
#   - MedQuAD   — NIH factual medical Q&A
#   - MEDIQA-Chat — clinical dialogues (optional)
#
# PubMedQA excluded: research paper titles as questions, journal-abstract answers.
# MedMCQA  excluded: exam language, not patient language.
#
# Run: python scripts/preprocess.py

import os
import sys
import re
import json
import glob
import random
import unicodedata
import xml.etree.ElementTree as ET
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MEDQUAD_DIR, MEDDIALOG_DIR, MEDIQA_DIR,
    SPLITS_DIR, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    MIN_Q_CHARS, MIN_A_CHARS, MAX_Q_CHARS, MAX_A_CHARS,
    MEDDIALOG_UPSAMPLE, MAX_EXAMPLES,
)

random.seed(RANDOM_SEED)

# ── Branding / greeting stripping ─────────────────────────────────────────────
# MedDialog answers often start with broken greeting lines caused by naive
# branding replacement, e.g. "Welcome to a doctor! Thanks for posting on a doctor!"
# We strip the entire opening greeting block before any other processing.

# Matches common opening salutation lines (the whole line up to the first real sentence)
_GREETING_LINE = re.compile(
    r'^(hi|hello|dear|welcome|thanks?\s+for\s+(posting|using|contacting|writing|reaching)|'
    r'good\s+(morning|afternoon|evening)|thank\s+you\s+for)[^.!?]*[.!?]\s*',
    re.IGNORECASE,
)

# Matches the specific broken branding pattern produced by naive replacement
_BROKEN_BRANDING = re.compile(
    r'(welcome\s+to\s+(a\s+doctor|healthcaremagic|chatdoctor)[.!,]?\s*'
    r'|thanks?\s+for\s+posting\s+(your\s+query\s+)?on\s+(a\s+doctor|healthcaremagic|chatdoctor)[.!,]?\s*'
    r'|hi\s*,?\s+welcome\s+to\s+[^.!?]*[.!?]\s*)',
    re.IGNORECASE,
)

# Closing filler lines that add no medical value
_CLOSING_FILLER = re.compile(
    r'(hope\s+(this\s+)?(helps?|answers?|clears?)[^.!?]*[.!?]\s*'
    r'|feel\s+free\s+to\s+(ask|consult)[^.!?]*[.!?]\s*'
    r'|let\s+me\s+know\s+if\s+(you\s+have\s+)?any\s+(other\s+)?questions?[^.!?]*[.!?]\s*'
    r'|regards[.!]?\s*'
    r'|take\s+care[.!]?\s*)',
    re.IGNORECASE,
)


def strip_greeting_and_closing(text: str) -> str:
    """
    Remove broken branding lines, opening greetings, and closing filler
    from doctor answers. These add noise without medical content.
    """
    # First strip the specific broken branding pattern
    text = _BROKEN_BRANDING.sub('', text)
    # Then strip any remaining generic opening greeting lines (first line only)
    text = _GREETING_LINE.sub('', text, count=2)
    # Strip closing filler anywhere in the text
    text = _CLOSING_FILLER.sub('', text)
    return text.strip()


# ── Tone normalisation ─────────────────────────────────────────────────────────
# Rewrites every answer toward a consistent doctor voice:
# warm, direct, second-person, jargon-light.
# Applied to answers only — patient questions are left as written.

TONE_REPLACEMENTS = [
    # Third-person patient → second person
    (r'\bthe patient should\b',          'you should',               re.IGNORECASE),
    (r'\bthe patient (can|may|must)\b',  r'you \1',                  re.IGNORECASE),
    (r'\bpatients should\b',             'you should',               re.IGNORECASE),
    (r'\bpatients (can|may|must)\b',     r'you \1',                  re.IGNORECASE),
    (r'\bin patients\b',                 'in people',                re.IGNORECASE),
    (r'\bthe patient\b',                 'you',                      re.IGNORECASE),
    (r'\bpatients\b',                    'people',                   re.IGNORECASE),

    # Passive / impersonal → active doctor voice
    (r'\bit is recommended( that)?\b',   "I'd recommend",            re.IGNORECASE),
    (r'\bit is advised( that)?\b',       "I'd advise",               re.IGNORECASE),
    (r'\bone should\b',                  'you should',               re.IGNORECASE),
    (r'\bmay be administered\b',         'can be given',             re.IGNORECASE),
    (r'\bis indicated\b',                'is the right approach',    re.IGNORECASE),
    (r'\bcontraindicated\b',             'not recommended',          re.IGNORECASE),

    # Academic / jargon → plain English
    (r'\bthe study (found|showed|demonstrated)\b', 'research shows', re.IGNORECASE),
    (r'\bstatistically significant\b',   'meaningful',               re.IGNORECASE),
    (r'\betiology\b',                    'cause',                    re.IGNORECASE),
    (r'\bpathophysiology\b',             'how this condition works',  re.IGNORECASE),
    (r'\bpresents with\b',               'has symptoms of',          re.IGNORECASE),
    (r'\bpresentation\b',                'symptoms',                 re.IGNORECASE),
    (r'\bcomorbid(ity|ities)?\b',        'other health conditions',  re.IGNORECASE),
    (r'\bprognosis\b',                   'outlook',                  re.IGNORECASE),
    (r'\bprophylaxis\b',                 'prevention',               re.IGNORECASE),
    (r'\badminister(ed|ing)?\b',         'give',                     re.IGNORECASE),
]

_TONE_PATTERNS = [
    (re.compile(pat, flags), repl)
    for pat, repl, flags in TONE_REPLACEMENTS
]

# General noise patterns
_HTML_TAG  = re.compile(r'<[^>]{1,100}>')
_URL       = re.compile(r'https?://\S+|www\.\S+')
_BRACKET   = re.compile(r'\[.*?\]|\(fig\.?[\s\d]+\)', re.IGNORECASE)
_MULTI_WS  = re.compile(r'[ \t]+')
_MULTI_NL  = re.compile(r'\n{3,}')
_ENDS_PUNC = re.compile(r'[.!?]$')


def _fix_punctuation_spacing(text: str) -> str:
    text = re.sub(r'\s+([,\.!?;:])', r'\1', text)
    text = re.sub(r'([,\.!?;:])\s{2,}', r'\1 ', text)
    return text


def _capitalise_sentences(text: str) -> str:
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p[0].upper() + p[1:] if p else p for p in parts]
    return ' '.join(parts)


def _ensure_sentence_end(text: str) -> str:
    t = text.rstrip()
    if t and not _ENDS_PUNC.search(t):
        t += '.'
    return t


def apply_tone(text: str) -> str:
    for pattern, replacement in _TONE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def clean(text: str, is_answer: bool = False) -> str:
    """
    Full cleaning pipeline:
      1. Unicode normalisation
      2. Strip HTML, URLs, bracketed refs
      3. Strip greeting/closing/branding (answers only)
      4. Tone normalisation (answers only)
      5. Whitespace normalisation
      6. Punctuation spacing
      7. Sentence capitalisation
      8. Sentence-final punctuation
    """
    if not text:
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = _HTML_TAG.sub(' ', text)
    text = _URL.sub('', text)
    text = _BRACKET.sub('', text)
    text = _MULTI_WS.sub(' ', text)
    text = _MULTI_NL.sub('\n\n', text)
    text = text.strip()
    if is_answer:
        text = strip_greeting_and_closing(text)
        text = apply_tone(text)
    text = _fix_punctuation_spacing(text)
    text = _capitalise_sentences(text)
    text = _ensure_sentence_end(text)
    return text.strip()


def is_valid(q: str, a: str) -> bool:
    if len(q) < MIN_Q_CHARS or len(q) > MAX_Q_CHARS:
        return False
    if len(a) < MIN_A_CHARS or len(a) > MAX_A_CHARS:
        return False
    if a.lower().strip() == q.lower().strip():
        return False
    if len(a.split()) < 10:
        return False
    return True


def to_conversation(q: str, a: str) -> list[dict]:
    return [
        {'role': 'user',      'content': q},
        {'role': 'assistant', 'content': a},
    ]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_meddialog(path: str) -> list:
    """
    Gold source. Real patient messages and real doctor replies.
    Greeting/closing stripped first, then tone-normalised.
    Upsampled MEDDIALOG_UPSAMPLE times with reshuffling between copies.
    """
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        q = clean(item.get('input', ''),  is_answer=False)
        a = clean(item.get('output', ''), is_answer=True)
        if is_valid(q, a):
            pairs.append(to_conversation(q, a))

    upsampled = pairs.copy()
    for _ in range(MEDDIALOG_UPSAMPLE - 1):
        sample = pairs.copy()
        random.shuffle(sample)
        upsampled.extend(sample)

    print(f'  MedDialog:  {len(pairs):>7,} unique  →  {len(upsampled):>7,} after {MEDDIALOG_UPSAMPLE}x upsample')
    return upsampled


def load_medquad(path: str) -> list:
    """
    NIH medical Q&A from XML. Factual and accurate; supports MedDialog
    with domain vocabulary. Not conversational but tone-normalised.
    """
    pairs = []
    for xml_file in glob.glob(os.path.join(path, '**/*.xml'), recursive=True):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue
        for qa in root.findall('.//QAPair'):
            q_el = qa.find('Question')
            a_el = qa.find('Answer')
            if a_el is None or not (a_el.text or '').strip():
                continue
            q = clean(q_el.text or '', is_answer=False)
            a = clean(a_el.text or '', is_answer=True)
            if is_valid(q, a):
                pairs.append(to_conversation(q, a))
    print(f'  MedQuAD:    {len(pairs):>7,} examples')
    return pairs


def load_mediqa(path: str) -> list:
    """
    Clinical doctor-patient dialogues with structured summaries (optional).
    """
    if not os.path.exists(path):
        print('  MEDIQA:     NOT FOUND — skipped (optional)')
        return []
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        dialogue = item.get('dialogue', '') or ''
        summary  = clean(item.get('note', '') or item.get('summary', '') or '', is_answer=True)
        if not dialogue or not summary:
            continue
        q = clean('Based on this conversation:\n' + dialogue, is_answer=False)
        if is_valid(q, summary):
            pairs.append(to_conversation(q, summary))
    print(f'  MEDIQA:     {len(pairs):>7,} examples')
    return pairs


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate(data: list) -> list:
    """
    Remove cross-source duplicate questions.
    Intentional MedDialog upsample copies (up to MEDDIALOG_UPSAMPLE) are kept.
    """
    counts: dict[str, int] = {}
    unique = []
    for conv in data:
        key = conv[0]['content'].lower().strip()
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= MEDDIALOG_UPSAMPLE:
            unique.append(conv)
    return unique


# ── Cap & split ────────────────────────────────────────────────────────────────

def cap_and_split(data: list, out_dir: str) -> None:
    random.shuffle(data)
    if len(data) > MAX_EXAMPLES:
        print(f'  Capping {len(data):,} → {MAX_EXAMPLES:,} examples (quality cap)')
        data = data[:MAX_EXAMPLES]

    os.makedirs(out_dir, exist_ok=True)
    n     = len(data)
    train = data[:int(n * TRAIN_RATIO)]
    val   = data[int(n * TRAIN_RATIO):int(n * (TRAIN_RATIO + VAL_RATIO))]
    test  = data[int(n * (TRAIN_RATIO + VAL_RATIO)):]

    for name, split in [('train', train), ('val', val), ('test', test)]:
        out_path = os.path.join(out_dir, f'{name}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'  {name:5s}: {len(split):>7,} examples → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading and cleaning data sources...\n')

    meddialog = load_meddialog(MEDDIALOG_DIR)
    others    = load_medquad(MEDQUAD_DIR) + load_mediqa(MEDIQA_DIR)
    data      = meddialog + others

    before = len(data)
    data   = deduplicate(data)
    print(f'\nAfter dedup: {before:,} → {len(data):,} ({before - len(data):,} removed)')

    print('\nSaving splits...')
    cap_and_split(data, SPLITS_DIR)

    print('\n── Final dataset summary ──────────────────────────────')
    total = sum(
        sum(1 for _ in open(os.path.join(SPLITS_DIR, f'{s}.jsonl'), encoding='utf-8'))
        for s in ('train', 'val', 'test')
    )
    print(f'  Total examples : {total:,}')
    print(f'  Vocab size     : 16,384  (nano-optimised)')
    print(f'  MedDialog weight: {MEDDIALOG_UPSAMPLE}x upsampled (dominant training signal)')
    print('\nDone.')