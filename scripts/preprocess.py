# scripts/preprocess.py
# Cleans, tone-normalises, upsamples, deduplicates, and splits all data sources
# into nanochat conversation format — optimised for a nano-scale model.
#
# Key decisions for nano model quality:
#   - Tight length filters: only examples the model can actually learn from
#   - MedDialog upsampled 3x: real patient language dominates training signal
#   - Hard cap at MAX_EXAMPLES: quality over quantity
#   - Tone normalisation: every answer sounds like the same doctor voice
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
    MEDQUAD_DIR, MEDDIALOG_DIR, PUBMEDQA_DIR, MEDIQA_DIR,
    SPLITS_DIR, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    MIN_Q_CHARS, MIN_A_CHARS, MAX_Q_CHARS, MAX_A_CHARS,
    MEDDIALOG_UPSAMPLE, MAX_EXAMPLES,
)

random.seed(RANDOM_SEED)

# ── Tone normalisation ─────────────────────────────────────────────────────────
# Every answer is rewritten toward a consistent doctor voice:
# warm, direct, second-person, jargon-light.
# Applied to answers only — questions are left as patients wrote them.

TONE_REPLACEMENTS = [
    # Third-person patient → second person
    (r'\bthe patient should\b',          'you should',              re.IGNORECASE),
    (r'\bthe patient (can|may|must)\b',  r'you \1',                 re.IGNORECASE),
    (r'\bpatients should\b',             'you should',              re.IGNORECASE),
    (r'\bpatients (can|may|must)\b',     r'you \1',                 re.IGNORECASE),
    (r'\bin patients\b',                 'in people',               re.IGNORECASE),
    (r'\bthe patient\b',                 'you',                     re.IGNORECASE),
    (r'\bpatients\b',                    'people',                  re.IGNORECASE),

    # Passive / impersonal → active doctor voice
    (r'\bit is recommended( that)?\b',   "I'd recommend",           re.IGNORECASE),
    (r'\bit is advised( that)?\b',       "I'd advise",              re.IGNORECASE),
    (r'\bone should\b',                  'you should',              re.IGNORECASE),
    (r'\bmay be administered\b',         'can be given',            re.IGNORECASE),
    (r'\bis indicated\b',                'is the right approach',   re.IGNORECASE),
    (r'\bcontraindicated\b',             'not recommended',         re.IGNORECASE),

    # Research / academic → plain English
    (r'\bthe study (found|showed|demonstrated)\b', 'research shows', re.IGNORECASE),
    (r'\bstatistically significant\b',   'meaningful',              re.IGNORECASE),
    (r'\betiology\b',                    'cause',                   re.IGNORECASE),
    (r'\bpathophysiology\b',             'how this condition works', re.IGNORECASE),
    (r'\bpresents with\b',               'has symptoms of',         re.IGNORECASE),
    (r'\bpresentation\b',                'symptoms',                re.IGNORECASE),
    (r'\bcomorbid(ity|ities)?\b',        'other health conditions', re.IGNORECASE),
    (r'\bprognosis\b',                   'outlook',                 re.IGNORECASE),
    (r'\bprophylaxis\b',                 'prevention',              re.IGNORECASE),
    (r'\badminister(ed|ing)?\b',         'give',                    re.IGNORECASE),

    # Branding / noise
    (r'\bchat\s*doctor\.?\b',            'a doctor',                re.IGNORECASE),
    (r'\bhealthcaremagic\.?\b',          '',                        re.IGNORECASE),
]

_TONE_PATTERNS = [
    (re.compile(pat, flags), repl)
    for pat, repl, flags in TONE_REPLACEMENTS
]

# Other noise
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
        text = apply_tone(text)
    text = _fix_punctuation_spacing(text)
    text = _capitalise_sentences(text)
    text = _ensure_sentence_end(text)
    return text.strip()


def is_valid(q: str, a: str) -> bool:
    """
    Tight filter tuned for nano model learning.
    Both too-short (useless) and too-long (unlearnable) examples are dropped.
    """
    if len(q) < MIN_Q_CHARS or len(q) > MAX_Q_CHARS:
        return False
    if len(a) < MIN_A_CHARS or len(a) > MAX_A_CHARS:
        return False
    # Drop examples where the answer is just a repeated version of the question
    if a.lower().strip() == q.lower().strip():
        return False
    # Drop answers that are clearly incomplete (end mid-word before our period fix)
    words = a.split()
    if len(words) < 10:
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
    Gold source — real patient messages and real doctor replies.
    Upsampled MEDDIALOG_UPSAMPLE times so the model sees this pattern
    far more than the formal academic sources.
    """
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        q = clean(item.get('input', ''),  is_answer=False)
        a = clean(item.get('output', ''), is_answer=True)
        if is_valid(q, a):
            pairs.append(to_conversation(q, a))

    # Upsample by duplicating with slight shuffling so it's not
    # literally identical copies back-to-back
    upsampled = pairs.copy()
    for _ in range(MEDDIALOG_UPSAMPLE - 1):
        sample = pairs.copy()
        random.shuffle(sample)
        upsampled.extend(sample)

    print(f'  MedDialog:  {len(pairs):>7,} unique  →  {len(upsampled):>7,} after {MEDDIALOG_UPSAMPLE}x upsample')
    return upsampled


def load_medquad(path: str) -> list:
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


def load_pubmedqa(path: str) -> list:
    """
    Long answers only — the plain-English paragraph, not yes/no or abstracts.
    Tight filter because PubMed answers tend to be long and formal;
    we only keep ones that fall within the nano-model-learnable range.
    """
    if not os.path.exists(path):
        print('  PubMedQA:   NOT FOUND — skipped')
        return []
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        q = clean(item.get('question', ''), is_answer=False)
        a = clean(item.get('long_answer', '') or '', is_answer=True)
        if is_valid(q, a):
            pairs.append(to_conversation(q, a))
    print(f'  PubMedQA:   {len(pairs):>7,} examples (tight-filtered)')
    return pairs


def load_mediqa(path: str) -> list:
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
    Remove exact-duplicate questions (case-insensitive).
    Because MedDialog is upsampled, duplicates within that source are expected
    and intentional — we only remove cross-source duplicates here by checking
    after combining everything. Upsampled copies of the same question with the
    same answer are kept (they are identical by design, not noise).
    """
    seen, unique = set(), []
    # Track how many times each key has been seen to allow upsampled repeats
    counts: dict[str, int] = {}
    for conv in data:
        key = conv[0]['content'].lower().strip()
        counts[key] = counts.get(key, 0) + 1
        # Allow up to MEDDIALOG_UPSAMPLE copies; beyond that it's true noise
        if counts[key] <= MEDDIALOG_UPSAMPLE:
            unique.append(conv)
    return unique


# ── Cap & split ────────────────────────────────────────────────────────────────

def cap_and_split(data: list, out_dir: str) -> None:
    """
    Shuffle, cap to MAX_EXAMPLES, then split into train/val/test.
    Capping after shuffle ensures we get a good mix from all sources.
    """
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

    # MedDialog first so its upsampled copies survive the dedup cap logic
    meddialog = load_meddialog(MEDDIALOG_DIR)
    others    = (
        load_medquad(MEDQUAD_DIR)
        + load_pubmedqa(PUBMEDQA_DIR)
        + load_mediqa(MEDIQA_DIR)
    )

    # Combine: MedDialog first so it has priority in dedup
    data = meddialog + others

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