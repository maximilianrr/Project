# scripts/prepare_tokenizer_data.py
# Builds tokenizer_text.txt from train + val splits and WTND book.
# Uses only the cleaner, filtered splits (not raw data) so the tokenizer
# learns vocabulary from the same quality text the model will train on.
#
# Run: python scripts/prepare_tokenizer_data.py

import os
import sys
import re
import json
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPLITS_DIR, WTND_CLEAN, TOKENIZER_TEXT, DOC_CAP

os.makedirs(os.path.dirname(TOKENIZER_TEXT), exist_ok=True)

def normalise(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

total_lines = 0

with open(TOKENIZER_TEXT, 'w', encoding='utf-8') as out:

    # Train + val splits — same quality-filtered text the model trains on
    # (never test — no leakage)
    for split in ('train', 'val'):
        path = os.path.join(SPLITS_DIR, f'{split}.jsonl')
        if not os.path.exists(path):
            print(f'  MISSING: {path} — run preprocess.py first')
            continue
        with open(path, encoding='utf-8') as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    conv = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                for msg in conv:
                    content = normalise(msg.get('content', ''))
                    if content:
                        out.write(content[:DOC_CAP] + '\n')
                        total_lines += 1

    # WTND — plain-language medical prose for vocabulary breadth
    if os.path.exists(WTND_CLEAN):
        with open(WTND_CLEAN, encoding='utf-8') as f:
            for paragraph in re.split(r'\n{2,}', f.read()):
                paragraph = normalise(paragraph)
                if len(paragraph) >= 30:
                    for i in range(0, len(paragraph), DOC_CAP):
                        chunk = paragraph[i:i + DOC_CAP].strip()
                        if chunk:
                            out.write(chunk + '\n')
                            total_lines += 1
        print('  Added WTND book text.')
    else:
        print('  WTND not found — run download_wtnd.py first.')

print(f'\nTotal lines : {total_lines:,}')
print(f'Saved to    : {TOKENIZER_TEXT}')
