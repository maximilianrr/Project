# python scripts/preprocess.py
# Preprocesses raw data into nanochat conversation format.
# Output: data/splits/train.jsonl, val.jsonl, test.jsonl

import os
import sys
import json
import glob
import random
import re
import xml.etree.ElementTree as ET
from datasets import load_from_disk

# Load config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MEDQUAD_DIR, MEDDIALOG_DIR, SPLITS_DIR, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED

random.seed(RANDOM_SEED)

def clean(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()   # Replace multiple spaces/newlines with single space
    text = re.sub(r'<[^>]+>', '', text)         # Remove HTML tags
    text = re.sub(r'\[.*?\]', '', text)         # Remove bracketed content
    text = re.sub(r'Chat\s*Doctor\.?', 'a doctor', text, flags=re.IGNORECASE)  # Remove ChatDoctor branding
    return text.strip()

def to_conversation(q, a):
    return [
        {"role": "user",      "content": q},
        {"role": "assistant", "content": a}
    ]  # Nanochat expects this format

def load_medquad(path):
    pairs = []
    for xml_file in glob.glob(os.path.join(path, "**/*.xml"), recursive=True):
        root = ET.parse(xml_file).getroot()
        for qa in root.findall(".//QAPair"):
            q = qa.find("Question")
            a = qa.find("Answer")
            if a is not None and a.text and len(a.text.strip()) > 20:
                q_text = clean(q.text or "")
                a_text = clean(a.text or "")
                if q_text and a_text:
                    pairs.append(to_conversation(q_text, a_text))
    print(f"  MedQuAD: {len(pairs)} examples")
    return pairs

def load_meddialog(path):
    ds = load_from_disk(path)
    pairs = []
    for item in ds:
        q = clean(item["input"])
        a = clean(item["output"])
        if q and a:
            pairs.append(to_conversation(q, a))
    print(f"  MedDialog: {len(pairs)} examples")
    return pairs

def save_splits(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    random.shuffle(data)
    n = len(data)
    train = data[:int(n * TRAIN_RATIO)]
    val   = data[int(n * TRAIN_RATIO):int(n * (TRAIN_RATIO + VAL_RATIO))]
    test  = data[int(n * (TRAIN_RATIO + VAL_RATIO)):]
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(out_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item) + "\n")
        print(f"  {name}: {len(split)} examples → {out_path}")

print("Loading data")
data = load_medquad(MEDQUAD_DIR) + load_meddialog(MEDDIALOG_DIR)
print(f"Total: {len(data)} examples")
print("Saving splits")
save_splits(data, SPLITS_DIR)
print("\nDone.")