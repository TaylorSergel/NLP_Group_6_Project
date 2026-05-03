import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────
EMOTIONS = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'disgust']
PROCESSED_PATH = 'data/processed'
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ── Step 1: Load BRIGHTER languages (have emotion labels) ─
print("Loading BRIGHTER datasets...")

def load_brighter_language(lang_name):
    splits = []
    for split in ['train', 'dev', 'test']:
        path = f'data/raw/brighter_{lang_name}_{split}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['split'] = split
            df['language'] = lang_name
            splits.append(df)
            print(f"  Loaded {path} ({len(df)} rows)")
    return pd.concat(splits, ignore_index=True) if splits else None

afrikaans_df = load_brighter_language('afrikaans')
english_df   = load_brighter_language('english')
isizulu_df   = load_brighter_language('isizulu')

# Combine all BRIGHTER data
brighter_df = pd.concat([afrikaans_df, english_df, isizulu_df], ignore_index=True)
print(f"\nTotal BRIGHTER rows: {len(brighter_df)}")

# ── Step 2: Inspect columns ───────────────────────────────
print("\nBRIGHTER columns:", brighter_df.columns.tolist())
print("\nSample row:")
print(brighter_df.iloc[0])

# ── Step 3: Check emotion columns ────────────────────────
available_emotions = [e for e in EMOTIONS if e in brighter_df.columns]
print(f"\nEmotion columns found: {available_emotions}")

# ── Step 4: Check intensity annotations ──────────────────
intensity_cols = [c for c in brighter_df.columns if 'intensity' in c.lower()]
print(f"\nIntensity columns found: {intensity_cols}")
if intensity_cols:
    print("\nLanguages with intensity data:")
    for lang in ['afrikaans', 'english', 'isizulu']:
        lang_df = brighter_df[brighter_df['language'] == lang]
        non_null = lang_df[intensity_cols].notna().any(axis=1).sum()
        print(f"  {lang}: {non_null} rows with intensity annotations")

# ── Step 5: Emotion class distribution ───────────────────
print("\nEmotion distributions per language:")
for lang in ['afrikaans', 'english', 'isizulu']:
    lang_df = brighter_df[brighter_df['language'] == lang]
    print(f"\n--- {lang} ({len(lang_df)} samples) ---")
    if available_emotions:
        print(lang_df[available_emotions].sum())

# ── Step 6: Preprocess text ───────────────────────────────
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nPreprocessing BRIGHTER text...")
brighter_df['text_clean'] = brighter_df['text'].apply(preprocess_text)

# Fill missing emotion values with 0
brighter_df[available_emotions] = brighter_df[available_emotions].fillna(0)
print("\nMissing values after fill:")
print(brighter_df[available_emotions].isnull().sum())

# Check remaining missing values
print("\nAll missing values:")
print(brighter_df.isnull().sum())

# Remove duplicates
before = len(brighter_df)
brighter_df = brighter_df.drop_duplicates(subset=['text_clean'])
print(f"\nRemoved {before - len(brighter_df)} duplicates. Remaining: {len(brighter_df)}")

# ── Step 7: Split into train/val/test ─────────────────────
print("\nSplitting data...")
train_list, val_list, test_list = [], [], []

for lang in ['afrikaans', 'english', 'isizulu']:
    lang_df = brighter_df[brighter_df['language'] == lang]

    if len(lang_df) < 10:
        print(f"Warning: {lang} has very few samples ({len(lang_df)}) — skipping")
        continue

    # Use existing splits if available and train split exists
    has_train = 'train' in lang_df['split'].values
    if 'split' in lang_df.columns and lang_df['split'].nunique() > 1 and has_train:
        train_list.append(lang_df[lang_df['split'] == 'train'])
        val_list.append(lang_df[lang_df['split'] == 'dev'])
        test_list.append(lang_df[lang_df['split'] == 'test'])
        print(f"  {lang}: using existing splits")
    else:
        # isiZulu has no train split — manually split from dev+test
        print(f"  {lang}: no train split found — manually splitting")
        train, temp = train_test_split(lang_df, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

train_df = pd.concat(train_list).reset_index(drop=True)
val_df   = pd.concat(val_list).reset_index(drop=True)
test_df  = pd.concat(test_list).reset_index(drop=True)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Step 8: Process augmentation languages ────────────────
print("\nProcessing augmentation datasets (Sesotho & Setswana)...")

# Sesotho
sesotho_sa   = pd.read_csv('data/raw/sesotho_sa.csv')
sesotho_absa = pd.read_csv('data/raw/sesotho_absa.csv')
sesotho_df   = pd.concat([sesotho_sa, sesotho_absa], ignore_index=True)
sesotho_df['text_clean'] = sesotho_df['text'].apply(preprocess_text)
sesotho_df = sesotho_df.drop_duplicates(subset=['text_clean'])
print(f"  Sesotho: {len(sesotho_df)} rows")

# Setswana
setswana_df = pd.read_csv('data/raw/setswana_train.csv')
setswana_df['language'] = 'setswana'
setswana_df['text_clean'] = setswana_df['text'].apply(preprocess_text)
setswana_df = setswana_df.drop_duplicates(subset=['text_clean'])
print(f"  Setswana: {len(setswana_df)} rows")

# ── Step 9: Save all processed files ─────────────────────
print("\nSaving processed files...")

train_df.to_csv(f'{PROCESSED_PATH}/train.csv', index=False)
val_df.to_csv(f'{PROCESSED_PATH}/val.csv', index=False)
test_df.to_csv(f'{PROCESSED_PATH}/test.csv', index=False)
sesotho_df.to_csv(f'{PROCESSED_PATH}/sesotho_augmentation.csv', index=False)
setswana_df.to_csv(f'{PROCESSED_PATH}/setswana_augmentation.csv', index=False)

print(f"\nSaved to {PROCESSED_PATH}/:")
print(f"  train.csv                 ({len(train_df)} rows)")
print(f"  val.csv                   ({len(val_df)} rows)")
print(f"  test.csv                  ({len(test_df)} rows)")
print(f"  sesotho_augmentation.csv  ({len(sesotho_df)} rows)")
print(f"  setswana_augmentation.csv ({len(setswana_df)} rows)")

# ── Step 10: Print data summary ───────────────────────────
print("\n" + "="*50)
print("DATA SUMMARY")
print("="*50)
print(f"\nLanguages with emotion labels: Afrikaans, English, isiZulu")
print(f"Languages for augmentation only: Sesotho, Setswana")
print(f"\nEmotion columns: {available_emotions}")
print(f"Intensity columns: {intensity_cols if intensity_cols else 'None found'}")
print("\nSample counts:")
for lang in ['afrikaans', 'english', 'isizulu']:
    lang_train = train_df[train_df['language'] == lang]
    lang_val   = val_df[val_df['language'] == lang]
    lang_test  = test_df[test_df['language'] == lang]
    print(f"  {lang}: train={len(lang_train)}, val={len(lang_val)}, test={len(lang_test)}")
print("\nPhase 1 complete.")