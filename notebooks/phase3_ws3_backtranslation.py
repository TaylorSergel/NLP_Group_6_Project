# =============================================================================
# PHASE 3 — WORKSTREAM 3: Back-Translation Data Augmentation
# COS 760 — Group 5 | Alisha leads Phase 3
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
# Generates additional training data for Sesotho and Setswana using
# back-translation: translate text → English → back to source language.
# The result is paraphrased text that preserves the original meaning
# and emotion, increasing training data diversity.
#
# WHY BACK-TRANSLATION?
# ─────────────────────
# Your sesotho_augmentation.csv (4193 rows) and setswana_augmentation.csv
# (110103 rows) have NO emotion labels — they're raw text from news sources.
# Back-translation + LLM annotation (Workstream 4) is how you create usable
# training examples from this unlabelled data.
#
# Back-translation specifically helps because:
#   1. It creates surface-form variety from existing text (paraphrasing)
#   2. The semantic content and emotion is preserved through translation
#   3. It exposes the model to more linguistic patterns in Sesotho/Setswana
#
# HOW IT WORKS (step by step)
# ────────────────────────────
# 1. Take Sesotho/Setswana text
# 2. Translate → English using NLLB-200 (a Meta multilingual translation model)
# 3. Translate English → back to original language using NLLB-200
# 4. You now have a paraphrase of the original text
# 5. Feed the back-translated text + original labels into training
#    (labels come from Workstream 4 LLM annotation)
#
# WHY NLLB-200 NOT GOOGLE TRANSLATE?
# ───────────────────────────────────
# NLLB-200 (No Language Left Behind) from Meta supports Sesotho and Setswana
# natively and is free to run locally. Google Translate requires an API key
# and costs money at scale. For 4193 Sesotho samples NLLB-200 is practical.
#
# HOW TO RUN
# ──────────────
# Run on Colab T4 GPU — NLLB-200-distilled-600M fits comfortably.
# The full NLLB-200-1.3B would be better quality but requires more memory.
# Estimated runtime: ~2-4 hours for all Sesotho samples on T4.
# For Setswana (110K rows), process a SUBSET (e.g. 5000 rows) only.
#
# EXPECTED OUTPUT
# ───────────────
# sesotho_backtranslated.csv  — original text + back-translated paraphrase
# setswana_backtranslated.csv — same for Setswana subset
# These get combined with LLM-annotated labels in a later step.
# =============================================================================


# ── CELL 1: Install dependencies ─────────────────────────────────────────────

# !pip install -q transformers==4.40.0 sentencepiece sacremoses accelerate \
#              pandas numpy torch


# ── CELL 2: Imports ───────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# NLLB-200 language codes for our target languages
# Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANG_CODES = {
    "sesotho":  "sot_Latn",   # Southern Sotho (Sesotho) — Latin script
    "setswana": "tsn_Latn",   # Tswana (Setswana) — Latin script
    "english":  "eng_Latn",   # English — Latin script
}


# ── CELL 3: Configuration ─────────────────────────────────────────────────────

CONFIG = {
    "data_dir": "/content/drive/MyDrive/project_data",
    "output_dir": "/content/drive/MyDrive/project_data/results/phase3_backtranslation",

    # NLLB-200-distilled-600M — good balance of quality vs speed/memory
    # Use "facebook/nllb-200-1.3B" for better quality if you have A100
    "model_name": "facebook/nllb-200-distilled-600M",

    # How many rows to process from each augmentation file
    # Sesotho: process all 4193 (manageable)
    # Setswana: process a subset — 110K rows would take days
    "sesotho_sample_size": 4193,   # All available
    "setswana_sample_size": 5000,  # Random sample — adjust based on time budget

    "batch_size": 16,       # Reduce to 8 if OOM
    "max_input_length": 256,
    "max_output_length": 256,
    "num_beams": 4,          # Beam search — higher = better quality, slower
    "seed": 42,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)


# ── CELL 4: Load NLLB-200 model ───────────────────────────────────────────────
# This downloads ~2.5GB on first run. Colab caches it for the session.

print(f"\nLoading NLLB-200 model: {CONFIG['model_name']}")
print("(First run downloads ~2.5GB — may take a few minutes)")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model     = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"]).to(DEVICE)
model.eval()
print("Model loaded.")


# ── CELL 5: Translation function ─────────────────────────────────────────────

def translate_batch(texts, src_lang, tgt_lang, batch_size=16):
    """
    Translate a list of texts from src_lang to tgt_lang using NLLB-200.

    Args:
        texts:      List of strings to translate
        src_lang:   NLLB language code for source (e.g. "sot_Latn")
        tgt_lang:   NLLB language code for target (e.g. "eng_Latn")
        batch_size: How many texts to translate at once

    Returns:
        List of translated strings (same length as input)
    """
    tokenizer.src_lang = src_lang
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    translations = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        if batch_num % 10 == 1:
            print(f"  Translating batch {batch_num}/{total_batches}...", end="\r")

        # Tokenize
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["max_input_length"],
        ).to(DEVICE)

        # Generate translation
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                num_beams=CONFIG["num_beams"],
                max_length=CONFIG["max_output_length"],
                early_stopping=True,
            )

        # Decode — skip_special_tokens removes language tags and padding
        batch_translations = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
        translations.extend(batch_translations)

    print(f"\n  Done. {len(translations)} texts translated.")
    return translations


def back_translate(texts, src_lang_code, batch_size=16):
    """
    Full back-translation pipeline: src → English → src.

    Returns a dict with:
        "english":          intermediate English translations
        "back_translated":  final back-translated text in source language
    """
    print(f"\n  Step 1/2: {src_lang_code} → English")
    english_texts = translate_batch(
        texts, src_lang=src_lang_code, tgt_lang="eng_Latn",
        batch_size=batch_size
    )

    print(f"\n  Step 2/2: English → {src_lang_code}")
    back_texts = translate_batch(
        english_texts, src_lang="eng_Latn", tgt_lang=src_lang_code,
        batch_size=batch_size
    )

    return {"english": english_texts, "back_translated": back_texts}


# ── CELL 6: Quality check on small sample ────────────────────────────────────
# Before processing all data, verify translation quality on 5 examples.
# Check that the back-translated text is plausible and not garbled.

print("\n" + "="*60)
print("QUALITY CHECK — 5 sample sentences per language")
print("="*60)

sample_sesotho = [
    "Baahloli ba re ho na le matšoao a taba e kgolo.",
    "Lefapha la bophelo bo botle le re moreki o lokela ho tseba ditokelo tsa hae.",
    "Baahi ba motse ba ntse ba phuthelana ka lebaka la ditaba tsa tšebelisano.",
    "Ho na le le dikgang tse ngata tse sa utloahaleng kahara setjhaba.",
    "Mmuso o phatlalletse ditaelo tse ntjha mabapi le tsamaiso ya merafong.",
]

print("\n--- Sesotho back-translation sample ---")
for i, text in enumerate(sample_sesotho):
    en = translate_batch([text], src_lang="sot_Latn", tgt_lang="eng_Latn", batch_size=1)[0]
    bt = translate_batch([en],   src_lang="eng_Latn", tgt_lang="sot_Latn", batch_size=1)[0]
    print(f"\n[{i+1}] Original:         {text}")
    print(f"     → English:        {en}")
    print(f"     → Back-translated: {bt}")


# ── CELL 7: Process Sesotho augmentation data ────────────────────────────────

print("\n" + "="*60)
print("PROCESSING SESOTHO AUGMENTATION DATA")
print("="*60)

sesotho_path = Path(CONFIG["data_dir"]) / "sesotho_augmentation.csv"
sesotho_df   = pd.read_csv(sesotho_path)
print(f"Loaded {len(sesotho_df)} Sesotho rows")
print(f"Columns: {sesotho_df.columns.tolist()}")

# Use text_clean if available, otherwise fall back to first text column
text_col = "text_clean" if "text_clean" in sesotho_df.columns else sesotho_df.columns[0]
print(f"Using text column: '{text_col}'")

# Sample if needed
if CONFIG["sesotho_sample_size"] < len(sesotho_df):
    sesotho_df = sesotho_df.sample(n=CONFIG["sesotho_sample_size"], random_state=CONFIG["seed"])
    print(f"Sampled {CONFIG['sesotho_sample_size']} rows")

sesotho_texts = sesotho_df[text_col].astype(str).tolist()

# Filter out very short texts (likely not full sentences)
sesotho_texts = [t for t in sesotho_texts if len(t.split()) >= 3]
print(f"After filtering short texts: {len(sesotho_texts)} rows")

t0 = time.time()
sesotho_results = back_translate(
    sesotho_texts,
    src_lang_code=NLLB_LANG_CODES["sesotho"],
    batch_size=CONFIG["batch_size"],
)
elapsed = time.time() - t0
print(f"Sesotho back-translation complete in {elapsed/60:.1f} minutes")

sesotho_out = pd.DataFrame({
    "original_text": sesotho_texts,
    "english_translation": sesotho_results["english"],
    "back_translated_text": sesotho_results["back_translated"],
    "language": "sesotho",
    "source": "backtranslation",
})
out_path = Path(CONFIG["output_dir"]) / "sesotho_backtranslated.csv"
sesotho_out.to_csv(out_path, index=False)
print(f"Saved {len(sesotho_out)} rows → {out_path}")


# ── CELL 8: Process Setswana augmentation data ───────────────────────────────

print("\n" + "="*60)
print("PROCESSING SETSWANA AUGMENTATION DATA")
print("="*60)

setswana_path = Path(CONFIG["data_dir"]) / "setswana_augmentation.csv"
setswana_df   = pd.read_csv(setswana_path)
print(f"Loaded {len(setswana_df)} Setswana rows (using {CONFIG['setswana_sample_size']})")

text_col = "text_clean" if "text_clean" in setswana_df.columns else setswana_df.columns[0]

# Sample subset — 110K rows is too many for a Colab session
np.random.seed(CONFIG["seed"])
setswana_df = setswana_df.sample(n=min(CONFIG["setswana_sample_size"], len(setswana_df)),
                                   random_state=CONFIG["seed"])
setswana_texts = setswana_df[text_col].astype(str).tolist()
setswana_texts = [t for t in setswana_texts if len(t.split()) >= 3]
print(f"After filtering short texts: {len(setswana_texts)} rows")

t0 = time.time()
setswana_results = back_translate(
    setswana_texts,
    src_lang_code=NLLB_LANG_CODES["setswana"],
    batch_size=CONFIG["batch_size"],
)
elapsed = time.time() - t0
print(f"Setswana back-translation complete in {elapsed/60:.1f} minutes")

setswana_out = pd.DataFrame({
    "original_text": setswana_texts,
    "english_translation": setswana_results["english"],
    "back_translated_text": setswana_results["back_translated"],
    "language": "setswana",
    "source": "backtranslation",
})
out_path = Path(CONFIG["output_dir"]) / "setswana_backtranslated.csv"
setswana_out.to_csv(out_path, index=False)
print(f"Saved {len(setswana_out)} rows → {out_path}")


# ── CELL 9: Quality metrics ───────────────────────────────────────────────────
# Compute simple lexical similarity between original and back-translated text.
# High overlap = back-translation preserved meaning (good).
# Very high overlap = back-translation is just copying (bad — no paraphrase).
# Aim for 40-70% word overlap.

from collections import Counter

def word_overlap(text1, text2):
    """Jaccard similarity between word sets of two strings."""
    w1 = set(str(text1).lower().split())
    w2 = set(str(text2).lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)

print("\n" + "="*60)
print("BACK-TRANSLATION QUALITY METRICS")
print("="*60)

for lang_name, out_df in [("Sesotho", sesotho_out), ("Setswana", setswana_out)]:
    overlaps = [
        word_overlap(r["original_text"], r["back_translated_text"])
        for _, r in out_df.iterrows()
    ]
    print(f"\n{lang_name}:")
    print(f"  Mean word overlap (original vs back-translated): {np.mean(overlaps):.3f}")
    print(f"  Median: {np.median(overlaps):.3f}  |  Std: {np.std(overlaps):.3f}")
    print(f"  (Target range: 0.30–0.70; too high = no paraphrase, too low = garbled)")

    # Show 3 worst quality examples (lowest overlap) for manual inspection
    out_df_copy = out_df.copy()
    out_df_copy["overlap"] = overlaps
    worst = out_df_copy.nsmallest(3, "overlap")
    print(f"\n  3 lowest-quality back-translations for manual review:")
    for _, row in worst.iterrows():
        print(f"    Original:    {row['original_text'][:80]}")
        print(f"    Back-transl: {row['back_translated_text'][:80]}")
        print(f"    Overlap:     {row['overlap']:.3f}")
        print()


# ── CELL 10: Summary ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("PHASE 3 WORKSTREAM 3 COMPLETE")
print("="*60)
print(f"\nOutputs saved to: {CONFIG['output_dir']}/")
print("  sesotho_backtranslated.csv  — Sesotho paraphrases")
print("  setswana_backtranslated.csv — Setswana paraphrases")
print()
print("NEXT STEPS:")
print("  1. Run Workstream 4 (LLM annotation) to label these back-translated texts")
print("     with emotion categories.")
print("  2. Combine labelled back-translated data with main train.csv")
print("  3. Re-run Workstreams 1 & 2 with augmented training data")
print("  4. Compare augmented vs non-augmented macro F1 for Sesotho/Setswana")
print("="*60)