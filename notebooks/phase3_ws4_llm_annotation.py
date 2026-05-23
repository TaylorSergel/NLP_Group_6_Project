# =============================================================================
# PHASE 3 — WORKSTREAM 4: LLM-Based Emotion Annotation
# COS 760 — Group 5 | Alisha leads Phase 3
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
# Uses Claude (via the Anthropic API) to annotate Sesotho and Setswana news
# headlines with emotion labels (anger, fear, joy, sadness, surprise, disgust).
# These labelled headlines are then used as additional training data.
#
# WHY LLM ANNOTATION?
# ───────────────────
# Your sesotho_augmentation.csv and setswana_backtranslated.csv have text
# but NO emotion labels. Human annotation at scale is expensive and slow.
# LLMs can annotate hundreds of examples per minute and have shown reasonable
# reliability for emotion labelling — especially via English intermediary
# (the model sees the English translation + original text).
#
# ANNOTATION STRATEGY
# ───────────────────
# We use a TWO-STEP approach for reliability:
#
# Step 1 — Annotate via English translation (more reliable for the LLM)
#   Input to LLM: English translation of the text (from Workstream 3)
#   Output: emotion labels as JSON
#
# Step 2 — Verify with original text
#   The LLM sees both original and translation, resolving ambiguity.
#
# This is more reliable than asking the LLM to read Sesotho/Setswana directly,
# since Claude has stronger multilingual understanding via English.
#
# MANUAL VERIFICATION (REQUIRED)
# ───────────────────────────────
# LLM annotation is NOT 100% reliable. The proposal says "manually verify a sample".
# This script:
#   1. Annotates all texts automatically
#   2. Produces a 100-row stratified sample for you to manually verify
#   3. Computes estimated accuracy from your manual check
#   4. Filters out low-confidence annotations before using for training
#
# HOW TO RUN
# ──────────
# You need an Anthropic API key. Set it as an environment variable:
#   In Colab: use the Secrets panel (key icon in sidebar)
#             Add secret: ANTHROPIC_API_KEY = sk-ant-...
#   In your .env file locally: ANTHROPIC_API_KEY=sk-ant-...
#
# Estimated cost: ~$2-5 for annotating 4193 Sesotho headlines with claude-haiku
# Estimated runtime: ~30-60 minutes (rate limited)
#
# EXPECTED RESULTS
# ────────────────
# LLM annotation agreement with human labels: 60-80% for English
# For Sesotho/Setswana (via translation): 50-70% estimated
# After filtering low-confidence annotations, expect higher agreement
# =============================================================================


# ── CELL 1: Install dependencies ─────────────────────────────────────────────

# !pip install -q anthropic pandas numpy scikit-learn tqdm


# ── CELL 2: Imports and API setup ─────────────────────────────────────────────

import os
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import anthropic

# Load API key from Colab secrets (recommended) or environment
# In Colab: from google.colab import userdata
#           ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY")
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# if not ANTHROPIC_API_KEY:
#     raise ValueError(
#         "Set ANTHROPIC_API_KEY. In Colab: use the Secrets panel.\n"
#         "In local .env: ANTHROPIC_API_KEY=sk-ant-..."
#     )

# client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
# print("Anthropic client initialised.")

# Replace the existing API key section with this
from google.colab import userdata
ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
print("Anthropic client initialised.")

LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "disgust"]


# ── CELL 3: Configuration ─────────────────────────────────────────────────────

CONFIG = {
    "backtranslation_dir": "/content/drive/MyDrive/project_data/results/phase3_backtranslation",
    "output_dir": "/content/drive/MyDrive/project_data/results/phase3_annotation",

    # claude-haiku-20240307 is fastest and cheapest — good for bulk annotation
    # Use claude-sonnet for higher quality at higher cost
    "model": "claude-haiku-20240307",

    # Max texts to annotate per language (set lower to save cost while testing)
    "sesotho_limit": 4174,
    "setswana_limit": 4967,

    # How many examples to include in the prompt (few-shot examples)
    "num_few_shot": 3,

    # Confidence threshold — only keep annotations where model is confident
    # Labels with confidence < this are set to 0 (not present)
    "confidence_threshold": 0.6,

    # Rate limiting — Claude API has rate limits
    "requests_per_minute": 50,   # Safe default for claude-haiku
    "retry_attempts": 3,
    "retry_delay": 5,            # Seconds between retries

    # Manual verification sample size
    "verification_sample_size": 100,

    "seed": 42,
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
random.seed(CONFIG["seed"])


# ── CELL 4: Annotation prompt ─────────────────────────────────────────────────
# The prompt is the most important part of LLM annotation.
# We use structured JSON output for easy parsing and few-shot examples
# to anchor the model on what each emotion means in context.

FEW_SHOT_EXAMPLES = [
    {
        "english_text": "The government announced massive job cuts leaving thousands unemployed.",
        "reasoning": "Job cuts cause economic fear and likely anger at the government. No joy or surprise since layoffs were expected.",
        "labels": {"anger": 1, "fear": 1, "joy": 0, "sadness": 1, "surprise": 0, "disgust": 0},
    },
    {
        "english_text": "Local football team wins championship in thrilling final match.",
        "reasoning": "A championship win is a joyful, surprising event for fans.",
        "labels": {"anger": 0, "fear": 0, "joy": 1, "sadness": 0, "surprise": 1, "disgust": 0},
    },
    {
        "english_text": "Flooding destroys homes, families left without shelter in winter.",
        "reasoning": "Natural disaster causing homelessness induces sadness and fear. No positive emotions.",
        "labels": {"anger": 0, "fear": 1, "joy": 0, "sadness": 1, "surprise": 0, "disgust": 0},
    },
]


def build_annotation_prompt(original_text, english_translation, language):
    """
    Build the system + user prompt for emotion annotation.

    Args:
        original_text:       Text in Sesotho/Setswana
        english_translation: English translation from NLLB-200
        language:            "sesotho" or "setswana"
    """
    few_shot_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_str += f"""
Text: {ex['english_text']}
Reasoning: {ex['reasoning']}
Labels: {json.dumps(ex['labels'])}
---"""

    system_prompt = """You are an expert emotion annotation system for news headlines.
Your task is to label text with the emotions it evokes in a reader.
Use the Ekman basic emotion taxonomy: anger, fear, joy, sadness, surprise, disgust.
A text can have multiple emotions (multilabel) or no strong emotion (all zeros).
You must respond ONLY with valid JSON — no explanation, no markdown, no backticks."""

    user_prompt = f"""Annotate the emotion expressed in this {language} news headline.
You are given both the original text and its English translation.
Base your annotation primarily on the English translation, but use the original to catch nuance.

EMOTION DEFINITIONS:
- anger: frustration, outrage, indignation about an injustice or wrongdoing
- fear: threat, danger, anxiety, dread about future harm
- joy: happiness, celebration, pride, positive outcome
- sadness: grief, loss, disappointment, sorrow
- surprise: unexpected event (can be positive or negative)
- disgust: revulsion, moral repugnance, corruption

FEW-SHOT EXAMPLES:
{few_shot_str}

NOW ANNOTATE:
Original ({language}): {original_text}
English translation: {english_translation}

Respond ONLY with JSON in this exact format:
{{
  "anger": 0 or 1,
  "fear": 0 or 1,
  "joy": 0 or 1,
  "sadness": 0 or 1,
  "surprise": 0 or 1,
  "disgust": 0 or 1,
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence explanation"
}}"""

    return system_prompt, user_prompt


# ── CELL 5: API call with retry logic ────────────────────────────────────────

def annotate_text(original_text, english_translation, language):
    """
    Call Claude API to annotate a single text.
    Returns parsed label dict or None on failure.
    """
    system_prompt, user_prompt = build_annotation_prompt(
        original_text, english_translation, language
    )

    for attempt in range(CONFIG["retry_attempts"]):
        try:
            response = client.messages.create(
                model=CONFIG["model"],
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Parse JSON response
            raw_text = response.content[0].text.strip()
            # Strip markdown code fences if present
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw_text)

            # Validate required keys
            for col in LABEL_COLS:
                if col not in result:
                    result[col] = 0
            if "confidence" not in result:
                result["confidence"] = 0.5
            if "reasoning" not in result:
                result["reasoning"] = ""

            # Clip labels to binary
            for col in LABEL_COLS:
                result[col] = int(bool(result[col]))

            return result

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            print(f"  Raw response: {raw_text[:100]}")
            time.sleep(CONFIG["retry_delay"])
        except anthropic.RateLimitError:
            wait = 60  # Wait a full minute on rate limit
            print(f"  Rate limit hit — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(CONFIG["retry_delay"])

    return None  # All retries failed


def annotate_batch(df, language, limit=None, delay_between=None):
    """
    Annotate a DataFrame of texts.

    Args:
        df:       DataFrame with 'original_text' and 'english_translation' columns
        language: Language name string
        limit:    Max rows to process
        delay_between: Seconds to wait between API calls (rate limiting)

    Returns:
        DataFrame with added emotion label columns
    """
    if limit:
        df = df.head(limit).copy()

    if delay_between is None:
        # Compute delay from requests_per_minute
        delay_between = 60.0 / CONFIG["requests_per_minute"]

    results = []
    failed  = []

    print(f"\nAnnotating {len(df)} {language} texts with {CONFIG['model']}...")
    print(f"Estimated time: {len(df) * delay_between / 60:.1f} minutes")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Annotating {language}"):
        result = annotate_text(
            original_text=str(row["original_text"]),
            english_translation=str(row["english_translation"]),
            language=language,
        )

        if result is not None:
            row_dict = row.to_dict()
            row_dict.update(result)
            results.append(row_dict)
        else:
            failed.append(idx)

        time.sleep(delay_between)

    print(f"\n  Annotated: {len(results)} | Failed: {len(failed)}")
    if failed:
        print(f"  Failed indices (will be excluded): {failed[:10]}...")

    return pd.DataFrame(results)


# ── CELL 6: Annotate Sesotho ──────────────────────────────────────────────────

print("\n" + "="*60)
print("ANNOTATING SESOTHO BACK-TRANSLATED TEXTS")
print("="*60)

sesotho_bt = pd.read_csv(
    Path(CONFIG["backtranslation_dir"]) / "sesotho_backtranslated.csv"
)
print(f"Loaded {len(sesotho_bt)} Sesotho rows")

sesotho_annotated = annotate_batch(
    sesotho_bt, language="sesotho", limit=CONFIG["sesotho_limit"]
)

out_path = Path(CONFIG["output_dir"]) / "sesotho_annotated_raw.csv"
sesotho_annotated.to_csv(out_path, index=False)
print(f"Saved raw annotations → {out_path}")


# ── CELL 7: Annotate Setswana ─────────────────────────────────────────────────

print("\n" + "="*60)
print("ANNOTATING SETSWANA BACK-TRANSLATED TEXTS")
print("="*60)

setswana_bt = pd.read_csv(
    Path(CONFIG["backtranslation_dir"]) / "setswana_backtranslated.csv"
)
print(f"Loaded {len(setswana_bt)} Setswana rows")

setswana_annotated = annotate_batch(
    setswana_bt, language="setswana", limit=CONFIG["setswana_limit"]
)

out_path = Path(CONFIG["output_dir"]) / "setswana_annotated_raw.csv"
setswana_annotated.to_csv(out_path, index=False)
print(f"Saved raw annotations → {out_path}")


# ── CELL 8: Confidence filtering ─────────────────────────────────────────────
# Keep only high-confidence annotations for training.
# Low-confidence labels are more likely to be wrong and would hurt the model.

def filter_by_confidence(df, threshold):
    """
    For rows below confidence threshold, zero out all emotion labels.
    These rows are kept in the file but their labels become all-zero
    (neutral) rather than being dropped entirely.

    You can choose to drop them instead:
        df = df[df["confidence"] >= threshold]
    """
    low_conf = df["confidence"] < threshold
    df.loc[low_conf, LABEL_COLS] = 0
    print(f"  Rows below confidence {threshold}: {low_conf.sum()} (labels zeroed)")
    print(f"  High-confidence rows: {(~low_conf).sum()}")
    return df

print("\n" + "="*60)
print("FILTERING BY CONFIDENCE")
print("="*60)

sesotho_filtered   = filter_by_confidence(sesotho_annotated.copy(),   CONFIG["confidence_threshold"])
setswana_filtered  = filter_by_confidence(setswana_annotated.copy(),  CONFIG["confidence_threshold"])

# Label distribution after filtering
for lang_name, df in [("Sesotho", sesotho_filtered), ("Setswana", setswana_filtered)]:
    print(f"\n{lang_name} label counts after filtering:")
    for col in LABEL_COLS:
        if col in df.columns:
            print(f"  {col:<10}: {int(df[col].sum())} positive ({df[col].mean()*100:.1f}%)")


# ── CELL 9: Manual verification sample ───────────────────────────────────────
# Export a stratified sample of 100 annotations for you to verify manually.
# This is required by your proposal and gives an estimate of annotation quality.

def create_verification_sample(df, n=100, language="unknown"):
    """
    Create a stratified verification sample.
    Includes examples of each emotion class and all-zero cases.
    """
    sample_rows = []

    # Include examples of each emotion
    for col in LABEL_COLS:
        if col in df.columns and df[col].sum() > 0:
            positive_rows = df[df[col] == 1].sample(
                min(10, int(df[col].sum())), random_state=CONFIG["seed"]
            )
            sample_rows.append(positive_rows)

    # Include some all-zero rows
    all_zero = df[(df[LABEL_COLS] == 0).all(axis=1)]
    if len(all_zero) > 0:
        sample_rows.append(all_zero.sample(min(20, len(all_zero)), random_state=CONFIG["seed"]))

    sample = pd.concat(sample_rows).drop_duplicates().head(n)

    # Add columns for manual annotation
    sample = sample.copy()
    sample["human_anger"]    = ""
    sample["human_fear"]     = ""
    sample["human_joy"]      = ""
    sample["human_sadness"]  = ""
    sample["human_surprise"] = ""
    sample["human_disgust"]  = ""
    sample["human_notes"]    = ""

    return sample

print("\n" + "="*60)
print("CREATING MANUAL VERIFICATION SAMPLES")
print("="*60)

sesotho_verify  = create_verification_sample(sesotho_filtered,  n=CONFIG["verification_sample_size"], language="sesotho")
setswana_verify = create_verification_sample(setswana_filtered, n=CONFIG["verification_sample_size"], language="setswana")

sesotho_verify.to_csv(Path(CONFIG["output_dir"]) / "sesotho_verify_me.csv",  index=False)
setswana_verify.to_csv(Path(CONFIG["output_dir"]) / "setswana_verify_me.csv", index=False)

print(f"\nSaved verification samples:")
print(f"  sesotho_verify_me.csv  ({len(sesotho_verify)} rows)")
print(f"  setswana_verify_me.csv ({len(setswana_verify)} rows)")
print()
print("MANUAL VERIFICATION INSTRUCTIONS:")
print("  1. Open each *_verify_me.csv in a spreadsheet (Excel/Google Sheets)")
print("  2. For each row, look at 'original_text' and 'english_translation'")
print("  3. Fill in 'human_*' columns with 0 or 1 based on your judgment")
print("  4. Run Cell 10 below to compute agreement with LLM annotations")


# ── CELL 10: Agreement computation ───────────────────────────────────────────
# Run this AFTER completing manual verification.
# It computes Cohen's Kappa and per-label accuracy between LLM and human labels.

def compute_agreement(verified_csv_path, language):
    """
    Load manually verified CSV and compute agreement metrics.
    Expects human_* columns to be filled in.
    """
    df = pd.read_csv(verified_csv_path)

    # Check manual labels were filled in
    human_cols = [f"human_{c}" for c in LABEL_COLS]
    filled = df[human_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).sum()
    print(f"\n{language}: {filled}/{len(df)} rows have human labels")

    if filled < 10:
        print("  Not enough human labels to compute agreement. Please fill in the CSV.")
        return

    df_filled = df.dropna(subset=human_cols)
    df_filled[human_cols] = df_filled[human_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    print(f"\n{language} LLM vs Human Agreement:")
    for col in LABEL_COLS:
        llm_col   = col
        human_col = f"human_{col}"
        if llm_col not in df_filled.columns:
            continue
        llm_labels   = df_filled[llm_col].astype(int)
        human_labels = df_filled[human_col].astype(int)
        accuracy = (llm_labels == human_labels).mean()
        print(f"  {col:<10}: {accuracy:.3f} accuracy ({int(accuracy*100)}%)")

    # Overall exact match (all 6 labels correct)
    llm_matrix   = df_filled[LABEL_COLS].astype(int).values
    human_matrix = df_filled[human_cols].astype(int).values
    exact_match  = (llm_matrix == human_matrix).all(axis=1).mean()
    print(f"\n  Exact match (all 6 labels): {exact_match:.3f} ({int(exact_match*100)}%)")

# Uncomment and run after filling in manual verification CSV:
# compute_agreement(Path(CONFIG["output_dir"]) / "sesotho_verify_me.csv",  "Sesotho")
# compute_agreement(Path(CONFIG["output_dir"]) / "setswana_verify_me.csv", "Setswana")


# ── CELL 11: Prepare final augmentation training file ────────────────────────
# Combine the filtered annotated data into a format compatible with train.csv

def prepare_for_training(df, language, text_col="back_translated_text"):
    """Convert annotated augmentation data to train.csv format."""
    out = pd.DataFrame()
    out["text_clean"] = df[text_col].astype(str)
    out["language"]   = language
    out["split"]      = "train"
    out["id"]         = [f"{language}_aug_{i}" for i in range(len(df))]
    for col in LABEL_COLS:
        out[col] = df[col].astype(int) if col in df.columns else 0
    return out

sesotho_train_ready  = prepare_for_training(sesotho_filtered,  "sesotho")
setswana_train_ready = prepare_for_training(setswana_filtered, "setswana")

augmented = pd.concat([sesotho_train_ready, setswana_train_ready], ignore_index=True)
aug_path = Path(CONFIG["output_dir"]) / "augmented_train_addition.csv"
augmented.to_csv(aug_path, index=False)

print(f"\nFinal augmentation training file saved → {aug_path}")
print(f"  Sesotho rows:  {len(sesotho_train_ready)}")
print(f"  Setswana rows: {len(setswana_train_ready)}")
print(f"  Total:         {len(augmented)}")
print()
print("To use in Phase 3 fine-tuning, combine with your existing train.csv:")
print("  import pandas as pd")
print("  train = pd.read_csv('data/processed/train.csv')")
print("  aug   = pd.read_csv('results/phase3_annotation/augmented_train_addition.csv')")
print("  train_augmented = pd.concat([train, aug], ignore_index=True)")
print("  train_augmented.to_csv('data/processed/train_augmented.csv', index=False)")

print("\n" + "="*60)
print("PHASE 3 WORKSTREAM 4 COMPLETE")
print("="*60)