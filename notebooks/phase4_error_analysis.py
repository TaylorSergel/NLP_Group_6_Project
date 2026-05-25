# =============================================================================
# PHASE 4 — ANALYSIS & ERROR EVALUATION
# COS 760 — Group 5 | Shared phase
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
# Loads the saved WS1 (XLM-RoBERTa) and WS2 (AfriBERTa) model checkpoints
# and runs comprehensive error analysis on the test set. No retraining needed.
#
# Produces:
#   1. Full classification reports (precision, recall, F1 per emotion per language)
#   2. Multilabel confusion matrices per emotion class
#   3. Per-language error heatmaps
#   4. Misclassification examples for qualitative analysis
#   5. Model comparison summary table
#   6. Attention visualisation for selected examples (bertviz)
#
# HOW TO RUN
# ──────────
# Run in the same Colab notebook as WS1/WS2, after those are complete.
# All outputs save to Drive under results/phase4_analysis/
#
# EXPECTED RUNTIME
# ────────────────
# ~5-10 minutes (inference only, no training)
# =============================================================================


# ── CELL 1: Install dependencies ─────────────────────────────────────────────

# !pip install -q transformers peft scikit-learn pandas numpy matplotlib \
#              seaborn bertviz torch


# ── CELL 2: Imports ───────────────────────────────────────────────────────────

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, multilabel_confusion_matrix,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

LABEL_COLS   = ["anger", "fear", "joy", "sadness", "surprise", "disgust"]
NUM_LABELS   = len(LABEL_COLS)
LANGUAGES    = ["english", "afrikaans", "isizulu"]

DRIVE_BASE   = "/content/drive/MyDrive/project_data"
OUTPUT_DIR   = f"{DRIVE_BASE}/results/phase4_analysis"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Saved model paths from Phase 3
XLMR_MODEL_DIR = f"{DRIVE_BASE}/models/xlmroberta_lora"
AFRI_MODEL_DIR = f"{DRIVE_BASE}/models/afriberta_lora"
XLMR_BASE      = "xlm-roberta-base"
AFRI_BASE      = "castorini/afriberta_large"

# Best thresholds from WS1 threshold tuning
XLMR_THRESHOLDS = {
    "anger": 0.5, "fear": 0.6, "joy": 0.55,
    "sadness": 0.45, "surprise": 0.6, "disgust": 0.3,
}
# Use same thresholds for AfriBERTa (can tune separately if needed)
AFRI_THRESHOLDS = XLMR_THRESHOLDS


# ── CELL 3: Dataset class ─────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts      = df["text_clean"].astype(str).tolist()
        self.labels     = df[LABEL_COLS].values.astype(np.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── CELL 4: Load data ─────────────────────────────────────────────────────────

print("Loading test data...")
test_df = pd.read_csv(f"{DRIVE_BASE}/test.csv").dropna(subset=["text_clean"])
test_df["text_clean"] = test_df["text_clean"].astype(str)
print(f"Test set: {len(test_df)} rows")
print(test_df["language"].value_counts())


# ── CELL 5: Inference function ────────────────────────────────────────────────

def get_predictions(model, tokenizer, df, thresholds, batch_size=32):
    """
    Run inference on a DataFrame and return predictions and true labels.
    Uses per-label thresholds for binary prediction.
    """
    dataset = EmotionDataset(df, tokenizer)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2)

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
            logits = outputs.logits.float().cpu().numpy()
            probs  = 1 / (1 + np.exp(-logits))
            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Apply per-label thresholds
    all_preds = np.zeros_like(all_probs, dtype=int)
    for i, label in enumerate(LABEL_COLS):
        all_preds[:, i] = (all_probs[:, i] >= thresholds[label]).astype(int)

    return all_probs, all_preds, all_labels.astype(int)


# ── CELL 6: Load models ───────────────────────────────────────────────────────

print("\nLoading XLM-RoBERTa checkpoint...")
xlmr_tokenizer = AutoTokenizer.from_pretrained(XLMR_BASE)
xlmr_base      = AutoModelForSequenceClassification.from_pretrained(
    XLMR_BASE, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
xlmr_model     = PeftModel.from_pretrained(xlmr_base, XLMR_MODEL_DIR).to(DEVICE)
xlmr_model.eval()
print("XLM-RoBERTa loaded.")

print("\nLoading AfriBERTa checkpoint...")
afri_tokenizer = AutoTokenizer.from_pretrained(AFRI_BASE)
afri_base      = AutoModelForSequenceClassification.from_pretrained(
    AFRI_BASE, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
afri_model     = PeftModel.from_pretrained(afri_base, AFRI_MODEL_DIR).to(DEVICE)
afri_model.eval()
print("AfriBERTa loaded.")


# ── CELL 7: Run inference for both models ─────────────────────────────────────

print("\nRunning XLM-RoBERTa inference...")
xlmr_probs, xlmr_preds, true_labels = get_predictions(
    xlmr_model, xlmr_tokenizer, test_df, XLMR_THRESHOLDS
)

print("Running AfriBERTa inference...")
afri_probs, afri_preds, _ = get_predictions(
    afri_model, afri_tokenizer, test_df, AFRI_THRESHOLDS
)

print("Inference complete.")


# ── CELL 8: Full classification reports ───────────────────────────────────────

print("\n" + "="*70)
print("FULL CLASSIFICATION REPORTS")
print("="*70)

def print_and_save_report(preds, labels, model_name, suffix="overall"):
    """Print sklearn classification report and save to CSV."""
    rows = []
    for i, label in enumerate(LABEL_COLS):
        p = precision_score(labels[:, i], preds[:, i], zero_division=0)
        r = recall_score(   labels[:, i], preds[:, i], zero_division=0)
        f = f1_score(       labels[:, i], preds[:, i], zero_division=0)
        support = int(labels[:, i].sum())
        rows.append({"emotion": label, "precision": round(p, 4),
                     "recall": round(r, 4), "f1": round(f, 4),
                     "support": support})

    df_report = pd.DataFrame(rows)
    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    df_report.loc[len(df_report)] = {
        "emotion": "MACRO AVG", "precision": round(
            precision_score(labels, preds, average="macro", zero_division=0), 4),
        "recall": round(recall_score(labels, preds, average="macro", zero_division=0), 4),
        "f1": round(macro_f1, 4), "support": int(labels.sum()),
    }

    print(f"\n{model_name} — {suffix}")
    print(df_report.to_string(index=False))
    df_report.to_csv(f"{OUTPUT_DIR}/{model_name}_{suffix}_report.csv", index=False)
    return df_report, macro_f1

# Overall reports
xlmr_report, xlmr_macro = print_and_save_report(xlmr_preds, true_labels, "xlmroberta")
afri_report, afri_macro  = print_and_save_report(afri_preds, true_labels, "afriberta")

# Per-language reports
lang_results = {}
for lang in LANGUAGES:
    mask      = test_df["language"] == lang
    lang_true = true_labels[mask]
    lang_xlmr = xlmr_preds[mask]
    lang_afri = afri_preds[mask]

    xlmr_lang_report, xlmr_lang_f1 = print_and_save_report(
        lang_xlmr, lang_true, "xlmroberta", suffix=lang)
    afri_lang_report, afri_lang_f1 = print_and_save_report(
        lang_afri, lang_true, "afriberta",  suffix=lang)

    lang_results[lang] = {
        "xlmr_f1": xlmr_lang_f1,
        "afri_f1": afri_lang_f1,
        "xlmr_report": xlmr_lang_report,
        "afri_report": afri_lang_report,
    }


# ── CELL 9: Multilabel confusion matrices ────────────────────────────────────
# For multilabel classification, each label gets its own 2x2 confusion matrix.
# This shows false positives and false negatives per emotion.

def plot_confusion_matrices(preds, labels, model_name):
    """Plot 2x2 confusion matrix for each emotion label."""
    mcm = multilabel_confusion_matrix(labels, preds)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"{model_name} — Per-Emotion Confusion Matrices (Test Set)",
                 fontsize=14, fontweight="bold")

    for i, (ax, label) in enumerate(zip(axes.flat, LABEL_COLS)):
        cm = mcm[i]  # [[TN, FP], [FN, TP]]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0",    "Actual 1"],
        )
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        total_pos = tp + fn
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        ax.set_title(
            f"{label.upper()}\n"
            f"F1={f1:.3f}  |  Support={total_pos}",
            fontsize=11,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/{model_name}_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

plot_confusion_matrices(xlmr_preds, true_labels, "xlmroberta")
plot_confusion_matrices(afri_preds, true_labels, "afriberta")


# ── CELL 10: Per-language F1 heatmap ─────────────────────────────────────────
# Heatmap showing F1 score per emotion per language for both models.
# This is the key visualisation for your report.

def build_heatmap_data(preds, labels, df):
    """Build emotion x language F1 matrix."""
    data = {}
    for lang in LANGUAGES:
        mask      = df["language"] == lang
        lang_true = labels[mask]
        lang_pred = preds[mask]
        data[lang] = {}
        for i, label in enumerate(LABEL_COLS):
            data[lang][label] = round(
                f1_score(lang_true[:, i], lang_pred[:, i], zero_division=0), 3
            )
    return pd.DataFrame(data).T  # languages as rows, emotions as columns

xlmr_heatmap = build_heatmap_data(xlmr_preds, true_labels, test_df)
afri_heatmap = build_heatmap_data(afri_preds, true_labels, test_df)

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle("Per-Language × Per-Emotion F1 Scores (Test Set)",
             fontsize=14, fontweight="bold")

for ax, heatmap_df, title in zip(
    axes,
    [xlmr_heatmap, afri_heatmap],
    ["XLM-RoBERTa + LoRA", "AfriBERTa + LoRA"]
):
    sns.heatmap(
        heatmap_df, annot=True, fmt=".3f", cmap="RdYlGn",
        vmin=0, vmax=0.8, ax=ax,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Emotion", fontsize=11)
    ax.set_ylabel("Language", fontsize=11)
    ax.set_xticklabels(LABEL_COLS, rotation=30, ha="right")
    ax.set_yticklabels(LANGUAGES, rotation=0)

plt.tight_layout()
path = f"{OUTPUT_DIR}/per_language_emotion_heatmap.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {path}")

# Save heatmap data as CSV for report
xlmr_heatmap.to_csv(f"{OUTPUT_DIR}/xlmroberta_language_emotion_f1.csv")
afri_heatmap.to_csv(f"{OUTPUT_DIR}/afriberta_language_emotion_f1.csv")


# ── CELL 11: Model comparison bar chart ───────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Comparison: Macro F1 per Language",
             fontsize=14, fontweight="bold")

baseline = {"english": 0.4276, "afrikaans": 0.1820, "isizulu": 0.2911}
xlmr_f1  = {lang: lang_results[lang]["xlmr_f1"] for lang in LANGUAGES}
afri_f1  = {lang: lang_results[lang]["afri_f1"] for lang in LANGUAGES}

colors = {"Baseline (LR)": "#95a5a6", "XLM-RoBERTa": "#2ecc71", "AfriBERTa": "#3498db"}

for ax, lang in zip(axes, LANGUAGES):
    models  = ["Baseline\n(LR)", "XLM-\nRoBERTa", "AfriBERTa"]
    scores  = [baseline[lang], xlmr_f1[lang], afri_f1[lang]]
    bar_colors = ["#95a5a6", "#2ecc71", "#3498db"]
    bars = ax.bar(models, scores, color=bar_colors, width=0.5, edgecolor="white")

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title(lang.capitalize(), fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro F1" if lang == "english" else "")
    ax.axhline(y=baseline[lang], color="#95a5a6",
               linestyle="--", alpha=0.5, linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
path = f"{OUTPUT_DIR}/model_comparison_by_language.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {path}")


# ── CELL 12: Emotion-level comparison chart ───────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 6))

x      = np.arange(NUM_LABELS)
width  = 0.25

xlmr_emotion_f1 = [
    f1_score(true_labels[:, i], xlmr_preds[:, i], zero_division=0)
    for i in range(NUM_LABELS)
]
afri_emotion_f1 = [
    f1_score(true_labels[:, i], afri_preds[:, i], zero_division=0)
    for i in range(NUM_LABELS)
]

# Baseline per-emotion F1 from Phase 2 results
baseline_emotion_f1 = {
    "anger": 0.0, "fear": 0.45, "joy": 0.52,
    "sadness": 0.41, "surprise": 0.38, "disgust": 0.0
}
baseline_vals = [baseline_emotion_f1[l] for l in LABEL_COLS]

bars1 = ax.bar(x - width, baseline_vals,   width, label="Baseline (LR)",  color="#95a5a6")
bars2 = ax.bar(x,          xlmr_emotion_f1, width, label="XLM-RoBERTa",   color="#2ecc71")
bars3 = ax.bar(x + width,  afri_emotion_f1, width, label="AfriBERTa",     color="#3498db")

ax.set_xlabel("Emotion", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Per-Emotion F1: Baseline vs XLM-RoBERTa vs AfriBERTa",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([l.capitalize() for l in LABEL_COLS], fontsize=11)
ax.set_ylim(0, 0.9)
ax.legend(fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
path = f"{OUTPUT_DIR}/per_emotion_comparison.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {path}")


# ── CELL 13: Qualitative error analysis ───────────────────────────────────────
# Extract and save misclassified examples for each language and emotion.
# These go in your report as qualitative analysis.

print("\n" + "="*70)
print("QUALITATIVE ERROR ANALYSIS — MISCLASSIFIED EXAMPLES")
print("="*70)

def get_error_examples(df, preds, labels, model_name, n_per_type=5):
    """
    For each emotion, find:
    - False negatives (model missed a real emotion)
    - False positives (model predicted an emotion that isn't there)
    """
    rows = []
    for i, label in enumerate(LABEL_COLS):
        pred_col  = preds[:, i]
        true_col  = labels[:, i]

        # False negatives — missed detections
        fn_mask = (true_col == 1) & (pred_col == 0)
        fn_rows = df[fn_mask].copy()
        fn_rows["error_type"]  = "False Negative (missed)"
        fn_rows["emotion"]     = label
        fn_rows["true_label"]  = 1
        fn_rows["pred_label"]  = 0
        rows.append(fn_rows.head(n_per_type))

        # False positives — hallucinated emotions
        fp_mask = (true_col == 0) & (pred_col == 1)
        fp_rows = df[fp_mask].copy()
        fp_rows["error_type"]  = "False Positive (hallucinated)"
        fp_rows["emotion"]     = label
        fp_rows["true_label"]  = 0
        fp_rows["pred_label"]  = 1
        rows.append(fp_rows.head(n_per_type))

    errors = pd.concat(rows, ignore_index=True)
    keep_cols = ["text_clean", "language", "emotion",
                 "error_type", "true_label", "pred_label"] + LABEL_COLS
    keep_cols = [c for c in keep_cols if c in errors.columns]
    errors = errors[keep_cols]
    path = f"{OUTPUT_DIR}/{model_name}_error_examples.csv"
    errors.to_csv(path, index=False)
    print(f"Saved {len(errors)} error examples → {path}")
    return errors

xlmr_errors = get_error_examples(
    test_df.reset_index(drop=True), xlmr_preds, true_labels, "xlmroberta"
)
afri_errors = get_error_examples(
    test_df.reset_index(drop=True), afri_preds,  true_labels, "afriberta"
)

# Print sample errors for each language for the report
print("\n--- Sample False Negatives by Language (XLM-RoBERTa) ---")
for lang in LANGUAGES:
    lang_fn = xlmr_errors[
        (xlmr_errors["language"] == lang) &
        (xlmr_errors["error_type"] == "False Negative (missed)")
    ].head(2)
    if len(lang_fn) > 0:
        print(f"\n{lang.upper()}:")
        for _, row in lang_fn.iterrows():
            print(f"  Emotion: {row['emotion']}")
            print(f"  Text:    {str(row['text_clean'])[:120]}")
            print()


# ── CELL 14: isiZulu deep dive ────────────────────────────────────────────────
# isiZulu is the most interesting failure case — deep dive into what went wrong.

print("\n" + "="*70)
print("isiZULU DEEP DIVE — WHY DID BOTH MODELS STRUGGLE?")
print("="*70)

isizulu_mask = test_df["language"] == "isizulu"
isizulu_df   = test_df[isizulu_mask].reset_index(drop=True)
isizulu_true = true_labels[isizulu_mask]
isizulu_xlmr = xlmr_preds[isizulu_mask]
isizulu_afri = afri_preds[isizulu_mask]

print(f"\nisiZulu test samples: {len(isizulu_df)}")
print("\nTrue label distribution:")
for i, label in enumerate(LABEL_COLS):
    n = int(isizulu_true[:, i].sum())
    print(f"  {label:<10}: {n} positive ({n/len(isizulu_df)*100:.1f}%)")

print("\nXLM-RoBERTa isiZulu predictions:")
for i, label in enumerate(LABEL_COLS):
    n_pred = int(isizulu_xlmr[:, i].sum())
    n_true = int(isizulu_true[:, i].sum())
    f1     = f1_score(isizulu_true[:, i], isizulu_xlmr[:, i], zero_division=0)
    print(f"  {label:<10}: predicted {n_pred} (true={n_true})  F1={f1:.3f}")

print("\nAfriBERTa isiZulu predictions:")
for i, label in enumerate(LABEL_COLS):
    n_pred = int(isizulu_afri[:, i].sum())
    n_true = int(isizulu_true[:, i].sum())
    f1     = f1_score(isizulu_true[:, i], isizulu_afri[:, i], zero_division=0)
    print(f"  {label:<10}: predicted {n_pred} (true={n_true})  F1={f1:.3f}")

# Token length analysis — are isiZulu texts harder to encode?
print("\nTokenisation analysis:")
xlmr_isizulu_lengths = [
    len(xlmr_tokenizer.encode(t, truncation=True, max_length=128))
    for t in isizulu_df["text_clean"].head(100)
]
xlmr_english_lengths = [
    len(xlmr_tokenizer.encode(t, truncation=True, max_length=128))
    for t in test_df[test_df["language"]=="english"]["text_clean"].head(100)
]
print(f"  XLM-R avg tokens — isiZulu: {np.mean(xlmr_isizulu_lengths):.1f}  "
      f"English: {np.mean(xlmr_english_lengths):.1f}")

afri_isizulu_lengths = [
    len(afri_tokenizer.encode(t, truncation=True, max_length=128))
    for t in isizulu_df["text_clean"].head(100)
]
afri_english_lengths = [
    len(afri_tokenizer.encode(t, truncation=True, max_length=128))
    for t in test_df[test_df["language"]=="english"]["text_clean"].head(100)
]
print(f"  AfriBERTa avg tokens — isiZulu: {np.mean(afri_isizulu_lengths):.1f}  "
      f"English: {np.mean(afri_english_lengths):.1f}")
print("  (More tokens per text = more fragmented subwords = harder for model)")


# ── CELL 15: Summary comparison table ────────────────────────────────────────
# The main table for your report — all models, all languages, all emotions.

print("\n" + "="*70)
print("FINAL SUMMARY TABLE (for report)")
print("="*70)

summary_rows = []

# Baseline from Phase 2
summary_rows.append({
    "model": "TF-IDF + Logistic Regression",
    "overall_macro_f1": 0.4356,
    "english_f1": 0.4276,
    "afrikaans_f1": 0.1820,
    "isizulu_f1": 0.2911,
    "anger_f1": 0.0,
    "fear_f1": 0.45,
    "joy_f1": 0.52,
    "sadness_f1": 0.41,
    "surprise_f1": 0.38,
    "disgust_f1": 0.0,
})

# XLM-RoBERTa
xlmr_row = {
    "model": "XLM-RoBERTa-base + LoRA",
    "overall_macro_f1": round(f1_score(true_labels, xlmr_preds,
                                        average="macro", zero_division=0), 4),
    "english_f1":   round(lang_results["english"]["xlmr_f1"], 4),
    "afrikaans_f1": round(lang_results["afrikaans"]["xlmr_f1"], 4),
    "isizulu_f1":   round(lang_results["isizulu"]["xlmr_f1"], 4),
}
for i, label in enumerate(LABEL_COLS):
    xlmr_row[f"{label}_f1"] = round(
        f1_score(true_labels[:, i], xlmr_preds[:, i], zero_division=0), 4)
summary_rows.append(xlmr_row)

# AfriBERTa
afri_row = {
    "model": "AfriBERTa-large + LoRA",
    "overall_macro_f1": round(f1_score(true_labels, afri_preds,
                                        average="macro", zero_division=0), 4),
    "english_f1":   round(lang_results["english"]["afri_f1"], 4),
    "afrikaans_f1": round(lang_results["afrikaans"]["afri_f1"], 4),
    "isizulu_f1":   round(lang_results["isizulu"]["afri_f1"], 4),
}
for i, label in enumerate(LABEL_COLS):
    afri_row[f"{label}_f1"] = round(
        f1_score(true_labels[:, i], afri_preds[:, i], zero_division=0), 4)
summary_rows.append(afri_row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUTPUT_DIR}/phase4_summary_table.csv", index=False)

# Print readable table
print("\nOverall and per-language macro F1:")
display_cols = ["model", "overall_macro_f1", "english_f1",
                "afrikaans_f1", "isizulu_f1"]
print(summary_df[display_cols].to_string(index=False))

print("\nPer-emotion F1 (overall test set):")
emotion_cols = ["model"] + [f"{l}_f1" for l in LABEL_COLS]
print(summary_df[emotion_cols].to_string(index=False))


# ── CELL 16: Attention visualisation (bertviz) ────────────────────────────────
# Visualise attention weights for selected examples.
# This shows WHAT the model focuses on when predicting each emotion.
# Run this cell separately — it opens an interactive widget.

print("\n" + "="*70)
print("ATTENTION VISUALISATION (optional — run separately)")
print("="*70)

def visualise_attention(text, model, tokenizer, model_name, true_labels_str=""):
    """
    Show attention weights for a single text using bertviz.
    Open in Colab cell output — interactive HTML.
    """
    try:
        from bertviz import head_view
        from transformers import AutoModel
    except ImportError:
        print("Install bertviz: !pip install bertviz")
        return

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=128).to(DEVICE)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get attention weights from base model
    base = model.base_model
    with torch.no_grad():
        outputs = base(**inputs, output_attentions=True)

    attention = outputs.attentions  # tuple of tensors per layer
    print(f"\nModel: {model_name}")
    print(f"Text: {text[:80]}")
    print(f"True emotions: {true_labels_str}")
    print(f"Tokens: {tokens}")
    print(f"Layers: {len(attention)}  |  Heads per layer: {attention[0].shape[1]}")

    # Display interactive attention view
    head_view(attention, tokens)


# Select interesting examples for visualisation
# 1. isiZulu example where model got sadness right
isizulu_sadness = test_df[
    (test_df["language"] == "isizulu") &
    (test_df["sadness"] == 1)
]["text_clean"].iloc[0]

# 2. English example with multiple emotions
english_multi = test_df[
    (test_df["language"] == "english") &
    (test_df[LABEL_COLS].sum(axis=1) >= 2)
]["text_clean"].iloc[0]

print("Example texts for attention visualisation:")
print(f"\n1. isiZulu (sadness): {isizulu_sadness[:100]}")
print(f"\n2. English (multi-emotion): {english_multi[:100]}")
print("\nTo visualise attention, run:")
print("  visualise_attention(isizulu_sadness, xlmr_model, xlmr_tokenizer, 'XLM-RoBERTa')")
print("  visualise_attention(english_multi,   xlmr_model, xlmr_tokenizer, 'XLM-RoBERTa')")


# ── CELL 17: Save all outputs summary ────────────────────────────────────────

print("\n" + "="*70)
print("PHASE 4 COMPLETE — ALL OUTPUTS SAVED")
print("="*70)
print(f"\nAll files saved to: {OUTPUT_DIR}/")
print("""
Generated files:
  phase4_summary_table.csv              ← Main results table for report
  xlmroberta_overall_report.csv         ← Full classification report
  afriberta_overall_report.csv
  xlmroberta_english/afrikaans/isizulu_report.csv   ← Per-language reports
  afriberta_english/afrikaans/isizulu_report.csv
  xlmroberta_confusion_matrices.png     ← Per-emotion confusion matrices
  afriberta_confusion_matrices.png
  per_language_emotion_heatmap.png      ← KEY FIGURE for report
  model_comparison_by_language.png      ← KEY FIGURE for report
  per_emotion_comparison.png            ← KEY FIGURE for report
  xlmroberta_error_examples.csv         ← Qualitative error analysis
  afriberta_error_examples.csv
  xlmroberta_language_emotion_f1.csv    ← Raw data for heatmap
  afriberta_language_emotion_f1.csv
""")