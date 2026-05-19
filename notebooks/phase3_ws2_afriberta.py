# =============================================================================
# PHASE 3 — WORKSTREAM 2: AfriBERTa Adapter Fine-Tuning
# COS 760 — Group 5 | Alisha leads Phase 3
# =============================================================================
#
# WHAT THIS SCRIPT DOES
# ─────────────────────
# Fine-tunes AfriBERTa (castorini/afriberta_large) on multilabel emotion
# classification using the same LoRA adapter approach as Workstream 1.
#
# WHY AfriBERTa IN ADDITION TO XLM-RoBERTa?
# ──────────────────────────────────────────
# XLM-RoBERTa is pretrained on 100 languages but most of its data is from
# high-resource languages (English, German, French, etc.). African languages
# are severely under-represented.
#
# AfriBERTa was pretrained specifically on 11 African languages including
# Afrikaans and several Bantu languages related to isiZulu. This means:
#   - Better subword tokenisation for African languages (fewer UNK tokens)
#   - Richer contextual representations for morphologically complex languages
#   - Expected to outperform XLM-RoBERTa on Afrikaans and possibly isiZulu
#
# ARCHITECTURE NOTE
# ─────────────────
# AfriBERTa-large uses a RoBERTa-style architecture (no token_type_ids).
# The target_modules for LoRA are "query" and "value" — same as XLM-RoBERTa.
# The classification head is also re-trained from scratch.
#
# HOW TO RUN (Google Colab)
# ─────────────────────────
# Same workflow as Workstream 1:
# 1. Runtime → T4 GPU
# 2. Mount Drive and clone repo
# 3. Run install cell, then cells in order
# 4. Outputs → results/phase3_afriberta/ and models/afriberta_lora/
#
# EXPECTED RUNTIME
# ────────────────
# ~60-90 minutes on Colab T4 GPU
# AfriBERTa-large is bigger than XLM-RoBERTa-base so it runs slightly slower.
#
# EXPECTED RESULTS
# ────────────────
# AfriBERTa may outperform XLM-RoBERTa on Afrikaans due to pretraining data.
# On isiZulu, results may be comparable or slightly better.
# On English, XLM-RoBERTa is likely to win (more English pretraining data).
#
# Rough estimates:
# Overall test macro F1:  0.50 – 0.65
# English macro F1:       0.55 – 0.68
# isiZulu macro F1:       0.40 – 0.56
# Afrikaans macro F1:     0.38 – 0.55  ← may beat XLM-RoBERTa here
# =============================================================================


# ── CELL 1: Install dependencies ─────────────────────────────────────────────

# !pip install -q transformers==4.40.0 peft==0.10.0 datasets accelerate \
#              scikit-learn pandas numpy torch sentencepiece


# ── CELL 2: Imports ───────────────────────────────────────────────────────────

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── CELL 3: Configuration ─────────────────────────────────────────────────────
# KEY DIFFERENCE from WS1: model_name is AfriBERTa.
# The rest of the config is nearly identical — this makes comparison fair.

CONFIG = {
    "data_dir": "/content/drive/MyDrive/project_data",
    "output_dir": "/content/drive/MyDrive/project_data/results/phase3_afriberta",
    "model_save_dir": "/content/drive/MyDrive/project_data/models/afriberta_lora",

    # AfriBERTa-large — the flagship model from Ogueji et al. 2021
    # https://huggingface.co/castorini/afriberta_large
    "model_name": "castorini/afriberta_large",

    # LoRA config — same as WS1 for fair comparison
    # AfriBERTa uses the same attention module naming as RoBERTa
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "value"],

    # Training — same hyperparameters as WS1 for fair comparison
    # Reduce batch_size to 8 if OOM (AfriBERTa-large is bigger)
    "max_length": 128,
    "batch_size": 8,           # Smaller than WS1 due to larger model
    "grad_accum_steps": 4,     # Effective batch = 32 (same as WS1's 16×2)
    "num_epochs": 10,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,

    "label_cols": ["anger", "fear", "joy", "sadness", "surprise", "disgust"],
    "threshold": 0.5,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

LABEL_COLS = CONFIG["label_cols"]
NUM_LABELS = len(LABEL_COLS)


# ── CELL 4: Load data ─────────────────────────────────────────────────────────

def load_splits(data_dir):
    splits = {}
    for split in ["train", "val", "test"]:
        path = Path(data_dir) / f"{split}.csv"
        df = pd.read_csv(path)
        df = df.dropna(subset=["text_clean"])
        df["text_clean"] = df["text_clean"].astype(str)
        splits[split] = df
        print(f"Loaded {split}: {len(df)} rows")
    return splits["train"], splits["val"], splits["test"]

train_df, val_df, test_df = load_splits(CONFIG["data_dir"])

# Add this after load_splits() call
def compute_class_weights(df, label_cols, device):
    weights = []
    for col in label_cols:
        pos = df[col].sum()
        neg = len(df) - pos
        weight = min(neg / (pos + 1e-6), 10.0)
        weights.append(weight)
        print(f"  {col:<10}: pos={int(pos)}  neg={int(neg)}  weight={weight:.2f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

print("Class weights:")
pos_weights = compute_class_weights(train_df, LABEL_COLS, DEVICE)


# ── CELL 5: Tokenization check ────────────────────────────────────────────────
# IMPORTANT: AfriBERTa uses SentencePiece tokenization.
# This means it handles African language morphology differently from XLM-RoBERTa.
# Run this cell to verify tokenization works for all three languages.

print("Loading AfriBERTa tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

sample_texts = {
    "Afrikaans": "Die situasie het baie emosionele reaksies veroorsaak.",
    "English":   "The news caused a wave of fear and sadness across the country.",
    "isiZulu":   "Izindaba zenze abantu babesaba kakhulu.",
}

print("\nTokenization samples (fewer tokens = better vocab coverage):")
for lang, text in sample_texts.items():
    tokens = tokenizer.tokenize(text)
    print(f"  {lang}: {len(tokens)} tokens → {tokens}")

# Compare with XLM-RoBERTa if you ran WS1 already:
# AfriBERTa should produce fewer UNK tokens for Afrikaans/isiZulu


# ── CELL 6: Dataset class ─────────────────────────────────────────────────────
# Identical to WS1 — reused without changes

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts  = df["text_clean"].tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── CELL 7: Load model and apply LoRA ────────────────────────────────────────
# AfriBERTa note: it may emit a warning about missing classification head
# weights — this is expected. We're adding a new head on top.

print(f"\nLoading AfriBERTa model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["lora_target_modules"],
    bias="none",
    modules_to_save=["classifier"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model = model.to(DEVICE)


# ── CELL 8: DataLoaders ───────────────────────────────────────────────────────

train_dataset = EmotionDataset(train_df, tokenizer, CONFIG["max_length"])
val_dataset   = EmotionDataset(val_df,   tokenizer, CONFIG["max_length"])
test_dataset  = EmotionDataset(test_df,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"] * 2,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"] * 2,
                          shuffle=False, num_workers=2, pin_memory=True)


# ── CELL 9: Optimizer, scheduler, training loop ───────────────────────────────
# Identical logic to WS1 — same training procedure for fair comparison

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
total_steps  = (len(train_loader) // CONFIG["grad_accum_steps"]) * CONFIG["num_epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = GradScaler()


def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float().cpu().numpy()
            probs  = 1 / (1 + np.exp(-logits))
            preds  = (probs >= threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_label  = {LABEL_COLS[i]: round(float(
                      f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
                  ), 4) for i in range(NUM_LABELS)}
    return macro_f1, per_label, all_preds, all_labels


print("\n" + "="*60)
print("TRAINING AfriBERTa WITH LORA ADAPTERS")
print("="*60)

best_val_f1 = 0.0
patience_counter = 0
patience = 3
history  = []

for epoch in range(1, CONFIG["num_epochs"] + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            loss = loss_fn(outputs.logits, labels) / CONFIG["grad_accum_steps"]

        scaler.scale(loss).backward()
        total_loss += loss.item() * CONFIG["grad_accum_steps"]

        if (step + 1) % CONFIG["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    val_f1, val_per_label, _, _ = evaluate(model, val_loader, CONFIG["threshold"])
    history.append({"epoch": epoch, "train_loss": avg_loss, "val_macro_f1": val_f1})
    print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}")
    print("  Per-label:", val_per_label)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        model.save_pretrained(CONFIG["model_save_dir"])
        tokenizer.save_pretrained(CONFIG["model_save_dir"])
        print(f"  ✓ Best model saved")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break


# ── CELL 10: Test evaluation ──────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL TEST EVALUATION — AfriBERTa")
print("="*60)

base_for_eval = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,
)
best_model = PeftModel.from_pretrained(
    base_for_eval,
    CONFIG["model_save_dir"],
).to(DEVICE)
best_model.eval()
print("AfriBERTa reloaded successfully")

# Per-label thresholds tuned on validation set
best_thresholds = {
    "anger":    0.5,
    "fear":     0.6,
    "joy":      0.55,
    "sadness":  0.45,
    "surprise": 0.6,
    "disgust":  0.3,
}

def evaluate_with_per_label_thresholds(model, loader, thresholds):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float().cpu().numpy()
            probs  = 1 / (1 + np.exp(-logits))
            preds  = np.zeros_like(probs, dtype=int)
            for i, label in enumerate(LABEL_COLS):
                preds[:, i] = (probs[:, i] >= thresholds[label]).astype(int)
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_label  = {LABEL_COLS[i]: round(float(
                    f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
                  ), 4) for i in range(NUM_LABELS)}
    return macro_f1, per_label, all_preds, all_labels

test_macro_f1, test_per_label, test_preds, test_labels = evaluate_with_per_label_thresholds(
    best_model, test_loader, best_thresholds
)

print(f"\nOverall test macro F1: {test_macro_f1:.4f}")
print("\nPer-emotion test F1:")
for label, score in test_per_label.items():
    print(f"  {label:<10}: {score:.4f}")

print("\nPer-language test macro F1:")
lang_results = {}
for lang in test_df["language"].unique():
    lang_df      = test_df[test_df["language"] == lang].reset_index(drop=True)
    lang_dataset = EmotionDataset(lang_df, tokenizer, CONFIG["max_length"])
    lang_loader  = DataLoader(lang_dataset, batch_size=CONFIG["batch_size"] * 2,
                               shuffle=False, num_workers=2)
    macro_f1, per_label, _, _ = evaluate_with_per_label_thresholds(
    best_model, lang_loader, best_thresholds)    
    lang_results[lang] = {"macro_f1": round(macro_f1, 4), "per_label": per_label}
    print(f"  {lang:<12}: {macro_f1:.4f}")
    for emotion, f1 in per_label.items():
        print(f"    {emotion:<10}: {f1:.4f}")


# ── CELL 11: Save results ─────────────────────────────────────────────────────

pd.DataFrame(history).to_csv(
    f"{CONFIG['output_dir']}/afriberta_training_history.csv", index=False)

summary = {
    "model": "afriberta-large-lora",
    "test_macro_f1": round(test_macro_f1, 4),
    "baseline_macro_f1": 0.4356,
    "improvement": round(test_macro_f1 - 0.4356, 4),
}
summary.update({f"f1_{k}": v for k, v in test_per_label.items()})
pd.DataFrame([summary]).to_csv(
    f"{CONFIG['output_dir']}/afriberta_summary.csv", index=False)

lang_rows = []
for lang, res in lang_results.items():
    row = {"language": lang, "model": "afriberta-large-lora",
           "macro_f1": res["macro_f1"]}
    row.update({f"f1_{k}": v for k, v in res["per_label"].items()})
    lang_rows.append(row)
pd.DataFrame(lang_rows).to_csv(
    f"{CONFIG['output_dir']}/afriberta_per_language.csv", index=False)

with open(f"{CONFIG['output_dir']}/afriberta_config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

print(f"\nAll results saved to {CONFIG['output_dir']}/")
print("\n" + "="*60)
print("PHASE 3 WORKSTREAM 2 COMPLETE")
print(f"AfriBERTa test macro F1:        {test_macro_f1:.4f}")
print(f"Logistic Regression (baseline): 0.4356")
print(f"Improvement:                    {test_macro_f1 - 0.4356:+.4f}")
print("="*60)