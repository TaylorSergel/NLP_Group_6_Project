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

LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "disgust"]
NUM_LABELS = len(LABEL_COLS)


# ── Config ─────────────────────────────────────────────────────

DRIVE_BASE = "/content/drive/MyDrive/project_data"

CONFIG = {
    "data_dir":         f"{DRIVE_BASE}/processed",
    "annotation_dir":   f"{DRIVE_BASE}/results/phase3_annotation",
    "output_dir":       f"{DRIVE_BASE}/results/phase3_transfer",
    "xlmr_model_dir":   f"{DRIVE_BASE}/models/xlmroberta_lora",
    "afri_model_dir":   f"{DRIVE_BASE}/models/afriberta_lora",

    "xlmr_base":   "xlm-roberta-base",
    "afri_base":   "castorini/afriberta_large",

    "max_length":       128,
    "batch_size":       16,
    "num_epochs":       5,
    "learning_rate":    2e-4,
    "weight_decay":     0.01,
    "warmup_ratio":     0.1,
    "grad_accum_steps": 2,
    "threshold":        0.5,

    # ── LoRA config ─────────────────────────────────────
    "lora_r":                16,
    "lora_alpha":            32,
    "lora_dropout":          0.1,
    "lora_target_modules":   ["query", "value"],
}

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)


# ── Dataset class ────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts  = df["text_clean"].astype(str).tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def make_loader(df, tokenizer, batch_size, shuffle=False):
    ds = EmotionDataset(df, tokenizer, CONFIG["max_length"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


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


def train_model(model, train_loader, val_loader, save_dir, num_epochs):
    """Reusable training loop for cross-lingual experiments."""
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"],
    )
    total_steps  = (len(train_loader) // CONFIG["grad_accum_steps"]) * num_epochs
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()
    best_val_f1  = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / CONFIG["grad_accum_steps"]
            scaler.scale(loss).backward()
            total_loss += loss.item() * CONFIG["grad_accum_steps"]
            if (step + 1) % CONFIG["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        val_f1, _, _, _ = evaluate(model, val_loader, CONFIG["threshold"])
        print(f"  Epoch {epoch}  loss={total_loss/len(train_loader):.4f}  val_F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(save_dir)
            print(f"  ✓ Best saved (val_F1={val_f1:.4f})")

    return best_val_f1


def build_lora_model(base_model_name, ignore_size=False):
    """Load a fresh base model and apply LoRA."""
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=ignore_size,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none", modules_to_save=["classifier"],
    )
    return get_peft_model(base, lora_cfg).to(DEVICE)


# ── Load data ─────────────────────────────────────────────────────────

train_df = pd.read_csv(Path(CONFIG["data_dir"]) / "train.csv").dropna(subset=["text_clean"])
val_df   = pd.read_csv(Path(CONFIG["data_dir"]) / "val.csv").dropna(subset=["text_clean"])
test_df  = pd.read_csv(Path(CONFIG["data_dir"]) / "test.csv").dropna(subset=["text_clean"])

# Cross-lingual split: train only on English + Afrikaans
xling_train = train_df[train_df["language"].isin(["english", "afrikaans"])].reset_index(drop=True)
xling_val   = val_df[  val_df["language"].isin(  ["english", "afrikaans"])].reset_index(drop=True)
# Test on isiZulu only
isizulu_test = test_df[test_df["language"] == "isizulu"].reset_index(drop=True)

print(f"Cross-lingual train: {len(xling_train)} rows (English + Afrikaans)")
print(f"Cross-lingual val:   {len(xling_val)} rows")
print(f"isiZulu zero-shot test: {len(isizulu_test)} rows")


# ── Zero-shot test — use existing WS1 model on isiZulu ────────────────

print("\n" + "="*60)
print("EXPERIMENT A: Zero-shot cross-lingual transfer")
print("Train: English + Afrikaans  |  Test: isiZulu (zero-shot)")
print("="*60)

# XLM-RoBERTa cross-lingual experiment
xlmr_tokenizer = AutoTokenizer.from_pretrained(CONFIG["xlmr_base"])
xlmr_xling_model = build_lora_model(CONFIG["xlmr_base"])

xling_train_loader = make_loader(xling_train, xlmr_tokenizer, CONFIG["batch_size"], shuffle=True)
xling_val_loader   = make_loader(xling_val,   xlmr_tokenizer, CONFIG["batch_size"] * 2)
isizulu_loader     = make_loader(isizulu_test, xlmr_tokenizer, CONFIG["batch_size"] * 2)

save_dir = "models/xlmroberta_xling"
print("\nTraining XLM-RoBERTa on English+Afrikaans only...")
train_model(xlmr_xling_model, xling_train_loader, xling_val_loader,
            save_dir, CONFIG["num_epochs"])

# Load best checkpoint and evaluate zero-shot on isiZulu
xlmr_xling_best = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(
        CONFIG["xlmr_base"], num_labels=NUM_LABELS,
        problem_type="multi_label_classification"),
    save_dir,
).to(DEVICE)

zs_f1, zs_per_label, _, _ = evaluate(xlmr_xling_best, isizulu_loader, CONFIG["threshold"])
print(f"\nXLM-RoBERTa zero-shot isiZulu macro F1: {zs_f1:.4f}")
print(f"Baseline LR isiZulu macro F1:           0.2911")
print(f"Improvement over baseline:              {zs_f1 - 0.2911:+.4f}")
print("Per-label F1:", zs_per_label)


# ── Augmented training experiment ─────────────────────────────────────

print("\n" + "="*60)
print("EXPERIMENT B: Augmented training (original + Sesotho/Setswana)")
print("="*60)

aug_path = Path(CONFIG["annotation_dir"]) / "augmented_train_addition.csv"
if aug_path.exists():
    aug_df = pd.read_csv(aug_path)
    train_augmented = pd.concat([train_df, aug_df], ignore_index=True)
    print(f"Original train: {len(train_df)} | Augmented addition: {len(aug_df)}")
    print(f"Total augmented train: {len(train_augmented)}")

    # Retrain XLM-RoBERTa with augmented data
    xlmr_tokenizer_aug = AutoTokenizer.from_pretrained(CONFIG["xlmr_base"])
    xlmr_aug_model     = build_lora_model(CONFIG["xlmr_base"])

    aug_train_loader = make_loader(train_augmented, xlmr_tokenizer_aug,
                                   CONFIG["batch_size"], shuffle=True)
    aug_val_loader   = make_loader(val_df,           xlmr_tokenizer_aug,
                                   CONFIG["batch_size"] * 2)

    save_dir_aug = "models/xlmroberta_lora_augmented"
    print("\nTraining XLM-RoBERTa with augmented data...")
    train_model(xlmr_aug_model, aug_train_loader, aug_val_loader,
                save_dir_aug, CONFIG["num_epochs"])

    # Evaluate on test set
    xlmr_aug_best = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained(
            CONFIG["xlmr_base"], num_labels=NUM_LABELS,
            problem_type="multi_label_classification"),
        save_dir_aug,
    ).to(DEVICE)

    test_loader_aug = make_loader(test_df, xlmr_tokenizer_aug, CONFIG["batch_size"] * 2)
    aug_f1, aug_per_label, _, _ = evaluate(xlmr_aug_best, test_loader_aug, CONFIG["threshold"])
    print(f"\nXLM-RoBERTa + augmentation test macro F1: {aug_f1:.4f}")
    print("Per-label F1:", aug_per_label)
else:
    print(f"Augmented training file not found at {aug_path}")
    print("Run Workstreams 3 and 4 first, then re-run this cell.")
    aug_f1, aug_per_label = None, None


# ── Aggregate all results ────────────────────────────────────────────

print("\n" + "="*60)
print("COLLECTING ALL RESULTS INTO COMPARISON TABLE")
print("="*60)

results_rows = []

results_rows.append({
    "model": "TF-IDF + Logistic Regression",
    "phase": 2,
    "experiment": "baseline",
    "val_macro_f1":  0.4064,
    "test_macro_f1": 0.4356,
    "english_f1":    0.4276,
    "isizulu_f1":    0.2911,
    "afrikaans_f1":  0.1820,
    "augmented":     False,
    "cross_lingual": False,
})
results_rows.append({
    "model": "TF-IDF + SVM",
    "phase": 2,
    "experiment": "baseline",
    "val_macro_f1":  0.4003,
    "test_macro_f1": 0.4220,
    "english_f1":    None,
    "isizulu_f1":    None,
    "afrikaans_f1":  None,
    "augmented":     False,
    "cross_lingual": False,
})

ws1_summary_path = Path("results/phase3_xlmroberta/xlmroberta_summary.csv")
if ws1_summary_path.exists():
    ws1 = pd.read_csv(ws1_summary_path).iloc[0]
    ws1_lang_path = Path("results/phase3_xlmroberta/xlmroberta_per_language.csv")
    ws1_lang = pd.read_csv(ws1_lang_path) if ws1_lang_path.exists() else None

    row = {
        "model": "XLM-RoBERTa-base + LoRA",
        "phase": 3,
        "experiment": "adapter_fine_tune",
        "test_macro_f1": ws1["test_macro_f1"],
        "augmented": False,
        "cross_lingual": False,
    }
    if ws1_lang is not None:
        for _, r in ws1_lang.iterrows():
            row[f"{r['language']}_f1"] = r["macro_f1"]
    results_rows.append(row)

ws2_summary_path = Path("results/phase3_afriberta/afriberta_summary.csv")
if ws2_summary_path.exists():
    ws2 = pd.read_csv(ws2_summary_path).iloc[0]
    ws2_lang_path = Path("results/phase3_afriberta/afriberta_per_language.csv")
    ws2_lang = pd.read_csv(ws2_lang_path) if ws2_lang_path.exists() else None

    row = {
        "model": "AfriBERTa-large + LoRA",
        "phase": 3,
        "experiment": "adapter_fine_tune",
        "test_macro_f1": ws2["test_macro_f1"],
        "augmented": False,
        "cross_lingual": False,
    }
    if ws2_lang is not None:
        for _, r in ws2_lang.iterrows():
            row[f"{r['language']}_f1"] = r["macro_f1"]
    results_rows.append(row)

# Zero-shot cross-lingual result
results_rows.append({
    "model": "XLM-RoBERTa-base + LoRA (cross-lingual)",
    "phase": 3,
    "experiment": "zero_shot_transfer",
    "test_macro_f1": None,  
    "isizulu_f1": round(zs_f1, 4),
    "english_f1": None,
    "afrikaans_f1": None,
    "augmented": False,
    "cross_lingual": True,
})

# Augmented training result
if aug_f1 is not None:
    results_rows.append({
        "model": "XLM-RoBERTa-base + LoRA + Augmentation",
        "phase": 3,
        "experiment": "augmented_fine_tune",
        "test_macro_f1": round(aug_f1, 4),
        "augmented": True,
        "cross_lingual": False,
    })

results_df = pd.DataFrame(results_rows)
results_path = Path(CONFIG["output_dir"]) / "phase3_all_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nFull results table saved → {results_path}")

print("\n" + "="*70)
print("PHASE 3 RESULTS SUMMARY (for your report)")
print("="*70)
display_cols = ["model", "test_macro_f1", "english_f1", "isizulu_f1", "afrikaans_f1"]
display_df = results_df[[c for c in display_cols if c in results_df.columns]].fillna("—")
print(display_df.to_string(index=False))
print("="*70)
print("\nKey findings to report:")
print("  1. Did XLM-RoBERTa beat baseline by ≥5 F1 points on low-resource languages?")
print("  2. Did AfriBERTa outperform XLM-RoBERTa on Afrikaans?")
print("  3. What was the zero-shot transfer gap vs trained-on-isiZulu?")
print("  4. Did augmentation improve Sesotho/Setswana performance?")