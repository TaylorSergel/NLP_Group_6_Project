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


# ── Config ─────────────────────────────────────────────────────
CONFIG = {
    "data_dir": "/content/drive/MyDrive/project_data",
    "output_dir": "/content/drive/MyDrive/project_data/results/phase3_afriberta",
    "model_save_dir": "/content/drive/MyDrive/project_data/models/afriberta_lora",

    "model_name": "castorini/afriberta_large",

    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "value"],

    "max_length": 128,
    "batch_size": 8,           
    "grad_accum_steps": 4,     
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


# ── Load data ─────────────────────────────────────────────────────────

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


# ── Tokenization ────────────────────────────────────────────────
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

# ── Dataset class ─────────────────────────────────────────────────────

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


# ── Load model and apply LoRA ────────────────────────────────────────

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


# ── DataLoaders ───────────────────────────────────────────────────────

train_dataset = EmotionDataset(train_df, tokenizer, CONFIG["max_length"])
val_dataset   = EmotionDataset(val_df,   tokenizer, CONFIG["max_length"])
test_dataset  = EmotionDataset(test_df,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"] * 2,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"] * 2,
                          shuffle=False, num_workers=2, pin_memory=True)


# ── Optimizer, scheduler, training loop ───────────────────────────────

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


# ── Test evaluation ──────────────────────────────────────────────────

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


# ── Save results ─────────────────────────────────────────────────────

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