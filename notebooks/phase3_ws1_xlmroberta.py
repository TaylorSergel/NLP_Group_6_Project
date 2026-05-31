import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler  # mixed precision for speed

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Config ─────────────────────────────────────────────────────

DRIVE_BASE = "/content/drive/MyDrive/project_data"

CONFIG = {
    "data_dir": f"{DRIVE_BASE}",
    "output_dir": f"{DRIVE_BASE}/results/phase3_xlmroberta",
    "model_save_dir": f"{DRIVE_BASE}/models/xlmroberta_lora",

    "model_name": "xlm-roberta-base",

    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "value"],

    # Training
    "max_length": 128,       
    "batch_size": 16,        
    "num_epochs": 10,
    "learning_rate": 2e-4,  
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,     
    "grad_accum_steps": 2,   

    "label_cols": ["anger", "fear", "joy", "sadness", "surprise", "disgust"],

    "threshold": 0.5,       # Classification threshold
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

LABEL_COLS = CONFIG["label_cols"]
NUM_LABELS = len(LABEL_COLS)


# ── Load data ─────────────────────────────────────────────────────────

def load_splits(data_dir):
    """Load train/val/test CSVs. Validates expected columns exist."""
    splits = {}
    for split in ["train", "val", "test"]:
        path = Path(data_dir) / f"{split}.csv"
        df = pd.read_csv(path)

        missing = [c for c in ["text_clean"] + LABEL_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{split}.csv missing columns: {missing}")

        df = df.dropna(subset=["text_clean"])
        df["text_clean"] = df["text_clean"].astype(str)

        splits[split] = df
        print(f"Loaded {split}: {len(df)} rows")

    return splits["train"], splits["val"], splits["test"]


train_df, val_df, test_df = load_splits(CONFIG["data_dir"])

# Quick sanity check on label distribution
print("\nLabel distribution (train):")
print(train_df[LABEL_COLS].sum().to_string())

def compute_class_weights(df, label_cols, device):
    """
    Compute positive class weights for BCEWithLogitsLoss.
    Weight = (num_negative / num_positive) per label.
    This penalises the model more for missing rare classes.
    """
    weights = []
    for col in label_cols:
        pos = df[col].sum()
        neg = len(df) - pos
        # Cap weight at 10 to avoid extreme values for disgust
        weight = min(neg / (pos + 1e-6), 10.0)
        weights.append(weight)
        print(f"  {col:<10}: pos={int(pos)}  neg={int(neg)}  weight={weight:.2f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)

print("Class weights:")
pos_weights = compute_class_weights(train_df, LABEL_COLS, DEVICE)


# ── Dataset class ─────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text_clean"].tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)
        self.tokenizer = tokenizer
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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ── Load tokenizer and model with LoRA adapters ───────────────────────

print(f"\nLoading tokenizer: {CONFIG['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

print(f"Loading base model: {CONFIG['model_name']}")
base_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=NUM_LABELS,
)

# ── Apply LoRA adapters ──────────────────────────────────────────────────────

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


# ── Create DataLoaders ────────────────────────────────────────────────

train_dataset = EmotionDataset(train_df, tokenizer, CONFIG["max_length"])
val_dataset   = EmotionDataset(val_df,   tokenizer, CONFIG["max_length"])
test_dataset  = EmotionDataset(test_df,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=CONFIG["batch_size"] * 2,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"] * 2,
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"\nBatches per epoch: {len(train_loader)}")


# ── Optimizer and scheduler ──────────────────────────────────────────

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)

total_steps = (len(train_loader) // CONFIG["grad_accum_steps"]) * CONFIG["num_epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

scaler = GradScaler()

print(f"Total training steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")


# ── Evaluation helper ─────────────────────────────────────────────────

def evaluate(model, loader, threshold=0.5):
    """
    Run model on a dataloader, return macro F1, per-label F1, and per-language F1.
    Uses sigmoid + threshold for multilabel predictions.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits.float().cpu().numpy()
            probs  = 1 / (1 + np.exp(-logits))  # sigmoid
            preds  = (probs >= threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-label F1
    per_label_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_label = {LABEL_COLS[i]: round(float(per_label_f1[i]), 4) for i in range(NUM_LABELS)}

    return macro_f1, per_label, all_preds, all_labels


def evaluate_per_language(model, df, threshold=0.5):
    """Evaluate macro F1 separately for each language in the DataFrame."""
    results = {}
    for lang in df["language"].unique():
        lang_df = df[df["language"] == lang].reset_index(drop=True)
        lang_dataset = EmotionDataset(lang_df, tokenizer, CONFIG["max_length"])
        lang_loader  = DataLoader(lang_dataset, batch_size=CONFIG["batch_size"] * 2,
                                  shuffle=False, num_workers=2)
        macro_f1, per_label, _, _ = evaluate_with_per_label_thresholds(model, lang_loader, best_thresholds)
        results[lang] = {"macro_f1": round(macro_f1, 4), "per_label": per_label}
    return results


# ── Training loop ────────────────────────────────────────────────────

print("\n" + "="*60)
print("TRAINING XLM-RoBERTa WITH LORA ADAPTERS")
print("="*60)

best_val_f1   = 0.0
patience      = 3   
patience_counter = 0
history       = []

for epoch in range(1, CONFIG["num_epochs"] + 1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        with autocast():
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

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

    # Validate
    val_f1, val_per_label, _, _ = evaluate(model, val_loader, CONFIG["threshold"])

    history.append({"epoch": epoch, "train_loss": avg_loss, "val_macro_f1": val_f1})
    print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}  "
          f"loss={avg_loss:.4f}  val_macro_F1={val_f1:.4f}")
    print("  Per-label val F1:", val_per_label)

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        model.save_pretrained(CONFIG["model_save_dir"])
        tokenizer.save_pretrained(CONFIG["model_save_dir"])
        print(f"  ✓ New best model saved (val F1={best_val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


# ── Final evaluation on test set ────────────────────────────────────

print("\n" + "="*60)
print("FINAL TEST EVALUATION (best checkpoint)")
print("="*60)

base_for_eval = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
)

lora_config_eval = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=CONFIG["lora_target_modules"],
    bias="none",
    modules_to_save=["classifier"],
)

best_model = PeftModel.from_pretrained(
    base_for_eval,
    CONFIG["model_save_dir"],
).to(DEVICE)

best_model.eval()
print("Model reloaded successfully — classifier weights restored from checkpoint")

# Per-label thresholds tuned on validation set
best_thresholds = {
    "anger":    0.6,
    "fear":     0.55,
    "joy":      0.55,
    "sadness":  0.5,
    "surprise": 0.65,
    "disgust":  0.3,
}

# Modified evaluation using per-label thresholds
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

            preds = np.zeros_like(probs, dtype=int)
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
lang_results = evaluate_per_language(best_model, test_df, CONFIG["threshold"])
for lang, res in lang_results.items():
    print(f"  {lang:<12}: {res['macro_f1']:.4f}")
    for emotion, f1 in res["per_label"].items():
        print(f"    {emotion:<10}: {f1:.4f}")


# Training history
history_df = pd.DataFrame(history)
history_df.to_csv(f"{CONFIG['output_dir']}/xlmroberta_training_history.csv", index=False)

summary = {
    "model": "xlm-roberta-base-lora",
    "test_macro_f1": round(test_macro_f1, 4),
    "baseline_macro_f1": 0.4356,
    "improvement": round(test_macro_f1 - 0.4356, 4),
}
summary.update({f"f1_{k}": v for k, v in test_per_label.items()})
pd.DataFrame([summary]).to_csv(
    f"{CONFIG['output_dir']}/xlmroberta_summary.csv", index=False
)

# Per-language results
lang_rows = []
for lang, res in lang_results.items():
    row = {"language": lang, "model": "xlm-roberta-base-lora",
           "macro_f1": res["macro_f1"]}
    row.update({f"f1_{k}": v for k, v in res["per_label"].items()})
    lang_rows.append(row)
pd.DataFrame(lang_rows).to_csv(
    f"{CONFIG['output_dir']}/xlmroberta_per_language.csv", index=False
)

# Save config for reproducibility
with open(f"{CONFIG['output_dir']}/xlmroberta_config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

print(f"\nAll results saved to {CONFIG['output_dir']}/")
print("\n" + "="*60)
print("PHASE 3 WORKSTREAM 1 COMPLETE")
print(f"XLM-RoBERTa test macro F1:  {test_macro_f1:.4f}")
print(f"Logistic Regression (baseline): 0.4356")
print(f"Improvement:                {test_macro_f1 - 0.4356:+.4f}")
print("="*60)


def tune_thresholds(model, loader, thresholds=np.arange(0.3, 0.71, 0.05)):
    """Find the threshold per label that maximises F1 on a given set."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float().cpu().numpy()
            probs  = 1 / (1 + np.exp(-logits))
            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    best_thresholds = {}
    for i, label in enumerate(LABEL_COLS):
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = t
        best_thresholds[label] = {"threshold": round(float(best_t), 2),
                                  "val_f1": round(best_f1, 4)}

    return best_thresholds

print("\nTuning classification thresholds on validation set...")
best_thresholds = tune_thresholds(best_model, val_loader)
print("\nOptimal thresholds per label:")
for label, info in best_thresholds.items():
    print(f"  {label:<10}: threshold={info['threshold']}  val_F1={info['val_f1']}")

with open(f"{CONFIG['output_dir']}/xlmroberta_best_thresholds.json", "w") as f:
    json.dump(best_thresholds, f, indent=2)