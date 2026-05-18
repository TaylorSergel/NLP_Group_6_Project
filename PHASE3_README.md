# Phase 3 — Transfer Learning & Augmentation
## COS 760 Group 5 | Led by Alisha

---

## Overview

Phase 3 implements four parallel workstreams to improve on the Phase 2 baselines
(best: Logistic Regression, test macro F1 = **0.4356**).

| Workstream | Script | Goal |
|---|---|---|
| WS1 — XLM-RoBERTa LoRA | `phase3_ws1_xlmroberta.py` | Fine-tune a powerful multilingual model with adapters |
| WS2 — AfriBERTa LoRA | `phase3_ws2_afriberta.py` | Fine-tune an Africa-focused model for comparison |
| WS3 — Back-translation | `phase3_ws3_backtranslation.py` | Generate paraphrased Sesotho/Setswana training data |
| WS4 — LLM Annotation | `phase3_ws4_llm_annotation.py` | Label unlabelled data using Claude API |
| Transfer + Results | `phase3_transfer_and_results.py` | Zero-shot transfer experiments + final comparison table |

---

## Prerequisites

- Phase 1 complete: `data/processed/train.csv`, `val.csv`, `test.csv`, `sesotho_augmentation.csv`, `setswana_augmentation.csv`
- Phase 2 complete: baseline results in `results/phase2_baselines/`
- Google Colab with T4 GPU (free tier is sufficient for WS1 and WS2)
- Google Drive mounted with your project data
- Anthropic API key (for WS4 only)

---

## Recommended Execution Order

```
WS3 (back-translation) ──┐
                          ├──▶ WS4 (LLM annotation) ──┐
                                                        ├──▶ Transfer + Results
WS1 (XLM-RoBERTa) ───────────────────────────────────┤
WS2 (AfriBERTa) ─────────────────────────────────────┘
```

WS1 and WS2 can run in parallel (on separate Colab sessions).
WS3 must finish before WS4.
The Transfer + Results script runs last.

---

## Execution Guide

### Step 1 — Set up Colab environment (all workstreams)

In every new Colab session, run these cells first:

```python
# 1. Install dependencies
!pip install -q transformers==4.40.0 peft==0.10.0 datasets accelerate \
             scikit-learn pandas numpy torch sentencepiece sacremoses

# 2. Mount Google Drive (where your data lives)
from google.colab import drive
drive.mount('/content/drive')

# 3. Clone your repo
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# 4. If data is on Drive, copy it over:
# !cp -r /content/drive/MyDrive/project_data/processed data/
```

### Step 2 — Run WS1 (XLM-RoBERTa) — ~60-90 min on T4

```python
# Either paste the script into Colab cells, or run as a script:
!python phase3_ws1_xlmroberta.py

# Outputs to:
#   results/phase3_xlmroberta/xlmroberta_summary.csv
#   results/phase3_xlmroberta/xlmroberta_per_language.csv
#   models/xlmroberta_lora/   ← best model checkpoint
```

### Step 3 — Run WS2 (AfriBERTa) — ~75-90 min on T4

Open a **separate Colab session** so WS1 and WS2 run in parallel.

```python
!python phase3_ws2_afriberta.py

# Outputs to:
#   results/phase3_afriberta/afriberta_summary.csv
#   models/afriberta_lora/
```

### Step 4 — Run WS3 (Back-translation) — ~2-4 hrs on T4

Back-translation is slow. Start this in a third session or overnight.

```python
!pip install -q transformers sentencepiece sacremoses torch
!python phase3_ws3_backtranslation.py

# Outputs to:
#   results/phase3_backtranslation/sesotho_backtranslated.csv
#   results/phase3_backtranslation/setswana_backtranslated.csv
```

### Step 5 — Run WS4 (LLM Annotation) — ~45-90 min

Requires Anthropic API key. Set in Colab Secrets panel.

```python
# In Colab Secrets (lock icon in sidebar):
# Add secret: ANTHROPIC_API_KEY = sk-ant-...

import os
from google.colab import userdata
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

!pip install -q anthropic pandas
!python phase3_ws4_llm_annotation.py

# Outputs to:
#   results/phase3_annotation/sesotho_annotated_raw.csv
#   results/phase3_annotation/setswana_annotated_raw.csv
#   results/phase3_annotation/sesotho_verify_me.csv   ← MANUALLY VERIFY THIS
#   results/phase3_annotation/setswana_verify_me.csv  ← MANUALLY VERIFY THIS
#   results/phase3_annotation/augmented_train_addition.csv
```

**After WS4:** Open `sesotho_verify_me.csv` and `setswana_verify_me.csv` in
Google Sheets. Fill in the `human_*` columns for all 100 rows per file.
This is your manual verification step (required by the proposal).
Then run `compute_agreement()` in WS4 Cell 10.

### Step 6 — Run Transfer + Results

After WS1, WS2, WS4 are all done:

```python
!python phase3_transfer_and_results.py

# Outputs to:
#   results/phase3_transfer/phase3_all_results.csv  ← the main comparison table
```

---

## Output Files Reference

```
results/
├── phase3_xlmroberta/
│   ├── xlmroberta_summary.csv          # Overall test scores
│   ├── xlmroberta_per_language.csv     # Per-language breakdown
│   ├── xlmroberta_training_history.csv # Loss + val F1 per epoch
│   ├── xlmroberta_config.json          # Hyperparameters (reproducibility)
│   └── xlmroberta_best_thresholds.json # Optimal per-label thresholds
├── phase3_afriberta/
│   ├── afriberta_summary.csv
│   ├── afriberta_per_language.csv
│   └── afriberta_training_history.csv
├── phase3_backtranslation/
│   ├── sesotho_backtranslated.csv
│   └── setswana_backtranslated.csv
├── phase3_annotation/
│   ├── sesotho_annotated_raw.csv
│   ├── setswana_annotated_raw.csv
│   ├── sesotho_verify_me.csv           ← manually verify
│   ├── setswana_verify_me.csv          ← manually verify
│   └── augmented_train_addition.csv    # Final training data addition
└── phase3_transfer/
    └── phase3_all_results.csv          ← MAIN COMPARISON TABLE

models/
├── xlmroberta_lora/          # Best XLM-RoBERTa checkpoint
├── afriberta_lora/           # Best AfriBERTa checkpoint
└── xlmroberta_xling/         # Cross-lingual transfer checkpoint
```

---

## Expected Results

| Model | Test Macro F1 | vs Baseline |
|---|---|---|
| LR Baseline (Phase 2) | 0.4356 | — |
| XLM-RoBERTa + LoRA | ~0.55–0.68 | +0.11–0.24 |
| AfriBERTa + LoRA | ~0.50–0.65 | +0.07–0.21 |
| XLM-RoBERTa + LoRA + Augmentation | ~0.57–0.70 | +0.13–0.26 |
| XLM-RoBERTa zero-shot isiZulu | ~0.25–0.45 | vs 0.29 baseline |

These are estimates. Success criterion from the proposal: **≥5 macro F1 points** improvement on low-resource languages.

### Per-language expectations

| Language | LR Baseline | Expected (XLM-R) | Notes |
|---|---|---|---|
| English | 0.4276 | 0.60–0.72 | Biggest absolute gain |
| isiZulu | 0.2911 | 0.40–0.55 | Main beneficiary of transfer learning |
| Afrikaans | 0.1820 | 0.35–0.50 | AfriBERTa may outperform XLM-R here |

---

## Troubleshooting

**CUDA OOM (out of memory)**
- Reduce `batch_size` from 16 → 8 in CONFIG
- Increase `grad_accum_steps` from 2 → 4 to keep effective batch size the same
- Use `xlm-roberta-base` not `xlm-roberta-large`

**AfriBERTa download fails**
- Try: `!pip install -q huggingface_hub` and re-run
- Check you have internet access in your Colab session

**NLLB-200 produces garbage translations**
- Check NLLB language codes: `sot_Latn` for Sesotho, `tsn_Latn` for Setswana
- Increase `num_beams` from 4 → 5 in WS3 CONFIG

**Anthropic API rate limit (WS4)**
- Reduce `requests_per_minute` to 30 in WS4 CONFIG
- Use `claude-haiku-20240307` (fastest/cheapest) not Sonnet

**Low annotation agreement (WS4)**
- Increase `confidence_threshold` to 0.7 to filter more aggressively
- Manually annotate additional verification rows to get a better estimate

---

## Notes for Report

- Report macro F1 as **primary metric** (per proposal)
- Also report precision, recall, accuracy per emotion class
- Use `phase3_all_results.csv` as the source for your results table
- For error analysis (Phase 4), load `test_preds` and `test_labels` arrays
  from WS1/WS2 evaluate() and examine misclassified rows
- Compare: augmented vs non-augmented, XLM-R vs AfriBERTa, trained vs zero-shot