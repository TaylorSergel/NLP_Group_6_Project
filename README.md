# COS 760 — Emotion Analysis using BRIGHTER + EthioEmo Datasets
**Group 5 | Alisha Perumal, Junior Motsepe, Taylor Sergel**

Multilabel emotion classification across Afrikaans, English, isiZulu, Sesotho, and Setswana using transfer learning (XLM-RoBERTa, AfriBERTa) and data augmentation (back-translation, LLM annotation).

---

## Contents of the Zip File

```
NLP_Group_5_COS760/
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── notebooks/
│   ├── phase1_preprocessing.py        ← Dataset acquisition and preprocessing
│   ├── phase2_baseline.py             ← TF-IDF + Logistic Regression + SVM baselines
│   ├── phase2_visualisations.py       ← Baseline result visualisations
│   ├── phase3_ws1_xlmroberta.py       ← XLM-RoBERTa LoRA fine-tuning
│   ├── phase3_ws2_afriberta.py        ← AfriBERTa LoRA fine-tuning
│   ├── phase3_ws3_backtranslation.py  ← NLLB-200 back-translation augmentation
│   ├── phase3_ws4_llm_annotation.py   ← Claude LLM-based emotion annotation
│   ├── phase3_transfer_and_results.py ← Cross-lingual transfer + results aggregation
│   └── phase4_error_analysis.py       ← Error analysis and visualisations
├── results/
│   ├── phase2_baselines/              ← Baseline result CSVs
│   └── phase2_visualisations/         ← Baseline charts and graphs
└── models/
    └── (empty — model checkpoints saved to Google Drive during training)
```

> **Note on data and model checkpoints:** All processed datasets and trained model checkpoints are stored on Google Drive due to file size constraints (datasets: ~500MB; model checkpoints: ~3GB). See the Data Information section below for download and setup instructions.

---

## Project Overview

This project investigates multilabel emotion classification across five South African languages using the BRIGHTER + EthioEmo dataset. The pipeline covers five phases:

| Phase | Description | 
|---|---|---|
| Phase 1 | Dataset acquisition and preprocessing | 
| Phase 2 | TF-IDF baseline models (Logistic Regression, SVM) |
| Phase 3 | Transfer learning (XLM-RoBERTa, AfriBERTa) + augmentation | 
| Phase 4 | Error analysis and visualisation |
| Phase 5 | Report and presentation |

**Primary result:** XLM-RoBERTa + LoRA achieved test macro F1 of 0.5433 (+0.1077 over baseline). AfriBERTa outperformed XLM-RoBERTa on isiZulu (0.2444 vs 0.0902), demonstrating that language-specific Bantu pretraining benefits low-resource African NLP.

---

## Software Requirements

| Software | Version |
|---|---|
| Python | 3.10+ |
| torch | 2.10.0+cu128 |
| transformers | 4.40.0+ |
| peft | 0.10.0 |
| accelerate | latest |
| scikit-learn | latest |
| pandas | latest |
| numpy | latest |
| sentencepiece | latest |
| matplotlib | latest |
| seaborn | latest |
| anthropic | latest (Phase 3 WS4 only) |

### Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** Phases 3 and 4 require an NVIDIA GPU with at least 16GB VRAM. All GPU experiments were run on Google Colab (Tesla T4). Phases 1 and 2 can run on CPU.

---

## Data Information

### Primary dataset — BRIGHTER + EthioEmo
Loaded automatically from Hugging Face during Phase 1 preprocessing:
```
brighter-dataset/BRIGHTER-emotion-categories
```

### Supplementary datasets
- **Sesotho News Headlines** — download manually from Zenodo:
  `https://doi.org/10.5281/zenodo.10531959`
  Place the downloaded files in `data/raw/sesotho/`

- **PuoData (Setswana)** — loaded automatically from Hugging Face during Phase 1:
  `dsfsi/PuoData`

### Processed data files
The processed CSVs produced by Phase 1 are available on Google Drive and should **not** be re-run from scratch unless necessary (Phase 1 takes ~30 minutes):

**Google Drive folder:** `project_data/` (contact group members for access link)

```
project_data/
├── train.csv                    (6,018 rows — Afrikaans, English, isiZulu)
├── val.csv                      (648 rows)
├── test.csv                     (4,231 rows)
├── sesotho_augmentation.csv     (4,193 rows — no emotion labels)
├── setswana_augmentation.csv    (110,103 rows — no emotion labels)
├── results/
│   ├── phase2_baselines/
│   ├── phase3_xlmroberta/
│   ├── phase3_afriberta/
│   ├── phase3_backtranslation/
│   ├── phase3_annotation/
│   └── phase4_analysis/
└── models/
    ├── xlmroberta_lora/         (XLM-RoBERTa best checkpoint)
    └── afriberta_lora/          (AfriBERTa best checkpoint)
```

To use the processed data without re-running Phase 1, download the Google Drive folder and update the `DRIVE_BASE` path at the top of each Phase 3/4 script.

---

## Setup Instructions

### Local setup (Phase 1 and 2 — runs on CPU in VS Code)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/NLP_Group_5_COS760.git
cd NLP_Group_5_COS760

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Phase 1 preprocessing (downloads datasets automatically)
python notebooks/phase1_preprocessing.py

# 4. Run Phase 2 baselines
python notebooks/phase2_baseline.py
python notebooks/phase2_visualisations.py
```

### Google Colab setup (Phases 3 and 4 — requires GPU)

All Phase 3 and 4 scripts are designed to run on Google Colab with Google Drive for data storage. Follow these steps at the start of every Colab session:

**Step 1 — Set runtime to GPU:**
Runtime → Change runtime type → T4 GPU → Save

**Step 2 — Run setup cells in order:**

```python
# Cell 1 — Verify GPU
import torch
print(torch.cuda.is_available())       # Must print True
print(torch.cuda.get_device_name(0))   # Must print Tesla T4

# Cell 2 — Install dependencies
!pip install -q peft==0.10.0 accelerate scikit-learn \
             pandas numpy sentencepiece matplotlib seaborn

# Cell 3 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 4 — Clone repository
!git clone https://github.com/TaylorSergel/NLP_Group_6_Project.git
%cd NLP_Group_6_Project
print("Repo cloned")
```

**Step 3 — Verify data is accessible:**

```python
import os
DRIVE_BASE = "/content/drive/MyDrive/project_data"
print(os.listdir(DRIVE_BASE))
# Must show: train.csv, val.csv, test.csv, sesotho_augmentation.csv, setswana_augmentation.csv
```

---

## Running the Code

All Phase 3 and 4 scripts read the `DRIVE_BASE` variable at the top of each file. Confirm this matches your Google Drive folder name before running.

### Phase 3 — Transfer Learning

Run each workstream in a separate Colab session. WS1 and WS2 can run in parallel.

```python
# Workstream 1 — XLM-RoBERTa fine-tuning (~6-10 minutes on T4)
!python notebooks/phase3_ws1_xlmroberta.py

# Workstream 2 — AfriBERTa fine-tuning (~8-12 minutes on T4)
!python notebooks/phase3_ws2_afriberta.py

# Workstream 3 — Back-translation augmentation (~110 minutes on T4)
# Run WS1 and WS2 before this
!python notebooks/phase3_ws3_backtranslation.py

# Workstream 4 — LLM annotation (requires Anthropic API key)
# Run WS3 before this
# Set your API key in Colab Secrets (key icon in sidebar): ANTHROPIC_API_KEY
from google.colab import userdata
import os
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
!python notebooks/phase3_ws4_llm_annotation.py

# Results aggregation — run after WS1 and WS2
!python notebooks/phase3_transfer_and_results.py
```

### Phase 4 — Error Analysis

```python
# Run after Phase 3 WS1 and WS2 are complete
# (~5-10 minutes — inference only, no training)
!python notebooks/phase4_error_analysis.py
```

Outputs are saved to `project_data/results/phase4_analysis/` on Drive including:
- Confusion matrix figures (PNG)
- Per-language × per-emotion heatmap (PNG)
- Model comparison bar charts (PNG)
- Full classification report CSVs
- Misclassification example CSVs

### Phase 2 — Baselines (local, no GPU needed)

```python
python notebooks/phase2_baseline.py
python notebooks/phase2_visualisations.py
```

Results saved to `results/phase2_baselines/` and `results/phase2_visualisations/`.

---

## Key Configuration

Each Phase 3/4 script has a `DRIVE_BASE` and `CONFIG` block at the top. The only variable you need to change is `DRIVE_BASE` to match your Google Drive folder name:

```python
# In each Phase 3/4 script — update this one line
DRIVE_BASE = "/content/drive/MyDrive/project_data"  # change if your folder is named differently
```

All other paths are derived automatically from `DRIVE_BASE`.

---

## Reproducibility Notes

- All random seeds are fixed at `seed=42` throughout all scripts
- All hyperparameters are documented in the `CONFIG` dict at the top of each script
- Training used mixed precision (float16) via `torch.cuda.amp`
- Best model checkpoints are saved after every epoch improvement and reloaded for final evaluation
- All result CSVs include the config used to produce them for full traceability

### Key hyperparameters (Phase 3)

| Parameter | XLM-RoBERTa | AfriBERTa |
|---|---|---|
| LoRA rank (r) | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.1 | 0.1 |
| Batch size | 16 | 8 |
| Gradient accumulation | 2 | 4 |
| Effective batch size | 32 | 32 |
| Learning rate | 2e-4 | 2e-4 |
| Epochs | 10 (early stop) | 10 (early stop) |
| Max sequence length | 128 | 128 |
| Optimiser | AdamW | AdamW |

---

## Notes on What to Expect

- **Phase 1** takes ~20-30 minutes on first run (downloading datasets from Hugging Face and Zenodo)
- **Phase 3 WS1 and WS2** each take ~6-12 minutes on a T4 GPU
- **Phase 3 WS3** takes ~110 minutes on a T4 GPU (back-translation of ~9,000 texts)
- **Phase 3 WS4** takes ~90 minutes (API rate-limited to 50 requests/minute)
- **Phase 4** takes ~5-10 minutes (inference only)
- FutureWarnings from `torch.cuda.amp` are expected and do not affect results
- UNEXPECTED/MISSING weight warnings when loading checkpoints are expected and safe to ignore

---

## Contact

For access to the Google Drive folder containing processed datasets and model checkpoints, follow the link for access : https://drive.google.com/drive/folders/1qW-AWSCy1yGBBPOguGdPm9PrYMRO7Etv?usp=sharing