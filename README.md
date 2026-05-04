<div align="center">

# Emotion Analysis in African Languages
### COS 760 — Group 5

*Exploring Emotion Analysis in African Languages using Transfer Learning and Data Augmentation*

**Alisha Perumal · Junior Motsepe · Taylor Sergel**

</div>

---

<div align="center">

## Project Overview

</div>

This repository contains all code, data processing scripts, and experimental results for our COS 760 project on multilingual emotion analysis across five African languages. Using the **BRIGHTER + EthioEmo** datasets as the primary resource, we investigate how effectively transfer learning using **XLM-RoBERTa** and **AfriBERTa** improves multilabel emotion classification for low-resource languages including isiZulu, Sesotho and Setswana.

---

<div align="center">

## Languages

</div>

| Language | Region | Resource Level |
|---|---|---|
| English | United States/Canada | High-resource |
| Afrikaans | South Africa | High-resource |
| isiZulu | South Africa | Low-resource |
| Sesotho | South Africa | Low-resource |
| Setswana | South Africa | Low-resource |

---

<div align="center">

## Datasets

</div>

| Dataset | Purpose | Source |
|---|---|---|
| BRIGHTER + EthioEmo | Primary — multilabel emotion annotation across 17+ African languages | Hugging Face |
| Sesotho News Headlines | Supplementary — supports augmentation for Sesotho | Paper / Dataset |

---

<div align="center">

## Project Phases

</div>

| Phase | Description | Lead |
|---|---|---|
| Phase 1 | Dataset acquisition & preprocessing | Taylor |
| Phase 2 | Baseline model training (TF-IDF + LR/SVM) | Junior |
| Phase 3 | Transfer learning & data augmentation | Alisha |
| Phase 4 | Analysis & error evaluation | All |
| Phase 5 | Report & presentation preparation | All |

---

<div align="center">

## Methodology

</div>

**Research Question:**
> *"How effective is transfer learning using multilingual models (e.g. XLM-RoBERTa, AfriBERTa) in improving emotion classification performance for isiZulu, Sesotho and Setswana compared to Afrikaans and English?"*

**Approach:**
1. **Baseline Modelling** — TF-IDF representations with Logistic Regression and SVM classifiers trained across all five languages
2. **Transfer Learning** — Adapter-based fine-tuning of XLM-RoBERTa and AfriBERTa; cross-lingual transfer from high-resource to low-resource languages
3. **Data Augmentation** — Back-translation and LLM-based paraphrasing to address data scarcity for Sesotho and Setswana
4. **Explanatory Evaluation** — Attention visualisation and error analysis on misclassified instances

---

<div align="center">

## Evaluation Metrics

</div>

| Task | Metric |
|---|---|
| Multilabel classification | Macro F1-score, Precision, Recall, Accuracy |
| Emotion intensity prediction | Pearson r (where intensity annotations are available) |

---

<div align="center">

## Repository Structure

</div>

```
cos760-group5/
├── data/
│   ├── raw/               ← Downloaded datasets
│   ├── processed/         ← Cleaned and split data
│   └── augmented/         ← Augmented training data
├── notebooks/
│   ├── phase1_preprocessing.ipynb
│   ├── phase2_baselines.ipynb
│   ├── phase3_transfer_learning.ipynb
│   └── phase4_analysis.ipynb
├── models/                ← Saved model checkpoints
├── results/               ← Evaluation outputs and metrics
└── README.md
```

---

<div align="center">

## Getting Started

</div>

### Prerequisites
- Python 3.9+
- Google Colab (recommended) or a local environment with GPU access

### Install Dependencies
```bash
pip install datasets transformers adapters torch scikit-learn pandas numpy
```

### Clone the Repository
```bash
git clone https://github.com/<your-username>/cos760-group5.git
cd cos760-group5
```

### Run in Google Colab
Each phase has a corresponding notebook in the `notebooks/` folder. Open the relevant notebook in Google Colab, switch the runtime to **GPU** (Runtime → Change runtime type → GPU), and run the cells in order.

---

<div align="center">

## Timeline

</div>

| Milestone | Date |
|---|---|
| Proposal submission | 10 April 2025 |
| Phase 1 — Data acquisition & preprocessing | 11–18 April 2025 |
| Phase 2 — Baseline model training | 19–25 April 2025 |
| Phase 3 — Transfer learning & augmentation | 26 April – 9 May 2025 |
| Phase 4 — Analysis & error evaluation | 10–16 May 2025 |
| Phase 5 — Report & presentation | 17–23 May 2025 |
| **Final deliverables due** | **27 May 2025** |

---

<div align="center">

## License

</div>

This project is created for educational purposes as part of **COS 760**.
