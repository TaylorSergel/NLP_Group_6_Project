"""Phase 2 baseline pipeline for multilabel emotion analysis.

How to run in terminal:
python phase2_baseline.py \
  --train data/processed/train.csv \
  --val data/processed/val.csv \
  --test data/processed/test.csv \
  --output outputs/phase2_baselines
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TEXT_COL = "text_clean"
LANG_COL = "language"
LABEL_COLS = ["anger", "fear", "joy", "sadness", "surprise", "disgust"]


def load_split(path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [TEXT_COL, LANG_COL, *LABEL_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("").str.strip()
    df = df[df[TEXT_COL].ne("")]
    df[LANG_COL] = df[LANG_COL].astype(str).str.lower().str.strip()
    for col in LABEL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["phase2_split"] = split_name
    return df.reset_index(drop=True)


def make_pipeline(model_name: str, max_features: int, min_df: int) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    if model_name == "logreg":
        classifier = MultiOutputClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
        )
    elif model_name == "svm":
        classifier = MultiOutputClassifier(
            LinearSVC(class_weight="balanced", max_iter=5000)
        )
    else:
        raise ValueError("model_name must be 'logreg' or 'svm'")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def get_xy(df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
    return df[TEXT_COL].astype(str), df[LABEL_COLS].astype(int).values


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "subset_accuracy": accuracy_score(y_true, y_pred),
    }


def per_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(LABEL_COLS):
        rows.append({
            "emotion": label,
            "support_positive": int(y_true[:, i].sum()),
            "f1": f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "precision": precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "recall": recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
            "accuracy": accuracy_score(y_true[:, i], y_pred[:, i]),
        })
    return pd.DataFrame(rows)


def per_language_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for lang in sorted(df[LANG_COL].unique()):
        idx = df.index[df[LANG_COL] == lang].to_numpy()
        metrics = overall_metrics(y_true[idx], y_pred[idx])
        metrics["language"] = lang
        metrics["samples"] = len(idx)
        rows.append(metrics)
    return pd.DataFrame(rows)[["language", "samples", "macro_f1", "micro_f1", "macro_precision", "macro_recall", "subset_accuracy"]]


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, split_df in df.groupby("phase2_split"):
        for lang, lang_df in split_df.groupby(LANG_COL):
            row = {"split": split_name, "language": lang, "samples": len(lang_df)}
            for label in LABEL_COLS:
                row[label] = int(lang_df[label].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_and_save(model: Pipeline, df: pd.DataFrame, split_name: str, model_name: str, output_dir: Path) -> Dict[str, float]:
    X, y_true = get_xy(df)
    y_pred = model.predict(X)
    metrics = overall_metrics(y_true, y_pred)
    pd.DataFrame([{**{"model": model_name, "split": split_name}, **metrics}]).to_csv(
        output_dir / f"{model_name}_{split_name}_overall.csv", index=False
    )
    per_emotion_metrics(y_true, y_pred).to_csv(
        output_dir / f"{model_name}_{split_name}_per_emotion.csv", index=False
    )
    per_language_metrics(df.reset_index(drop=True), y_true, y_pred).to_csv(
        output_dir / f"{model_name}_{split_name}_per_language.csv", index=False
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 TF-IDF baselines.")
    parser.add_argument("--train", required=True, help="Path to data/processed/train.csv")
    parser.add_argument("--val", required=True, help="Path to data/processed/val.csv")
    parser.add_argument("--test", required=True, help="Path to data/processed/test.csv")
    parser.add_argument("--output", default="outputs/phase2_baselines")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output)
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.train, "train")
    val_df = load_split(args.val, "val")
    test_df = load_split(args.test, "test")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    class_distribution(all_df).to_csv(output_dir / "class_distribution_by_split_language.csv", index=False)

    X_train, y_train = get_xy(train_df)
    summary_rows: List[Dict[str, object]] = []

    for model_name in ["logreg", "svm"]:
        model = make_pipeline(model_name, args.max_features, args.min_df)
        model.fit(X_train, y_train)
        joblib.dump(model, models_dir / f"tfidf_{model_name}.joblib")

        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            metrics = evaluate_and_save(model, split_df, split_name, model_name, output_dir)
            summary_rows.append({"model": model_name, "split": split_name, **metrics})
            print(f"{model_name.upper()} {split_name}: macro F1 = {metrics['macro_f1']:.4f}")

    pd.DataFrame(summary_rows).to_csv(output_dir / "phase2_summary.csv", index=False)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"\nDone. Reports saved to: {output_dir}")


if __name__ == "__main__":
    main()
