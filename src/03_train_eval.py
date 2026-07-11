#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outputs:
    metrics/m1_<MODEL>.json
    metrics/m1_ALL.json
    metrics/m2_r05_<MODEL>.json
    metrics/m2_r05_ALL.json

Supported models:
    LinearSVC, ClassWeightedSVC, LogisticRegression, CodeBERT, CNN, BiLSTM
"""

import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

try:    import yaml
except Exception as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc

try:    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
except Exception:
    torch = None
    Dataset = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    Trainer = None
    TrainingArguments = None

try:    import tensorflow as tf
    from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except Exception:
    tf = None

KNOWN_MODELS = ["LinearSVC", "ClassWeightedSVC", "LogisticRegression","CodeBERT", "CNN", "BiLSTM"]


# ---------------------------------------------------------------------
# Logging and IO
# ---------------------------------------------------------------------
def setup_logging(workdir: Path) -> None:
    log_dir = workdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_eval_m1m2.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Logging to: %s", log_file)


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        rows = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Bad JSONL at {path}:{i}: {exc}") from exc
        return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------
# Dataset shaping
# ---------------------------------------------------------------------
def make_text_series(df: pd.DataFrame, summary_col: str, comments_col: str) -> pd.Series:
    s = df.get(summary_col, pd.Series([""] * len(df))).fillna("").astype(str)
    c = df.get(comments_col, pd.Series([""] * len(df))).fillna("").astype(str)
    return (s + "\n" + c).astype(str)


def normalize_text_label(df: pd.DataFrame, summary_col: str, comments_col: str, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Available columns: {sorted(df.columns)}")
    out = pd.DataFrame()
    out["text"] = make_text_series(df, summary_col, comments_col)
    out["label"] = df[label_col].fillna("UNKNOWN").astype(str)
    return out.dropna().reset_index(drop=True)


def load_base_splits(workdir: Path, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean_path = workdir / "clean.json"
    train_ids_path = workdir / "splits" / "train_ids.json"
    test_ids_path = workdir / "splits" / "test_ids.json"

    df = pd.DataFrame(load_json_or_jsonl(clean_path))
    train_ids = pd.Series(load_json_or_jsonl(train_ids_path)).astype(int).to_numpy()
    test_ids = pd.Series(load_json_or_jsonl(test_ids_path)).astype(int).to_numpy()

    cols = cfg["dataset"]["columns"]
    summary_col = cols["summary"]
    comments_col = cols["comments"]
    label_col = cols["label"]

    train_raw = df.iloc[train_ids].reset_index(drop=True)
    test_raw = df.iloc[test_ids].reset_index(drop=True)
    train_df = normalize_text_label(train_raw, summary_col, comments_col, label_col)
    test_df = normalize_text_label(test_raw, summary_col, comments_col, label_col)
    logging.info("Base train rows=%d test rows=%d", len(train_df), len(test_df))
    return train_df, test_df


def load_aug_df(workdir: Path, cfg: Dict[str, Any], ratio: str) -> pd.DataFrame:
    aug_path = workdir / "augmented" / f"m2_{ratio}.jsonl"
    logging.info("Loading M2 augmentation: %s", aug_path)
    rows = load_json_or_jsonl(aug_path)
    cols = cfg["dataset"]["columns"]
    return normalize_text_label(pd.DataFrame(rows), cols["summary"], cols["comments"], cols["label"])


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def metrics_base_name(mode: str, ratio: str, model_name: str) -> str:
    if mode == "m1":
        return f"m1_{model_name}"
    if mode == "m2":
        return f"m2_{ratio}_{model_name}"
    if mode == "m3":
        return f"m3_{model_name}"
    raise ValueError(f"Unknown mode: {mode}")


def save_metrics(workdir: Path, base: str, payload: Dict[str, Any], report_text: str) -> None:
    metrics_dir = workdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(metrics_dir / f"{base}.json", json.dumps(payload, indent=2, ensure_ascii=False))
    atomic_write_text(metrics_dir / f"{base}.txt", report_text)
    logging.info("Saved %s", metrics_dir / f"{base}.json")


def summarize_report(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "accuracy": report_dict.get("accuracy"),
        "macro_avg": report_dict.get("macro avg", {}),
        "weighted_avg": report_dict.get("weighted avg", {}),
        "top_k": report_dict.get("top_k", {}),
    }


def safe_topk_scores(model, x_test, y_true, y_train) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)
            score_labels = list(model.classes_)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(x_test)
            score_labels = list(model.classes_)
        else:
            return out
        if len(scores.shape) == 1:
            scores = np.vstack([-scores, scores]).T
        valid_labels = [l for l in sorted(set(pd.Series(y_train).astype(str)) | set(pd.Series(y_true).astype(str))) if l in score_labels]
        for k in [1, 3, 5]:
            kk = min(k, len(valid_labels))
            if kk > 0:
                out[f"top_{k}_accuracy"] = float(top_k_accuracy_score(y_true, scores, k=kk, labels=valid_labels))
    except Exception as exc:
        logging.warning("Could not compute top-k metrics: %s", exc)
    return out




# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def run_tfidf_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, use_smote: bool = False, smote_target_count: int = 0, synthetic_weight: float = 1.0) -> Tuple[float, str, Dict[str, Any]]:
    vectorizer = TfidfVectorizer(max_features=30000)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    y_train = train_df["label"].astype(str)
    y_test = test_df["label"].astype(str)
    sample_weight = None
    if "is_synthetic" in train_df.columns and not use_smote and float(synthetic_weight) != 1.0:
        sample_weight = np.where(train_df["is_synthetic"].astype(int).values == 1, float(synthetic_weight), 1.0)
        logging.info("Using synthetic sample weight %.3f for %s", float(synthetic_weight), model_name)
    if use_smote:
        x_train, y_train = apply_smote_to_tfidf(x_train, y_train, target_count=smote_target_count)
        sample_weight = None

    if model_name == "LinearSVC":
        model = LinearSVC()
    elif model_name == "ClassWeightedSVC":
        model = LinearSVC(class_weight="balanced")
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=2000)
    elif model_name == "ClassWeightedLR":
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "ComplementNB":
        model = ComplementNB()
    else:
        raise ValueError(f"Unknown TF-IDF model: {model_name}")

    logging.info("Training %s", model_name)
    if sample_weight is not None:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    rep_text = classification_report(y_test, preds, digits=4, zero_division=0)
    rep_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
    extra = safe_topk_scores(model, x_test, y_test, y_train)
    if extra:
        rep_dict["top_k"] = extra
        rep_text += "\n\nTop-k metrics:\n" + json.dumps(extra, indent=2) + "\n"
    return acc, rep_text, rep_dict


def run_transformer_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          max_len: int, epochs: int, batch_size: int) -> Tuple[float, str, Dict[str, Any]]:
    if Trainer is None or Dataset is None:
        raise RuntimeError("Transformer dependencies are not available. Install transformers datasets torch accelerate.")

    hf_map = {
        "BERT": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "CodeBERT": "microsoft/codebert-base",
        "DistilBERT": "distilbert-base-uncased",
    }
    if model_name not in hf_map:
        raise ValueError(model_name)

    labels = sorted(set(train_df["label"].astype(str)))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tr = train_df.copy()
    te = test_df[test_df["label"].astype(str).isin(label2id)].copy()
    tr["label_id"] = tr["label"].astype(str).map(label2id)
    te["label_id"] = te["label"].astype(str).map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(hf_map[model_name])
    train_ds = Dataset.from_dict({"text": tr["text"].tolist(), "labels": tr["label_id"].astype(int).tolist()})
    test_ds = Dataset.from_dict({"text": te["text"].tolist(), "labels": te["label_id"].astype(int).tolist()})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    clf = AutoModelForSequenceClassification.from_pretrained(
        hf_map[model_name], num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=1)
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(rep["macro avg"]["f1-score"])}

    try:        args = TrainingArguments(
            output_dir=f"./transformer_{model_name}", eval_strategy="epoch", save_strategy="no",
            learning_rate=2e-5, per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size, num_train_epochs=epochs,
            weight_decay=0.01, logging_steps=50, report_to="none",
        )
    except TypeError:
        args = TrainingArguments(
            output_dir=f"./transformer_{model_name}", evaluation_strategy="epoch", save_strategy="no",
            learning_rate=2e-5, per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size, num_train_epochs=epochs,
            weight_decay=0.01, logging_steps=50, report_to="none",
        )

    trainer = Trainer(
        model=clf, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        processing_class=tokenizer, compute_metrics=compute_metrics,
    )
    logging.info("Training transformer %s", model_name)
    trainer.train()
    pred = trainer.predict(test_ds)
    logits = pred.predictions
    y_true_ids = pred.label_ids
    y_pred_ids = np.argmax(logits, axis=1)
    y_true = [id2label[int(i)] for i in y_true_ids]
    y_pred = [id2label[int(i)] for i in y_pred_ids]
    acc = accuracy_score(y_true, y_pred)
    rep_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    try:        for k in [1, 3, 5]:
            kk = min(k, len(labels))
            rep_dict.setdefault("top_k", {})[f"top_{k}_accuracy"] = float(
                top_k_accuracy_score(y_true_ids, logits, k=kk, labels=np.arange(len(labels)))
            )
        rep_text += "\n\nTop-k metrics:\n" + json.dumps(rep_dict["top_k"], indent=2) + "\n"
    except Exception as exc:
        logging.warning("Transformer top-k failed: %s", exc)
    return acc, rep_text, rep_dict


def encode_labels(train_labels: pd.Series, test_labels: pd.Series):
    labels = sorted(set(train_labels.astype(str)))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    y_train = np.array([label2id.get(x, -1) for x in train_labels.astype(str)], dtype=np.int32)
    y_test = np.array([label2id.get(x, -1) for x in test_labels.astype(str)], dtype=np.int32)
    return y_train, y_test, label2id, id2label


def run_keras_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
                    max_len: int, vocab_size: int, embed_dim: int, epochs: int, batch_size: int) -> Tuple[float, str, Dict[str, Any]]:
    if tf is None:
        raise RuntimeError("TensorFlow is not available.")

    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tok.fit_on_texts(train_df["text"].tolist())
    x_train = pad_sequences(tok.texts_to_sequences(train_df["text"].tolist()), maxlen=max_len, padding="post", truncating="post")
    x_test = pad_sequences(tok.texts_to_sequences(test_df["text"].tolist()), maxlen=max_len, padding="post", truncating="post")

    y_train, y_test, label2id, id2label = encode_labels(train_df["label"], test_df["label"])
    valid = y_test >= 0
    x_test = x_test[valid]
    y_test = y_test[valid]
    n_classes = len(label2id)

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len))
    if model_name == "CNN":
        model.add(Conv1D(128, 5, activation="relu"))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(n_classes, activation="softmax"))
    elif model_name == "BiLSTM":
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(n_classes, activation="softmax"))
    else:
        raise ValueError(model_name)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    logging.info("Training Keras model %s", model_name)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    probs = model.predict(x_test, verbose=0)
    pred_ids = probs.argmax(axis=1)
    y_true = [id2label[int(i)] for i in y_test]
    y_pred = [id2label[int(i)] for i in pred_ids]
    acc = accuracy_score(y_true, y_pred)
    rep_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
    rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return acc, rep_text, rep_dict


def run_model(model_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Dict[str, Any], mode: str = "m1", smote_target_count: int = 0, synthetic_weight: float = 1.0):
    use_smote = False
    if model_name in {"LinearSVC", "ClassWeightedSVC", "LogisticRegression", "ClassWeightedLR", "ComplementNB"}:
        return run_tfidf_model(model_name, train_df, test_df, synthetic_weight=synthetic_weight)
    if model_name in {"CodeBERT", "BERT", "RoBERTa", "DistilBERT"}:
        tc = cfg.get("transformer", {})
        return run_transformer_model(
            model_name, train_df, test_df,
            max_len=int(tc.get("max_len", 256)),
            epochs=int(tc.get("epochs", 3)),
            batch_size=int(tc.get("batch_size", 8)),
        )
    if model_name in {"CNN", "BiLSTM"}:
        dc = cfg.get("deep", {})
        return run_keras_model(
            model_name, train_df, test_df,
            max_len=int(dc.get("max_len", 300)),
            vocab_size=int(dc.get("vocab_size", 30000)),
            embed_dim=int(dc.get("embed_dim", 128)),
            epochs=int(dc.get("epochs", 5)),
            batch_size=int(dc.get("batch_size", 64)),
        )
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--mode", choices=["m1", "m2"], required=True)
    ap.add_argument("--ratio", default="r05", help="Only used for m2, e.g., r05, r10, r25, r50")
    ap.add_argument("--smote_target_count", type=int, default=0, help="Only used for m3. 0=median eligible class count; >0=oversample eligible classes up to this count.")
    ap.add_argument("--synthetic_weight", type=float, default=1.0, help="Only used for m2 TF-IDF/Ensemble models. Weight for synthetic rows, e.g. 0.3.")
    ap.add_argument("--models", nargs="+", default=["LinearSVC", "ClassWeightedSVC", "CodeBERT", "LogisticRegression", "BiLSTM", "CNN"])
    ap.add_argument("--aug_only", type=int, default=0, help="For diagnostics only. 1=train on synthetic rows only.")
    ap.add_argument("--resume", type=int, default=1, help="Skip a model if its metrics JSON already exists.")
    args = ap.parse_args()

    if yaml is None:
        raise RuntimeError(f"PyYAML is required: {YAML_IMPORT_ERROR}")

    workdir = Path(args.workdir)
    setup_logging(workdir)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    base_train_df, test_df = load_base_splits(workdir, cfg)
    base_train_df = base_train_df.copy()
    base_train_df["is_synthetic"] = 0
    ratio = args.ratio.lower()

    if args.mode == "m1":
        train_df = base_train_df
        all_base = "m1_ALL"
        all_payload = {"mode": "m1", "ratio": None, "models": {}}
    elif args.mode == "m2":
        aug_df = load_aug_df(workdir, cfg, ratio)
        aug_df = aug_df.copy()
        aug_df["is_synthetic"] = 1
        train_df = aug_df if args.aug_only else pd.concat([base_train_df, aug_df], ignore_index=True)
        logging.info(
            "M2 training rows=%d base=%d synthetic=%d aug_only=%s synthetic_weight=%.3f",
            len(train_df), len(base_train_df), len(aug_df), args.aug_only, float(args.synthetic_weight),
        )
        all_base = f"m2_{ratio}_ALL"
        all_payload = {
            "mode": "m2",
            "ratio": ratio,
            "models": {},
            "aug_only": bool(args.aug_only),
            "synthetic_rows": int(len(aug_df)),
            "synthetic_weight": float(args.synthetic_weight),
        }

    for model_name in args.models:
        if model_name not in KNOWN_MODELS:
            logging.warning("Skipping unknown model: %s", model_name)
            continue
        base = metrics_base_name(args.mode, ratio, model_name)
        json_path = workdir / "metrics" / f"{base}.json"
        if args.resume and json_path.exists():
            logging.info("Resume: loading existing metrics for %s", model_name)
            try:                payload = json.loads(json_path.read_text(encoding="utf-8"))
                all_payload["models"][model_name] = {
                    "accuracy": payload.get("accuracy"),
                    "macro_avg": payload.get("macro_avg", {}),
                    "weighted_avg": payload.get("weighted_avg", {}),
                    "top_k": payload.get("top_k", {}),
                }
                continue
            except Exception:
                logging.warning("Existing metrics unreadable. Recomputing: %s", json_path)

        try:            acc, rep_text, rep_dict = run_model(model_name, train_df, test_df, cfg, mode=args.mode, smote_target_count=args.smote_target_count, synthetic_weight=args.synthetic_weight)
            summary = summarize_report(rep_dict)
            payload = {
                "mode": args.mode,
                "ratio": ratio if args.mode == "m2" else None,
                "model": model_name,
                "accuracy": acc,
                "macro_avg": summary["macro_avg"],
                "weighted_avg": summary["weighted_avg"],
                "top_k": summary["top_k"],
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "aug_only": bool(args.aug_only) if args.mode == "m2" else False,
                "synthetic_weight": float(args.synthetic_weight) if args.mode == "m2" else None,
                "synthetic_rows": int(train_df.get("is_synthetic", pd.Series([0] * len(train_df))).sum()) if args.mode == "m2" else None,
            }
            save_metrics(workdir, base, payload, rep_text)
            all_payload["models"][model_name] = summarize_report(rep_dict)
        except Exception as exc:
            logging.error("Model failed: %s | %s", model_name, exc)
            logging.error(traceback.format_exc())
            all_payload["models"][model_name] = {"error": str(exc)}

    save_metrics(workdir, all_base, all_payload, json.dumps(all_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
