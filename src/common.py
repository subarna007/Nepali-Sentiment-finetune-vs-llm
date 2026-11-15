"""Shared utilities: data loading, fixed splits, metrics and result saving."""
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

DEFAULT_DATASET = "raygx/NepCov19Tweets"
LABELS = ["negative", "neutral", "positive"]
RAW_TO_ID = {-1: 0, 0: 1, 1: 2}          # dataset uses -1 / 0 / 1
TEXT_NAMES = {"sentences", "sentence", "text", "tweet", "tweets"}
LABEL_NAMES = {"sentiment", "label", "labels"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _find_columns(df: pd.DataFrame):
    text_col = next((c for c in df.columns if c.lower() in TEXT_NAMES), None)
    label_col = next((c for c in df.columns if c.lower() in LABEL_NAMES), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns in {list(df.columns)}")
    return text_col, label_col


def load_splits(dataset_name: str = DEFAULT_DATASET, data_csv: str = None, seed: int = 42):
    """Return (train, val, test) DataFrames with columns `text` and `label` (0/1/2).

    All available rows are merged, cleaned and de-duplicated *before* splitting,
    so the same tweet can never appear in both train and test. The split is a
    stratified 80/10/10 with a fixed seed, so every method sees the same test set.
    """
    if data_csv:
        df = pd.read_csv(data_csv)
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name)
        df = pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    text_col, label_col = _find_columns(df)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")

    raw = df["label"].astype(int)
    if set(raw.unique()) <= set(RAW_TO_ID):
        df["label"] = raw.map(RAW_TO_ID)
    elif set(raw.unique()) <= {0, 1, 2}:
        df["label"] = raw
    else:
        raise ValueError(f"Unexpected label values: {sorted(raw.unique())}")
    df = df.reset_index(drop=True)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def compute_metrics(y_true, y_pred) -> dict:
    """Metrics over the three real classes. A prediction of -1 (unparsed LLM
    output) counts as an error for accuracy and for every class."""
    ids = list(range(len(LABELS)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ids, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=ids, average="weighted", zero_division=0)),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in ids},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def plot_confusion(cm, title: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(key: str, method: str, test_df: pd.DataFrame, y_pred, out_dir: str,
                 extra: dict = None) -> dict:
    """Write results/<key>.json, a confusion-matrix PNG and a predictions CSV."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = test_df["label"].tolist()
    y_pred = [int(x) for x in y_pred]
    metrics = compute_metrics(y_true, y_pred)
    record = {"key": key, "method": method, "n_test": len(y_true), **metrics, **(extra or {})}
    with open(os.path.join(out_dir, f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    plot_confusion(metrics["confusion_matrix"], method, os.path.join(out_dir, f"{key}_confusion.png"))
    out = test_df.copy()
    out["pred"] = y_pred
    out["true_name"] = [LABELS[i] for i in y_true]
    out["pred_name"] = [LABELS[i] if i >= 0 else "unparsed" for i in y_pred]
    out.to_csv(os.path.join(out_dir, f"{key}_predictions.csv"), index=False, encoding="utf-8")
    print(f"[{method}] accuracy={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}  "
          f"(n={len(y_true)})")
    return record
