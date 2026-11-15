"""Fine-tune a multilingual transformer (default: XLM-RoBERTa base) for
three-class Nepali sentiment classification.

The best epoch is selected by macro-F1 on the validation set; the test set is
touched once, at the end.
"""
import argparse
import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import DEFAULT_DATASET, LABELS, load_splits, save_results, set_seed


class TextDataset(Dataset):
    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_loader(df, tokenizer, batch_size, max_len, shuffle):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding=True,
                        return_tensors="pt")
        enc["labels"] = torch.tensor(labels)
        return enc
    return DataLoader(TextDataset(df), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=2)


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):            # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict(model, loader, device, use_amp):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--data_csv", default=None)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--class_weights", action="store_true",
                    help="Weight the loss by inverse class frequency")
    ap.add_argument("--save_model", default=None, help="Directory to save the best model")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    train, val, test = load_splits(args.dataset, args.data_csv, args.seed)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)

    train_loader = make_loader(train, tokenizer, args.batch_size, args.max_len, True)
    val_loader = make_loader(val, tokenizer, args.batch_size * 2, args.max_len, False)
    test_loader = make_loader(test, tokenizer, args.batch_size * 2, args.max_len, False)

    weight = None
    if args.class_weights:
        counts = np.bincount(train["label"], minlength=len(LABELS))
        weight = torch.tensor(len(train) / (len(LABELS) * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps),
                                                total_steps)
    scaler = make_scaler(use_amp)

    history, best_f1, best_state, start = [], -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * labels.size(0)

        val_f1 = f1_score(val["label"], predict(model, val_loader, device, use_amp), average="macro")
        history.append({"epoch": epoch, "train_loss": running / len(train), "val_macro_f1": float(val_f1)})
        print(history[-1])
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    test_pred = predict(model, test_loader, device, use_amp)
    key = "finetuned_" + args.model.split("/")[-1].replace("-", "_")
    save_results(key, f"Fine-tuned {args.model}", test, test_pred, args.out_dir, extra={
        "val_macro_f1": float(best_f1), "history": history, "trained_on": len(train),
        "train_minutes": round((time.time() - start) / 60, 1),
        "hyperparameters": {k: getattr(args, k) for k in
                            ("epochs", "batch_size", "lr", "max_len", "weight_decay",
                             "warmup_ratio", "class_weights", "seed")},
    })
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == "__main__":
    main()
