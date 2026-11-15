| Method | Labelled examples used | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive | Test size |
|---|---|---|---|---|---|---|---|
| TF-IDF (char 1-4 grams) + Logistic Regression | 26,764 | 0.7029 | 0.6568 | 0.7527 | 0.4639 | 0.7538 | 3,346 |
| Qwen2.5-3B-Instruct (zero-shot, no training) | 0 | 0.5475 | 0.4799 | 0.5272 | 0.2664 | 0.6462 | 3,346 |
| Qwen2.5-3B-Instruct (2-shot per class, no training) | 6 | 0.6372 | 0.5391 | 0.6966 | 0.2507 | 0.6700 | 3,346 |
| Fine-tuned xlm-roberta-base | 26,764 | 0.7764 | 0.7074 | 0.8205 | 0.4857 | 0.8161 | 3,346 |
