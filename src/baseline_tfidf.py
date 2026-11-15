"""Classical baseline: character n-gram TF-IDF + logistic regression.

Character n-grams suit Nepali well because the language is morphologically
rich and there is no reliable stemmer; no tokeniser is required.
"""
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from common import DEFAULT_DATASET, load_splits, save_results, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--data_csv", default=None, help="Optional local CSV instead of the Hub dataset")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    train, val, test = load_splits(args.dataset, args.data_csv, args.seed)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True)
    x_train, x_val, x_test = (vec.fit_transform(train["text"]), vec.transform(val["text"]),
                              vec.transform(test["text"]))

    best = None
    for c in (0.3, 1.0, 3.0, 10.0):          # regularisation chosen on the validation set
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        clf.fit(x_train, train["label"])
        score = f1_score(val["label"], clf.predict(x_val), average="macro")
        print(f"C={c}: val macro-F1={score:.4f}")
        if best is None or score > best[0]:
            best = (score, c, clf)

    val_f1, c, clf = best
    save_results("tfidf_logreg", "TF-IDF (char 1-4 grams) + Logistic Regression", test,
                 clf.predict(x_test), args.out_dir,
                 extra={"val_macro_f1": float(val_f1), "C": c, "trained_on": len(train)})


if __name__ == "__main__":
    main()
