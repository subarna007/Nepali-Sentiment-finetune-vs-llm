"""Collect results/*.json into one table and write it into results/results.md
and into README.md (between the RESULTS markers)."""
import argparse
import glob
import json
import os
import re

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def order(rec):
    key = rec["key"]
    group = 0 if key.startswith("tfidf") else 1 if key.startswith("llm") else 2
    return group, rec.get("shots_per_class", 0), key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    records = []
    for path in glob.glob(os.path.join(args.results_dir, "*.json")):
        with open(path, encoding="utf-8") as fh:
            records.append(json.load(fh))
    if not records:
        raise SystemExit("No result files found. Run the experiments first.")
    records.sort(key=order)

    lines = ["| Method | Labelled examples used | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive | Test size |",
             "|---|---|---|---|---|---|---|---|"]
    for r in records:
        pc = r["per_class"]
        lines.append(f"| {r['method']} | {r.get('trained_on', '-'):,} | {r['accuracy']:.4f} | "
                     f"{r['macro_f1']:.4f} | {pc['negative']['f1']:.4f} | {pc['neutral']['f1']:.4f} | "
                     f"{pc['positive']['f1']:.4f} | {r['n_test']:,} |")
    notes = [f"- {r['method']}: {r['unparsed_outputs']} of {r['n_test']} outputs "
             f"({r['unparsed_rate']:.1%}) could not be mapped to a label and were counted as errors."
             for r in records if r.get("unparsed_outputs")]
    table = "\n".join(lines + ([""] + notes if notes else []))

    with open(os.path.join(args.results_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    if os.path.exists(args.readme):
        with open(args.readme, encoding="utf-8") as fh:
            readme = fh.read()
        if START in readme and END in readme:
            readme = re.sub(re.escape(START) + r".*?" + re.escape(END),
                            lambda _: f"{START}\n{table}\n{END}", readme, flags=re.S)
            with open(args.readme, "w", encoding="utf-8") as fh:
                fh.write(readme)
            print("README.md results table updated.")
    print(table)


if __name__ == "__main__":
    main()
