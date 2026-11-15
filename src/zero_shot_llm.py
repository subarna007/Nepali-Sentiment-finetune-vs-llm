"""Zero-shot and few-shot sentiment classification with an open instruction-tuned LLM.

The LLM is *not* trained on this dataset. It only receives an instruction (and,
with --shots k, k labelled examples per class taken from the training split).
Outputs that cannot be mapped to a label are counted as errors and reported.
"""
import argparse
import time

from common import DEFAULT_DATASET, LABELS, load_splits, save_results, set_seed

SYSTEM = ("You are a sentiment classifier for tweets written in Nepali. "
          "Classify the sentiment of the tweet as exactly one of: positive, negative, neutral. "
          "Reply with only that one English word.")

# English labels plus common Nepali equivalents the model may answer with.
ALIASES = {
    "negative": 0, "नकारात्मक": 0,
    "neutral": 1, "तटस्थ": 1,
    "positive": 2, "सकारात्मक": 2,
}


def parse_label(text: str) -> int:
    """Map raw model output to a label id; -1 if no label word is found."""
    text = text.strip().lower()
    hits = [(text.find(word), idx) for word, idx in ALIASES.items() if word in text]
    return min(hits)[1] if hits else -1


def build_messages(tweet: str, examples) -> list:
    messages = [{"role": "system", "content": SYSTEM}]
    for ex_text, ex_label in examples:
        messages.append({"role": "user", "content": f"Tweet: {ex_text}"})
        messages.append({"role": "assistant", "content": LABELS[ex_label]})
    messages.append({"role": "user", "content": f"Tweet: {tweet}"})
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--data_csv", default=None)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--shots", type=int, default=0, help="Labelled examples per class in the prompt")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Evaluate on a stratified subset of the test set (0 = full test set)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    train, _, test = load_splits(args.dataset, args.data_csv, args.seed)
    if args.max_samples and args.max_samples < len(test):
        from sklearn.model_selection import train_test_split
        test, _ = train_test_split(test, train_size=args.max_samples, stratify=test["label"],
                                   random_state=args.seed)
        test = test.reset_index(drop=True)

    examples = []
    if args.shots:
        picked = train.groupby("label").sample(n=args.shots, random_state=args.seed)
        picked = picked.sample(frac=1.0, random_state=args.seed)      # mix class order
        examples = list(zip(picked["text"], picked["label"]))

    cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"                 # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if cuda else torch.float32,
        device_map="auto" if cuda else None)
    model.eval()

    prompts = [tokenizer.apply_chat_template(build_messages(t, examples), tokenize=False,
                                             add_generation_prompt=True) for t in test["text"]]
    preds, raw, start = [], [], time.time()
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="LLM inference"):
        enc = tokenizer(prompts[i:i + args.batch_size], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raw.extend(texts)
        preds.extend(parse_label(t) for t in texts)

    unparsed = sum(p == -1 for p in preds)
    mode = "zero-shot" if args.shots == 0 else f"{args.shots}-shot per class"
    short = args.model.split("/")[-1]
    key = f"llm_{short.replace('-', '_').replace('.', '_').lower()}_{args.shots}shot"
    test = test.copy()
    test["raw_output"] = raw
    save_results(key, f"{short} ({mode}, no training)", test, preds, args.out_dir, extra={
        "shots_per_class": args.shots, "unparsed_outputs": int(unparsed),
        "unparsed_rate": unparsed / len(preds), "trained_on": args.shots * len(LABELS),
        "inference_minutes": round((time.time() - start) / 60, 1), "system_prompt": SYSTEM,
    })


if __name__ == "__main__":
    main()
