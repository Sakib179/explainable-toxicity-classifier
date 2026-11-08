"""Classify a new comment with a locally saved experiment checkpoint."""

import argparse
import json
from pathlib import Path

from .model import choose_device, load_artifact, predict_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=["frozen", "untuned", "tuned"], default="tuned")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must contain a nonempty comment")
    config = json.loads((args.run / "config.json").read_text())
    thresholds = json.loads((args.run / "thresholds.json").read_text())
    device = choose_device(args.device)
    model, tokenizer = load_artifact(args.run, args.model, device)
    score = float(predict_texts(model, tokenizer, [args.text], device, config["max_length"])[0, 1])
    token_length = len(tokenizer(args.text, truncation=False, verbose=False)["input_ids"])
    print(json.dumps({"text": args.text, "model": args.model, "toxic_score": score,
                      "decision_threshold": thresholds[args.model],
                      "label": "toxic" if score >= thresholds[args.model] else "not toxic",
                      "truncated": token_length > config["max_length"]}, indent=2))


if __name__ == "__main__":
    main()
