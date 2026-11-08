"""Tokenizer-only EDA on reproducible stratified samples; no model weights loaded."""

import argparse
from pathlib import Path

from .data import ROOT, SPLITS, load_civil, stratified_indices, write_json

import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["hub", "local"], default="hub")
    parser.add_argument("--local-dir", type=Path, default=ROOT)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased", use_fast=True)
    raw = load_civil(args.source, args.local_dir)
    report = {"source": args.source, "seed": 42, "sampling": "stratified within each official raw split",
              "tokenizer": "distilbert/distilbert-base-uncased", "special_tokens_included": True,
              "model_weights_loaded": False, "splits": {}}
    for name in SPLITS:
        labels = (np.asarray(raw[name]["toxicity"]) >= 0.5).astype(int)
        rows = stratified_indices(np.arange(len(raw[name])), labels, args.sample_size, 42)
        lengths = []
        for batch in raw[name].select(rows.tolist()).iter(batch_size=256):
            encoded = tokenizer(batch["text"], truncation=False, verbose=False)
            lengths.extend(map(len, encoded["input_ids"]))
        lengths = np.asarray(lengths)
        report["splits"][name] = {
            "sample_rows": len(rows), "source_rows": rows.tolist(),
            "toxic_fraction": float(labels[rows].mean()),
            "length_quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0.5, 0.9, 0.95, 0.99, 1)},
            "truncation_fraction": {str(t): float((lengths > t).mean()) for t in (128, 192, 256, 384, 512)},
            "truncation_fraction_at_256_by_label": {
                str(c): float((lengths[labels[rows] == c] > 256).mean()) for c in (0, 1)},
        }
        print(name, report["splits"][name]["length_quantiles"],
              report["splits"][name]["truncation_fraction"], flush=True)
    write_json(args.output / "tokenizer_audit.json", report)
    lines = ["# Tokenizer-only length audit", "",
             "DistilBERT WordPiece tokenizer; special tokens included. Seed 42; 5,000 stratified raw rows per split by default.",
             "Only tokenizer files were loaded. No pretrained model weights, training, or model inference.", "",
             "| Split | Sample rows | Median tokens | P95 tokens | P99 tokens | >128 | >192 | >256 | >384 | >512 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, item in report["splits"].items():
        q, t = item["length_quantiles"], item["truncation_fraction"]
        lines.append(f"| {name} | {item['sample_rows']:,} | {q['0.5']:.0f} | {q['0.95']:.0f} | {q['0.99']:.0f} | " +
                     " | ".join(f"{t[str(limit)]:.2%}" for limit in (128, 192, 256, 384, 512)) + " |")
    lines += ["", "Rates are sample estimates. The experiment also records exact coverage on its selected train and model-selection subsets.",
              "Maximum length is a compute tradeoff; a small rate of truncated comments can still affect difficult examples.",
              "Test token lengths are descriptive; the training subset is the basis for choosing the sequence limit."]
    (args.output / "tokenizer_analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
