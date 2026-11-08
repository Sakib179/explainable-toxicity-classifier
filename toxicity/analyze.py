"""Dataset-only EDA. Does not load a model or run any training."""

import argparse
from itertools import combinations
from pathlib import Path

from .data import ROOT, SPLITS, clean_indices, load_civil, scan_split, write_json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def quantiles(values):
    return {str(q): float(np.quantile(values, q)) for q in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)}


def conflicting_groups(keys, labels):
    return int(len(np.intersect1d(np.unique(keys[labels == 0]), np.unique(keys[labels == 1]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["hub", "local"], default="hub")
    parser.add_argument("--local-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    raw = load_civil(args.source, args.local_dir)
    scans = {}
    report = {"source": args.source, "label_rule": "toxicity >= 0.5", "splits": {}, "overlaps": {}}
    bins = [0, 0.1, 0.3, 0.4, 0.5, 0.6, 0.8, 1.000001]
    for name in SPLITS:
        print(f"Auditing {name}: {len(raw[name]):,} rows", flush=True)
        scan = scans[name] = scan_split(raw[name])
        keys, labels, scores = scan["keys"], scan["labels"], scan["scores"]
        report["splits"][name] = {
            "rows": len(raw[name]), "schema": str(raw[name].features),
            "nulls": scan["nulls"], "invalid_scores": scan["invalid_scores"],
            "invalid_or_empty_rows": int((~scan["valid"]).sum()),
            "toxic": int(labels.sum()), "not_toxic": int((labels == 0).sum()),
            "toxic_fraction": float(labels.mean()), "toxicity_quantiles": quantiles(scores),
            "exactly_0_5": int((scores == 0.5).sum()),
            "ambiguous_0_4_to_0_6_inclusive": int(((scores >= 0.4) & (scores <= 0.6)).sum()),
            "threshold_sensitivity": {str(t): int((scores >= t).sum()) for t in (0.4, 0.5, 0.6)},
            "toxicity_histogram": {f"[{a}, {b})": int(n) for a, b, n in zip(bins[:-1], bins[1:], np.histogram(scores, bins)[0])},
            "characters": quantiles(scan["chars"]), "whitespace_words": quantiles(scan["words"]),
            "word_length_by_label": {str(c): quantiles(scan["words"][labels == c]) for c in (0, 1)},
            "word_count_above": {str(t): int((scan["words"] > t).sum()) for t in (128, 192, 256, 512)},
            "duplicate_rows_exact": int(len(keys) - len(np.unique(scan["exact_keys"]))),
            "duplicate_rows_normalized": int(len(keys) - len(np.unique(keys))),
            "normalized_groups_with_conflicting_binary_labels": conflicting_groups(keys, labels),
        }
    for a, b in combinations(SPLITS, 2):
        overlap = np.intersect1d(scans[a]["keys"], scans[b]["keys"])
        report["overlaps"][f"{a}__{b}"] = {
            "unique_normalized_texts": len(overlap),
            f"rows_in_{a}": int(np.isin(scans[a]["keys"], overlap).sum()),
            f"rows_in_{b}": int(np.isin(scans[b]["keys"], overlap).sum()),
        }
    clean = clean_indices(scans)
    report["clean_rows"] = {s: len(clean[s]) for s in SPLITS}
    report["majority_only_accuracy"] = {s: float(1 - scans[s]["labels"].mean()) for s in SPLITS}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "dataset_audit.json", report)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in SPLITS:
        scan = scans[name]
        axes[0].hist(scan["scores"], bins=np.linspace(0, 1, 21), weights=np.ones(len(scan["scores"])) / len(scan["scores"]),
                     histtype="step", label=name)
        axes[1].hist(scan["words"], bins=np.arange(0, 551, 10), density=True, histtype="step", label=name)
    axes[0].set(xlabel="Toxicity annotation score", ylabel="Fraction of rows", yscale="log")
    axes[1].set(xlabel="Whitespace words (display clipped at 550)", ylabel="Density")
    axes[2].bar(SPLITS, [scans[s]["labels"].mean() for s in SPLITS])
    axes[2].set(ylabel="Toxic fraction at score ≥ 0.5", ylim=(0, 0.12))
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output / "dataset_distributions.png", dpi=160)
    plt.close(fig)
    lines = ["# Civil Comments: observed split audit", "", "Label: **toxic if toxicity ≥ 0.5**.", "",
             f"Source for this audit: `{args.source}`. All rows were scanned; no model was loaded.", "",
             "| Official split | Rows | Toxic | Toxic % | Median words | 95th percentile words | Normalized duplicate rows |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s, item in report["splits"].items():
        lines.append(f"| {s} | {item['rows']:,} | {item['toxic']:,} | {100 * item['toxic_fraction']:.3f} | "
                     f"{item['whitespace_words']['0.5']:.0f} | {item['whitespace_words']['0.95']:.0f} | {item['duplicate_rows_normalized']:,} |")
    lines += ["", "![Split distributions](dataset_distributions.png)", "", "## Integrity and ambiguity", ""]
    for s, item in report["splits"].items():
        lines.append(f"- {s}: {item['invalid_or_empty_rows']:,} invalid/empty rows; {item['exactly_0_5']:,} scores exactly 0.5; "
                     f"{item['ambiguous_0_4_to_0_6_inclusive']:,} scores in [0.4, 0.6]; "
                     f"{item['normalized_groups_with_conflicting_binary_labels']:,} normalized text groups with conflicting labels.")
    for pair, item in report["overlaps"].items():
        lines.append(f"- {pair}: {item['unique_normalized_texts']:,} shared normalized texts.")
    lines += ["", f"Rows after invalid-row removal and deduplication (test > validation > train): `{report['clean_rows']}`.",
              "", "See `dataset_audit.json` for score ranges, missingness, threshold sensitivity, and full length quantiles.",
              "Whitespace counts are not WordPiece token counts. Sequence-length coverage must be measured with the model tokenizer.",
              "This audit examines test integrity and aggregate distributions only. Model choices and threshold selection use validation, never test outcomes."]
    (args.output / "dataset_analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Saved dataset-only audit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
