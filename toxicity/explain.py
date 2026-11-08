"""Sourced out-of-domain examples and local LIME explanations after model selection."""

import argparse
import json
from pathlib import Path

from .data import ROOT, text_key, write_json
from .model import choose_device, load_artifact, predict_texts, seed_everything

import numpy as np
import matplotlib
matplotlib.use("Agg")
from datasets import load_from_disk
from lime.lime_text import LimeTextExplainer


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def selected_test_cases(test, probability, threshold):
    labels = np.asarray(test["label"])
    predictions = probability >= threshold
    cases = []
    for name, truth, prediction in (("true_positive", 1, 1), ("true_negative", 0, 0),
                                     ("false_positive", 0, 1), ("false_negative", 1, 0)):
        indices = np.flatnonzero((labels == truth) & (predictions == prediction))
        if len(indices) == 0:
            continue  # A missing error category is reported, never manufactured.
        short = [i for i in indices if len(test[int(i)]["text"].split()) <= 80]
        if short:
            indices = np.asarray(short)
        i = int(indices[np.argmax(np.abs(probability[indices] - threshold))])
        row = test[i]
        cases.append({"id": f"test_{name}", "kind": "held_out_test_diagnostic",
                      "text": row["text"], "expected_label": int(row["label"]),
                      "annotation_score": row["toxicity"], "source_row": row["source_row"],
                      "note": "Most confident example of this category, preferring ≤80 whitespace words; selected after evaluation for diagnosis only."})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text())
    thresholds = json.loads((args.run / "thresholds.json").read_text())
    output = args.run / "interpretability"
    output.mkdir(exist_ok=True)
    seed_everything(config["seed"])
    device = choose_device(args.device)
    parts = load_from_disk(str(args.run / "data"))
    observed_keys = {name: set(part["text_hash"]) for name, part in parts.items()}
    examples = read_jsonl(ROOT / "examples" / "real_world.jsonl") + read_jsonl(ROOT / "examples" / "stress_cases.jsonl")
    for example in examples:
        key = text_key(example["text"])
        example["overlap_with_experiment_splits"] = [s for s, keys in observed_keys.items() if key in keys]
    predictions = []
    for name in ("frozen", "untuned", "tuned"):
        model, tokenizer = load_artifact(args.run, name, device)
        scores = predict_texts(model, tokenizer, [e["text"] for e in examples], device,
                               config["max_length"], config["batch_size"])
        for example, score in zip(examples, scores[:, 1]):
            predictions.append({**example, "model": name, "toxic_score": float(score),
                                "threshold": thresholds[name], "predicted_label": int(score >= thresholds[name])})
        del model
    write_json(output / "external_predictions.json", predictions)
    model, tokenizer = load_artifact(args.run, "tuned", device)
    archive = np.load(args.run / "tuned_predictions.npz")
    cases = selected_test_cases(parts["test"], archive["test_scores"], thresholds["tuned"])
    # Include one authentic hostile forum excerpt and one authentic anti-abuse statement.
    cases += [e for e in examples if e["id"] in ("real_03", "real_06")]

    def predict(texts):
        return predict_texts(model, tokenizer, texts, device, config["max_length"], config["batch_size"])

    details = []
    for case in cases:
        explanations = []
        for repeat, seed in enumerate((config["seed"], config["seed"] + 137)):
            explainer = LimeTextExplainer(class_names=["not toxic", "toxic"], random_state=seed,
                                          bow=False, mask_string=tokenizer.mask_token)
            exp = explainer.explain_instance(case["text"], predict, labels=(1,),
                                             num_features=config["lime_features"], num_samples=config["lime_samples"])
            if repeat == 0:
                exp.save_to_file(str(output / f"{case['id']}.html"))
                fig = exp.as_pyplot_figure(label=1)
                fig.tight_layout()
                fig.savefig(output / f"{case['id']}.png", dpi=160)
                import matplotlib.pyplot as plt
                plt.close(fig)
            explanations.append({"seed": seed, "local_surrogate_r2": float(exp.score),
                                 "surrogate_prediction_at_original": float(exp.local_pred[0]),
                                 "intercept": float(exp.intercept[1]),
                                 "contributions": [[word, float(value)] for word, value in exp.as_list(label=1, positions=True)]})
        a, b = [set(word for word, _ in exp["contributions"]) for exp in explanations]
        score = float(predict([case["text"]])[0, 1])
        details.append({**case, "toxic_score": score, "threshold": thresholds["tuned"],
                        "predicted_label": int(score >= thresholds["tuned"]),
                        "top_feature_jaccard_across_seeds": len(a & b) / max(1, len(a | b)),
                        "explanations": explanations})
        print(f"Explained {case['id']}: toxic_score={score:.4f}", flush=True)
    write_json(output / "lime_explanations.json", details)
    lines = ["# External examples and interpretation", "",
             "Real-world excerpts include source URLs; stress cases are explicitly authored examples.",
             "Expected labels are single-reviewer judgments and can be disputed. These hand-picked examples do not estimate deployment accuracy.",
             "Overlap checks cover the experiment subsets, not the entire source corpus or pretraining data.", "",
             "| Example | Type | Human label | Frozen score | Untuned score | Tuned score | Tuned label |",
             "|---|---|---|---:|---:|---:|---|"]
    for example in examples:
        entries = [next(p for p in predictions if p["id"] == example["id"] and p["model"] == name)
                   for name in ("frozen", "untuned", "tuned")]
        label = "toxic" if entries[-1]["predicted_label"] else "not toxic"
        lines.append(f"| {example['id']} | {example['kind']} | {example['expected_label']} | " +
                     " | ".join(f"{p['toxic_score']:.4f}" for p in entries) + f" | {label} |")
    lines += ["", "## Selected LIME explanations", "",
              "Positive contributions raise the local surrogate's toxic score; negative contributions lower it.",
              "LIME approximates this model near masked variants of one text. It does not explain causality or human intent.",
              "Check surrogate R² and the agreement across two random seeds before trusting a word ranking.",
              "Masking can create unnatural language. Low fidelity or unstable rankings weaken the explanation.", ""]
    for case in details:
        lines += [f"### {case['id']}", "", case["text"], "",
                  f"Toxic score: {case['toxic_score']:.4f}; decision threshold: {case['threshold']:.4f}; "
                  f"expected label: {case['expected_label']}; predicted label: {case['predicted_label']}.",
                  f"Local surrogate R²: {case['explanations'][0]['local_surrogate_r2']:.3f}; "
                  f"top-feature Jaccard across seeds: {case['top_feature_jaccard_across_seeds']:.3f}.", "",
                  f"[Interactive explanation]({case['id']}.html)", "", f"![Word contributions]({case['id']}.png)", ""]
    lines += ["## External sources", ""]
    lines.extend(f"- {e['id']}: [{e.get('source_author') or 'public forum excerpt'}]({e['source_url']}); {e['note']}"
                 for e in examples if e.get("source_url"))
    (output / "interpretation.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {output / 'interpretation.md'}", flush=True)


if __name__ == "__main__":
    main()
