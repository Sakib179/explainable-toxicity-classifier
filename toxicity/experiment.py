"""Run frozen transfer, end-to-end fine-tuning, random search, and final evaluation."""

import argparse
import csv
import gc
import json
import platform
import random
import shutil
import time
from importlib.metadata import version
from itertools import product
from pathlib import Path

from .data import ROOT, load_civil, prepare_data, write_json
from .metrics import bootstrap_differences, evaluate, select_threshold
from .model import (choose_device, loader, load_weights, new_pretrained, predict_loader,
                    seed_everything, tokenize_splits)
from .train import encoder_digest, extract_features, feature_loader, fit

import numpy as np
import torch


def truncation_audit(parts, tokenizer, max_length):
    result = {}
    for name in ("train", "selection"):
        lengths = []
        for batch in parts[name].iter(batch_size=512):
            lengths.extend(map(len, tokenizer(batch["text"], truncation=False, verbose=False)["input_ids"]))
        lengths = np.asarray(lengths)
        result[name] = {"rows": len(lengths), "max_length": max_length,
                        "fraction_truncated": float((lengths > max_length).mean()),
                        "token_length_quantiles": {str(q): float(np.quantile(lengths, q)) for q in (0.5, 0.95, 0.99, 1)}}
    return result


def validate_config(config):
    positive = ("train_size", "validation_size", "test_size", "batch_size", "baseline_epochs",
                "fine_tune_epochs", "gradient_accumulation_steps", "early_stopping_patience", "bootstrap_repetitions")
    if any(not isinstance(config[k], int) or config[k] < 1 for k in positive):
        raise ValueError("Sample sizes, epochs, batch sizes, patience, and bootstrap count must be positive integers.")
    if config["validation_size"] < 4 or not 0 < config["label_threshold"] < 1:
        raise ValueError("Invalid validation size or label threshold.")
    if not 4 <= config["max_length"] <= 512 or not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("Invalid sequence length or warmup ratio.")
    if config["class_weighting"] not in ("balanced", "none"):
        raise ValueError("class_weighting must be balanced or none.")
    candidates = list(product(config["search"]["learning_rate"], config["search"]["weight_decay"], config["search"]["dropout"]))
    if not 1 <= config["search"]["trials"] <= len(set(candidates)):
        raise ValueError("The search must contain the requested number of unique trials.")


def release_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--source", choices=["hub", "local"], default="hub")
    parser.add_argument("--local-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "run")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new --output directory; {output} contains results.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    seed_everything(config["seed"])
    device = choose_device(args.device)
    print(f"Using {device}; preparing official splits and removing normalized duplicates.", flush=True)
    raw = load_civil(args.source, args.local_dir)
    parts = prepare_data(raw, config, output)
    del raw
    # Persist the exact subset so explanations never reload or resample the dataset.
    parts.save_to_disk(str(output / "data"))
    model, tokenizer, revision = new_pretrained(config)
    model.encoder.config.save_pretrained(output / "encoder_config")
    tokenizer.save_pretrained(output / "tokenizer")
    write_json(output / "provenance.json", {
        "dataset": "google/civil_comments", "dataset_source": args.source,
        "model": config["model_name"], "resolved_model_revision": revision,
        "python": platform.python_version(), "platform": platform.platform(), "device": str(device),
        "packages": {p: version(p) for p in ("datasets", "transformers", "torch", "numpy", "scikit-learn")},
    })
    coverage = truncation_audit(parts, tokenizer, config["max_length"])
    write_json(output / "token_lengths.json", coverage)
    print(f"Train truncation fraction: {coverage['train']['fraction_truncated']:.2%}", flush=True)
    encoded = tokenize_splits(parts, tokenizer, config["max_length"])
    batches = {s: loader(d, tokenizer, config["batch_size"], config["seed"]) for s, d in encoded.items()}
    labels = {s: np.asarray(d["label"], dtype=int) for s, d in parts.items()}
    counts = np.bincount(labels["train"], minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both training labels are required; increase train_size.")
    weights = (len(labels["train"]) / (2 * counts)).tolist() if config["class_weighting"] == "balanced" else [1.0, 1.0]
    write_json(output / "class_weights.json", {"computed_from": "train only", "not_toxic": weights[0], "toxic": weights[1]})
    model.to(device)
    initial_digest = encoder_digest(model)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    print("Frozen baseline: caching deterministic encoder features once.", flush=True)
    started = time.perf_counter()
    features = {s: extract_features(model, batches[s], device, output / f"{s}_features.pt") for s in ("train", "selection")}
    extraction_seconds = time.perf_counter() - started
    baseline_hyperparameters = {"learning_rate": config["baseline_learning_rate"],
                                "weight_decay": config["baseline_weight_decay"], "dropout": 0.1}
    baseline = fit(model, feature_loader(features["train"], config, shuffle=True),
                   feature_loader(features["selection"], config), device, config, baseline_hyperparameters,
                   weights, output / "frozen.pt", frozen=True)
    if encoder_digest(model) != initial_digest or any(p.grad is not None for p in model.encoder.parameters()):
        raise AssertionError("The frozen baseline modified the encoder.")
    baseline.update({"feature_extraction_seconds": extraction_seconds, "encoder_sha256": initial_digest,
                     "encoder_unchanged_verified": True})
    write_json(output / "frozen_training.json", baseline)
    # Baseline is fixed and recorded before any fine-tuning hyperparameter search.
    del features
    release_cache(device)

    def train_candidate(name, hyperparameters):
        # Every candidate starts from the SAME pretrained encoder and frozen-trained head.
        # No trial continues from another trial's weights.
        load_weights(model, output / "frozen.pt")
        train_batches = loader(encoded["train"], tokenizer, config["batch_size"], config["seed"], shuffle=True)
        print(f"{name}: {hyperparameters}", flush=True)
        result = fit(model, train_batches, batches["selection"], device, config, hyperparameters,
                     weights, output / f"{name}.pt", frozen=False)
        if encoder_digest(model) == initial_digest:
            raise AssertionError("End-to-end fine-tuning did not update the encoder.")
        result["encoder_updated_verified"] = True
        write_json(output / f"{name}_training.json", result)
        release_cache(device)
        return result

    untuned = train_candidate("untuned", config["untuned"])
    shutil.copyfile(output / "untuned.pt", output / "tuned.pt")
    selected = {"origin": "untuned", **untuned}
    search = config["search"]
    grid = list(product(search["learning_rate"], search["weight_decay"], search["dropout"]))
    candidates = random.Random(config["seed"]).sample(list(dict.fromkeys(grid)), search["trials"])
    trials = []
    for i, (lr, wd, dropout) in enumerate(candidates, 1):
        result = train_candidate("candidate", {"learning_rate": lr, "weight_decay": wd, "dropout": dropout})
        result["trial"] = i
        trials.append(result)
        write_json(output / "search_trials.json", trials)
        if result["selection_average_precision"] > selected["selection_average_precision"] + 1e-6:
            shutil.copyfile(output / "candidate.pt", output / "tuned.pt")
            selected = {"origin": f"random_search_trial_{i}", **result}
    write_json(output / "tuned_training.json", selected)
    # Freeze the chosen model identity before computing thresholds or test metrics.
    write_json(output / "selection_lock.json", {
        "criterion": "selection average precision", "selected": selected["origin"],
        "selection_average_precision": selected["selection_average_precision"],
        "untuned_average_precision": untuned["selection_average_precision"],
        "test_used_for_selection": False,
    })

    thresholds, probabilities, results = {}, {}, {}
    for name in ("frozen", "untuned", "tuned"):
        load_weights(model, output / f"{name}.pt")
        calibration_scores = predict_loader(model, batches["calibration"], device)
        threshold = select_threshold(labels["calibration"], calibration_scores)
        thresholds[name] = threshold
        # Each fixed checkpoint is evaluated on the same held-out test subset once.
        test_scores = predict_loader(model, batches["test"], device)
        probabilities[name] = test_scores
        results[name] = {"fixed_0_5": evaluate(labels["test"], test_scores, 0.5),
                         "validation_threshold": evaluate(labels["test"], test_scores, threshold)}
        np.savez_compressed(output / f"{name}_predictions.npz", test_scores=test_scores,
                            test_labels=labels["test"], source_rows=np.asarray(parts["test"]["source_row"]),
                            calibration_scores=calibration_scores, calibration_labels=labels["calibration"])
    write_json(output / "thresholds.json", thresholds)
    write_json(output / "test_metrics.json", results)
    intervals = {name: bootstrap_differences(labels["test"], probabilities["frozen"], probabilities[name],
                                             thresholds["frozen"], thresholds[name],
                                             config["bootstrap_repetitions"], config["seed"])
                 for name in ("untuned", "tuned")}
    write_json(output / "paired_bootstrap.json", intervals)
    with (output / "comparison.csv").open("w", newline="") as handle:
        columns = ["model", "operating_point", "threshold", "average_precision", "roc_auc", "accuracy",
                   "balanced_accuracy", "toxic_precision", "toxic_recall", "toxic_f1", "macro_f1"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, operating_points in results.items():
            for point, metrics in operating_points.items():
                writer.writerow({"model": name, "operating_point": point,
                                 **{k: metrics[k] for k in columns[2:]}})
    from .report import write_report
    write_report(output, results, probabilities, labels["test"], intervals, selected)
    print(f"Completed. Results: {output / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
