"""Shared loading, integrity checks, and split preparation."""

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Keep new caches in this project, including on machines with a shared UV environment.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

SPLITS = ("train", "validation", "test")
SCORE_COLUMNS = (
    "toxicity", "severe_toxicity", "obscene", "threat", "insult",
    "identity_attack", "sexual_explicit",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_civil(source="hub", local_dir=ROOT):
    if source == "hub":
        # Required assignment loader. The Hub's official splits are preserved.
        return load_dataset("google/civil_comments")
    local_dir = Path(local_dir)
    files = {s: sorted(str(p) for p in local_dir.glob(f"{s}-*.parquet")) for s in SPLITS}
    if any(not paths for paths in files.values()):
        raise FileNotFoundError(f"Missing Civil Comments Parquet splits in {local_dir}")
    return load_dataset("parquet", data_files=files)


def text_key(text):
    # Normalization is ONLY for duplicate detection. Model input remains the original text.
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def scan_split(split, threshold=0.5):
    """Bounded text batches; compact numeric arrays for the full-split audit."""
    n = len(split)
    keys = np.empty(n, dtype="S32")
    exact_keys = np.empty(n, dtype="S32")
    scores = np.empty(n, dtype=np.float32)
    chars = np.zeros(n, dtype=np.int32)
    words = np.zeros(n, dtype=np.int32)
    valid = np.ones(n, dtype=bool)
    nulls = {c: 0 for c in split.column_names}
    invalid_scores = {c: 0 for c in SCORE_COLUMNS}
    offset = 0
    for batch in split.iter(batch_size=8192):
        count = len(batch["text"])
        for col in split.column_names:
            nulls[col] += sum(x is None for x in batch[col])
        for col in SCORE_COLUMNS:
            values = np.asarray(batch[col], dtype=float)
            invalid_scores[col] += int((~np.isfinite(values) | (values < 0) | (values > 1)).sum())
        scores[offset:offset + count] = np.asarray(batch["toxicity"], dtype=np.float32)
        for j, value in enumerate(batch["text"], offset):
            text = value if isinstance(value, str) else ""
            keys[j] = text_key(text)
            exact_keys[j] = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            chars[j] = len(text)
            words[j] = len(text.split())
            valid[j] = bool(text.strip())
        offset += count
    valid &= np.isfinite(scores) & (scores >= 0) & (scores <= 1)
    return {"keys": keys, "exact_keys": exact_keys, "scores": scores, "chars": chars,
            "words": words, "valid": valid, "labels": (scores >= threshold).astype(np.int64),
            "nulls": nulls, "invalid_scores": invalid_scores}


def clean_indices(scans):
    """Preserve test over validation over train, without inspecting model results.

    Remove within-split repeats and overlaps with higher-priority official splits.
    Original row order determines which occurrence is retained within a split.
    """
    kept = {}
    reserved = np.empty(0, dtype="S32")
    for name in reversed(SPLITS):
        scan = scans[name]
        candidates = np.flatnonzero(scan["valid"])
        _, first = np.unique(scan["keys"][candidates], return_index=True)
        candidates = np.sort(candidates[first])
        candidates = candidates[~np.isin(scan["keys"][candidates], reserved)]
        kept[name] = candidates
        # Reserve all valid higher-priority keys, even if repeated in those splits.
        reserved = np.union1d(reserved, scan["keys"][scan["valid"]])
    return kept


def stratified_indices(indices, labels, size, seed):
    if size is None or size >= len(indices):
        return np.sort(indices)
    if size < 2 or len(np.unique(labels[indices])) != 2:
        raise ValueError("A stratified subset must support both labels.")
    selected, _ = train_test_split(indices, train_size=size, random_state=seed,
                                   stratify=labels[indices])
    return np.sort(selected)


def prepare_data(raw, config, output):
    scans = {s: scan_split(raw[s], config["label_threshold"]) for s in SPLITS}
    clean = clean_indices(scans)
    indices = {s: stratified_indices(clean[s], scans[s]["labels"], config[f"{s}_size"],
                                      config["seed"]) for s in SPLITS}
    selection, calibration = train_test_split(
        indices["validation"], test_size=0.5, random_state=config["seed"],
        stratify=scans["validation"]["labels"][indices["validation"]],
    )
    indices["selection"] = np.sort(selection)
    indices["calibration"] = np.sort(calibration)
    manifest = {"label_threshold": config["label_threshold"], "seed": config["seed"],
                "dataset_fingerprints": {s: raw[s]._fingerprint for s in SPLITS},
                "rows_before_cleaning": {s: len(raw[s]) for s in SPLITS},
                "rows_after_cleaning": {s: len(clean[s]) for s in SPLITS}, "splits": {}}
    prepared = DatasetDict()
    for name in ("train", "selection", "calibration", "test"):
        origin = "validation" if name in ("selection", "calibration") else name
        rows = indices[name]
        part = raw[origin].select(rows.tolist()).select_columns(["text", "toxicity"])
        part = part.add_column("label", scans[origin]["labels"][rows].tolist())
        part = part.add_column("source_row", rows.tolist())
        part = part.add_column("text_hash", scans[origin]["keys"][rows].astype(str).tolist())
        prepared[name] = part
        manifest["splits"][name] = {
            "official_split": origin, "rows": len(rows),
            "toxic_rows": int(scans[origin]["labels"][rows].sum()),
            "source_rows": rows.tolist(),
        }
    # Fail before any model training if duplicate contamination survived.
    seen = set()
    for part in prepared.values():
        keys = set(part["text_hash"])
        if keys & seen:
            raise ValueError("Normalized text overlap survived preprocessing.")
        seen.update(keys)
    write_json(Path(output) / "split_manifest.json", manifest)
    return prepared
