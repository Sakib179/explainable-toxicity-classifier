"""The same mean-pooled BERT-family encoder and linear head in all experiments."""

import random
from pathlib import Path

from . import data  # Configure local caches before importing transformers.

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ToxicityModel(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.config.hidden_size, 2))

    def encode(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def forward(self, **inputs):
        return self.head(self.encode(**inputs))


def new_pretrained(config):
    encoder = AutoModel.from_pretrained(config["model_name"], revision=config["model_revision"],
                                         attn_implementation="eager")
    model = ToxicityModel(encoder)
    # Resolve the tokenizer to the very same checkpoint commit when available.
    revision = getattr(encoder.config, "_commit_hash", None) or config["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=revision, use_fast=True)
    return model, tokenizer, revision


def save_weights(model, path):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def load_artifact(directory, run_name, device):
    directory = Path(directory)
    config = AutoConfig.from_pretrained(directory / "encoder_config", local_files_only=True)
    encoder = AutoModel.from_config(config, attn_implementation="eager")
    model = load_weights(ToxicityModel(encoder), directory / f"{run_name}.pt")
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
    return model.to(device).eval(), tokenizer


def tokenize_splits(parts, tokenizer, max_length):
    encoded = {}
    for name, part in parts.items():
        encoded[name] = part.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True, remove_columns=part.column_names, desc=f"Tokenize {name}",
        ).add_column("labels", list(part["label"]))
    return encoded


def loader(dataset, tokenizer, batch_size, seed=42, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator, collate_fn=DataCollatorWithPadding(tokenizer))


@torch.inference_mode()
def predict_loader(model, batches, device):
    model.eval()
    predictions = []
    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        predictions.append(model(**inputs).softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(predictions)


@torch.inference_mode()
def predict_texts(model, tokenizer, texts, device, max_length=256, batch_size=16):
    if isinstance(texts, str):
        texts = [texts]
    model.eval()
    probabilities = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        probabilities.append(model(**batch).softmax(-1).cpu().numpy())
    return np.concatenate(probabilities) if probabilities else np.empty((0, 2))
