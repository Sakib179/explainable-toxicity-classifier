"""Explicit training loops with cached frozen features and validation-only selection."""

import hashlib
import math
import time
from itertools import islice

from .model import load_weights, save_weights, seed_everything

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup


def encoder_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.encoder.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def extract_features(model, batches, device, path):
    model.encoder.eval()
    features, labels = [], []
    for batch in batches:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        features.append(model.encode(**inputs).cpu())
        labels.append(batch["labels"])
    features, labels = torch.cat(features), torch.cat(labels)
    torch.save({"features": features, "labels": labels}, path)
    return TensorDataset(features, labels)


def feature_loader(features, config, shuffle=False):
    return DataLoader(features, batch_size=config["batch_size"], shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(config["seed"]))


@torch.no_grad()
def validation_scores(model, batches, device, frozen):
    model.eval()
    probabilities, labels = [], []
    for batch in batches:
        if frozen:
            features, y = batch
            logits = model.head(features.to(device))
        else:
            y = batch["labels"]
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs)
        probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def optimizer_groups(model, hyperparameters, config, frozen):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        head = name.startswith("head.")
        lr = hyperparameters["learning_rate"]
        if head and not frozen:
            lr *= config["head_learning_rate_multiplier"]
        decay = 0.0 if name.endswith("bias") or "layernorm" in name.lower() or "layer_norm" in name.lower() else hyperparameters["weight_decay"]
        groups.setdefault((lr, decay), []).append(param)
    return [{"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in groups.items()]


def fit(model, train_batches, selection_batches, device, config, hyperparameters, class_weights,
        checkpoint, frozen=False):
    """Return the best selection-AP epoch, never the final epoch by assumption."""
    seed_everything(config["seed"])
    for param in model.encoder.parameters():
        param.requires_grad_(not frozen)
    model.head[0].p = hyperparameters["dropout"]
    model.to(device)
    epochs = config["baseline_epochs"] if frozen else config["fine_tune_epochs"]
    accumulation = config["gradient_accumulation_steps"]
    steps = math.ceil(len(train_batches) / accumulation) * epochs
    optimizer = torch.optim.AdamW(optimizer_groups(model, hyperparameters, config, frozen))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(steps * config["warmup_ratio"]), num_training_steps=steps,
    )
    weights_cpu = torch.tensor(class_weights, dtype=torch.float32)
    weights = weights_cpu.to(device)
    best_ap, best_epoch, stale = -1.0, 0, 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if frozen:
            model.encoder.eval()
        epoch_numerator, epoch_denominator = 0.0, 0.0
        iterator = iter(train_batches)
        while group := list(islice(iterator, accumulation)):
            optimizer.zero_grad(set_to_none=True)
            # Normalize over the complete effective batch, including a short final batch.
            group_labels = [b[1] if frozen else b["labels"] for b in group]
            denominator = sum(float(weights_cpu[y].sum()) for y in group_labels)
            for batch, y_cpu in zip(group, group_labels):
                y = y_cpu.to(device)
                if frozen:
                    logits = model.head(batch[0].to(device))
                else:
                    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                    logits = model(**inputs)
                numerator = F.cross_entropy(logits, y, weight=weights, reduction="sum")
                (numerator / denominator).backward()
                epoch_numerator += float(numerator.detach())
            epoch_denominator += denominator
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
        y, probability = validation_scores(model, selection_batches, device, frozen)
        ap = float(average_precision_score(y, probability))
        if not math.isfinite(ap) or not math.isfinite(epoch_numerator):
            raise RuntimeError("Non-finite training result; experiment stopped before test evaluation.")
        history.append({"epoch": epoch, "weighted_train_loss": epoch_numerator / epoch_denominator,
                        "selection_average_precision": ap})
        print(f"  epoch={epoch} loss={history[-1]['weighted_train_loss']:.4f} selection_AP={ap:.4f}", flush=True)
        if ap > best_ap + 1e-6:
            best_ap, best_epoch, stale = ap, epoch, 0
            save_weights(model, checkpoint)
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                break
    load_weights(model, checkpoint)
    model.eval()
    return {"best_epoch": best_epoch, "selection_average_precision": best_ap, "history": history,
            "fit_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "hyperparameters": hyperparameters, "encoder_frozen": frozen}
