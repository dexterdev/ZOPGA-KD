"""Evaluation helpers: loss, top-1 accuracy, per-class accuracy, agreement."""

import contextlib

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx=None):
    """Full evaluation pass: CE loss, top-1 accuracy and per-class accuracy.

    Returns {"loss", "acc", "per_class_acc": {label: acc}, "n"}.
    autocast_ctx: optional callable returning an autocast context (AMP).
    """
    model.eval()
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext
    loss_sum, correct, total = 0.0, 0, 0
    class_correct, class_total = {}, {}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with ctx():
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
        loss_sum += loss.float().item()
        pred = logits.argmax(dim=1)
        hits = pred == y
        correct += hits.sum().item()
        total += y.numel()
        for c in y.unique():
            ci = int(c)
            mask = y == c
            class_correct[ci] = class_correct.get(ci, 0) + hits[mask].sum().item()
            class_total[ci] = class_total.get(ci, 0) + int(mask.sum())
    per_class = {str(c): class_correct[c] / max(1, class_total[c])
                 for c in sorted(class_total)}
    return {"loss": loss_sum / max(1, total),
            "acc": correct / max(1, total),
            "per_class_acc": per_class,
            "n": total}


@torch.no_grad()
def accuracy(model, loader, device):
    """Top-1 accuracy of a model on a dataloader."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def agreement(model_a, model_b, loader, device):
    """Fraction of samples where two models produce the same argmax label."""
    model_a.eval()
    model_b.eval()
    same, total = 0, 0
    for x, _ in loader:
        x = x.to(device)
        same += (model_a(x).argmax(1) == model_b(x).argmax(1)).sum().item()
        total += x.size(0)
    return same / max(1, total)
