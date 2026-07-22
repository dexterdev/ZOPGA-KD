"""Baselines on real data with the same student architecture.

(a) 'ce': student trained from scratch with cross-entropy on real labels.
(b) 'kd': classical Hinton KD on real data with teacher soft labels
          (alpha * KD_T + (1 - alpha) * CE).

Per-epoch metrics (loss components, train acc, val loss/acc, lr, time,
throughput) are logged and persisted to baseline_<method>_metrics.csv / .json.
"""

import time

import torch
import torch.nn as nn

from .data import dataset_info, get_dataloaders
from .distill import kd_loss
from .evaluate import evaluate
from .hardware import default_hardware
from .models import get_model
from .utils import (AverageMeter, MetricsLogger, count_parameters,
                    format_metrics, save_checkpoint)


def train_baseline(cfg, device, out_dir, logger, method, teacher=None, hw=None):
    """Train one baseline student. method: 'ce' or 'kd'."""
    if hw is None:
        hw = default_hardware(device, logger)
    info = dataset_info(cfg["dataset"]["name"])
    bcfg = cfg["baseline"]
    seed = cfg.get("seed", 42)
    loaders = get_dataloaders(cfg, seed=seed,
                              batch_size=bcfg.get("batch_size", 128),
                              with_aug=True, hw=hw)

    student = get_model(cfg["student"]["arch"],
                        num_classes=info["num_classes"]).to(device)
    logger.info(f"Baseline '{method}' student '{cfg['student']['arch']}' "
                f"params: {count_parameters(student):,}  amp={hw.use_amp}")

    optimizer = torch.optim.SGD(
        student.parameters(), lr=bcfg.get("lr", 0.01),
        momentum=bcfg.get("momentum", 0.9),
        weight_decay=float(bcfg.get("weight_decay", 5e-4)))
    epochs = int(bcfg.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=epochs)
    ce = nn.CrossEntropyLoss()
    temp = float(bcfg.get("temperature", 10))
    alpha = float(bcfg.get("alpha", 0.9))
    scaler = hw.grad_scaler()
    metrics = MetricsLogger(out_dir, f"baseline_{method}")

    for epoch in range(1, epochs + 1):
        student.train()
        loss_m, ce_m, kd_m, acc_m = (AverageMeter(), AverageMeter(),
                                     AverageMeter(), AverageMeter())
        n_seen, t0 = 0, time.time()
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with hw.autocast():
                s_logits = student(x)
            ce_term = ce(s_logits.float(), y)
            if method == "ce":
                loss = ce_term
                kd_term = torch.zeros((), device=device)
            else:
                with torch.no_grad(), hw.autocast():
                    t_logits = teacher(x)
                kd_term = kd_loss(s_logits.float(), t_logits.float(), temp)
                loss = alpha * kd_term + (1.0 - alpha) * ce_term
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = (s_logits.argmax(1) == y).float().mean()
            loss_m.update(loss.item(), x.size(0))
            ce_m.update(ce_term.item(), x.size(0))
            kd_m.update(kd_term.item(), x.size(0))
            acc_m.update(acc.item(), x.size(0))
            n_seen += x.size(0)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.time() - t0
        val = evaluate(student, loaders["val"], device, hw.autocast)
        row = {"epoch": epoch, "loss": loss_m.avg, "ce_loss": ce_m.avg,
               "train_acc": acc_m.avg, "val_loss": val["loss"],
               "val_acc": val["acc"], "lr": lr_now,
               "epoch_time_s": epoch_time,
               "imgs_per_s": n_seen / max(1e-9, epoch_time)}
        if method == "kd":
            row["kd_loss"] = kd_m.avg
        metrics.log(**row)
        logger.info(f"[baseline-{method}] epoch {epoch:3d}/{epochs}  "
                    f"{format_metrics(row)}")

    test = evaluate(student, loaders["test"], device, hw.autocast)
    logger.info(f"Baseline '{method}' test_loss {test['loss']:.4f}  "
                f"test_acc {test['acc']:.4f}")
    save_checkpoint({"state_dict": student.state_dict(),
                     "arch": cfg["student"]["arch"], "method": method,
                     "test_acc": test["acc"]},
                    f"{out_dir}/student_{method}.pt")
    result = {"test_acc": test["acc"], "test_loss": test["loss"],
              "per_class_test_acc": test["per_class_acc"],
              "params": count_parameters(student)}
    metrics.save(extra=result)
    return result
