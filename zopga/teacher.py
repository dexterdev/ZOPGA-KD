"""Teacher training and loading.

The teacher is trained with standard cross-entropy on real data and the best
checkpoint by validation accuracy is kept. Once loaded for synthesis or
distillation the teacher is frozen: eval mode and requires_grad_(False) on
every parameter -- it is only ever queried through forward passes.

Per-epoch metrics (train loss/acc, val loss/acc, lr, time, throughput) are
logged and persisted to teacher_metrics.csv / teacher_metrics.json.
"""

import time

import torch
import torch.nn as nn

from .data import dataset_info, get_dataloaders
from .evaluate import evaluate
from .hardware import default_hardware
from .models import get_model
from .utils import (AverageMeter, MetricsLogger, count_parameters,
                    format_metrics, load_checkpoint, save_checkpoint)


def _make_scheduler(optimizer, cfg, epochs):
    name = cfg.get("schedule", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "multistep":
        milestones = [int(0.5 * epochs), int(0.75 * epochs)]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)
    if name == "none":
        return None
    raise ValueError(f"Unknown schedule '{name}'")


def train_teacher(cfg, device, out_dir, logger, hw=None):
    """Train the teacher with CE; save best-by-val checkpoint to out_dir/best.pt."""
    if hw is None:
        hw = default_hardware(device, logger)
    tcfg = cfg["teacher"]
    info = dataset_info(cfg["dataset"]["name"])
    loaders = get_dataloaders(cfg, seed=cfg.get("seed", 42),
                              batch_size=tcfg.get("batch_size", 128), hw=hw)

    model = get_model(tcfg["arch"], num_classes=info["num_classes"]).to(device)
    logger.info(f"Teacher '{tcfg['arch']}' params: {count_parameters(model):,}  "
                f"batch_size={hw.batch_size(tcfg.get('batch_size', 128))}  "
                f"amp={hw.use_amp}")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=tcfg.get("lr", 0.01),
        momentum=tcfg.get("momentum", 0.9),
        weight_decay=float(tcfg.get("weight_decay", 5e-4)))
    epochs = int(tcfg.get("epochs", 10))
    scheduler = _make_scheduler(optimizer, tcfg, epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = hw.grad_scaler()
    metrics = MetricsLogger(out_dir, "teacher")

    best_val = 0.0
    ckpt_path = f"{out_dir}/best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()
        n_seen, t0 = 0, time.time()
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with hw.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = (logits.argmax(1) == y).float().mean()
            loss_m.update(loss.item(), x.size(0))
            acc_m.update(acc.item(), x.size(0))
            n_seen += x.size(0)
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        epoch_time = time.time() - t0
        val = evaluate(model, loaders["val"], device, hw.autocast)
        row = metrics.log(epoch=epoch, train_loss=loss_m.avg, train_acc=acc_m.avg,
                          val_loss=val["loss"], val_acc=val["acc"], lr=lr_now,
                          epoch_time_s=epoch_time,
                          imgs_per_s=n_seen / max(1e-9, epoch_time))
        logger.info(f"[teacher] epoch {epoch:3d}/{epochs}  {format_metrics(row)}")
        if val["acc"] >= best_val:
            best_val = val["acc"]
            save_checkpoint(
                {"state_dict": model.state_dict(), "arch": tcfg["arch"],
                 "val_acc": best_val, "epoch": epoch}, ckpt_path)

    model.load_state_dict(load_checkpoint(ckpt_path)["state_dict"])
    test = evaluate(model, loaders["test"], device, hw.autocast)
    logger.info(f"Teacher done. best_val_acc {best_val:.4f}  "
                f"test_loss {test['loss']:.4f}  test_acc {test['acc']:.4f}")
    result = {"best_val_acc": best_val, "test_acc": test["acc"],
              "test_loss": test["loss"],
              "per_class_test_acc": test["per_class_acc"],
              "params": count_parameters(model), "ckpt": ckpt_path}
    metrics.save(extra=result)
    return result


def load_teacher(cfg, device, ckpt_path):
    """Load a frozen black-box teacher: eval mode, no grads ever."""
    info = dataset_info(cfg["dataset"]["name"])
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = get_model(ckpt.get("arch", cfg["teacher"]["arch"]),
                      num_classes=info["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
