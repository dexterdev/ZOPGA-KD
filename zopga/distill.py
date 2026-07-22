"""Hinton knowledge distillation on the synthetic dataset.

The student trains on synthetic data only. Each batch is augmented on the fly
(dataset-appropriate augmentation pool), the frozen teacher relabels the
*augmented* images with soft targets at temperature T, and the student
minimizes the temperature-scaled KL divergence (KD-only loss, T^2 factor --
no CE term, since synthetic labels are correct by construction).

Per-epoch metrics (kd loss, teacher/student agreement, accuracy vs the
synthetic labels, lr, time, throughput, and optionally real-test loss/acc
every `eval_every` epochs) are logged and persisted to
distill_metrics.csv / distill_metrics.json.
"""

import time

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T

from .data import dataset_info, get_dataloaders
from .evaluate import evaluate
from .hardware import default_hardware
from .models import get_model
from .utils import (AverageMeter, MetricsLogger, count_parameters,
                    format_metrics, save_checkpoint)


def kd_loss(student_logits, teacher_logits, temperature):
    """KL(student_T || teacher_T) * T^2 (Hinton et al., 2015)."""
    t = temperature
    log_p_s = F.log_softmax(student_logits / t, dim=1)
    p_t = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)


def build_augment(names, image_size):
    """Build a batched-tensor augmentation pipeline (operates in [0,1] pixel space).

    Returns None when no augmentations are requested."""
    tfms = []
    for name in names or []:
        if name == "crop":
            tfms.append(T.RandomCrop(image_size, padding=4))
        elif name == "flip":
            tfms.append(T.RandomHorizontalFlip())
        elif name == "rotate":
            tfms.append(T.RandomRotation(10))
        elif name == "affine":
            tfms.append(T.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                       scale=(0.9, 1.1)))
        elif name == "cutout":
            tfms.append(T.RandomErasing(p=0.5, scale=(0.02, 0.15), value=0))
        else:
            raise ValueError(f"Unknown augmentation '{name}'")
    return T.Compose(tfms) if tfms else None


def train_student_kd(student, teacher, images, labels, cfg, data_info, device,
                     logger=None, tag="distill", hw=None, eval_loader=None,
                     metrics=None):
    """Core KD loop on a synthetic tensor dataset.

    images: normalized-space tensor (N, C, H, W); labels: synthetic labels
    (correct by construction, used only as a training metric).
    eval_loader: optional real-data loader evaluated every
    cfg['eval_every'] epochs (0/None disables per-epoch evaluation).
    Returns final train stats.
    """
    if hw is None:
        hw = default_hardware(device, logger)
    dcfg = cfg
    epochs = int(dcfg.get("epochs", 30))
    batch_size = hw.batch_size(int(dcfg.get("batch_size", 256)))
    temp = float(dcfg.get("temperature", 10))
    eval_every = int(dcfg.get("eval_every", 0) or 0)
    info = data_info

    mean = torch.tensor(info["mean"], device=device).view(1, -1, 1, 1)
    std = torch.tensor(info["std"], device=device).view(1, -1, 1, 1)
    augment = build_augment(dcfg.get("augmentations"), info["image_size"])

    images = images.to(device)
    labels = labels.to(device)
    n = images.size(0)
    optimizer = torch.optim.SGD(
        student.parameters(), lr=dcfg.get("lr", 0.01),
        momentum=dcfg.get("momentum", 0.9),
        weight_decay=float(dcfg.get("weight_decay", 5e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=epochs)
    scaler = hw.grad_scaler()

    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 42) + 7)
    stats = {}
    for epoch in range(1, epochs + 1):
        student.train()
        loss_m, agree_m, label_m = AverageMeter(), AverageMeter(), AverageMeter()
        t0 = time.time()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = images[idx], labels[idx]
            xb_px = (xb * std + mean).clamp(0.0, 1.0)
            if augment is not None:
                xb_px = augment(xb_px)
            xb_aug = (xb_px - mean) / std
            with torch.no_grad(), hw.autocast():
                t_logits = teacher(xb_aug)
            with hw.autocast():
                s_logits = student(xb_aug)
                loss = kd_loss(s_logits.float(), t_logits.float(), temp)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            agree = (s_logits.argmax(1) == t_logits.argmax(1)).float().mean()
            label_acc = (s_logits.argmax(1) == yb).float().mean()
            loss_m.update(loss.item(), xb.size(0))
            agree_m.update(agree.item(), xb.size(0))
            label_m.update(label_acc.item(), xb.size(0))
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        epoch_time = time.time() - t0

        row = {"epoch": epoch, "kd_loss": loss_m.avg,
               "teacher_agreement": agree_m.avg,
               "synthetic_label_acc": label_m.avg, "lr": lr_now,
               "epoch_time_s": epoch_time,
               "imgs_per_s": n / max(1e-9, epoch_time)}
        if eval_loader is not None and eval_every > 0 and \
                (epoch % eval_every == 0 or epoch == epochs):
            test = evaluate(student, eval_loader, device, hw.autocast)
            row["test_loss"] = test["loss"]
            row["test_acc"] = test["acc"]
        elif eval_loader is not None and eval_every > 0:
            row["test_loss"] = None
            row["test_acc"] = None
        if metrics is not None:
            metrics.log(**row)
        if logger is not None:
            logger.info(f"[{tag}] epoch {epoch:3d}/{epochs}  "
                        f"{format_metrics(row)}")
        stats = {"final_kd_loss": loss_m.avg, "final_agreement": agree_m.avg,
                 "final_synthetic_label_acc": label_m.avg}
    return stats


def distill_student(cfg, device, out_dir, logger, teacher, synthetic, hw=None):
    """Train the ZO-PGA student on synthetic data; save checkpoint + test metrics."""
    if hw is None:
        hw = default_hardware(device, logger)
    info = dataset_info(cfg["dataset"]["name"])
    dcfg = dict(cfg["distill"])
    dcfg.setdefault("seed", cfg.get("seed", 42))
    images, labels, _ = synthetic

    student = get_model(cfg["student"]["arch"],
                        num_classes=info["num_classes"]).to(device)
    logger.info(f"Student '{cfg['student']['arch']}' params: "
                f"{count_parameters(student):,}  amp={hw.use_amp}")

    loaders = get_dataloaders(cfg, seed=cfg.get("seed", 42),
                              batch_size=cfg["teacher"].get("batch_size", 128),
                              with_aug=False, hw=hw)
    metrics = MetricsLogger(out_dir, "distill")
    stats = train_student_kd(student, teacher, images, labels, dcfg, info,
                             device, logger, tag="zo-pga-distill", hw=hw,
                             eval_loader=loaders["test"], metrics=metrics)

    test = evaluate(student, loaders["test"], device, hw.autocast)
    logger.info(f"ZO-PGA student test_loss {test['loss']:.4f}  "
                f"test_acc {test['acc']:.4f}")
    save_checkpoint({"state_dict": student.state_dict(),
                     "arch": cfg["student"]["arch"], "test_acc": test["acc"]},
                    f"{out_dir}/student.pt")
    result = {"test_acc": test["acc"], "test_loss": test["loss"],
              "per_class_test_acc": test["per_class_acc"],
              "params": count_parameters(student), **stats}
    metrics.save(extra=result)
    return result
