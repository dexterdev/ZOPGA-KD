"""Diagnostics for the synthetic dataset.

1. Subsample scaling curves: student test accuracy vs fraction of synthetic
   data (shorter KD runs).
2. Effective rank of teacher penultimate features of the synthetic images:
   erank = exp(entropy of normalized singular values).
3. Nearest-neighbor duplicate check (mode-collapse detection): distribution
   of per-image nearest-neighbor distances and the fraction of near-duplicate
   pairs, computed in both pixel and teacher-feature space.
"""

import torch
import torch.nn as nn

from .data import dataset_info, get_dataloaders
from .distill import train_student_kd
from .evaluate import evaluate
from .hardware import default_hardware
from .models import get_model
from .utils import save_json


def _subsample_per_class(images, labels, fraction, seed):
    """Class-balanced subsample of a fraction of the synthetic dataset."""
    gen = torch.Generator().manual_seed(seed)
    idxs = []
    for c in labels.unique():
        c_idx = (labels == c).nonzero(as_tuple=True)[0]
        k = max(1, int(round(c_idx.numel() * fraction)))
        perm = c_idx[torch.randperm(c_idx.numel(), generator=gen)[:k]]
        idxs.append(perm)
    idx = torch.cat(idxs)
    return images[idx], labels[idx]


def scaling_curves(cfg, device, teacher, synthetic, logger, hw=None):
    """Student accuracy vs fraction of synthetic data (subsample scaling)."""
    if hw is None:
        hw = default_hardware(device, logger)
    info = dataset_info(cfg["dataset"]["name"])
    dcfg_base = dict(cfg["distill"])
    dcfg_base["epochs"] = int(cfg["diagnostics"].get("epochs", 15))
    dcfg_base.setdefault("seed", cfg.get("seed", 42))
    dcfg_base["eval_every"] = 0  # only evaluate at the end of each short run
    images, labels, _ = synthetic
    loaders = get_dataloaders(cfg, seed=cfg.get("seed", 42), with_aug=False,
                              hw=hw)

    results = {}
    for frac in cfg["diagnostics"].get("fractions", [0.25, 0.5, 1.0]):
        frac = float(frac)
        sub_imgs, sub_labels = _subsample_per_class(
            images, labels, frac, seed=cfg.get("seed", 42))
        student = get_model(cfg["student"]["arch"],
                            num_classes=info["num_classes"]).to(device)
        train_student_kd(student, teacher, sub_imgs, sub_labels, dcfg_base,
                         info, device, logger, tag=f"scaling-{frac:g}", hw=hw)
        test = evaluate(student, loaders["test"], device, hw.autocast)
        results[str(frac)] = {"n_samples": int(sub_imgs.size(0)),
                              "test_acc": test["acc"],
                              "test_loss": test["loss"]}
        logger.info(f"[diagnose] fraction {frac:g}: n={sub_imgs.size(0)} "
                    f"test_loss {test['loss']:.4f} test_acc {test['acc']:.4f}")
        del student
    return results


@torch.no_grad()
def penultimate_features(model, images, device, chunk=512):
    """Inputs to the final Linear layer, for a (N, C, H, W) image tensor."""
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("No Linear layer found for penultimate features")
    feats = []

    def hook(module, inputs):
        feats.append(inputs[0].detach().cpu())

    handle = last_linear.register_forward_pre_hook(hook)
    try:
        for i in range(0, images.size(0), chunk):
            model(images[i:i + chunk].to(device))
    finally:
        handle.remove()
    return torch.cat(feats)


def effective_rank(features):
    """erank = exp(H(p)), p = normalized singular values of the feature matrix."""
    feats = features - features.mean(dim=0, keepdim=True)
    svals = torch.linalg.svdvals(feats)
    p = svals / svals.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy))


def _nn_stats(x, threshold_rel):
    """Nearest-neighbor distance stats for rows of x.

    threshold_rel: near-duplicate threshold as a fraction of the median NN
    distance. Returns min/median/mean NN distance and near-duplicate fraction.
    """
    n = x.size(0)
    if n < 2:
        return {"n": n}
    mins = []
    chunk = 1024
    xf = x.float()
    for i in range(0, n, chunk):
        d = torch.cdist(xf[i:i + chunk], xf)
        rows = torch.arange(d.size(0))
        d[rows, i + rows] = float("inf")  # exclude self-matches only
        mins.append(d.min(dim=1).values)
    nn_dist = torch.cat(mins)
    median = nn_dist.median()
    thresh = threshold_rel * median
    return {
        "n": n,
        "nn_dist_min": float(nn_dist.min()),
        "nn_dist_median": float(median),
        "nn_dist_mean": float(nn_dist.mean()),
        "near_duplicate_fraction": float((nn_dist < thresh).float().mean()),
        "threshold": float(thresh),
    }


def duplicate_check(teacher, images, labels, device, logger, threshold_rel=0.1):
    """Nearest-neighbor duplicate check per class, in pixel and feature space."""
    feats = penultimate_features(teacher, images, device)
    flat_px = images.view(images.size(0), -1)
    per_class = {}
    dup_fracs_px, dup_fracs_ft = [], []
    for c in labels.unique():
        mask = labels == c
        px_stats = _nn_stats(flat_px[mask], threshold_rel)
        ft_stats = _nn_stats(feats[mask], threshold_rel)
        per_class[str(int(c))] = {"pixel": px_stats, "feature": ft_stats}
        if "near_duplicate_fraction" in px_stats:
            dup_fracs_px.append(px_stats["near_duplicate_fraction"])
            dup_fracs_ft.append(ft_stats["near_duplicate_fraction"])
    summary = {
        "per_class": per_class,
        "mean_near_duplicate_fraction_pixel":
            sum(dup_fracs_px) / max(1, len(dup_fracs_px)),
        "mean_near_duplicate_fraction_feature":
            sum(dup_fracs_ft) / max(1, len(dup_fracs_ft)),
    }
    logger.info(f"[diagnose] near-duplicate fraction "
                f"pixel {summary['mean_near_duplicate_fraction_pixel']:.4f}  "
                f"feature {summary['mean_near_duplicate_fraction_feature']:.4f}")
    return summary


def run_diagnostics(cfg, device, out_dir, logger, teacher, synthetic, hw=None):
    """Run all diagnostics; write diagnostics.json and return its content."""
    if hw is None:
        hw = default_hardware(device, logger)
    images, labels, _ = synthetic
    results = {}

    logger.info("[diagnose] subsample scaling curves")
    results["scaling"] = scaling_curves(cfg, device, teacher, synthetic, logger,
                                        hw=hw)

    logger.info("[diagnose] effective rank (teacher penultimate features)")
    feats = penultimate_features(teacher, images, device)
    results["effective_rank"] = effective_rank(feats)
    logger.info(f"[diagnose] effective rank: {results['effective_rank']:.2f} "
                f"(feature dim {feats.size(1)})")

    logger.info("[diagnose] nearest-neighbor duplicate check")
    results["duplicates"] = duplicate_check(
        teacher, images, labels, device, logger,
        threshold_rel=float(cfg["diagnostics"].get("nn_threshold", 0.1)))

    save_json(results, f"{out_dir}/diagnostics.json")
    return results
