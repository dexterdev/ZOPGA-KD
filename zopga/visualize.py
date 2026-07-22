"""Visualization of the synthetic dataset.

Renders synthetic/synthetic.pt as a class x sample image mesh (one row per
class, `samples_per_class` columns -- 10x10 by default for the 10-class
datasets). Each cell shows the de-normalized image with the teacher's softmax
confidence for the constructed label above it; rows are labelled with the
class index. Selection per row: acceptance order ('first'), highest
confidence ('best') or a seeded random sample ('random').
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .synthesis import load_synthetic  # noqa: E402


def visualize_synthetic(pt_path, info, out_path, samples_per_class=10,
                        select="first", seed=42, title=None, logger=None):
    """Save a mesh figure of the synthetic dataset; returns the output path."""
    if select not in ("first", "best", "random"):
        raise ValueError(f"select must be first|best|random, got '{select}'")
    images, labels, confs = load_synthetic(pt_path)
    mean = torch.tensor(info["mean"]).view(-1, 1, 1)
    std = torch.tensor(info["std"]).view(-1, 1, 1)
    imgs_px = (images.cpu() * std + mean).clamp(0.0, 1.0)
    labels = labels.cpu()
    confs = confs.cpu() if confs is not None else None

    classes = sorted(int(c) for c in labels.unique())
    rows, cols = len(classes), int(samples_per_class)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.15, rows * 1.3), squeeze=False)
    gen = torch.Generator().manual_seed(seed)

    for r, c in enumerate(classes):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if select == "best" and confs is not None:
            order = torch.argsort(confs[idx], descending=True)
        elif select == "random":
            order = torch.randperm(idx.numel(), generator=gen)
        else:
            order = torch.arange(idx.numel())
        pick = idx[order[:cols]]
        for j in range(cols):
            ax = axes[r][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j >= pick.numel():
                continue
            img = imgs_px[pick[j]]
            if img.size(0) == 1:
                ax.imshow(img[0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(img.permute(1, 2, 0))
            if confs is not None:
                ax.set_title(f"{float(confs[pick[j]]):.2f}", fontsize=7, pad=2)
        axes[r][0].set_ylabel(f"class {c}", fontsize=8)

    if title is None:
        title = (f"Synthetic dataset ({select} {cols}/class) -- "
                 f"teacher softmax confidence above each image")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        n = int(images.size(0))
        mc = float(confs.mean()) if confs is not None else float("nan")
        logger.info(f"Saved synthetic mesh ({rows}x{cols}, {n} images total, "
                    f"mean conf {mc:.3f}) -> {out_path}")
    return out_path
