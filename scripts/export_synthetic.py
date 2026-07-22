#!/usr/bin/env python
"""Export a repo synthetic dataset to the external (imgs, labels, logits) format.

Converts runs/<case>/synthetic/synthetic.pt (dict with normalized-space
`images`, constructed `labels`, `teacher_confidences`) into the tuple format
    (synth_imgs, synth_labels, synth_logits)
used by the standalone ResNet34->ResNet18 pipeline, where
- synth_imgs   are float32 in [0, 1] pixel space (N, 3, 32, 32),
- synth_labels are the class labels (argmax-consistent with the logits), and
- synth_logits are the frozen repo teacher's logits on the clean images.

The external pipeline's student phase re-labels every augmented batch with
its own teacher on the fly, so the stored logits mainly carry the labels and
let it print class purity; the images are the payload being tested.

Usage:
  python scripts/export_synthetic.py --config configs/cifar10_resnet.yaml \
      --out synthetic_data/zopga_cifar10_synthetic.pt
  # then, in the external script's directory (with its own teacher_best.pth):
  python <external_script>.py --phase student

Options let you point at a specific .pt, teacher checkpoint, device or
label source (--labels constructed|teacher).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from zopga.data import dataset_info  # noqa: E402
from zopga.hardware import resolve_device  # noqa: E402
from zopga.synthesis import load_synthetic  # noqa: E402
from zopga.teacher import load_teacher  # noqa: E402
from zopga.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Export repo synthetic.pt to (imgs, labels, logits) tuple")
    ap.add_argument("--config", required=True, help="repo YAML config")
    ap.add_argument("--pt", default=None,
                    help="synthetic .pt (default runs/<case>/synthetic/synthetic.pt)")
    ap.add_argument("--teacher", default=None,
                    help="repo teacher ckpt (default runs/<case>/teacher/best.pt)")
    ap.add_argument("--out", default=None,
                    help="output path (default synthetic_data/"
                         "zopga_<dataset>_synthetic.pt)")
    ap.add_argument("--labels", choices=["constructed", "teacher"],
                    default="constructed",
                    help="use the synthesis labels or the teacher argmax")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    cfg = load_config(args.config)
    case = cfg.get("case") or os.path.splitext(os.path.basename(args.config))[0]
    info = dataset_info(cfg["dataset"]["name"])
    pt_path = args.pt or os.path.join("runs", case, "synthetic", "synthetic.pt")
    teacher_path = args.teacher or os.path.join("runs", case, "teacher", "best.pt")
    out_path = args.out or os.path.join(
        "synthetic_data", f"zopga_{info['name']}_synthetic.pt")
    device = resolve_device(args.device)

    images, labels, confs = load_synthetic(pt_path)
    print(f"Loaded {tuple(images.shape)} from {pt_path}")

    # Repo images are stored in normalized space; the external pipeline wants
    # [0, 1] pixel space and normalizes with its own statistics.
    mean = torch.tensor(info["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(info["std"]).view(1, -1, 1, 1)
    imgs_px = (images * std + mean).clamp(0.0, 1.0).float()

    if imgs_px.size(1) == 1:
        print("WARNING: single-channel dataset; replicating to 3 channels for "
              "the external RGB pipeline (its CIFAR-only teacher will still "
              "be a domain mismatch).")
        imgs_px = imgs_px.repeat(1, 3, 1, 1)

    # Teacher logits on the clean normalized images, as during generation.
    teacher = load_teacher(cfg, device, teacher_path)
    outs = []
    with torch.no_grad():
        for i in range(0, images.size(0), args.batch):
            outs.append(teacher(images[i:i + args.batch].to(device))
                        .float().cpu())
    logits = torch.cat(outs)

    teacher_labels = logits.argmax(1)
    purity = (teacher_labels == labels).float().mean().item()
    out_labels = labels.long() if args.labels == "constructed" else teacher_labels
    print(f"Teacher/constructed label agreement (purity): {purity:.4f}  "
          f"(exporting '{args.labels}' labels)")
    if confs is not None:
        print(f"Stored teacher confidences: mean {confs.mean():.4f}  "
              f"min {confs.min():.4f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save((imgs_px, out_labels, logits), out_path)
    print(f"Saved (imgs [0,1] {tuple(imgs_px.shape)}, labels, logits) -> "
          f"{out_path}")
    print("Point the external pipeline's --syn_dir at "
          f"'{os.path.dirname(out_path) or '.'}' (expects the file name "
          "zopga_cifar10_synthetic.pt for CIFAR-10) and run --phase student.")


if __name__ == "__main__":
    main()
