"""Random augmentation of teacher queries during ZO synthesis.

When enabled (synthesis.n_random_ops > 0) the ZO objective becomes
f(x) = log p(c | A(x)) with A a fresh random augmentation per gradient step,
which discourages adversarial-noise solutions and pushes candidates toward
images the teacher recognizes robustly. The pool is the classic 17-operation
set (AutoAugment's 16 operations plus identity):

    identity, shear-x, shear-y, translate-x, translate-y, rotate,
    auto-contrast, invert, equalize, solarize, posterize, contrast,
    color, brightness, sharpness, cutout, sample-pairing

Per gradient step, `n_random_ops` distinct operations are drawn with random
magnitudes and applied identically to the whole query batch. The antithetic
pair x+su / x-su is augmented with the *same* A (paired mode), so the central
finite difference stays a consistent directional derivative of f(A(.)).

The tau acceptance test is NOT augmented: candidates are accepted on the
teacher's confidence for the clean image (see synthesizer.py). Augmentation
is also skipped in whitebox mode, whose true-gradient objective is the clean
log-probability.

All operations act on batched images in [0, 1] pixel space; flat candidates
in normalized space are mapped through the dataset mean/std both ways.
"""

import math

import torch
import torchvision.transforms.v2.functional as TF

OPS = [
    "identity", "shear_x", "shear_y", "translate_x", "translate_y", "rotate",
    "auto_contrast", "invert", "equalize", "solarize", "posterize",
    "contrast", "color", "brightness", "sharpness", "cutout",
    "sample_pairing",
]

_SHEAR_MAX_DEG = math.degrees(math.atan(0.3))  # AutoAugment shear range 0.3
_TRANSLATE_MAX = 0.3                           # fraction of image side
_ROTATE_MAX_DEG = 30.0


def resolve_ops(spec):
    """Resolve the `synthesis.query_augment` config value into an op list.

    spec: "all" / None for the full 17-op pool, or a list (or single name)
    drawn from OPS. Raises on unknown names.
    """
    if spec in (None, "all"):
        return list(OPS)
    ops = [spec] if isinstance(spec, str) else list(spec)
    unknown = [o for o in ops if o not in OPS]
    if unknown or not ops:
        raise ValueError(f"synthesis.query_augment must be 'all' or a "
                         f"non-empty subset of {OPS}, got {spec!r}")
    return ops


class QueryAugment:
    """Applies n random ops from the pool to a flat normalized query batch."""

    def __init__(self, cfg, data_info, rng):
        self.ops = resolve_ops(cfg.get("query_augment", "all"))
        self.n_ops = int(cfg.get("n_random_ops", 0) or 0)
        self.rng = rng
        c = data_info["in_channels"]
        self.mean = torch.tensor(data_info["mean"]).view(1, c, 1, 1)
        self.std = torch.tensor(data_info["std"]).view(1, c, 1, 1)

    @property
    def enabled(self):
        return self.n_ops > 0

    def __call__(self, flat_x, shape, paired=False):
        """Augment a (N, D) flat batch; returns the same layout.

        paired=True marks a [plus; minus] antithetic batch: the two halves
        receive identical augmentation (sample-pairing partners are tiled
        so element i pairs with the same partner in both halves).
        """
        if not self.enabled:
            return flat_x
        x = flat_x.view(-1, *shape)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        px = (x * std + mean).clamp(0.0, 1.0)
        chosen = self.rng.choice(len(self.ops),
                                 size=min(self.n_ops, len(self.ops)),
                                 replace=False)
        for i in chosen:
            px = self._apply(self.ops[i], px, paired)
        px = px.clamp(0.0, 1.0)
        return ((px - mean) / std).reshape(flat_x.shape)

    # -- individual operations ---------------------------------------------

    def _apply(self, name, px, paired):
        r = self.rng
        size = px.size(-1)
        if name == "identity":
            return px
        if name == "shear_x":
            deg = float(r.uniform(-_SHEAR_MAX_DEG, _SHEAR_MAX_DEG))
            return TF.affine(px, angle=0.0, translate=[0, 0], scale=1.0,
                             shear=[deg, 0.0])
        if name == "shear_y":
            deg = float(r.uniform(-_SHEAR_MAX_DEG, _SHEAR_MAX_DEG))
            return TF.affine(px, angle=0.0, translate=[0, 0], scale=1.0,
                             shear=[0.0, deg])
        if name == "translate_x":
            t = int(r.uniform(-_TRANSLATE_MAX, _TRANSLATE_MAX) * size)
            return TF.affine(px, angle=0.0, translate=[t, 0], scale=1.0,
                             shear=[0.0, 0.0])
        if name == "translate_y":
            t = int(r.uniform(-_TRANSLATE_MAX, _TRANSLATE_MAX) * size)
            return TF.affine(px, angle=0.0, translate=[0, t], scale=1.0,
                             shear=[0.0, 0.0])
        if name == "rotate":
            return TF.rotate(px, float(r.uniform(-_ROTATE_MAX_DEG,
                                                 _ROTATE_MAX_DEG)))
        if name == "auto_contrast":
            return TF.autocontrast(px)
        if name == "invert":
            return 1.0 - px
        if name == "equalize":
            u8 = (px * 255.0).round().clamp(0, 255).to(torch.uint8)
            return TF.equalize(u8).float() / 255.0
        if name == "solarize":
            return TF.solarize(px, float(r.uniform(0.3, 1.0)))
        if name == "posterize":
            u8 = (px * 255.0).round().clamp(0, 255).to(torch.uint8)
            return TF.posterize(u8, int(r.integers(4, 9))).float() / 255.0
        if name == "contrast":
            return TF.adjust_contrast(px, float(r.uniform(0.6, 1.4)))
        if name == "color":
            if px.size(1) != 3:
                return px  # saturation is undefined for grayscale
            return TF.adjust_saturation(px, float(r.uniform(0.6, 1.4)))
        if name == "brightness":
            return TF.adjust_brightness(px, float(r.uniform(0.6, 1.4)))
        if name == "sharpness":
            return TF.adjust_sharpness(px, float(r.uniform(0.5, 2.0)))
        if name == "cutout":
            frac = float(r.uniform(0.1, 0.3))
            k = max(1, int(frac * size))
            y0 = int(r.integers(0, size - k + 1))
            x0 = int(r.integers(0, size - k + 1))
            out = px.clone()
            out[..., y0:y0 + k, x0:x0 + k] = 0.5
            return out
        if name == "sample_pairing":
            b = px.size(0)
            w = float(r.uniform(0.1, 0.4))
            if paired and b % 2 == 0:
                half = b // 2
                perm = torch.from_numpy(r.permutation(half))
                perm = torch.cat([perm, perm + half]).to(px.device)
            else:
                perm = torch.from_numpy(r.permutation(b)).to(px.device)
            return (1.0 - w) * px + w * px[perm]
        raise ValueError(f"Unknown augmentation op '{name}'")
