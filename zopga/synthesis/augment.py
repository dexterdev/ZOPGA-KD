"""Random augmentation of teacher queries during ZO synthesis.

When enabled (synthesis.n_random_ops > 0) the ZO objective becomes
f(x) = log p(c | A(x)) with A a fresh random augmentation per gradient step,
which discourages adversarial-noise solutions and pushes candidates toward
images the teacher recognizes robustly. The pool is a 17-operation set
covering geometric, photometric and noise corruptions (registry AUG_OPS):

    rotate, affine, perspective, zoom_crop, color_jitter, grayscale,
    gaussian_blur, sharpness, autocontrast, equalize, posterize, solarize,
    invert, gaussian_noise, salt_pepper, cutout, random_gray_erase

Per gradient step, `n_random_ops` distinct operations are drawn with random
magnitudes and applied identically to the whole query batch. The antithetic
pair x+su / x-su is augmented with the *same* A (paired mode): parameterized
ops sample one parameter set per call, and the stochastic pixel ops
(gaussian_noise, salt_pepper) draw their noise for one half and tile it, so
the central finite difference stays a consistent directional derivative of
f(A(.)).

The tau acceptance test is NOT augmented: candidates are accepted on the
teacher's confidence for the clean image (see synthesizer.py). Augmentation
is also skipped in whitebox mode, whose true-gradient objective is the clean
log-probability.

All operations act on batched images in [0, 1] pixel space; flat candidates
in normalized space are mapped through the dataset mean/std both ways.
Channel-dependent ops (color_jitter's saturation/hue, grayscale) reduce to
their channel-safe part on single-channel images.
"""

import torch
import torchvision.transforms.v2.functional as TF


def _to_uint8(px):
    return (px * 255.0).round().clamp(0, 255).to(torch.uint8)


def _tile_half(make, px, paired):
    """Draw a stochastic tensor for the batch; in paired mode draw it for the
    first half and tile so both antithetic halves get identical noise."""
    b = px.size(0)
    if paired and b % 2 == 0:
        half = make(px[:b // 2])
        return torch.cat([half, half])
    return make(px)


# -- geometric ---------------------------------------------------------------

def op_rotate(px, r, paired):
    return TF.rotate(px, float(r.uniform(-30.0, 30.0)))


def op_affine(px, r, paired):
    size = px.size(-1)
    return TF.affine(
        px, angle=float(r.uniform(-15.0, 15.0)),
        translate=[int(r.uniform(-0.15, 0.15) * size),
                   int(r.uniform(-0.15, 0.15) * size)],
        scale=float(r.uniform(0.85, 1.15)),
        shear=[float(r.uniform(-10.0, 10.0)), float(r.uniform(-10.0, 10.0))])


def op_perspective(px, r, paired):
    h, w = px.shape[-2:]
    d = float(r.uniform(0.1, 0.3))
    dx, dy = int(d * w / 2), int(d * h / 2)

    def j(hi):
        return int(r.integers(0, hi + 1))

    start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    end = [[j(dx), j(dy)],
           [w - 1 - j(dx), j(dy)],
           [w - 1 - j(dx), h - 1 - j(dy)],
           [j(dx), h - 1 - j(dy)]]
    return TF.perspective(px, start, end)


def op_zoom_crop(px, r, paired):
    h, w = px.shape[-2:]
    s = float(r.uniform(0.7, 0.95))
    ch, cw = max(1, int(s * h)), max(1, int(s * w))
    top = int(r.integers(0, h - ch + 1))
    left = int(r.integers(0, w - cw + 1))
    return TF.resized_crop(px, top, left, ch, cw, [h, w], antialias=True)


# -- photometric -------------------------------------------------------------

def op_color_jitter(px, r, paired):
    px = TF.adjust_brightness(px, float(r.uniform(0.7, 1.3)))
    px = TF.adjust_contrast(px, float(r.uniform(0.7, 1.3)))
    if px.size(1) == 3:
        px = TF.adjust_saturation(px, float(r.uniform(0.7, 1.3)))
        px = TF.adjust_hue(px, float(r.uniform(-0.1, 0.1)))
    return px


def op_grayscale(px, r, paired):
    if px.size(1) != 3:
        return px
    return TF.rgb_to_grayscale(px, num_output_channels=3)


def op_gaussian_blur(px, r, paired):
    k = int(r.choice([3, 5]))
    return TF.gaussian_blur(px, [k, k], [float(r.uniform(0.3, 1.5))])


def op_sharpness(px, r, paired):
    return TF.adjust_sharpness(px, float(r.uniform(0.5, 2.0)))


def op_autocontrast(px, r, paired):
    return TF.autocontrast(px)


def op_equalize(px, r, paired):
    return TF.equalize(_to_uint8(px)).float() / 255.0


def op_posterize(px, r, paired):
    return TF.posterize(_to_uint8(px), int(r.integers(4, 9))).float() / 255.0


def op_solarize(px, r, paired):
    return TF.solarize(px, float(r.uniform(0.3, 1.0)))


def op_invert(px, r, paired):
    return 1.0 - px


# -- noise / occlusion -------------------------------------------------------

def op_gaussian_noise(px, r, paired):
    sigma = float(r.uniform(0.02, 0.08))
    noise = _tile_half(lambda t: torch.randn_like(t) * sigma, px, paired)
    return px + noise


def op_salt_pepper(px, r, paired):
    p = float(r.uniform(0.01, 0.05))
    u = _tile_half(
        lambda t: torch.rand(t.size(0), 1, *t.shape[-2:], device=t.device),
        px, paired)
    salt = (u < p / 2).float()
    pepper = (u > 1.0 - p / 2).float()
    return px * (1.0 - salt) * (1.0 - pepper) + salt


def _erase_box(px, r, fill):
    size = px.size(-1)
    k = max(1, int(float(r.uniform(0.1, 0.3)) * size))
    y0 = int(r.integers(0, size - k + 1))
    x0 = int(r.integers(0, size - k + 1))
    out = px.clone()
    out[..., y0:y0 + k, x0:x0 + k] = fill
    return out


def op_cutout(px, r, paired):
    return _erase_box(px, r, 0.0)


def op_random_gray_erase(px, r, paired):
    return _erase_box(px, r, float(r.uniform(0.0, 1.0)))


AUG_OPS = {
    "rotate": op_rotate, "affine": op_affine, "perspective": op_perspective,
    "zoom_crop": op_zoom_crop, "color_jitter": op_color_jitter,
    "grayscale": op_grayscale, "gaussian_blur": op_gaussian_blur,
    "sharpness": op_sharpness, "autocontrast": op_autocontrast,
    "equalize": op_equalize, "posterize": op_posterize,
    "solarize": op_solarize, "invert": op_invert,
    "gaussian_noise": op_gaussian_noise, "salt_pepper": op_salt_pepper,
    "cutout": op_cutout, "random_gray_erase": op_random_gray_erase,
}

OPS = list(AUG_OPS)


def resolve_ops(spec):
    """Resolve the `synthesis.query_augment` config value into an op list.

    spec: "all" / None for the full 17-op pool, or a list (or single name)
    drawn from AUG_OPS. Raises on unknown names.
    """
    if spec in (None, "all"):
        return list(OPS)
    ops = [spec] if isinstance(spec, str) else list(spec)
    unknown = [o for o in ops if o not in AUG_OPS]
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
        receive identical augmentation (stochastic pixel noise is drawn for
        one half and tiled).
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
            px = AUG_OPS[self.ops[i]](px, self.rng, paired)
        px = px.clamp(0.0, 1.0)
        return ((px - mean) / std).reshape(flat_x.shape)
