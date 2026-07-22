"""Random 17-operation batch augmentation for distillation.

During distillation every synthetic batch is augmented on the fly and the
frozen teacher is queried on the *augmented* images for fresh soft targets,
so the student sees far more diverse views than the raw synthetic set. The
pool is a 17-operation set covering geometric, photometric and noise
corruptions (registry AUG_OPS):

    rotate, affine, perspective, zoom_crop, color_jitter, grayscale,
    gaussian_blur, sharpness, autocontrast, equalize, posterize, solarize,
    invert, gaussian_noise, salt_pepper, cutout, random_gray_erase

Per batch, `n_random_ops` distinct operations are drawn with random
magnitudes and applied to the whole batch (one parameter set per call).
Configured by `distill.query_augment` (op pool: "all" or a subset) and
`distill.n_random_ops` (0 disables). Synthesis (ZO-PGA / PGA) is untouched:
its objective and the tau acceptance test always use clean images.

All operations act on batched images in [0, 1] pixel space; channel-dependent
ops (color_jitter's saturation/hue, grayscale) reduce to their channel-safe
part on single-channel images.
"""

import torch
import torchvision.transforms.v2.functional as TF


def _to_uint8(px):
    return (px * 255.0).round().clamp(0, 255).to(torch.uint8)


# -- geometric ---------------------------------------------------------------

def op_rotate(px, r):
    return TF.rotate(px, float(r.uniform(-30.0, 30.0)))


def op_affine(px, r):
    size = px.size(-1)
    return TF.affine(
        px, angle=float(r.uniform(-15.0, 15.0)),
        translate=[int(r.uniform(-0.15, 0.15) * size),
                   int(r.uniform(-0.15, 0.15) * size)],
        scale=float(r.uniform(0.85, 1.15)),
        shear=[float(r.uniform(-10.0, 10.0)), float(r.uniform(-10.0, 10.0))])


def op_perspective(px, r):
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


def op_zoom_crop(px, r):
    h, w = px.shape[-2:]
    s = float(r.uniform(0.7, 0.95))
    ch, cw = max(1, int(s * h)), max(1, int(s * w))
    top = int(r.integers(0, h - ch + 1))
    left = int(r.integers(0, w - cw + 1))
    return TF.resized_crop(px, top, left, ch, cw, [h, w], antialias=True)


# -- photometric -------------------------------------------------------------

def op_color_jitter(px, r):
    px = TF.adjust_brightness(px, float(r.uniform(0.7, 1.3)))
    px = TF.adjust_contrast(px, float(r.uniform(0.7, 1.3)))
    if px.size(1) == 3:
        px = TF.adjust_saturation(px, float(r.uniform(0.7, 1.3)))
        px = TF.adjust_hue(px, float(r.uniform(-0.1, 0.1)))
    return px


def op_grayscale(px, r):
    if px.size(1) != 3:
        return px
    return TF.rgb_to_grayscale(px, num_output_channels=3)


def op_gaussian_blur(px, r):
    k = int(r.choice([3, 5]))
    return TF.gaussian_blur(px, [k, k], [float(r.uniform(0.3, 1.5))])


def op_sharpness(px, r):
    return TF.adjust_sharpness(px, float(r.uniform(0.5, 2.0)))


def op_autocontrast(px, r):
    return TF.autocontrast(px)


def op_equalize(px, r):
    return TF.equalize(_to_uint8(px)).float() / 255.0


def op_posterize(px, r):
    return TF.posterize(_to_uint8(px), int(r.integers(4, 9))).float() / 255.0


def op_solarize(px, r):
    return TF.solarize(px, float(r.uniform(0.3, 1.0)))


def op_invert(px, r):
    return 1.0 - px


# -- noise / occlusion -------------------------------------------------------

def op_gaussian_noise(px, r):
    sigma = float(r.uniform(0.02, 0.08))
    return px + torch.randn_like(px) * sigma


def op_salt_pepper(px, r):
    p = float(r.uniform(0.01, 0.05))
    u = torch.rand(px.size(0), 1, *px.shape[-2:], device=px.device)
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


def op_cutout(px, r):
    return _erase_box(px, r, 0.0)


def op_random_gray_erase(px, r):
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
    """Resolve the `distill.query_augment` config value into an op list.

    spec: "all" / None for the full 17-op pool, or a list (or single name)
    drawn from AUG_OPS. Raises on unknown names.
    """
    if spec in (None, "all"):
        return list(OPS)
    ops = [spec] if isinstance(spec, str) else list(spec)
    unknown = [o for o in ops if o not in AUG_OPS]
    if unknown or not ops:
        raise ValueError(f"distill.query_augment must be 'all' or a "
                         f"non-empty subset of {OPS}, got {spec!r}")
    return ops


class RandomOpsAugment:
    """Applies n random ops from the pool to a batch of images.

    ops_spec: "all" or a subset of AUG_OPS names; n_random_ops distinct ops
    with fresh random magnitudes are drawn per call (0 disables).
    """

    def __init__(self, ops_spec, n_random_ops, rng):
        self.ops = resolve_ops(ops_spec)
        self.n_ops = int(n_random_ops or 0)
        self.rng = rng

    @property
    def enabled(self):
        return self.n_ops > 0

    def apply_px(self, px):
        """Augment a (B, C, H, W) batch in [0, 1] pixel space."""
        if not self.enabled:
            return px
        chosen = self.rng.choice(len(self.ops),
                                 size=min(self.n_ops, len(self.ops)),
                                 replace=False)
        for i in chosen:
            px = AUG_OPS[self.ops[i]](px, self.rng)
        return px.clamp(0.0, 1.0)

    def describe(self):
        return (f"{self.n_ops} random ops/{len(self.ops)}-op pool"
                if self.enabled else "off")
