"""Low-frequency initialization families for synthesis candidates.

Candidates are initialized from diverse *smooth* patterns (never white noise):
linear gradients in several orientations, corner/radial bumps, and fractal
Perlin-style noise implemented from scratch in numpy/torch. Patterns are
produced in pixel space [0, 1] and then mapped into the dataset's normalized
space by the caller.
"""

import numpy as np
import torch
import torch.nn.functional as F

FAMILIES = ["corner", "horizontal", "vertical", "angled", "radial", "perlin"]


def _fractal_noise(rng, h, w, octaves=4, base_res=4):
    """Fractal (Perlin-style) noise: sum of bilinearly upsampled random grids
    with halving amplitude per octave. No external dependency."""
    out = np.zeros((h, w), dtype=np.float32)
    amp_total = 0.0
    for o in range(octaves):
        res = base_res * (2 ** o)
        grid = rng.random((1, 1, res, res), dtype=np.float32)
        grid_t = torch.from_numpy(grid)
        layer = F.interpolate(grid_t, size=(h, w), mode="bilinear",
                              align_corners=False)[0, 0].numpy()
        amp = 0.5 ** o
        out += amp * layer
        amp_total += amp
    out /= amp_total
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-8)


def _gradient_pattern(rng, h, w, angle):
    """Linear ramp along a given angle (radians), normalized to [0, 1]."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx, yy = xx / max(w - 1, 1), yy / max(h - 1, 1)
    proj = xx * np.cos(angle) + yy * np.sin(angle)
    proj -= proj.min()
    proj /= max(proj.max(), 1e-8)
    if rng.random() < 0.5:
        proj = 1.0 - proj
    return proj.astype(np.float32)


def sample_pattern(rng, family, h, w):
    """Draw one single-channel pattern in [0, 1] from the requested family."""
    if family == "horizontal":
        return _gradient_pattern(rng, h, w, 0.0)
    if family == "vertical":
        return _gradient_pattern(rng, h, w, np.pi / 2)
    if family == "angled":
        return _gradient_pattern(rng, h, w, rng.uniform(0, np.pi))
    if family == "corner":
        cy, cx = rng.integers(0, 2, size=2)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt((yy - cy * (h - 1)) ** 2 + (xx - cx * (w - 1)) ** 2)
        d /= max(d.max(), 1e-8)
        return (1.0 - d).astype(np.float32)
    if family == "radial":
        cy, cx = rng.uniform(0.2, 0.8, size=2)
        sigma = rng.uniform(0.15, 0.45)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yy, xx = yy / max(h - 1, 1), xx / max(w - 1, 1)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        return np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)
    if family == "perlin":
        return _fractal_noise(rng, h, w)
    raise ValueError(f"Unknown init family '{family}'")


def sample_init(rng, shape, mean, std):
    """Sample one candidate image in *normalized* space.

    shape: (C, H, W). mean/std: per-channel normalization constants (tuples).
    A pattern family is chosen at random; per-channel patterns are randomly
    scaled and offset so the whole valid pixel box is covered over draws.
    """
    c, h, w = shape
    family = FAMILIES[rng.integers(0, len(FAMILIES))]
    px = np.empty(shape, dtype=np.float32)
    for ch in range(c):
        pat = sample_pattern(rng, family, h, w)
        scale = rng.uniform(0.6, 1.0)
        offset = rng.uniform(0.0, 1.0 - scale)
        px[ch] = np.clip(offset + scale * pat, 0.0, 1.0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(c, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(c, 1, 1)
    return (torch.from_numpy(px) - mean_t) / std_t
