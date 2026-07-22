"""Initialization families for synthesis candidates.

Candidates are initialized from a diverse pool of pattern families, from
smooth low-frequency signals to structured textures and stochastic noise:

  smooth:   corner / horizontal / vertical / angled gradients, radial bumps,
            fractal Perlin-style noise
  textured: checkerboard gratings, Gabor patches, random straight edges
  noise:    uniform (white), gaussian, pink (1/f^alpha spectrum)

All patterns are produced in pixel space [0, 1] (single channel; the caller
draws one pattern per image channel) and then mapped into the dataset's
normalized space. The active pool is the `synthesis.init` hyperparameter.
"""

import numpy as np
import torch
import torch.nn.functional as F

FAMILIES = ["corner", "horizontal", "vertical", "angled", "radial", "perlin",
            "checkerboard", "gabor", "uniform", "gaussian", "pink", "edges"]


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


def _checkerboard(rng, h, w):
    """Axis-aligned checkerboard with random cell size and phase offset."""
    cell = int(rng.choice([2, 4, 8, 16]))
    oy, ox = int(rng.integers(0, cell)), int(rng.integers(0, cell))
    yy, xx = np.mgrid[0:h, 0:w]
    pat = (((yy + oy) // cell + (xx + ox) // cell) % 2).astype(np.float32)
    lo = rng.uniform(0.0, 0.3)
    hi = rng.uniform(0.7, 1.0)
    return lo + (hi - lo) * pat


def _gabor(rng, h, w):
    """Gabor patch: sinusoidal carrier under a Gaussian envelope, random
    orientation, frequency, phase, center and envelope width."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = yy / max(h - 1, 1), xx / max(w - 1, 1)
    theta = rng.uniform(0, np.pi)
    freq = rng.uniform(2.0, 8.0)          # cycles across the image
    phase = rng.uniform(0, 2 * np.pi)
    cy, cx = rng.uniform(0.25, 0.75, size=2)
    sigma = rng.uniform(0.15, 0.5)
    proj = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    env = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    pat = 0.5 + 0.5 * env * np.cos(2 * np.pi * freq * proj + phase)
    return pat.astype(np.float32)


def _pink_noise(rng, h, w):
    """Spectral (1/f^alpha) noise via FFT with a random spectral exponent."""
    alpha = rng.uniform(1.0, 2.5)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1.0                         # keep the DC term finite
    spectrum = (rng.normal(size=(h, w)) + 1j * rng.normal(size=(h, w)))
    spectrum /= f ** (alpha / 2.0)
    pat = np.real(np.fft.ifft2(spectrum)).astype(np.float32)
    lo, hi = pat.min(), pat.max()
    return (pat - lo) / max(hi - lo, 1e-8)


def _random_edges(rng, h, w):
    """Piecewise-constant field built from a few random straight step edges."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy, xx = yy / max(h - 1, 1), xx / max(w - 1, 1)
    pat = np.full((h, w), 0.5, dtype=np.float32)
    for _ in range(int(rng.integers(1, 5))):
        theta = rng.uniform(0, np.pi)
        proj = xx * np.cos(theta) + yy * np.sin(theta)
        thresh = rng.uniform(proj.min(), proj.max())
        step = rng.uniform(0.3, 1.0) * (1 if rng.random() < 0.5 else -1)
        pat[proj > thresh] += step
    lo, hi = pat.min(), pat.max()
    return (pat - lo) / max(hi - lo, 1e-8)


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
    if family == "checkerboard":
        return _checkerboard(rng, h, w)
    if family == "gabor":
        return _gabor(rng, h, w)
    if family == "uniform":
        return rng.random((h, w), dtype=np.float32)
    if family == "gaussian":
        g = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        return np.clip(0.5 + g / 6.0, 0.0, 1.0)
    if family == "pink":
        return _pink_noise(rng, h, w)
    if family == "edges":
        return _random_edges(rng, h, w)
    raise ValueError(f"Unknown init family '{family}'")


def resolve_families(spec):
    """Resolve the `synthesis.init` config value into a family list.

    spec: "all" / None for every family, or a list (or single name) drawn
    from FAMILIES. Raises on unknown names.
    """
    if spec in (None, "all"):
        return list(FAMILIES)
    families = [spec] if isinstance(spec, str) else list(spec)
    unknown = [f for f in families if f not in FAMILIES]
    if unknown or not families:
        raise ValueError(f"synthesis.init must be 'all' or a non-empty subset "
                         f"of {FAMILIES}, got {spec!r}")
    return families


def sample_init(rng, shape, mean, std, families=None):
    """Sample one candidate image in *normalized* space.

    shape: (C, H, W). mean/std: per-channel normalization constants (tuples).
    A pattern family is chosen at random from `families` (default: all);
    per-channel patterns are randomly scaled and offset so the whole valid
    pixel box is covered over draws.
    """
    c, h, w = shape
    families = families or FAMILIES
    family = families[rng.integers(0, len(families))]
    px = np.empty(shape, dtype=np.float32)
    for ch in range(c):
        pat = sample_pattern(rng, family, h, w)
        scale = rng.uniform(0.6, 1.0)
        offset = rng.uniform(0.0, 1.0 - scale)
        px[ch] = np.clip(offset + scale * pat, 0.0, 1.0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(c, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(c, 1, 1)
    return (torch.from_numpy(px) - mean_t) / std_t
