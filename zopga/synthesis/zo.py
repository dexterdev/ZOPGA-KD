"""Zeroth-order gradient estimation and the white-box reference.

The objective is f(x) = log p(c | x) under the frozen teacher (log-softmax of
the target class). ZO gradients use antithetic central finite differences over
q random directions drawn from a low-frequency subspace: noise sampled at a
reduced resolution, bilinearly upsampled to full resolution, L2-normalized.

    g ~= (1/q) * sum_i [ f(x + s u_i) - f(x - s u_i) ] / (2 s) * u_i

All teacher queries run under torch.no_grad(). The white-box variant is the
only place gradients flow, and only with respect to the *input image*.
"""

import torch
import torch.nn.functional as F


def sample_directions(n, shape, lowres, device, generator=None):
    """Sample n unit-norm directions in a low-frequency subspace.

    shape: (C, H, W) full image shape; lowres: reduced grid side length.
    Returns a (n, C*H*W) tensor of normalized directions.
    """
    c, h, w = shape
    noise = torch.randn(n, c, lowres, lowres, device=device, generator=generator)
    up = F.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)
    flat = up.reshape(n, -1)
    return flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def query_logprob(teacher, flat_x, class_idx, shape, chunk=512):
    """Batched f(x) = log p(c | x) for a (N, D) tensor of flat images."""
    outs = []
    for i in range(0, flat_x.size(0), chunk):
        xb = flat_x[i:i + chunk].view(-1, *shape)
        logp = F.log_softmax(teacher(xb), dim=1)
        outs.append(logp[:, class_idx])
    return torch.cat(outs)


@torch.no_grad()
def query_confidence(teacher, flat_x, class_idx, shape, chunk=512):
    """Batched softmax confidence of the target class."""
    outs = []
    for i in range(0, flat_x.size(0), chunk):
        xb = flat_x[i:i + chunk].view(-1, *shape)
        outs.append(F.softmax(teacher(xb), dim=1)[:, class_idx])
    return torch.cat(outs)


def estimate_gradient(teacher, x, class_idx, q, sigma, lowres, shape, device,
                      generator=None, chunk=512):
    """Antithetic central-difference ZO gradient for a pool of candidates.

    x: (P, D) flat candidates. Returns g: (P, D).
    """
    p, d = x.shape
    u = sample_directions(p * q, shape, lowres, device, generator).view(p, q, d)
    x_plus = (x.unsqueeze(1) + sigma * u).reshape(p * q, d)
    x_minus = (x.unsqueeze(1) - sigma * u).reshape(p * q, d)
    f_plus = query_logprob(teacher, x_plus, class_idx, shape, chunk).view(p, q)
    f_minus = query_logprob(teacher, x_minus, class_idx, shape, chunk).view(p, q)
    coeff = (f_plus - f_minus) / (2.0 * sigma)          # (P, q)
    return (coeff.unsqueeze(-1) * u).mean(dim=1)


def whitebox_gradient(teacher, x, class_idx, shape, chunk=512):
    """True gradient of sum_x log p(c|x) w.r.t. the input (upper-bound reference).

    This is the only function where autograd is used against the teacher, and
    gradients flow exclusively into the input images, never into teacher
    parameters (which are frozen and requires_grad=False).
    """
    grads = []
    for i in range(0, x.size(0), chunk):
        xb = x[i:i + chunk].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logp = F.log_softmax(teacher(xb.view(-1, *shape)), dim=1)
            loss = logp[:, class_idx].sum()
        grads.append(torch.autograd.grad(loss, xb)[0])
    return torch.cat(grads)
