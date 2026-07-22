"""Zeroth-order optimizers over a pool of candidates (vectorized state).

State tensors have one row per candidate so the whole candidate pool advances
in lockstep and individual rows can be reset on acceptance / restart.
"""

import torch


class HeavyBall:
    """Heavy-ball momentum with a normalized step: x <- x + lr * m / ||m||."""

    def __init__(self, n, dim, device, beta=0.9):
        self.beta = beta
        self.m = torch.zeros(n, dim, device=device)
        self.t = torch.zeros(n, device=device)

    def step(self, g, lr):
        self.m.mul_(self.beta).add_(g, alpha=1.0 - self.beta)
        direction = self.m / self.m.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.t += 1
        return lr * direction

    def reset(self, mask):
        self.m[mask] = 0.0
        self.t[mask] = 0.0


class ZOAdaMM:
    """Adam-style moments for ZO gradients, with a low second-moment decay
    (beta2 = 0.9 by default) and bias correction."""

    def __init__(self, n, dim, device, beta1=0.9, beta2=0.9, eps=1e-8):
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = torch.zeros(n, dim, device=device)
        self.v = torch.zeros(n, dim, device=device)
        self.t = torch.zeros(n, device=device)

    def step(self, g, lr):
        self.t += 1
        self.m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
        bc1 = (1.0 - self.beta1 ** self.t).clamp_min(1e-8).unsqueeze(1)
        bc2 = (1.0 - self.beta2 ** self.t).clamp_min(1e-8).unsqueeze(1)
        m_hat = self.m / bc1
        v_hat = self.v / bc2
        return lr * m_hat / (v_hat.sqrt() + self.eps)

    def reset(self, mask):
        self.m[mask] = 0.0
        self.v[mask] = 0.0
        self.t[mask] = 0.0


def make_optimizer(name, n, dim, device, cfg):
    if name == "heavyball":
        return HeavyBall(n, dim, device, beta=cfg.get("beta1", 0.9))
    if name == "adamm":
        return ZOAdaMM(n, dim, device, beta1=cfg.get("beta1", 0.9),
                       beta2=cfg.get("beta2", 0.9), eps=cfg.get("eps", 1e-8))
    raise ValueError(f"Unknown ZO optimizer '{name}' (heavyball|adamm)")
