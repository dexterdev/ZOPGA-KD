"""The ZO-PGA synthesis loop.

Class-conditional image synthesis by projected gradient ascent on the frozen
teacher's log p(c | x) in (normalized) pixel space:

1. init:     diverse low-frequency patterns (see initializers.py); the
             family pool is the `init` hyperparameter
2. gradient: antithetic central differences over q low-frequency directions,
             batched teacher queries (2q per candidate per step), or true
             input gradients in whitebox mode. With `n_random_ops` > 0 the
             queried objective is log p(c | A(x)) for a random augmentation
             A from the classic 17-op pool (see augment.py)
3. update:   heavy-ball or ZO-AdaMM
4. project:  clamp to the valid pixel box after every step
5. accept:   keep image when teacher confidence in the target class >= tau,
             always measured on the CLEAN (un-augmented) image; abandon and
             restart from a fresh init after a step budget.

A persistent pool of in-progress candidates is cycled until the per-class
target count is reached, yielding a balanced synthetic dataset labelled by
construction. A safety cap (max_total_iters) guarantees termination: if the
cap is hit, the best-confidence candidates seen so far are taken instead.
"""

import time

import numpy as np
import torch

from ..utils import ensure_dir, save_checkpoint, save_json
from .augment import QueryAugment
from .initializers import resolve_families, sample_init
from .optimizers import make_optimizer
from .zo import estimate_gradient, query_confidence, whitebox_gradient


class ZOPGASynthesizer:
    def __init__(self, teacher, cfg, data_info, device, seed=42, logger=None,
                 hw=None):
        self.teacher = teacher
        self.cfg = cfg
        self.info = data_info
        self.device = device
        self.logger = logger
        self.hw = hw
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 1)

        self.init_families = resolve_families(cfg.get("init", "all"))
        self.augment = QueryAugment(cfg, data_info, self.rng)

        c = data_info["in_channels"]
        s = data_info["image_size"]
        self.shape = (c, s, s)
        self.dim = c * s * s
        mean = torch.tensor(data_info["mean"], device=device).view(c, 1, 1)
        std = torch.tensor(data_info["std"], device=device).view(c, 1, 1)
        self.box_lo = ((0.0 - mean) / std).expand(c, s, s).reshape(-1)
        self.box_hi = ((1.0 - mean) / std).expand(c, s, s).reshape(-1)

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def _fresh_candidate(self):
        img = sample_init(self.rng, self.shape, self.info["mean"],
                          self.info["std"], families=self.init_families)
        return img.reshape(-1).to(self.device)

    def _restart(self, pool, opt, steps, best_imgs, best_confs, mask):
        """Re-initialize the pool slots selected by mask."""
        idxs = mask.nonzero(as_tuple=True)[0].tolist()
        for i in idxs:
            pool[i] = self._fresh_candidate()
        opt.reset(mask)
        steps[mask] = 0
        # best_imgs/best_confs are deliberately kept across restarts: they
        # track the best candidate *ever seen* in this slot (safety top-up).
        return idxs

    def _synthesize_class(self, class_idx):
        cfg = self.cfg
        per_class = int(cfg.get("per_class", 100))
        pool_size = int(cfg.get("pool_size", 32))
        q = int(cfg.get("q", 32))
        sigma = float(cfg.get("sigma", 0.01))
        lr = float(cfg.get("lr", 0.05))
        max_steps = int(cfg.get("steps", 500))
        tau = float(cfg.get("tau", 0.99))
        mode = cfg.get("mode", "zo")
        chunk = int(cfg.get("query_batch", 512))
        if self.hw is not None:
            chunk = self.hw.query_batch(chunk)
        max_total = int(cfg.get("max_total_iters", 20000))

        pool = torch.stack([self._fresh_candidate() for _ in range(pool_size)])
        opt = make_optimizer(cfg.get("optimizer", "heavyball"), pool_size,
                             self.dim, self.device, cfg)
        steps = torch.zeros(pool_size, dtype=torch.long, device=self.device)
        best_confs = torch.full((pool_size,), -1.0, device=self.device)
        best_imgs = pool.clone()

        accepted_imgs, accepted_confs = [], []
        it, queries, restarts = 0, 0, 0
        t_start = time.time()
        while len(accepted_imgs) < per_class and it < max_total:
            it += 1
            if mode == "whitebox":
                g = whitebox_gradient(self.teacher, pool, class_idx,
                                      self.shape, chunk)
                queries += pool_size
            else:
                g = estimate_gradient(self.teacher, pool, class_idx, q, sigma,
                                      int(cfg.get("lowres", 8)), self.shape,
                                      self.device, self.gen, chunk,
                                      transform=self.augment
                                      if self.augment.enabled else None)
                queries += pool_size * 2 * q
            pool += opt.step(g, lr)
            pool = torch.maximum(torch.minimum(pool, self.box_hi), self.box_lo)
            steps += 1

            conf = query_confidence(self.teacher, pool, class_idx, self.shape,
                                    chunk)
            queries += pool_size
            improved = conf > best_confs
            best_imgs[improved] = pool[improved]
            best_confs[improved] = conf[improved]

            acc_mask = conf >= tau
            if acc_mask.any():
                for i in acc_mask.nonzero(as_tuple=True)[0].tolist():
                    accepted_imgs.append(pool[i].detach().cpu().clone())
                    accepted_confs.append(float(conf[i]))
                self._restart(pool, opt, steps, best_imgs, best_confs, acc_mask)

            stale = steps >= max_steps
            if stale.any():
                restarts += int(stale.sum())
                self._restart(pool, opt, steps, best_imgs, best_confs, stale)

            if it % 20 == 0 or it == 1:
                elapsed = time.time() - t_start
                self._log(f"  class {class_idx} it {it}: accepted "
                          f"{len(accepted_imgs)}/{per_class}, "
                          f"mean_conf {conf.mean():.3f}, max_conf {conf.max():.3f}, "
                          f"queries {queries:,}, restarts {restarts}, "
                          f"{it / max(1e-9, elapsed):.2f} it/s")

        topped_up = max(0, per_class - len(accepted_imgs))
        if len(accepted_imgs) < per_class:
            # Safety cap reached: top up from best-seen candidates.
            self._log(f"  class {class_idx}: safety cap hit at iter {it}; "
                      f"topping up {per_class - len(accepted_imgs)} images "
                      f"from best-seen candidates")
            # Safety cap reached: top up from best-seen candidates (cycling
            # over the valid slots if the pool holds fewer than needed).
            order = torch.argsort(best_confs, descending=True).tolist()
            valid = [i for i in order if best_confs[i] >= 0] or order
            k = 0
            while len(accepted_imgs) < per_class:
                i = valid[k % len(valid)]
                accepted_imgs.append(best_imgs[i].detach().cpu().clone())
                accepted_confs.append(float(best_confs[i].clamp_min(0.0)))
                k += 1

        stats = {
            "iterations": it,
            "teacher_queries": queries,
            "restarts": restarts,
            "accepted": per_class - topped_up,
            "topped_up": topped_up,
            "wall_time_s": round(time.time() - t_start, 2),
            "query_batch": chunk,
        }
        return (torch.stack(accepted_imgs[:per_class]),
                torch.tensor(accepted_confs[:per_class]),
                stats)

    def synthesize(self, out_dir):
        """Run the full synthesis; save images/labels/confidences tensors."""
        ensure_dir(out_dir)
        num_classes = self.info["num_classes"]
        all_imgs, all_labels, all_confs = [], [], []
        per_class_stats = {}
        aug_desc = (f"{self.augment.n_ops} random ops/"
                    f"{len(self.augment.ops)}-op pool"
                    if self.augment.enabled else "off")
        for c in range(num_classes):
            self._log(f"Synthesizing class {c} "
                      f"(mode={self.cfg.get('mode', 'zo')}, "
                      f"optimizer={self.cfg.get('optimizer', 'heavyball')}, "
                      f"init={self.init_families}, query_augment={aug_desc})")
            imgs, confs, stats = self._synthesize_class(c)
            all_imgs.append(imgs.view(-1, *self.shape))
            all_labels.append(torch.full((imgs.size(0),), c, dtype=torch.long))
            all_confs.append(confs)
            stats["mean_confidence"] = float(confs.mean())
            stats["min_confidence"] = float(confs.min())
            per_class_stats[str(c)] = stats
            self._log(f"  class {c} done: {imgs.size(0)} images, "
                      f"mean teacher conf {confs.mean():.3f}, "
                      f"queries {stats['teacher_queries']:,}, "
                      f"restarts {stats['restarts']}, "
                      f"{stats['wall_time_s']:.1f}s")

        images = torch.cat(all_imgs)
        labels = torch.cat(all_labels)
        confs = torch.cat(all_confs)
        totals = {
            "images": int(images.size(0)),
            "teacher_queries": sum(s["teacher_queries"]
                                   for s in per_class_stats.values()),
            "restarts": sum(s["restarts"] for s in per_class_stats.values()),
            "topped_up": sum(s["topped_up"] for s in per_class_stats.values()),
            "wall_time_s": round(sum(s["wall_time_s"]
                                     for s in per_class_stats.values()), 2),
            "mean_confidence": float(confs.mean()),
        }
        stats_all = {"totals": totals, "per_class": per_class_stats}
        save_json(stats_all, f"{out_dir}/synthesis_stats.json")
        path = f"{out_dir}/synthetic.pt"
        save_checkpoint({"images": images, "labels": labels,
                         "teacher_confidences": confs, "config": dict(self.cfg)},
                        path)
        self._log(f"Saved synthetic dataset: {tuple(images.shape)} -> {path}  "
                  f"(total queries {totals['teacher_queries']:,}, "
                  f"mean conf {totals['mean_confidence']:.3f})")
        return {"images": images, "labels": labels,
                "teacher_confidences": confs, "path": path,
                "stats": stats_all}


def load_synthetic(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["images"], ckpt["labels"], ckpt.get("teacher_confidences")
