"""Shared utilities: seeding, logging, checkpointing, timing, config helpers."""

import csv
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import yaml


def set_seed(seed):
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """Total number of parameters (trainable at construction time)."""
    return sum(p.numel() for p in model.parameters())


class AverageMeter:
    """Tracks a running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


class Timer:
    """Context manager wall-clock timer."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        return False


def get_logger(name, log_path=None):
    """Logger writing to stdout and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_csv_row(path, fieldnames, row):
    """Append a row to a CSV, writing the header first if the file is new."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class MetricsLogger:
    """Per-epoch metric history for one training stage.

    Every `log(...)` call appends a row to `<out_dir>/<name>_metrics.csv`
    (header from the first row) and keeps it in memory; `save(extra)` writes
    the full history plus any summary fields to `<out_dir>/<name>_metrics.json`.
    Existing files from a previous run are replaced.
    """

    def __init__(self, out_dir, name):
        self.rows = []
        self._fieldnames = None
        ensure_dir(out_dir)
        self.csv_path = os.path.join(out_dir, f"{name}_metrics.csv")
        self.json_path = os.path.join(out_dir, f"{name}_metrics.json")
        for path in (self.csv_path, self.json_path):
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return v

    def log(self, **row):
        self.rows.append(row)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        append_csv_row(self.csv_path, self._fieldnames,
                       {k: self._fmt(row.get(k)) for k in self._fieldnames})
        return row

    def save(self, extra=None):
        save_json({"summary": extra or {}, "history": self.rows},
                  self.json_path)


def format_metrics(row, skip=("epoch", "epochs")):
    """Render a metrics row as 'key value' pairs for a log line."""
    parts = []
    for k, v in row.items():
        if k in skip:
            continue
        if isinstance(v, float):
            parts.append(f"{k} {v:.4f}")
        elif v is not None:
            parts.append(f"{k} {v}")
    return "  ".join(parts)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, pairs):
    """Apply dotted-key overrides, e.g. ['teacher.epochs=1', 'synthesis.tau=0.9'].

    Values are parsed with yaml so ints/floats/bools/lists/null work.
    """
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got: {pair}")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return cfg


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
