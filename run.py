#!/usr/bin/env python
"""ZO-PGA: single CLI entry point.

Subcommands:
  train-teacher   train the teacher on real data (CE), save best checkpoint
  synthesize      run ZO-PGA synthesis against the frozen teacher
  distill         Hinton KD of the student on the synthetic dataset
  baseline        student baselines on real data (--method ce|kd|both)
  diagnose        scaling curves, effective rank, duplicate check
  all             full pipeline for one config (teacher -> synthesize ->
                  distill -> baselines -> diagnose), writes results.json
  benchmark       same as 'all' plus per-stage timings and a row appended to
                  runs/benchmark_summary.csv

Common flags:
  --config PATH     YAML config (required)
  --device DEV      auto|cpu|cuda|cuda:0|mps ... (overrides hardware.device)
  --seed N          override the config seed
  --out DIR         output root (default: runs/<case>)
  --set k=v         dotted config override, repeatable
                    (e.g. --set teacher.epochs=1 --set hardware.precision=fp32)
  --force           re-run stages even if artifacts already exist

Hardware-aware settings (device, mixed precision, batch-size scaling,
dataloader workers, synthesis query batch) come from the optional
`hardware:` section of the config; every `auto` value is resolved against
the detected GPU VRAM / system RAM / CPU count at startup and logged.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402,F401

from zopga.baselines import train_baseline  # noqa: E402
from zopga.data import dataset_info, get_dataloaders  # noqa: E402
from zopga.diagnostics import run_diagnostics  # noqa: E402
from zopga.distill import distill_student  # noqa: E402
from zopga.evaluate import evaluate  # noqa: E402
from zopga.hardware import HardwareManager  # noqa: E402
from zopga.models import get_model  # noqa: E402
from zopga.synthesis import ZOPGASynthesizer, load_synthetic  # noqa: E402
from zopga.teacher import load_teacher, train_teacher  # noqa: E402
from zopga.utils import (Timer, append_csv_row, apply_overrides,  # noqa: E402
                         count_parameters, ensure_dir, get_logger, load_config,
                         save_json, set_seed)


def prepare(args):
    """Load config, apply overrides, seed, hardware, output dir, logger."""
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg.setdefault("seed", 42)
    set_seed(cfg["seed"])
    case = cfg.get("case") or os.path.splitext(os.path.basename(args.config))[0]
    out_root = args.out or os.path.join("runs", case)
    ensure_dir(out_root)
    logger = get_logger(case, os.path.join(out_root, "logs", f"{args.command}.log"))
    hw = HardwareManager(cfg.get("hardware"), cli_device=args.device,
                         logger=logger)
    hw.apply_global_settings()
    logger.info(f"case={case} device={hw.device} seed={cfg['seed']} "
                f"out={out_root}")
    hw.log_summary()
    return cfg, hw, out_root, logger


def teacher_ckpt_path(out_root):
    return os.path.join(out_root, "teacher", "best.pt")


def synthetic_path(out_root):
    return os.path.join(out_root, "synthetic", "synthetic.pt")


def stage_teacher(cfg, hw, out_root, logger, force=False):
    path = teacher_ckpt_path(out_root)
    if os.path.exists(path) and not force:
        logger.info(f"Teacher checkpoint exists, skipping training ({path})")
        return None
    return train_teacher(cfg, hw.device, os.path.join(out_root, "teacher"),
                         logger, hw=hw)


def stage_synthesize(cfg, hw, out_root, logger, force=False):
    path = synthetic_path(out_root)
    if os.path.exists(path) and not force:
        logger.info(f"Synthetic dataset exists, skipping synthesis ({path})")
        return load_synthetic(path)
    teacher = hw.maybe_compile_teacher(
        load_teacher(cfg, hw.device, teacher_ckpt_path(out_root)))
    info = dataset_info(cfg["dataset"]["name"])
    synth = ZOPGASynthesizer(teacher, cfg["synthesis"], info, hw.device,
                             seed=cfg["seed"], logger=logger, hw=hw)
    result = synth.synthesize(os.path.join(out_root, "synthetic"))
    del teacher
    return result["images"], result["labels"], result["teacher_confidences"]


def stage_distill(cfg, hw, out_root, logger):
    teacher = hw.maybe_compile_teacher(
        load_teacher(cfg, hw.device, teacher_ckpt_path(out_root)))
    synthetic = load_synthetic(synthetic_path(out_root))
    res = distill_student(cfg, hw.device,
                          os.path.join(out_root, "student_zopga"),
                          logger, teacher, synthetic, hw=hw)
    del teacher
    return res


def stage_baselines(cfg, hw, out_root, logger, method):
    teacher = None
    if method in ("kd", "both"):
        teacher = load_teacher(cfg, hw.device, teacher_ckpt_path(out_root))
    results = {}
    methods = ("ce", "kd") if method == "both" else (method,)
    for m in methods:
        results[m] = train_baseline(cfg, hw.device,
                                    os.path.join(out_root, "baselines"),
                                    logger, m, teacher=teacher, hw=hw)
    return results


def stage_diagnose(cfg, hw, out_root, logger):
    teacher = load_teacher(cfg, hw.device, teacher_ckpt_path(out_root))
    synthetic = load_synthetic(synthetic_path(out_root))
    res = run_diagnostics(cfg, hw.device, out_root, logger, teacher, synthetic,
                          hw=hw)
    del teacher
    return res


def run_full_pipeline(args, with_csv):
    cfg, hw, out_root, logger = prepare(args)
    timings, results = {}, {}

    with Timer() as t:
        stage_teacher(cfg, hw, out_root, logger, force=args.force)
    timings["teacher"] = t.elapsed

    with Timer() as t:
        stage_synthesize(cfg, hw, out_root, logger, force=args.force)
    timings["synthesis"] = t.elapsed

    with Timer() as t:
        results["student_zopga"] = stage_distill(cfg, hw, out_root, logger)
    timings["distill"] = t.elapsed

    with Timer() as t:
        results["baselines"] = stage_baselines(cfg, hw, out_root, logger,
                                               "both")
    timings["baselines"] = t.elapsed

    with Timer() as t:
        results["diagnostics"] = stage_diagnose(cfg, hw, out_root, logger)
    timings["diagnostics"] = t.elapsed
    timings["total"] = sum(timings.values())

    # Teacher test metrics + parameter counts
    teacher = load_teacher(cfg, hw.device, teacher_ckpt_path(out_root))
    loaders = get_dataloaders(cfg, seed=cfg["seed"], with_aug=False, hw=hw)
    teacher_test = evaluate(teacher, loaders["test"], hw.device, hw.autocast)
    del teacher
    info = dataset_info(cfg["dataset"]["name"])
    param_counts = {
        "teacher": count_parameters(get_model(cfg["teacher"]["arch"],
                                              info["num_classes"])),
        "student": count_parameters(get_model(cfg["student"]["arch"],
                                              info["num_classes"])),
    }

    zopga_res = results["student_zopga"]
    summary = {
        "case": cfg.get("case"),
        "config": cfg,
        "hardware": hw.summary(),
        "param_counts": param_counts,
        "accuracies": {
            "teacher": teacher_test["acc"],
            "student_zopga_kd": zopga_res["test_acc"],
            "student_ce_real": results["baselines"]["ce"]["test_acc"],
            "student_kd_real": results["baselines"]["kd"]["test_acc"],
        },
        "losses": {
            "teacher_test": teacher_test["loss"],
            "student_zopga_kd_test": zopga_res["test_loss"],
            "student_zopga_final_kd_loss": zopga_res["final_kd_loss"],
            "student_ce_real_test": results["baselines"]["ce"]["test_loss"],
            "student_kd_real_test": results["baselines"]["kd"]["test_loss"],
        },
        "per_class_test_acc": {
            "teacher": teacher_test["per_class_acc"],
            "student_zopga_kd": zopga_res["per_class_test_acc"],
            "student_ce_real": results["baselines"]["ce"]["per_class_test_acc"],
            "student_kd_real": results["baselines"]["kd"]["per_class_test_acc"],
        },
        "distill_final": {
            "kd_loss": zopga_res["final_kd_loss"],
            "teacher_agreement": zopga_res["final_agreement"],
            "synthetic_label_acc": zopga_res["final_synthetic_label_acc"],
        },
        "diagnostics": {
            "effective_rank": results["diagnostics"]["effective_rank"],
            "scaling": results["diagnostics"]["scaling"],
            "duplicates": {
                k: v for k, v in results["diagnostics"]["duplicates"].items()
                if k != "per_class"},
        },
        "timings_sec": timings,
    }
    save_json(summary, os.path.join(out_root, "results.json"))
    logger.info(f"results.json written to {out_root}/results.json")
    logger.info(
        f"[summary] teacher {teacher_test['acc']:.4f} | "
        f"zopga-kd {zopga_res['test_acc']:.4f} | "
        f"ce-real {results['baselines']['ce']['test_acc']:.4f} | "
        f"kd-real {results['baselines']['kd']['test_acc']:.4f} | "
        f"total {timings['total']:.1f}s")

    if with_csv:
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case": cfg.get("case"),
            "config": os.path.basename(args.config),
            "device": str(hw.device),
            "amp": hw.use_amp,
            "teacher_acc": f"{teacher_test['acc']:.4f}",
            "teacher_loss": f"{teacher_test['loss']:.4f}",
            "synth_kd_acc": f"{zopga_res['test_acc']:.4f}",
            "synth_kd_loss": f"{zopga_res['test_loss']:.4f}",
            "ce_acc": f"{results['baselines']['ce']['test_acc']:.4f}",
            "kd_acc": f"{results['baselines']['kd']['test_acc']:.4f}",
            "effective_rank": f"{results['diagnostics']['effective_rank']:.2f}",
            "nn_dup_frac_pixel":
                f"{results['diagnostics']['duplicates']['mean_near_duplicate_fraction_pixel']:.4f}",
            "time_teacher_s": f"{timings['teacher']:.1f}",
            "time_synthesis_s": f"{timings['synthesis']:.1f}",
            "time_distill_s": f"{timings['distill']:.1f}",
            "time_baselines_s": f"{timings['baselines']:.1f}",
            "time_diagnostics_s": f"{timings['diagnostics']:.1f}",
            "time_total_s": f"{timings['total']:.1f}",
        }
        append_csv_row("runs/benchmark_summary.csv", list(row.keys()), row)
        logger.info("Appended row to runs/benchmark_summary.csv")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="ZO-PGA: data-free KD via zeroth-order projected "
                    "gradient ascent")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ["train-teacher", "synthesize", "distill", "baseline",
                "diagnose", "all", "benchmark"]:
        p = sub.add_parser(cmd)
        p.add_argument("--config", required=True)
        p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--out", default=None)
        p.add_argument("--set", action="append", default=[],
                       help="dotted config override key=value (repeatable)")
        p.add_argument("--force", action="store_true")
        if cmd == "baseline":
            p.add_argument("--method", choices=["ce", "kd", "both"],
                           default="both")
        if cmd == "synthesize":
            p.add_argument("--mode", choices=["zo", "whitebox"], default=None,
                           help="override synthesis.mode")
    args = parser.parse_args()

    if getattr(args, "mode", None):
        args.set = list(args.set) + [f"synthesis.mode={args.mode}"]

    if args.command in ("all", "benchmark"):
        run_full_pipeline(args, with_csv=(args.command == "benchmark"))
        return

    cfg, hw, out_root, logger = prepare(args)
    if args.command == "train-teacher":
        stage_teacher(cfg, hw, out_root, logger, force=True)
    elif args.command == "synthesize":
        stage_synthesize(cfg, hw, out_root, logger, force=True)
    elif args.command == "distill":
        stage_distill(cfg, hw, out_root, logger)
    elif args.command == "baseline":
        stage_baselines(cfg, hw, out_root, logger, args.method)
    elif args.command == "diagnose":
        stage_diagnose(cfg, hw, out_root, logger)


if __name__ == "__main__":
    main()
