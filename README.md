# ZO-PGA: Data-Free Knowledge Distillation via Zeroth-Order Projected Gradient Ascent

ZO-PGA distills a student network from a **frozen black-box teacher without any
real training data**. Class-conditional synthetic images are synthesized by
(projected) gradient ascent on the teacher's log-probability `log p(c | x)` in
pixel space, using only **zeroth-order (query-based) gradient estimates** —
antithetic central finite differences over random directions drawn from a
low-frequency subspace. The student is then trained on the synthetic set with
Hinton knowledge distillation, where the teacher relabels augmented synthetic
images on the fly with soft targets.

## Method summary

1. **Teacher training.** Standard cross-entropy on real data; best checkpoint
   by validation accuracy is kept. Afterwards the teacher is frozen (`eval`
   mode, `requires_grad_(False)` on all parameters) and treated as a black box:
   synthesis only needs forward queries.
2. **Synthesis (ZO-PGA).** For each target class `c`, a persistent pool of
   candidate images is cycled until `per_class` images are accepted:
   - *Init:* diverse low-frequency patterns — smooth gradients (corner,
     horizontal, vertical, angled, radial) and fractal Perlin noise — never
     white noise.
   - *Gradient:* `g ≈ (1/q) Σ [f(x+σuᵢ) − f(x−σuᵢ)] / (2σ) · uᵢ`, with
     `f(x) = log p(c|x)` and directions `uᵢ` sampled at low resolution
     (e.g. 8×8), bilinearly upsampled and normalized. All 2q perturbations of
     all pool candidates are queried in batched forward passes.
   - *Update:* heavy-ball momentum with normalized step, or ZO-AdaMM
     (Adam moments with low `β2 = 0.9`, bias-corrected).
   - *Projection:* clamp to the valid pixel box (the `[0,1]` range mapped
     through dataset normalization) after every step.
   - *Acceptance:* accept when teacher confidence in class `c` is `≥ τ`
     (default 0.99); abandon and restart from a fresh init after a step budget.
   The result is a balanced synthetic dataset labeled by construction, saved
   with teacher confidences. A white-box variant (true input gradients,
     `--mode whitebox`) is included as an upper-bound reference.
3. **Distillation.** The student trains on synthetic data only. Each batch is
   augmented (crop/flip/rotation/cutout for CIFAR; affine/cutout for
   MNIST-like), the teacher relabels the augmented images with soft targets at
   temperature `T`, and the student minimizes `KL(student_T ‖ teacher_T) · T²`
   (KD-only loss — synthetic labels are correct by construction).
4. **Baselines & diagnostics.** Same student architecture trained (a) with CE
   on real data and (b) with classical KD on real data. Diagnostics on the
   synthetic set: subsample scaling curves (accuracy vs data fraction),
   effective rank of teacher penultimate features, and a nearest-neighbor
   duplicate check (mode-collapse detection) in pixel and feature space.

## Installation

```bash
pip install -r requirements.txt
```

## Quickstart

All commands go through `run.py` and write to `runs/<case>/`.

### Full pipeline (teacher → synthesis → distillation → baselines → diagnostics)

```bash
# MNIST: LeNet-5 -> LeNet-5-Half
python run.py all --config configs/mnist_lenet.yaml

# Fashion-MNIST: LeNet-5 -> LeNet-5-Half
python run.py all --config configs/fashionmnist_lenet.yaml

# CIFAR-10: AlexNet -> AlexNet-Half
python run.py all --config configs/cifar10_alexnet.yaml

# CIFAR-10: ResNet-34 -> ResNet-18
python run.py all --config configs/cifar10_resnet.yaml
```

### Benchmarks (all four cases, timings + CSV summary)

```bash
bash scripts/benchmark_all.sh            # runs `run.py benchmark` per config
python run.py benchmark --config configs/mnist_lenet.yaml   # single case
```

### Individual stages

```bash
python run.py train-teacher --config configs/mnist_lenet.yaml
python run.py synthesize    --config configs/mnist_lenet.yaml
python run.py synthesize    --config configs/mnist_lenet.yaml --mode whitebox  # upper-bound reference
python run.py distill       --config configs/mnist_lenet.yaml
python run.py baseline      --config configs/mnist_lenet.yaml --method both    # ce|kd|both
python run.py diagnose      --config configs/mnist_lenet.yaml
```

### Common flags and overrides

- `--device auto|cpu|cuda|cuda:N|mps` (overrides `hardware.device`),
  `--seed N`, `--out DIR`, `--force`
- `--set key=value` — repeatable dotted config override, e.g. a fast smoke run:

```bash
python run.py all --config configs/mnist_lenet.yaml --device cpu \
  --set dataset.max_train_samples=2000 --set teacher.epochs=1 \
  --set synthesis.per_class=10 --set synthesis.steps=100 --set synthesis.q=8 \
  --set synthesis.pool_size=8 --set distill.epochs=1 --set baseline.epochs=1 \
  --set diagnostics.epochs=1 --set 'diagnostics.fractions=[0.5,1.0]'
```

## Hardware-aware optimization

Every config has a `hardware:` section; each `auto` value is resolved once at
startup against the detected hardware (GPU name/VRAM via
`torch.cuda.mem_get_info`, system RAM, CPU count) and the effective settings
are logged and echoed into `results.json`:

```yaml
hardware:
  device: auto              # auto | cpu | cuda | cuda:N | mps
  precision: auto           # auto | amp | fp32  (amp = CUDA mixed precision;
                            #   auto enables AMP on GPUs with tensor cores)
  batch_size: auto          # auto | int | null  (auto scales per-stage batch
                            #   sizes 0.5x-4x to available VRAM/RAM; int
                            #   overrides them; null keeps config values)
  query_batch: auto         # auto | int | null  (synthesis teacher-query
                            #   batch, sized to free memory)
  num_workers: auto         # auto | int  (dataloader workers from CPU count)
  pin_memory: auto          # auto | true | false
  cpu_threads: auto         # auto | int  (torch CPU threads on cpu runs)
  cudnn_benchmark: true     # cuDNN autotuner (CUDA only)
  allow_tf32: true          # TF32 matmul on Ampere+ GPUs
  compile_teacher: false    # torch.compile the frozen teacher (PyTorch 2.x)
  memory_fraction: 0.8      # fraction of free memory auto sizing may plan for
```

All of it is overridable from the CLI like any other config key, e.g.
`--set hardware.precision=fp32 --set hardware.batch_size=64`. Mixed precision
(AMP autocast + GradScaler) is used in the teacher, distillation, baseline and
diagnostics training loops; ZO synthesis queries stay in fp32 so the
finite-difference gradient estimates are not degraded by half-precision noise.

## Metrics

Every training stage logs full per-epoch metrics and persists them next to
its checkpoint as `<stage>_metrics.csv` (one row per epoch) and
`<stage>_metrics.json` (history + final summary):

- **teacher**: train loss/acc, val loss/acc, lr, epoch time, images/sec;
  final test loss/acc + per-class test accuracy.
- **distill (ZO-PGA student)**: KD loss, teacher/student agreement, accuracy
  vs the synthetic labels, lr, epoch time, images/sec, and real-test
  loss/acc every `distill.eval_every` epochs (0 disables).
- **baselines (ce/kd)**: total loss, CE and KD components, train acc,
  val loss/acc, lr, epoch time, images/sec; final test loss/acc + per-class.
- **synthesis**: per-class teacher-query counts, acceptance/restart counts,
  iterations/sec and wall time, saved to `synthetic/synthesis_stats.json`.

`results.json` aggregates the final test accuracies *and* losses of all four
models (teacher, ZO-PGA student, CE and KD baselines), per-class test
accuracies, the distillation end-state (KD loss, agreement, label accuracy),
diagnostics and per-stage timings, plus the resolved hardware settings.

## Expected parameter counts

| Pair | Teacher | Student |
|---|---|---|
| AlexNet → AlexNet-Half (CIFAR-10) | 1,659,178 | 417,434 |
| ResNet-34 → ResNet-18 (CIFAR-10) | 21,282,122 | 11,173,962 |
| LeNet-5 → LeNet-5-Half (MNIST / Fashion-MNIST) | 61,706 | 35,820 |

Verify with:

```bash
python -c "from zopga.models import get_model; \
from zopga.utils import count_parameters; \
[print(n, count_parameters(get_model(n))) for n in \
 ['alexnet','alexnet_half','resnet34','resnet18','lenet5','lenet5_half']]"
```

The CIFAR AlexNet is the TF-port variant (conv 48/128/192/192/128 with 5×5,
5×5, 3×3, 3×3, 3×3 kernels, 3×3/stride-2 pools, BatchNorm throughout,
FC 1152→512→256→10); the student is exactly half-width
(24/64/96/96/64, FC 576→256→128→10). LeNet-5-Half halves only the conv1/conv2
filters (1→3→8→120); FC layers are unchanged.

## Outputs

Everything lands under `runs/<case>/`:

- `teacher/best.pt` — teacher checkpoint (best val accuracy)
- `teacher/teacher_metrics.{csv,json}` — per-epoch teacher training metrics
- `synthetic/synthetic.pt` — synthetic images, labels, teacher confidences
- `synthetic/synthesis_stats.json` — per-class query/acceptance/restart stats
- `student_zopga/student.pt`, `baselines/student_ce.pt`, `baselines/student_kd.pt` — students
- `student_zopga/distill_metrics.{csv,json}`,
  `baselines/baseline_{ce,kd}_metrics.{csv,json}` — per-epoch student metrics
- `diagnostics.json` — scaling curves, effective rank, duplicate stats
- `results.json` — test accuracies *and* losses (teacher / ZO-PGA-KD / CE /
  real-KD students), per-class accuracies, distill end-state, per-stage
  timings, parameter counts, resolved hardware settings, config echo
- `logs/<stage>.log` — per-stage logs
- `runs/benchmark_summary.csv` — one row per `benchmark` run (now includes
  device, AMP flag and test losses)

## Notes on scaling up

Default epoch counts are deliberately modest (e.g. 30 teacher epochs on
CIFAR-10, 100–200 synthetic images per class) so the pipeline is feasible on a
single GPU. For paper-quality results, scale up via config or `--set`:

- teacher epochs: 100–200 (CIFAR-10), `--set teacher.epochs=200`
- synthetic set size: 500–2000 per class, `--set synthesis.per_class=1000`
- distillation epochs: 100+, `--set distill.epochs=100`
- try `--set synthesis.optimizer=adamm` and the `--mode whitebox` reference
  to bound the ZO gradient-estimation gap.

## Repository layout

```
run.py                    # CLI entry point (all subcommands)
configs/                  # YAML configs for the four cases
scripts/benchmark_all.sh  # run all four cases end-to-end
zopga/
  models/                 # alexnet.py, resnet.py, lenet.py + registry
  data.py                 # CIFAR-10 / Fashion-MNIST / MNIST loaders & splits
  teacher.py              # teacher CE training + frozen black-box loading
  synthesis/
    initializers.py       # low-frequency init families (gradients, Perlin)
    zo.py                 # ZO gradient estimators + white-box reference
    optimizers.py         # heavy-ball, ZO-AdaMM
    synthesizer.py        # ZO-PGA loop (projection, acceptance, restarts, pool)
  distill.py              # Hinton KD on synthetic data (on-the-fly relabeling)
  baselines.py            # CE-from-scratch and classical-KD baselines
  diagnostics.py          # scaling curves, effective rank, duplicate check
  evaluate.py             # loss / accuracy / per-class / agreement helpers
  hardware.py             # hardware detection + auto settings (device, AMP,
                          #   batch scaling, workers, query batch, tf32, ...)
  utils.py                # seeding, logging, metrics history, checkpointing,
                          #   timing, config
```
