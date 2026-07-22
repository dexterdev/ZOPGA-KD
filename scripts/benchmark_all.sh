#!/usr/bin/env bash
# Run all four ZO-PGA cases end-to-end (benchmark mode: timings + CSV summary).
# Extra args are forwarded to every run.py invocation, e.g.:
#   bash scripts/benchmark_all.sh --device cuda --set teacher.epochs=100
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/mnist_lenet.yaml
  configs/fashionmnist_lenet.yaml
  configs/cifar10_alexnet.yaml
  configs/cifar10_resnet.yaml
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== Benchmark: ${cfg} ==="
  python run.py benchmark --config "${cfg}" "$@"
done

echo
echo "All benchmarks finished. Summary:"
echo "  runs/benchmark_summary.csv"
column -s, -t < runs/benchmark_summary.csv 2>/dev/null || cat runs/benchmark_summary.csv
