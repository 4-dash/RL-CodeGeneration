#!/usr/bin/env bash
# Run in the existing training environment. No downloads/setup are performed here.
# Example: RBAC_BENCHMARK_DIR=/path/to/benchmark bash run_rbac_train_sample.sh
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RBAC_PYTHON:-python3}"
SAMPLE_SIZE="${RBAC_SAMPLE_SIZE:-256}"
SEED="${RBAC_SAMPLE_SEED:-42}"
NUM_GPUS="${RBAC_NUM_GPUS:-2}"
RUNS_DIR="${RBAC_TRAIN_RUNS_DIR:-$PROJECT_DIR/savings/rbac_pilots}"

if [[ "${1:-}" == --help ]]; then
    cat <<'HELP'
Usage: RBAC_BENCHMARK_DIR=/path/to/benchmark bash run_rbac_train_sample.sh
Run after activating the existing training environment.
Optional environment variables:
  RBAC_SAMPLE_SIZE=256        Random training rows (must be less than the source).
  RBAC_SAMPLE_SEED=42         Reproducible sampling seed.
  RBAC_NUM_GPUS=2             Number of GPU processes; CUDA_VISIBLE_DEVICES applies.
  RBAC_TRAIN_DATA=...         Source training JSON/JSONL; defaults to Spider train.
  RBAC_TRAIN_DB_DIR=...       Spider SQLite database root.
  RBAC_TRAIN_RUNS_DIR=...     Parent for fresh, separate pilot run directories.
  RBAC_PYTHON=python3         Python in the activated training environment.
  WANDB_MODE=offline         Default; set online for live W&B dashboard reporting.
Each run retains its sample, checkpoints, and console log. No evaluation is run.
HELP
    exit 0
fi
[[ $# == 0 ]] || { printf 'Use --help for usage.\n' >&2; exit 1; }
: "${RBAC_BENCHMARK_DIR:?Set RBAC_BENCHMARK_DIR to the existing benchmark checkout}"
[[ "$SAMPLE_SIZE" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid RBAC_SAMPLE_SIZE\n' >&2; exit 1; }
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid RBAC_NUM_GPUS\n' >&2; exit 1; }
command -v "$PYTHON" >/dev/null
command -v accelerate >/dev/null

cd -- "$PROJECT_DIR"
export RBAC_BENCHMARK_DIR
export RBAC_TRAIN_DB_DIR="${RBAC_TRAIN_DB_DIR:-$RBAC_BENCHMARK_DIR/data/spider/database}"
SOURCE_DATA="${RBAC_TRAIN_DATA:-$RBAC_BENCHMARK_DIR/data/selected/spider/column_level_rbac_dataset_spider_train_v3_no_sm.json}"
[[ -f "$RBAC_BENCHMARK_DIR/rbac-exp/evaluation/prediction_policy.py" ]] \
    || { printf 'Current benchmark prediction_policy.py is missing.\n' >&2; exit 1; }
[[ -f "$SOURCE_DATA" ]] || { printf 'Training data not found: %s\n' "$SOURCE_DATA" >&2; exit 1; }
mkdir -p -- "$RUNS_DIR"
RUN_DIR="$(mktemp -d "$RUNS_DIR/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf 'Pilot files (retained even on failure): %s\n' "$RUN_DIR"

"$PYTHON" - "$SOURCE_DATA" "$RUN_DIR" "$SAMPLE_SIZE" "$SEED" <<'PY'
import json
import os
from pathlib import Path
import random
import sys
from collections import Counter

source, run = Path(sys.argv[1]), Path(sys.argv[2])
size, seed = int(sys.argv[3]), int(sys.argv[4])
with source.open(encoding="utf-8-sig") as handle:
    rows = ([json.loads(line) for line in handle if line.strip()]
            if source.suffix == ".jsonl" else json.load(handle))
if not isinstance(rows, list) or not 0 < size < len(rows):
    raise SystemExit("Sample size must be positive and smaller than the source training set.")
indices = random.Random(seed).sample(range(len(rows)), size)
sample = [rows[index] for index in indices]
root = Path(os.environ["RBAC_TRAIN_DB_DIR"]).expanduser().resolve()
for row in sample:
    db_id = row["db_id"]
    database = (root / db_id / f"{db_id}.sqlite").resolve()
    if not database.is_relative_to(root) or not database.is_file():
        raise SystemExit(f"Missing training database: {database}")
    if row["metadata"]["permission"] not in ("allowed", "denied"):
        raise SystemExit("Unknown permission label in sample")
with (run / "sample.json").open("x", encoding="utf-8") as handle:
    json.dump(sample, handle, ensure_ascii=False)
with (run / "sample_manifest.json").open("x", encoding="utf-8") as handle:
    json.dump({"source": str(source.resolve()), "seed": seed,
               "source_rows": len(rows), "sample_indices": indices}, handle, indent=2)
print(f"Sample: {size}/{len(rows)} rows, seed={seed}, "
      f"permissions={dict(Counter(row['metadata']['permission'] for row in sample))}")
PY

export RBAC_TRAIN_DATA="$RUN_DIR/sample.json"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
printf 'Checkpoints: %s/checkpoints/checkpoint-<step>/policy\n' "$RUN_DIR"
accelerate launch --config-file config/accelerate.yml --num_processes "$NUM_GPUS" \
    main.py --mode fine_tune --param ppo_sql --framework_params sql \
    --output-dir "$RUN_DIR/checkpoints" 2>&1 | tee "$RUN_DIR/training.log"
