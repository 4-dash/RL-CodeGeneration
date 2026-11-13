#!/usr/bin/env bash
# Generate/evaluate using the already prepared environments and benchmark.
# Usage: bash run_rbac.sh [all|generate|evaluate]

# --------------------------- Configuration ---------------------------
# Edit these defaults here, or override them with the RBAC_* variables.
PROJECT_DIR="${RBAC_PROJECT_DIR:-$PWD}"
BENCHMARK_DIR="${RBAC_BENCHMARK_DIR:-../RBAC-Text2SQL-Benchmark}"
RESULTS_DIR="${RBAC_RESULTS_DIR:-$PROJECT_DIR/rbac_results}"
OUTPUT_DIR="${RBAC_OUTPUT_DIR:-}"        # Empty = new run, or latest run in evaluate mode.

CONDA_BIN="${CONDA_EXE:-conda}"            # Conda executable, not an environment.
GENERATION_ENV="${RBAC_GENERATION_ENV:-code_gen}"
EVALUATION_ENV="${RBAC_EVALUATION_ENV:-rolesql}"

DATASET_NAMES="${RBAC_DATASETS:-spider bird}" # Space-separated: spider, bird, or both.
NUM_GPUS="${RBAC_NUM_GPUS:-2}"
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" # Visible GPU IDs; NUM_GPUS selects workers.
LIMIT="${RBAC_LIMIT:-0}"                 # First N rows PER DATASET; 0 = full split.
MAX_NEW_TOKENS="${RBAC_MAX_NEW_TOKENS:-512}"
MODEL="${RBAC_MODEL:-}"                   # Empty = Qwen/Qwen2.5-Coder-1.5B-Instruct,
                                          # or the adapter's base model if supplied.
ADAPTER="${RBAC_ADAPTER:-}"               # Empty = pretrained model; otherwise .../policy.
EXECUTE_SQL="${RBAC_EXECUTE_SQL:-1}"       # 1 = SQL execution + SafeEX; 0 = policy metrics.

SPIDER_DB_DIR="${RBAC_SPIDER_DB_DIR:-$BENCHMARK_DIR/data/spider/database}"
BIRD_DB_DIR="${RBAC_BIRD_DB_DIR:-$BENCHMARK_DIR/data/Bird/dev_databases}"
# Set the database paths above to your actual extracted database directories.
# We pass --db_dir explicitly; upstream's dated default directory is not required.
# For an older evaluation-only rerun, set OUTPUT_DIR to its generation directory.
# A generation run never writes into an existing OUTPUT_DIR, even if it is empty.
# ---------------------------------------------------------------------

set -euo pipefail

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

MODE="${1:-all}"
case "$MODE" in
    all|generate|evaluate) ;;
    -h|--help)
        printf '%s\n' \
            'Usage: bash run_rbac.sh [all|generate|evaluate]' \
            'Configure paths, environments, datasets, GPUs and limits at the top.' \
            'Generation always reserves a new directory and preserves older runs.' \
            'Evaluate reuses the latest completed generation by default.' \
            'Set RBAC_OUTPUT_DIR to evaluate an older saved generation run.' \
            'No downloads, installations or environment creation are performed.'
        exit 0
        ;;
    *) fail "Unknown mode '$MODE'; use all, generate, or evaluate." ;;
esac

[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || fail 'NUM_GPUS must be positive.'
[[ "$LIMIT" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'LIMIT must be 0 (all rows) or positive.'
[[ "$MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || fail 'MAX_NEW_TOKENS must be positive.'
[[ "$EXECUTE_SQL" == 0 || "$EXECUTE_SQL" == 1 ]] || fail 'EXECUTE_SQL must be 0 or 1.'
command -v "$CONDA_BIN" >/dev/null || fail 'Conda was not found; set CONDA_BIN to its path.'
[[ -f "$PROJECT_DIR/eval/rbac.py" ]] || fail "Project not found: $PROJECT_DIR"
EVALUATOR="$BENCHMARK_DIR/rbac-exp/evaluation/evaluate_protocol_v2.py"
[[ -f "$EVALUATOR" ]] || fail "Evaluator not found: $EVALUATOR"
[[ -z "$ADAPTER" || "$MODE" == evaluate || -f "$ADAPTER/adapter_model.safetensors" ]] \
    || fail "Adapter weights not found: $ADAPTER"

# Resolve relative paths before switching working directories.
PROJECT_DIR="$(cd -- "$PROJECT_DIR" && pwd)"
BENCHMARK_DIR="$(cd -- "$BENCHMARK_DIR" && pwd)"
EVALUATOR="$BENCHMARK_DIR/rbac-exp/evaluation/evaluate_protocol_v2.py"
[[ "$RESULTS_DIR" = /* ]] || RESULTS_DIR="$PWD/$RESULTS_DIR"
[[ -z "$OUTPUT_DIR" || "$OUTPUT_DIR" = /* ]] || OUTPUT_DIR="$PWD/$OUTPUT_DIR"
[[ -z "$ADAPTER" || "$ADAPTER" = /* ]] || ADAPTER="$PWD/$ADAPTER"
[[ "$SPIDER_DB_DIR" = /* ]] || SPIDER_DB_DIR="$PWD/$SPIDER_DB_DIR"
[[ "$BIRD_DB_DIR" = /* ]] || BIRD_DB_DIR="$PWD/$BIRD_DB_DIR"

if [[ "$MODE" == evaluate && -z "$OUTPUT_DIR" ]]; then
    [[ -f "$RESULTS_DIR/latest_generation.txt" ]] \
        || fail 'No completed generation recorded; set RBAC_OUTPUT_DIR to an existing run.'
    IFS= read -r OUTPUT_DIR < "$RESULTS_DIR/latest_generation.txt"
    [[ -d "$OUTPUT_DIR" ]] || fail "Saved run directory not found: $OUTPUT_DIR"
fi

# A saved run remembers which datasets it generated. Explicit selection wins.
if [[ "$MODE" == evaluate && -z "${RBAC_DATASETS:-}" && -f "$OUTPUT_DIR/datasets.txt" ]]; then
    IFS= read -r DATASET_NAMES < "$OUTPUT_DIR/datasets.txt"
fi
read -r -a DATASETS <<< "$DATASET_NAMES"
(( ${#DATASETS[@]} > 0 )) || fail 'Select at least one dataset.'

prediction_dir() {
    # Also support the flat role_data.json/predictions.jsonl layout in sss.md.
    if [[ "$MODE" == evaluate && ${#DATASETS[@]} == 1 \
        && ! -f "$OUTPUT_DIR/$1/predictions.jsonl" && -f "$OUTPUT_DIR/predictions.jsonl" ]]; then
        printf '%s\n' "$OUTPUT_DIR"
    else
        printf '%s/%s\n' "$OUTPUT_DIR" "$1"
    fi
}

# Check all selected inputs before spending time on generation.
for dataset in "${DATASETS[@]}"; do
    case "$dataset" in
        spider) db_dir="$SPIDER_DB_DIR" ;;
        bird) db_dir="$BIRD_DB_DIR" ;;
        *) fail "Unsupported dataset '$dataset'; use spider and/or bird." ;;
    esac
    if [[ "$MODE" != evaluate ]]; then
        data_file="$BENCHMARK_DIR/data/selected/$dataset/column_level_rbac_dataset_${dataset}_v3_no_sm.json"
        [[ -f "$data_file" ]] || fail "Dataset not found: $data_file"
    else
        dataset_output="$(prediction_dir "$dataset")"
        [[ -f "$dataset_output/role_data.json" && -f "$dataset_output/predictions.jsonl" ]] \
            || fail "Saved generation files not found in $dataset_output; for flat legacy results select one dataset with RBAC_DATASETS."
    fi
    if [[ "$MODE" != generate ]]; then
        [[ -d "$db_dir" ]] || fail "Database directory not found: $db_dir"
    fi
done

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

if [[ "$MODE" != evaluate ]]; then
    if [[ -z "$OUTPUT_DIR" ]]; then
        mkdir -p -- "$RESULTS_DIR"
        OUTPUT_DIR="$(mktemp -d "$RESULTS_DIR/run_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
    else
        mkdir -p -- "$(dirname -- "$OUTPUT_DIR")"
        mkdir -- "$OUTPUT_DIR" \
            || fail "Cannot reserve $OUTPUT_DIR; generation requires a NEW directory. Existing files were left untouched."
    fi
    # Exclusively create the manifest before any other files: a second writer
    # must stop even on systems where concurrent mkdir calls both report success.
    (
        set -o noclobber
        printf 'model=%s\nadapter=%s\nlimit=%s\nnum_gpus=%s\ngpu_devices=%s\nmax_new_tokens=%s\n' \
            "${MODEL:-adapter base model or Qwen/Qwen2.5-Coder-1.5B-Instruct}" "$ADAPTER" \
            "$LIMIT" "$NUM_GPUS" "$GPU_DEVICES" "$MAX_NEW_TOKENS" > "$OUTPUT_DIR/run_info.txt"
    ) || fail "Generation directory already reserved: $OUTPUT_DIR; existing files were left untouched."
    printf 'Generation directory (retained even on failure): %s\n' "$OUTPUT_DIR"
    printf '%s\n' "${DATASETS[*]}" > "$OUTPUT_DIR/datasets.txt"
    for dataset in "${DATASETS[@]}"; do
        dataset_output="$OUTPUT_DIR/$dataset"
        mkdir -- "$dataset_output"
        generation_args=(
            -m eval.rbac --benchmark-dir "$BENCHMARK_DIR"
            --dataset "$dataset" --output-dir "$dataset_output"
            --generate-only --num-gpus "$NUM_GPUS" --max-new-tokens "$MAX_NEW_TOKENS"
        )
        [[ "$LIMIT" == 0 ]] || generation_args+=(--limit "$LIMIT")
        [[ -z "$MODEL" ]] || generation_args+=(--model "$MODEL")
        [[ -z "$ADAPTER" ]] || generation_args+=(--adapter "$ADAPTER")
        printf '\nGenerating %s -> %s\n' "$dataset" "$dataset_output"
        (
            cd -- "$PROJECT_DIR"
            "$CONDA_BIN" run --no-capture-output -n "$GENERATION_ENV" python "${generation_args[@]}"
        ) 2>&1 | tee "$dataset_output/generation.log"
        [[ -s "$dataset_output/role_data.json" && -s "$dataset_output/predictions.jsonl" ]] \
            || fail "Generation did not save both output files in $dataset_output."
    done
    # Only this pointer is replaced, after ALL selected datasets finish generation.
    # An evaluation failure must not make completed predictions hard to find.
    mkdir -p -- "$RESULTS_DIR"
    latest_file="$(mktemp "$RESULTS_DIR/.latest_generation_XXXXXX")"
    printf '%s\n' "$OUTPUT_DIR" > "$latest_file"
    mv -f -- "$latest_file" "$RESULTS_DIR/latest_generation.txt"
fi

if [[ "$MODE" != generate ]]; then
    for dataset in "${DATASETS[@]}"; do
        dataset_output="$(prediction_dir "$dataset")"
        case "$dataset" in
            spider) db_dir="$SPIDER_DB_DIR" ;;
            bird) db_dir="$BIRD_DB_DIR" ;;
        esac
        mkdir -p -- "$dataset_output/evaluations"
        evaluation_output="$(mktemp -d "$dataset_output/evaluations/eval_$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
        evaluation_args=(
            "$EVALUATOR" --dataset "$dataset"
            --role_json "$dataset_output/role_data.json"
            --prediction_path "$dataset_output/predictions.jsonl"
            --db_dir "$db_dir" --output_dir "$evaluation_output"
        )
        [[ "$EXECUTE_SQL" == 0 ]] || evaluation_args+=(--execute_sql)
        printf '\nEvaluating %s -> %s\n' "$dataset" "$evaluation_output"
        (
            cd -- "$BENCHMARK_DIR/rbac-exp"
            "$CONDA_BIN" run --no-capture-output -n "$EVALUATION_ENV" python "${evaluation_args[@]}"
        ) 2>&1 | tee "$evaluation_output/evaluation.log"
    done
fi

printf '\nDone. Run directory: %s\n' "$OUTPUT_DIR"
