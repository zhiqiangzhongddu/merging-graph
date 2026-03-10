#!/bin/bash -l
#SBATCH --job-name "pt"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --nodes=1               # 1 node per dataset
#SBATCH --partition gpu         # partition
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --gpus-per-task=1       # one GPU per task
#SBATCH --ntasks-per-node=4     # one task per GPU
#SBATCH --cpus-per-task=8       # cores per task (adjust if needed)
#SBATCH --array=0-21            # one dataset per array element (update if experiment file changes)
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# This script runs pretraining as a SLURM array over datasets.
# Each array element allocates 1 node with 4 GPUs and runs only that dataset's
# experiments (across the 4 GPUs) before exiting.
# Expected experiment row format:
#   model dataset task_level induced method

# Resolve repo root based on submission directory (preferred) or script path.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
if [ "$(basename "${ROOT_DIR}")" = "slurm" ]; then
  ROOT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Activate conda environment (defaults to agae). Override CONDA_BASE if needed.
CONDA_BASE="${CONDA_BASE:-/project/home/p200709/conda_base_path/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV:-agae}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

mkdir -p "${ROOT_DIR}/slurm/output"

EXPERIMENT_FILE="${ROOT_DIR}/slurm/run_all_pretrain_experiments.tsv"
if [ ! -f "${EXPERIMENT_FILE}" ]; then
  echo "[pretrain] ERROR: Experiment file not found: ${EXPERIMENT_FILE}" >&2
  exit 2
fi

# Discover datasets from the experiment table (2nd column).
mapfile -t DATASETS < <(
  grep -v "^[#[:space:]]" "${EXPERIMENT_FILE}" \
    | awk -F'[ \t]+' 'NF {print $2}' \
    | sort -u
)

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( ARRAY_ID >= ${#DATASETS[@]} )); then
  echo "[pretrain] WARN: SLURM_ARRAY_TASK_ID=${ARRAY_ID} exceeds dataset count (${#DATASETS[@]}); skipping."
  exit 0
fi
DATASET="${DATASETS[ARRAY_ID]}"
echo "[pretrain] Selected dataset=${DATASET} (array_id=${ARRAY_ID}/${#DATASETS[@]})"

# Launch 4 tasks (one per GPU) on this node; each task runs a slice of this dataset's experiments.
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:---label}"
if srun ${SRUN_EXTRA_ARGS} --ntasks="${SLURM_NTASKS:-1}" bash <<SRUN_SCRIPT
set -euo pipefail

ROOT_DIR="${ROOT_DIR}"
EXPERIMENT_FILE="${EXPERIMENT_FILE}"
DATASET_FILTER="${DATASET}"
TOTAL_TASKS=\${SLURM_NTASKS:-1}
TASK_ID=\${SLURM_PROCID:-0}
HOST="\$(hostname)"
log() { echo "[\$(date +%F' '%T)] [task=\${TASK_ID}/\${TOTAL_TASKS} host=\${HOST} cuda=\${CUDA_VISIBLE_DEVICES:-unset}] \$*"; }
log "Starting pretrain for dataset=\${DATASET_FILTER}"

mapfile -t EXPERIMENTS < <(grep -v "^[#[:space:]]" "\${EXPERIMENT_FILE}")

local_idx=0
for EXP in "\${EXPERIMENTS[@]}"; do
  IFS=$' \t' read -r MODEL DATASET TASK INDUCED METHOD _ <<<"\${EXP}"
  if [ -z "\${METHOD:-}" ]; then
    log "Skipping malformed experiment row: \${EXP}"
    continue
  fi
  if [ "\${DATASET}" != "\${DATASET_FILTER}" ]; then
    continue
  fi
  if (( local_idx % TOTAL_TASKS != TASK_ID )); then
    local_idx=\$((local_idx + 1))
    continue
  fi

  CMD=(python "\${ROOT_DIR}/run_pretrain.py"
    model.name "\${MODEL}"
    pretrain.dataset.name "\${DATASET}"
    pretrain.dataset.task_level "\${TASK}"
    pretrain.dataset.induced "\${INDUCED}"
    pretrain.method "\${METHOD}"
    device 0)

  log "\${CMD[*]}"
  "\${CMD[@]}"
  local_idx=\$((local_idx + 1))
done
SRUN_SCRIPT
then
  :
else
  rc=$?
  echo "[pretrain] ERROR: srun failed with rc=${rc} for dataset=${DATASET}" >&2
  exit "${rc}"
fi
