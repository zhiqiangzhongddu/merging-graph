#!/bin/bash -l
#SBATCH --job-name "pt_zinc15"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --nodes=1               # reserve a single GPU node
#SBATCH --partition gpu         # partition
#SBATCH --gpus-per-node=4       # use all 4 GPUs on the node
#SBATCH --gpus-per-task=1       # one GPU per pretraining task
#SBATCH --ntasks-per-node=4     # launch 4 parallel tasks
#SBATCH --cpus-per-task=8       # cores per task (adjust if needed)
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%j.out
#SBATCH --error=slurm/output/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/run_pretrain.py" ]; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Activate conda environment (defaults to merging-graph). Override CONDA_BASE if needed.
CONDA_BASE="${CONDA_BASE:-/project/home/p200709/conda_base_path/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV:-merging-graph}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

mkdir -p "${ROOT_DIR}/slurm/output"
cd "${ROOT_DIR}"
export ROOT_DIR

SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:---label}"
if srun ${SRUN_EXTRA_ARGS} --ntasks=4 --kill-on-bad-exit=1 bash <<'SRUN_SCRIPT'
set -euo pipefail

METHODS=(edge_pred edge_pred context_pred context_pred)
BATCH_SIZES=(1024 2048 1024 2048)

TASK_ID="${SLURM_PROCID:-0}"
TOTAL_TASKS="${SLURM_NTASKS:-4}"
HOST="$(hostname)"

if (( TASK_ID < 0 || TASK_ID >= ${#METHODS[@]} )); then
  echo "[pretrain] ERROR: unexpected task id ${TASK_ID}" >&2
  exit 2
fi

METHOD="${METHODS[TASK_ID]}"
BATCH_SIZE="${BATCH_SIZES[TASK_ID]}"
log() { echo "[$(date +%F' '%T)] [task=${TASK_ID}/${TOTAL_TASKS} host=${HOST} cuda=${CUDA_VISIBLE_DEVICES:-unset}] $*"; }

cd "${ROOT_DIR}"
log "Starting zinc15 pretraining method=${METHOD} batch_size=${BATCH_SIZE}"

# Each task sees only its assigned GPU, so the local CUDA index is always 0.
python "${ROOT_DIR}/run_pretrain.py" \
  model.name gin \
  pretrain.dataset.name zinc15 \
  pretrain.dataset.task_level graph \
  pretrain.dataset.induced False \
  pretrain.method "${METHOD}" \
  pretrain.epochs 100 \
  pretrain.early_stopping 100 \
  pretrain.batch_size "${BATCH_SIZE}" \
  device 0
SRUN_SCRIPT
then
  :
else
  rc=$?
  echo "[pretrain] ERROR: srun failed with rc=${rc}" >&2
  exit "${rc}"
fi
