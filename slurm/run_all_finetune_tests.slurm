#!/bin/bash -l
#SBATCH --job-name "finetune_tests_by_batch"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --nodes=1               # one node per batch of finetune tests
#SBATCH --partition gpu         # partition
#SBATCH --gpus-per-node=4       # GPUs per node
#SBATCH --gpus-per-task=1       # one GPU per finetune task
#SBATCH --ntasks-per-node=4     # one task per GPU
#SBATCH --cpus-per-task=8       # cores per task (adjust if needed)
#SBATCH --array=0-22            # 90 commands => 23 array elements at 4 commands per job
#SBATCH --qos default           # SLURM qos
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# Derived from test/finetune.sh.
# If test/finetune.sh changes, update slurm/run_all_finetune_tests.commands.txt
# and the array range in this file together.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
if [ "$(basename "${ROOT_DIR}")" = "slurm" ]; then
  ROOT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

CONDA_BASE="${CONDA_BASE:-/project/home/p201211/conda_base_path/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV:-merging-graph}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

mkdir -p "${ROOT_DIR}/slurm/output"
cd "${ROOT_DIR}"

MANIFEST_FILE="${ROOT_DIR}/slurm/run_all_finetune_tests.commands.txt"
if [ ! -f "${MANIFEST_FILE}" ]; then
  echo "[finetune-tests] ERROR: Manifest file not found: ${MANIFEST_FILE}" >&2
  exit 2
fi

export ROOT_DIR
export MANIFEST_FILE

TASK_COUNT="${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-4}}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:---label}"

if srun ${SRUN_EXTRA_ARGS} --ntasks="${TASK_COUNT}" --kill-on-bad-exit=0 bash <<'SRUN_SCRIPT'
set -euo pipefail

ROOT_DIR="${ROOT_DIR:?}"
MANIFEST_FILE="${MANIFEST_FILE:?}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_ID="${SLURM_PROCID:-0}"
TOTAL_TASKS="${SLURM_NTASKS:-${SLURM_NTASKS_PER_NODE:-4}}"
BATCH_SIZE=4
HOST="$(hostname)"

mapfile -t COMMANDS < <(grep -Ev '^[[:space:]]*($|#)' "${MANIFEST_FILE}")
TOTAL_COMMANDS="${#COMMANDS[@]}"
GLOBAL_IDX=$(( ARRAY_ID * BATCH_SIZE + TASK_ID ))

log() {
  echo "[$(date +%F' '%T)] [array=${ARRAY_ID} task=${TASK_ID}/${TOTAL_TASKS} global=${GLOBAL_IDX} host=${HOST} cuda=${CUDA_VISIBLE_DEVICES:-unset}] $*"
}

if (( GLOBAL_IDX >= TOTAL_COMMANDS )); then
  log "No command assigned; total_commands=${TOTAL_COMMANDS}"
  exit 0
fi

CMD="${COMMANDS[GLOBAL_IDX]}"
cd "${ROOT_DIR}"
log "Using python=$(command -v python) conda_prefix=${CONDA_PREFIX:-unset}"
log "Starting command ${GLOBAL_IDX}/${TOTAL_COMMANDS}: ${CMD}"
eval "${CMD}"
RC=$?
if (( RC == 0 )); then
  log "Finished successfully"
else
  log "FAILED with rc=${RC}"
fi
exit "${RC}"
SRUN_SCRIPT
then
  :
else
  rc=$?
  echo "[finetune-tests] ERROR: srun failed with rc=${rc} for array_id=${SLURM_ARRAY_TASK_ID:-0}" >&2
  exit "${rc}"
fi
