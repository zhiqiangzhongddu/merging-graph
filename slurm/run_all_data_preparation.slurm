#!/bin/bash -l
#SBATCH --job-name "prep_data_all"
#SBATCH --time=2-00:00:00       # time (D-HH:MM:SS)
#SBATCH --account=p201211       # project account
#SBATCH --partition gpu         # partition
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1              # one dataset per array task
#SBATCH --gpus=1                # one GPU per dataset
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-70            # 39 node/edge datasets + 32 graph datasets
#SBATCH --qos default
#SBATCH --output=slurm/output/%x-%A_%a.out
#SBATCH --error=slurm/output/%x-%A_%a.err

set -euo pipefail

# Resolve repo root based on submission directory (preferred) or script path.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "${SUBMIT_DIR}" && pwd)"
if [ "$(basename "${ROOT_DIR}")" = "slurm" ]; then
  ROOT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Activate conda environment (defaults to agae). Override CONDA_BASE or CONDA_ENV if needed.
CONDA_BASE="${CONDA_BASE:-/project/home/p200709/conda_base_path/miniconda3}"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  . "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV:-agae}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

mkdir -p "${ROOT_DIR}/slurm/output"

NODE_DATASETS_TSV="${NODE_DATASETS_TSV:-${ROOT_DIR}/data/available_node_datasets.tsv}"
GRAPH_DATASETS_TSV="${GRAPH_DATASETS_TSV:-${ROOT_DIR}/data/available_graph_datasets.tsv}"

if [ ! -f "${NODE_DATASETS_TSV}" ]; then
  echo "[ERROR] Missing node dataset list: ${NODE_DATASETS_TSV}"
  exit 1
fi
if [ ! -f "${GRAPH_DATASETS_TSV}" ]; then
  echo "[ERROR] Missing graph dataset list: ${GRAPH_DATASETS_TSV}"
  exit 1
fi

TMP_DATASET_ROWS="$(mktemp)"
TMP_SINGLE_DATASET="$(mktemp)"
trap 'rm -f "${TMP_DATASET_ROWS}" "${TMP_SINGLE_DATASET}"' EXIT

# Build one merged list so each array task handles one dataset.
{
  grep -v '^[#[:space:]]' "${NODE_DATASETS_TSV}" | sed 's/$/\tnode-edge/'
  grep -v '^[#[:space:]]' "${GRAPH_DATASETS_TSV}" | sed 's/$/\tgraph/'
} > "${TMP_DATASET_ROWS}"

mapfile -t DATASET_ROWS < "${TMP_DATASET_ROWS}"
TOTAL=${#DATASET_ROWS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK_ID >= TOTAL )); then
  echo "[WARN] SLURM_ARRAY_TASK_ID=${TASK_ID} exceeds dataset count (${TOTAL}); skipping."
  exit 0
fi

IFS=$'\t' read -r DATASET DATASET_KIND <<< "${DATASET_ROWS[TASK_ID]}"
printf '%s\n' "${DATASET}" > "${TMP_SINGLE_DATASET}"

echo "========================================================================"
echo "[SLURM][DataPrep] task=${TASK_ID}/${TOTAL} dataset=${DATASET} kind=${DATASET_KIND}"
echo "========================================================================"

srun --ntasks=1 --gpus=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  python "${ROOT_DIR}/run_data_preparation.py" \
    data_preparation.target_datasets "${TMP_SINGLE_DATASET}" \
    data_preparation.dataset.root "${ROOT_DIR}/data/datasets"
