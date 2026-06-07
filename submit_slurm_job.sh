#!/usr/bin/env bash
#SBATCH -J real_swit
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p A40,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -o job_logs/%j.log
#SBATCH -e job_logs/%j.err

# =============================================================================
# Submit and run a REAL-SWIT workflow on a SLURM cluster
#
# Usage examples:
#   sbatch submit_slurm_job.sh ROCK1 train data/demo_data/rock1_train.csv data/demo_data/rock1_test.csv
#   sbatch submit_slurm_job.sh ROCK1 predict examples/ROCK1/lightning_logs/version_0/checkpoints/model.ckpt data/demo_data/rock1_test.csv
#   sbatch submit_slurm_job.sh ROCK1 generate run_001
#
# Supported modes:
#   train     Train a target-specific scoring model.
#   predict   Predict or evaluate molecules using a trained scoring model.
#   generate  Run target-guided molecular generation from an existing RL_config.json.
#
# Edit only the variables in the "User-editable settings" section for most runs.
# =============================================================================

set -euo pipefail

# =============================================================================
# User-editable settings
# Modify this section for your own computing environment and default parameters.
# =============================================================================

# Conda environment name used for REAL-SWIT.
CONDA_ENV_NAME="real_swit"

# Optional CUDA module used on some Linux clusters.
# Leave empty if your system does not use environment modules.
CUDA_MODULE="mathlib/cuda/10.1.168_418.67"

# Number of CPU cores passed to Python scripts.
NCPU=6

# Number of epochs for training the target-specific scoring model.
N_EPOCHS=100

# Default RL practice directory under examples/<TASK_NAME>/RL_practice/.
DEFAULT_RL_RUN_NAME="run_001"


# =============================================================================
# Internal code
# The code below usually does not need to be modified.
# =============================================================================

print_usage() {
    cat <<USAGE
Usage:
  bash submit_slurm_job.sh <TASK_NAME> <MODE> [MODE_ARGUMENTS]
  sbatch submit_slurm_job.sh <TASK_NAME> <MODE> [MODE_ARGUMENTS]

Modes:
  train <TRAIN_CSV> <TEST_CSV>
      Train a target-specific scoring model.
      TRAIN_CSV and TEST_CSV should contain SMILES and docking scores.

  predict <CKPT_PATH> <INPUT_CSV>
      Predict target-specific scores for molecules in INPUT_CSV.
      If INPUT_CSV contains target scores, RMSE/PCC and a scatter plot will be generated.
      If INPUT_CSV contains only SMILES, only predictions will be written.

  generate <RL_RUN_NAME>
      Run target-guided molecular generation using:
      examples/<TASK_NAME>/RL_practice/<RL_RUN_NAME>/RL_config.json
      If RL_RUN_NAME is omitted, DEFAULT_RL_RUN_NAME is used.

Examples:
  bash submit_slurm_job.sh ROCK1 train data/demo_data/rock1_train.csv data/demo_data/rock1_test.csv
  bash submit_slurm_job.sh ROCK1 predict examples/ROCK1/lightning_logs/version_0/checkpoints/epoch=9-step=159.ckpt data/demo_data/rock1_test.csv
  bash submit_slurm_job.sh ROCK1 generate run_001
USAGE
}

# Require at least task name and mode.
if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi

TASK_NAME="$1"
MODE="$2"
shift 2

# Locate the repository directory from the location of this script.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

# Create the SLURM log directory if it does not already exist.
mkdir -p job_logs

# Load the CUDA module if requested and if the module command is available.
if [[ -n "${CUDA_MODULE}" ]] && command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE}"
fi

# Activate the Conda environment.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "ERROR: conda was not found in PATH. Please install Conda or activate the environment manually." >&2
    exit 1
fi

# Add the repository to PYTHONPATH so internal modules can be imported.
export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"

case "${MODE}" in
    train)
        if [[ $# -lt 2 ]]; then
            echo "ERROR: train mode requires <TRAIN_CSV> and <TEST_CSV>." >&2
            print_usage
            exit 1
        fi

        TRAIN_CSV="$1"
        TEST_CSV="$2"

        if [[ ! -f "${TRAIN_CSV}" ]]; then
            echo "ERROR: training dataset not found: ${TRAIN_CSV}" >&2
            exit 1
        fi

        if [[ ! -f "${TEST_CSV}" ]]; then
            echo "ERROR: testing dataset not found: ${TEST_CSV}" >&2
            exit 1
        fi

        echo "Training target-specific scoring model for task: ${TASK_NAME}"
        time python train_scoring_model.py \
            "${TRAIN_CSV}" \
            "${TASK_NAME}" \
            --testing_dataset_path "${TEST_CSV}" \
            --ncpu "${NCPU}" \
            --epochs "${N_EPOCHS}"
        ;;

    predict)
        if [[ $# -lt 2 ]]; then
            echo "ERROR: predict mode requires <CKPT_PATH> and <INPUT_CSV>." >&2
            print_usage
            exit 1
        fi

        CKPT_PATH="$1"
        INPUT_CSV="$2"

        if [[ ! -f "${CKPT_PATH}" ]]; then
            echo "ERROR: checkpoint file not found: ${CKPT_PATH}" >&2
            exit 1
        fi

        if [[ ! -f "${INPUT_CSV}" ]]; then
            echo "ERROR: input dataset not found: ${INPUT_CSV}" >&2
            exit 1
        fi

        echo "Predicting target-specific scores for task: ${TASK_NAME}"
        time python evaluate_scoring_model.py \
            "${CKPT_PATH}" \
            "${INPUT_CSV}" \
            --task_name "${TASK_NAME}" \
            --ncpu "${NCPU}"
        ;;

    generate)
        RL_RUN_NAME="${1:-${DEFAULT_RL_RUN_NAME}}"
        RL_CONFIG="examples/${TASK_NAME}/RL_practice/${RL_RUN_NAME}/RL_config.json"

        if [[ ! -f "${RL_CONFIG}" ]]; then
            echo "ERROR: RL configuration file not found: ${RL_CONFIG}" >&2
            echo "Please create it first with prepare_rl_config.py." >&2
            exit 1
        fi

        echo "Running target-guided molecular generation for task: ${TASK_NAME}"
        time python gen_models/input.py "${RL_CONFIG}"
        ;;

    *)
        echo "ERROR: unsupported mode: ${MODE}" >&2
        print_usage
        exit 1
        ;;
esac

echo "REAL-SWIT ${MODE} workflow completed."