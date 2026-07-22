#!/usr/bin/env bash
#SBATCH -J real_swit
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -o job_logs/%j.log
#SBATCH -e job_logs/%j.err

# =============================================================================
# Submit and run a REAL-SWIT workflow on a SLURM cluster
#
# Usage examples:
#   sbatch submit_real_swit.sh ROCK1 train data/demo_data/rock1_train_demo.csv data/demo_data/rock1_test_demo.csv
#   sbatch submit_real_swit.sh ROCK1 predict examples/ROCK1/lightning_logs/version_0/checkpoints/model.ckpt data/demo_data/rock1_test_demo.csv
#   sbatch submit_real_swit.sh ROCK1 generate run_001
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

# Optional CUDA module provided by the local cluster.
# Leave empty if CUDA is available without loading an environment module.
CUDA_MODULE=""

# Number of CPU cores passed to Python scripts.
# Defaults to the number allocated by SLURM.
NCPU="${SLURM_CPUS_PER_TASK:-6}"

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
  bash submit_real_swit.sh <TASK_NAME> <MODE> [MODE_ARGUMENTS]
  sbatch submit_real_swit.sh <TASK_NAME> <MODE> [MODE_ARGUMENTS]

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
  bash submit_real_swit.sh ROCK1 train data/demo_data/rock1_train_demo.csv data/demo_data/rock1_test_demo.csv
  bash submit_real_swit.sh ROCK1 predict examples/ROCK1/lightning_logs/version_0/checkpoints/epoch=9-step=159.ckpt data/demo_data/rock1_test_demo.csv
  bash submit_real_swit.sh ROCK1 generate run_001
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

# Use the submission directory for SLURM jobs and the script directory otherwise.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REPO_DIR="${SLURM_SUBMIT_DIR}"
else
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "${REPO_DIR}"


# Load the CUDA module if requested and if the module command is available.
if [[ -n "${CUDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE}"
fi

# Activate the requested environment unless it is already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    MICROMAMBA_BIN=""

    # MAMBA_EXE is normally defined by "micromamba shell init".
    if [[ -n "${MAMBA_EXE:-}" && -x "${MAMBA_EXE}" ]]; then
        MICROMAMBA_BIN="${MAMBA_EXE}"
    else
        MICROMAMBA_BIN="$(type -P micromamba 2>/dev/null || true)"
    fi

    if [[ -n "${MICROMAMBA_BIN}" && -x "${MICROMAMBA_BIN}" ]]; then
        eval "$("${MICROMAMBA_BIN}" shell hook --shell bash)"
        micromamba activate "${CONDA_ENV_NAME}"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV_NAME}"
    else
        echo "ERROR: Neither micromamba nor Conda could be initialized." >&2
        echo "Activate '${CONDA_ENV_NAME}' before submitting the job, or ensure MAMBA_EXE is exported." >&2
        exit 1
    fi
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: Environment '${CONDA_ENV_NAME}' was not activated correctly." >&2
    exit 1
fi

# Prefer the C++ runtime provided by the activated environment.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Add the repository to PYTHONPATH so internal modules can be imported.
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

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