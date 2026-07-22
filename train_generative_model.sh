#!/usr/bin/env bash
#SBATCH -J train_generative_model
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p YOUR_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH -o job_logs/%j.log
#SBATCH -e job_logs/%j.err

# =============================================================================
# Train a molecular generative model for REAL-SWIT
#
# Usage:
#   sbatch train_generative_model.sh
# or:
#   bash train_generative_model.sh
#
# This script performs three optional steps:
#   1. Create randomized SMILES for the training and validation sets.
#   2. Create an empty generative model and vocabulary.
#   3. Train the generative model.
#
# Edit only the variables in the "User-editable settings" section for most runs.
# =============================================================================

set -euo pipefail

# =============================================================================
# User-editable settings
# Modify this section for your own dataset and computing environment.
# =============================================================================

# Conda environment name used for REAL-SWIT.
CONDA_ENV_NAME="real_swit"

# Optional CUDA module provided by the local cluster.
# Leave empty if CUDA is available without loading an environment module.
CUDA_MODULE=""

# Input training set located under data/demo_data/ by default.
# The file should contain SMILES used to train the generative model.
TRAINING_SET="generative_train_demo.csv"

# Input validation set located under data/demo_data/ by default.
# The file should contain SMILES used to monitor validation performance.
VALIDATION_SET="generative_validation_demo.csv"

# Input directory containing the training and validation files.
DATA_DIR="data/demo_data"

# Output directory for randomized SMILES, model checkpoints, and TensorBoard logs.
OUTPUT_DIR="gm_training_demo"

# Number of randomized SMILES generated for each input molecule. Default: 300. Set to 10 for a quick test run.
N_RANDOMIZED_SMILES=10

# LSTM model depth.
N_LAYERS=6

# LSTM hidden-state size.
HIDDEN_SIZE=2048

# Token embedding size.
EMBEDDING_SIZE=1024

# Number of training epochs. Default: 300. Set to 10 for a quick test run.
N_EPOCHS=10

# Training batch size.
BATCH_SIZE=128

# Number of validation samples used during each validation step.
VALIDATION_SAMPLE_SIZE=1000

# Whether to create randomized SMILES before training.
# Set to false if randomized SMILES have already been prepared.
CREATE_RANDOMIZED_SMILES=true

# Whether to create an empty model and vocabulary before training.
# Set to false if OUTPUT_DIR/models/model.empty already exists.
CREATE_EMPTY_MODEL=true

# SMILES file used to build the model vocabulary.
# By default, use the provided vocabulary dataset. Alternatively, set this to
# "${OUTPUT_DIR}/training/001.smi" to use the randomized training SMILES.
VOCABULARY_FILE="data/demo_data/data4vocabulary_test.smi"

# Whether to train the generative model.
TRAIN_MODEL=true

# Whether to select the saved checkpoint with the lowest validation NLL.
SELECT_BEST_CHECKPOINT=true

# Output path for the selected checkpoint.
BEST_MODEL_PATH="${OUTPUT_DIR}/models/model.best"

# Offset between TensorBoard steps and checkpoint epoch numbers.
# Keep 0 for a normal training run. Change only when selecting from logs created
# by a resumed run whose TensorBoard steps restarted from zero.
BEST_EPOCH_OFFSET=0

# Optional: sample molecules after training.
# Set to true only if you want to sample after checkpoint selection.
SAMPLE_AFTER_TRAINING=true

# Model checkpoint used for sampling.
# If TRAIN_MODEL=true and SAMPLE_AFTER_TRAINING=true, leave this empty to use
# BEST_MODEL_PATH automatically. Otherwise, set the checkpoint path explicitly
# when you want to sample from an existing model.
SAMPLE_MODEL_PATH=""

# Number of molecules to sample if SAMPLE_AFTER_TRAINING=true.
N_SAMPLED_MOLECULES=10000

# Output CSV file for optional sampling.
SAMPLED_OUTPUT="${OUTPUT_DIR}/sampled_molecules.csv"


# =============================================================================
# Internal code
# The code below usually does not need to be modified.
# =============================================================================

# Use the directory from which the job was submitted as the repository directory.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_DIR}"

echo "Working directory: $(pwd)"
echo "Repository directory: ${REPO_DIR}"

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

# Add the repository to PYTHONPATH so internal modules can be imported.
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Prefer the C++ runtime provided by the activated environment.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# Use the Java runtime installed in the environment for PySpark.
export JAVA_HOME="${CONDA_PREFIX}"
export PATH="${JAVA_HOME}/bin:${PATH}"

# Define frequently used paths.
TRAINING_INPUT="${DATA_DIR}/${TRAINING_SET}"
VALIDATION_INPUT="${DATA_DIR}/${VALIDATION_SET}"
TRAINING_RANDOMIZED_DIR="${OUTPUT_DIR}/training"
VALIDATION_RANDOMIZED_DIR="${OUTPUT_DIR}/validation"
MODEL_DIR="${OUTPUT_DIR}/models"
TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
EMPTY_MODEL="${MODEL_DIR}/model.empty"
TRAINED_MODEL_PREFIX="${MODEL_DIR}/model.trained"

# Check input files before running time-consuming steps.
if [[ ! -f "${TRAINING_INPUT}" ]]; then
    echo "ERROR: training set not found: ${TRAINING_INPUT}" >&2
    exit 1
fi

if [[ ! -f "${VALIDATION_INPUT}" ]]; then
    echo "ERROR: validation set not found: ${VALIDATION_INPUT}" >&2
    exit 1
fi

# Create output directories.
mkdir -p "${TRAINING_RANDOMIZED_DIR}" "${VALIDATION_RANDOMIZED_DIR}" "${MODEL_DIR}" "${TENSORBOARD_DIR}"

# Step 1: create randomized SMILES for the training and validation sets.
if [[ "${CREATE_RANDOMIZED_SMILES}" == true ]]; then
    echo "Creating randomized SMILES for the training set..."
    time python gen_models/create_randomized_smiles.py \
        -i "${TRAINING_INPUT}" \
        -o "${TRAINING_RANDOMIZED_DIR}" \
        -n "${N_RANDOMIZED_SMILES}"

    echo "Creating randomized SMILES for the validation set..."
    time python gen_models/create_randomized_smiles.py \
        -i "${VALIDATION_INPUT}" \
        -o "${VALIDATION_RANDOMIZED_DIR}" \
        -n "${N_RANDOMIZED_SMILES}"
fi

# Step 2: create an empty model and vocabulary from the randomized training SMILES.
if [[ "${CREATE_EMPTY_MODEL}" == true ]]; then
    echo "Creating an empty generative model and vocabulary..."
    time python gen_models/create_model.py \
        -l "${N_LAYERS}" \
        -s "${HIDDEN_SIZE}" \
        -e "${EMBEDDING_SIZE}" \
        -i "${VOCABULARY_FILE}" \
        -o "${EMPTY_MODEL}"
fi

# Step 3: train the generative model.
if [[ "${TRAIN_MODEL}" == true ]]; then
    echo "Training the molecular generative model..."
    time python gen_models/train_model.py \
        -i "${EMPTY_MODEL}" \
        -o "${TRAINED_MODEL_PREFIX}" \
        -s "${TRAINING_RANDOMIZED_DIR}" \
        -e "${N_EPOCHS}" \
        --lrm ada \
        --csl "${TENSORBOARD_DIR}" \
        --csv "${VALIDATION_RANDOMIZED_DIR}" \
        --csn "${VALIDATION_SAMPLE_SIZE}" \
        --batch-size "${BATCH_SIZE}"
fi

# Step 4: select the saved checkpoint with the lowest validation NLL.
if [[ "${SELECT_BEST_CHECKPOINT}" == true ]]; then
    echo "Selecting the checkpoint with the lowest validation NLL..."
    time python gen_models/select_best_checkpoint.py \
        --tensorboard-dir "${TENSORBOARD_DIR}" \
        --model-prefix "${TRAINED_MODEL_PREFIX}" \
        --output "${BEST_MODEL_PATH}" \
        --epoch-offset "${BEST_EPOCH_OFFSET}"
fi

# Step 5: optionally sample molecules from the selected checkpoint.
if [[ "${SAMPLE_AFTER_TRAINING}" == true ]]; then
    if [[ -z "${SAMPLE_MODEL_PATH}" ]]; then
        echo "ERROR: SAMPLE_MODEL_PATH must be set when sampling without training in the same run." >&2
        exit 1
    fi

    if [[ ! -f "${SAMPLE_MODEL_PATH}" ]]; then
        echo "ERROR: sampling checkpoint not found: ${SAMPLE_MODEL_PATH}" >&2
        exit 1
    fi

    echo "Sampling molecules from the trained generative model..."
    time python gen_models/sample_from_model.py \
        -m "${SAMPLE_MODEL_PATH}" \
        -n "${N_SAMPLED_MOLECULES}" \
        -o "${SAMPLED_OUTPUT}"
fi

echo "Generative model workflow completed."