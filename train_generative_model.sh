#!/usr/bin/env bash
#SBATCH -J train_generative_model
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p A40,gpu
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

# Optional CUDA module used on some Linux clusters.
# Leave empty if your system does not use environment modules.
CUDA_MODULE="mathlib/cuda/10.1.168_418.67"

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

# Number of randomized SMILES generated for each input molecule.
N_RANDOMIZED_SMILES=300

# LSTM model depth.
N_LAYERS=6

# LSTM hidden-state size.
HIDDEN_SIZE=2048

# Token embedding size.
EMBEDDING_SIZE=1024

# Number of training epochs.
N_EPOCHS=300

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

# Whether to train the generative model.
TRAIN_MODEL=true

# Optional: sample molecules after training.
# Set to true only if you want to sample from a specified trained model checkpoint.
SAMPLE_AFTER_TRAINING=false

# Trained model checkpoint used for optional sampling.
SAMPLE_MODEL_PATH="${OUTPUT_DIR}/models/model.trained"

# Number of molecules to sample if SAMPLE_AFTER_TRAINING=true.
N_SAMPLED_MOLECULES=10000

# Output CSV file for optional sampling.
SAMPLED_OUTPUT="${OUTPUT_DIR}/sampled_molecules.csv"


# =============================================================================
# Internal code
# The code below usually does not need to be modified.
# =============================================================================

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
# This works for most Conda/Miniconda installations.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "ERROR: conda was not found in PATH. Please install Conda or activate the environment manually." >&2
    exit 1
fi

# Add the repository to PYTHONPATH so internal modules can be imported.
export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"

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
        -i "${TRAINING_RANDOMIZED_DIR}/001.smi" \
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

# Optional step: sample molecules from a trained model.
if [[ "${SAMPLE_AFTER_TRAINING}" == true ]]; then
    echo "Sampling molecules from the trained generative model..."
    time python gen_models/sample_from_model.py \
        -m "${SAMPLE_MODEL_PATH}" \
        -n "${N_SAMPLED_MOLECULES}" \
        -o "${SAMPLED_OUTPUT}"
fi

echo "Generative model workflow completed."