# REAL-SWIT

REAL-SWIT is a target-guided molecular generation workflow for exploring synthesizable chemical space. It combines a generative model trained on REAL Space-derived molecules with a target-specific scoring model trained to approximate docking scores, enabling iterative reinforcement-learning-based generation of molecules prioritized for a given protein target.

This repository contains the code, example data, and pretrained components associated with the manuscript.
**Access to Synthesizable Chemical Space Through Generative Models Enables Ultra-Large Virtual Screening**  
Kaiyue Zhang *et al.*
---

## Pretrained models
Pretrained model weights are hosted on the Hugging Face Hub: [REAL‑SWIT checkpoints on Hugging Face](https://huggingface.co/KaiyueZhang957/REAL-SWIT-checkpoints). Please download the pretrained file `QBL_model.ckpt` from that repository and place it in this project's `checkpoints/` directory before running the generation or training workflows.

---

## Requirements

REAL-SWIT was developed and tested on a Linux-based GPU cluster. Most Python scripts should also run on other platforms if the required dependencies are properly installed.

- Conda or Miniconda
- Python environment specified in `environment.yml`
- CUDA-enabled GPU recommended for model training and generation

---

## Installation

We recommend installing REAL-SWIT with Conda or Mamba.

```bash
conda env create -f environment.yml
conda activate real_swit

Set the repository root as the Python path when running scripts from the command line:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Repository structure

A typical repository layout is:

```text
REAL_SWIT/
├── README.md
├── environment.yml
├── data/
│   └── demo_data/
├── examples/
├── checkpoints/
│   └── QBL_model.ckpt
├── gen_models/
├── mpn_models/
├── train_generative_model.sh
├── train_scoring_model.py
├── evaluate_target_scoring_model.py
├── prepare_rl_config.py
└── run_generation.sh
```

Main folders and files:

- `data/`: input datasets and demo datasets.
- `examples/`: task-specific outputs, including trained scoring models, predictions, RL configurations, and generated molecules.
- `checkpoints/`: pretrained model checkpoints, including the default generative model `QBL_model.ckpt`.
- `gen_models/`: scripts for training and sampling from the molecular generative model.
- `mpn_models/`: implementation of the target-specific scoring model.
- `train_generative_model.sh`: example workflow for training a generative model.
- `train_scoring_model.py`: trains a target-specific scoring model from docking-score-labeled molecules.
- `evaluate_target_scoring_model.py`: predicts target-specific scores and optionally evaluates performance when labels are provided.
- `prepare_rl_config.py`: creates the reinforcement learning configuration file for target-guided generation.

---

## Input data format

### Training data for the target-specific scoring model

The training file should be a CSV file with at least two columns:

```csv
SMILES,score
CCOc1ccc(...),-8.7
CCN1CC(...),-10.2
```

- The first column should contain SMILES strings.
- The second column should contain docking scores.
- Docking scores are expected to follow the usual convention where more negative values indicate better predicted binding.

## Usage

REAL-SWIT contains three main workflows:

1. Optional: train a molecular generative model from custom molecules.
2. Required: train a target-specific scoring model for the target of interest.
3. Required: create an RL configuration file and run target-guided molecular generation.

If you use the pretrained generative model provided with this repository, you can skip Step 1 and start from Step 2.

---

## Step 1. Optional: train a molecular generative model

This step is only needed if you want to train a new generative model using your own molecular dataset. If you use the pretrained model from the manuscript, the default checkpoint is:

```text
checkpoints/QBL_model.ckpt
```

The generative model training workflow consists of three stages:

1. Create randomized SMILES for the training and validation sets.
2. Create an empty model and prepare the vocabulary from the training data.
3. Train the generative model.

An example SLURM script is provided in:

```bash
train_generative_model.sh
```

The core commands are:

```bash
# 1. Create randomized SMILES for the training set
python gen_models/create_randomized_smiles.py \
    -i data/demo_data/generative_train_demo.csv \
    -o gm_training_demo/training \
    -n 150

# 2. Create randomized SMILES for the validation set
python gen_models/create_randomized_smiles.py \
    -i data/demo_data/generative_validation_demo.csv \
    -o gm_training_demo/validation \
    -n 150

# 3. Create an empty model and build the vocabulary
python gen_models/create_model.py \
    -l 6 \
    -s 2048 \
    -e 1024 \
    -i gm_training_demo/training/001.smi \
    -o gm_training_demo/models/model.empty

# 4. Train the generative model
python gen_models/train_model.py \
    -i gm_training_demo/models/model.empty \
    -o gm_training_demo/models/model.trained \
    -s gm_training_demo/training \
    -e 150 \
    --lrm ada \
    --csl gm_training_demo/tensorboard \
    --csv gm_training_demo/validation \
    --csn 100 \
    --batch-size 128
```

After training, select the desired trained checkpoint and use it as the prior and initial agent model in the RL configuration.

---

## Step 2. Train a target-specific scoring model

For each new target, train a target-specific scoring model using molecules labeled by docking scores.

Example command:

```bash
python train_scoring_model.py \
    data/demo_data/rock1_train_demo.csv \
    ROCK1_demo \
    --testing_dataset_path data/demo_data/rock1_test_demo.csv \
    --ncpu 6 \
    --epochs 10
```

Arguments:

- `data/demo_data/rock1_train_demo.csv`: training dataset containing SMILES and docking scores.
- `ROCK1_demo`: task name. Outputs will be saved under `examples/ROCK1_demo/`.
- `--testing_dataset_path`: optional test dataset for model evaluation.
- `--ncpu`: number of CPU cores available to each prediction worker.
- `--epochs`: number of training epochs.

The trained checkpoint is saved under:

```text
examples/<task_name>/lightning_logs/<version>/checkpoints/
```

If a test dataset with docking scores is provided, the script also evaluates the model and outputs predictions and plots under:

```text
examples/<task_name>/preds/
```

---

## Step 3. Create an RL configuration file

After training the target-specific scoring model, create an RL configuration file for target-guided generation.

Example command:

```bash
python prepare_rl_config.py ROCK1_demo run_001
```

This command creates:

```text
examples/ROCK1_demo/RL_practice/run_001/RL_config.json
```

By default, the configuration uses the pretrained generative model:

```text
checkpoints/QBL_model.ckpt
```

and automatically locates the target-specific scoring model checkpoint under:

```text
examples/ROCK1_demo/lightning_logs/<version>/checkpoints/
```

Before running large-scale generation, check the user-editable settings at the beginning of `prepare_rl_config.py`, including:

- `gen_model_name`: pretrained generative model checkpoint name.
- `n_steps`: number of RL optimization steps.
- `n_mols`: maximum number of generated molecules allowed in the RL run.
- `low_mw` and `high_mw`: molecular weight bounds.
- `max_inverted_score`: upper bound used for the sign-inverted target-specific score. The docking score is multiplied by `-1` before transformation, so a raw docking score of `-50` corresponds to `50` here.
- `target_score_weight`: weight of the target-specific scoring component.
- `sigma`: sigma value used in the augmented likelihood calculation.

---

## Step 4. Run target-guided molecular generation

Run reinforcement-learning-based molecular generation using the configuration file generated in Step 3:

```bash
python gen_models/input.py examples/ROCK1_demo/RL_practice/run_001/RL_config.json
```

Generated molecules and the associated scores are saved under:

```text
examples/ROCK1_demo/RL_practice/run_001/results/
```

The main output file is usually:

```text
scaffold_memory.csv
```

---

## Notes for SLURM users

Example SLURM scripts are provided for running training and generation on GPU clusters. Before submitting a job, update the following settings according to your local cluster environment:

- partition name
- GPU type
- conda environment path
- repository path
- log directory

Avoid hard-coded absolute paths when preparing a public GitHub release. Prefer paths relative to the repository root whenever possible.

---

## Citation

If you use REAL-SWIT in your work, please cite the associated manuscript:

```text
Kaiyue Zhang et al. Access to Synthesizable Chemical Space Through Molecular Generative Models Enables Ultra-Large Virtual Screening.
```

A BibTeX entry will be added after publication.

---

## Contact

For questions or issues, please contact:

- Kaiyue Zhang
- Jing Huang

Alternatively, please open an issue on GitHub.

---

## Acknowledgments

REAL-SWIT builds on ideas and code components from several open-source projects, including:

- [REINVENT](https://github.com/MolecularAI/Reinvent)
- [reinvent-randomized](https://github.com/undeadpixel/reinvent-randomized)
- [Chemprop](https://github.com/chemprop/chemprop)
- [MolPAL](https://github.com/coleygroup/molpal)