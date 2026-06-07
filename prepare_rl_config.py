#!/usr/bin/env python

"""
Prepare a reinforcement learning configuration file for REAL-SWIT.

This script automatically locates the trained target-specific scoring model
checkpoint under examples/<task_name>/lightning_logs/, and creates an
RL_config.json file for a target-guided generation run.
"""


# =============================================================================
# User-editable settings
# Modify the variables in this section for each task.
# The remaining sections usually do not need to be changed.
# =============================================================================

import sys
from pathlib import Path


# Name of the target/task folder under examples/.
# Example command:
#   python prepare_rl_config.py ROCK1 run_001
task_name = sys.argv[1]

# Name of the output subfolder under examples/<task_name>/RL_practice/.
# Example command:
#   python prepare_rl_config.py ROCK1 run_001
rl_run_name = sys.argv[2]

# Name of the pretrained generative model used as both prior and initial agent.
gen_model_name = "QBL_model.ckpt"

# Number of RL optimization steps.
n_steps = 20

# Maximum number of molecules to generate during the RL run.
n_mols = 1000

# Lower molecular weight bound used in the molecular weight scoring component.
low_mw = 200

# Upper molecular weight bound used in the molecular weight scoring component.
high_mw = 600

# Upper bound used for the sign-inverted target-specific score.
# The docking score is multiplied by -1 before transformation, so a raw docking score of -50 corresponds to 50 here.
max_inverted_score = 50

# Job name stored in the REINVENT-style configuration file.
job_name = "gen"

# Weight of the target-specific scoring model in the multi-component scoring function.
target_score_weight = 2

# Sigma value used in the augmented likelihood calculation during RL.
sigma = 128

# Learning rate for reinforcement learning.
learning_rate = 0.0001

# Batch size used during RL generation.
batch_size = 128

# Logging frequency during RL.
logging_frequency = 20000


# =============================================================================
# Internal code
# The code below usually does not need to be modified.
# =============================================================================

import json


def find_single_checkpoint(task_dir: Path) -> Path:
    """
    Automatically find a single checkpoint file from:
    examples/<task_name>/lightning_logs/<version_or_job>/checkpoints/
    """

    lightning_logs_dir = task_dir / "lightning_logs"

    if not lightning_logs_dir.exists():
        raise FileNotFoundError(
            f"Cannot find lightning_logs directory: {lightning_logs_dir}"
        )

    job_dirs = [p for p in lightning_logs_dir.iterdir() if p.is_dir()]

    if len(job_dirs) == 0:
        raise FileNotFoundError(
            f"No job/version folder found under: {lightning_logs_dir}"
        )

    if len(job_dirs) > 1:
        job_names = [p.name for p in job_dirs]
        raise RuntimeError(
            "More than one job/version folder was found under "
            f"{lightning_logs_dir}:\n{job_names}\n"
            "Please keep only one folder or modify the script to specify one explicitly."
        )

    checkpoint_dir = job_dirs[0] / "checkpoints"

    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Cannot find checkpoints directory: {checkpoint_dir}"
        )

    checkpoint_files = sorted(checkpoint_dir.glob("*.ckpt"))

    if len(checkpoint_files) == 0:
        raise FileNotFoundError(
            f"No .ckpt file found under: {checkpoint_dir}"
        )

    if len(checkpoint_files) > 1:
        checkpoint_names = [p.name for p in checkpoint_files]
        raise RuntimeError(
            "More than one checkpoint file was found under "
            f"{checkpoint_dir}:\n{checkpoint_names}\n"
            "Please keep only one checkpoint or modify the script to specify one explicitly."
        )

    return checkpoint_files[0]


def build_rl_configuration(
    output_dir: Path,
    prior_model_path: Path,
    scoring_model_path: Path,
) -> dict:
    """
    Build the REINVENT-style reinforcement learning configuration dictionary.
    """

    configuration = {
        "version": 2,
        "run_type": "reinforcement_learning",
    }

    configuration["logging"] = {
        "sender": "http://127.0.0.1",
        "recipient": "local",
        "logging_frequency": logging_frequency,
        "logging_path": str(output_dir / "progress.log"),
        "resultdir": str(output_dir / "results"),
        "job_name": job_name,
        "job_id": "demo",
    }

    configuration["parameters"] = {}

    configuration["parameters"]["diversity_filter"] = {
        "name": "IdenticalTopologicalScaffold",
        "nbmax": 10,
        "minscore": 0,
        "minsimilarity": 0.2,
    }

    configuration["parameters"]["inception"] = {
        "smiles": [],
        "memory_size": 100,
        "sample_size": 10,
    }

    configuration["parameters"]["reinforcement_learning"] = {
        "prior": str(prior_model_path),
        "agent": str(prior_model_path),
        "n_steps": n_steps,
        "n_mols": n_mols,
        "sigma": sigma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "reset": 0,
        "reset_score_cutoff": 0.5,
        "margin_threshold": 50,
    }

    configuration["parameters"]["scoring_function"] = {
        "name": "custom_sum",
        "parallel": False,
        "parameters": [
            {
                "component_type": "molecular_weight",
                "name": "Molecular weight",
                "weight": 1,
                "model_path": None,
                "smiles": [],
                "specific_parameters": {
                    "transformation_type": "double_sigmoid",
                    "high": high_mw,
                    "low": low_mw,
                    "coef_div": high_mw,
                    "coef_si": 20,
                    "coef_se": 20,
                    "transformation": True,
                },
            },
            {
                "component_type": "cdock_score",
                "name": "CDock Score",
                "weight": target_score_weight,
                "model_path": str(scoring_model_path),
                "smiles": [],
                "specific_parameters": {
                    "transformation_type": "sigmoid",
                    "high": max_inverted_score,
                    "low": 0,
                    "k": 0.2,
                    "scikit": "regression",
                    "transformation": True,
                },
            },
        ],
    }

    return configuration


def main() -> None:
    """
    Main workflow:
    1. Locate the REAL-SWIT repository directory.
    2. Find the trained target-specific scoring model checkpoint.
    3. Create the RL output directory.
    4. Write RL_config.json.
    """

    swit_dir = Path(__file__).resolve().parent

    task_dir = swit_dir / "examples" / task_name
    gen_model_dir = swit_dir / "checkpoints"
    output_dir = task_dir / "RL_practice" / rl_run_name

    scoring_model_path = find_single_checkpoint(task_dir)
    prior_model_path = gen_model_dir / gen_model_name

    if not prior_model_path.exists():
        raise FileNotFoundError(
            f"Cannot find pretrained generative model: {prior_model_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = build_rl_configuration(
        output_dir=output_dir,
        prior_model_path=prior_model_path,
        scoring_model_path=scoring_model_path,
    )

    configuration_json_path = output_dir / "RL_config.json"

    with open(configuration_json_path, "w") as f:
        json.dump(configuration, f, indent=4, sort_keys=True)

    print(f"Target task: {task_name}")
    print(f"Scoring model checkpoint: {scoring_model_path}")
    print(f"Prior generative model: {prior_model_path}")
    print(f"RL output directory: {output_dir}")
    print(f"Configuration file written to: {configuration_json_path}")


if __name__ == "__main__":
    main()