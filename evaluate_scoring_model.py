#!/usr/bin/env python
"""Evaluate or apply a target-specific scoring model.

The input file can contain either:
1. SMILES only: predictions are written without calculating metrics or plotting.
2. SMILES and target scores: predictions, RMSE, PCC, and a scatter plot are generated.

Expected input formats:
- CSV with one column: SMILES
- CSV with two columns: SMILES, score
- Header rows are allowed.
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ray
import scipy.stats as stats
import torch
from joblib import load
from tqdm import tqdm

from mpn_models import mpnn, utils
from mpn_models.dmpnn import MPNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate or apply a target-specific scoring model."
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to a trained scoring model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Path to an input CSV/SMI file. The first column should contain SMILES. "
            "If the second column contains target scores, RMSE/PCC and a scatter plot are generated."
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="task1",
        help="Name of the task-specific output folder under examples/. Default: task1.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for prediction outputs. Default: examples/<task_name>/preds.",
    )
    parser.add_argument(
        "--scaler_path",
        type=Path,
        default=None,
        help="Path to std_scaler.bin. By default, it is inferred from the checkpoint directory.",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="Number of CPU cores available to each Ray prediction worker. Default: 1.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size used inside the MPNN prediction function. Default: 50.",
    )
    parser.add_argument(
        "--smiles_batch_size",
        type=int,
        default=10000,
        help="Number of SMILES assigned to each Ray prediction task. Default: 10000.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable scatter plot generation even when target scores are available.",
    )
    return parser.parse_args()


def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def read_smiles_and_optional_targets(input_path: Path) -> Tuple[list[str], Optional[list[float]]]:
    """Read SMILES and optional target scores from a CSV/SMI-like file.

    A target column is considered available only when every non-header row contains
    a numeric second column. Otherwise, the file is treated as SMILES-only.
    """
    smiles: list[str] = []
    targets: list[Optional[float]] = []

    with input_path.open("r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t ;")
        except csv.Error:
            dialect = csv.excel

        reader = csv.reader(f, dialect)
        for row in reader:
            row = [item.strip() for item in row if item.strip()]
            if not row:
                continue

            first = row[0].lower()
            if first in {"smiles", "smile", "smi", "canonical_smiles"}:
                continue

            smiles.append(row[0])
            if len(row) >= 2 and is_float(row[1]):
                targets.append(float(row[1]))
            else:
                targets.append(None)

    if not smiles:
        raise ValueError(f"No valid SMILES were found in {input_path}.")

    has_complete_targets = all(target is not None for target in targets)
    if has_complete_targets:
        return smiles, [float(target) for target in targets if target is not None]
    return smiles, None


def infer_scaler_path(model_path: Path) -> Path:
    """Infer std_scaler.bin from a Lightning checkpoint path."""
    model_path = model_path.resolve()
    if model_path.parent.name == "checkpoints":
        return model_path.parent.parent / "std_scaler.bin"
    return model_path.parent / "std_scaler.bin"


def load_scoring_model(model_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    mpnn_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("mpnn."):
            mpnn_state_dict[key.replace("mpnn.", "", 1)] = value
        else:
            mpnn_state_dict[key] = value

    scoring_model = MPNN()
    scoring_model.model.load_state_dict(mpnn_state_dict)
    scoring_model.model.eval()
    return scoring_model.model


def predict_scores(
    model: torch.nn.Module,
    smiles: Sequence[str],
    scaler,
    ncpu: int,
    batch_size: int,
    smiles_batch_size: int,
) -> list[float]:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    use_gpu = ray.cluster_resources().get("GPU", 0) > 0
    num_gpus = 1 if use_gpu else 0
    remote_predict = ray.remote(num_cpus=ncpu, num_gpus=num_gpus)(mpnn.predict)

    model_ref = ray.put(model)
    scaler_ref = ray.put(scaler)
    smiles_batches = utils.batches(smiles, smiles_batch_size)

    refs = [
        remote_predict.remote(
            model_ref,
            batch_smiles,
            batch_size,
            ncpu,
            True,
            scaler_ref,
            use_gpu,
            True,
        )
        for batch_smiles in smiles_batches
    ]

    pred_chunks = [ray.get(ref) for ref in tqdm(refs, desc="Prediction", leave=False)]
    return [float(score[0]) for chunk in pred_chunks for score in chunk[0]]


def write_predictions(
    output_path: Path,
    smiles: Sequence[str],
    predictions: Sequence[float],
    targets: Optional[Sequence[float]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if targets is None:
            writer.writerow(["SMILES", "prediction"])
            writer.writerows((smi, round(pred, 4)) for smi, pred in zip(smiles, predictions))
        else:
            writer.writerow(["SMILES", "prediction", "target"])
            writer.writerows(
                (smi, round(pred, 4), target)
                for smi, pred, target in zip(smiles, predictions, targets)
            )


def calculate_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
) -> tuple[float, float, int, list[float], list[float]]:
    filtered_predictions = []
    filtered_targets = []

    for pred, target in zip(predictions, targets):
        # Keep the original behavior: non-negative target values are ignored.
        if target >= 0:
            continue
        filtered_predictions.append(float(pred))
        filtered_targets.append(float(target))

    if len(filtered_targets) < 2:
        raise ValueError("At least two valid target scores are required to calculate PCC/RMSE.")

    errors = [(target - pred) ** 2 for pred, target in zip(filtered_predictions, filtered_targets)]
    rmse = math.sqrt(float(np.mean(errors)))
    pcc, _ = stats.pearsonr(filtered_predictions, filtered_targets)
    return rmse, float(pcc), len(filtered_targets), filtered_predictions, filtered_targets


def plot_predictions(
    output_path: Path,
    predictions: Sequence[float],
    targets: Sequence[float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lower = min(min(predictions), min(targets), -50)
    upper = max(max(predictions), max(targets), 0)
    diagonal = np.linspace(lower, upper, 100)

    plt.figure(figsize=(8, 8))
    plt.scatter(predictions, targets, s=6)
    plt.plot(diagonal, diagonal, "--")
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.xlabel("Predicted score", fontsize=14)
    plt.ylabel("Target score", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    suffix = datetime.now().strftime("%Y_%m_%d")

    output_dir = args.output_dir or Path("examples") / args.task_name / "preds"
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles, targets = read_smiles_and_optional_targets(args.input_path)
    print(f"Input file: {args.input_path}")
    print(f"Input data size: {len(smiles)}")
    print(f"Target scores detected: {targets is not None}")

    scaler_path = args.scaler_path or infer_scaler_path(args.model_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file was not found: {scaler_path}")

    model = load_scoring_model(args.model_path)
    scaler = load(scaler_path)
    predictions = predict_scores(
        model=model,
        smiles=smiles,
        scaler=scaler,
        ncpu=args.ncpu,
        batch_size=args.batch_size,
        smiles_batch_size=args.smiles_batch_size,
    )

    if len(predictions) != len(smiles):
        raise RuntimeError(
            f"The number of predictions ({len(predictions)}) does not match "
            f"the number of input SMILES ({len(smiles)})."
        )

    output_csv = output_dir / f"predictions_{suffix}.csv"
    write_predictions(output_csv, smiles, predictions, targets)
    print(f"Output data size: {len(predictions)}")
    print(f"Prediction file: {output_csv}")
    print(f"Scaler file: {scaler_path}")

    if targets is None:
        print("SMILES-only input detected. Metrics and scatter plot were skipped.")
        return

    rmse, pcc, n_valid, filtered_predictions, filtered_targets = calculate_metrics(predictions, targets)
    print(f"Number of valid target scores: {n_valid}")
    print(f"RMSE: {rmse:.2f}")
    print(f"PCC : {pcc:.2f}")

    if not args.no_plot:
        plot_path = output_dir / f"pred_target_scatter_{suffix}.png"
        plot_predictions(plot_path, filtered_predictions, filtered_targets)
        print(f"Scatter plot: {plot_path}")


if __name__ == "__main__":
    main()