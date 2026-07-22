#!/usr/bin/env python3
"""Select the saved checkpoint with the lowest validation NLL."""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read validation NLL values from TensorBoard logs and copy the "
            "best saved checkpoint to a stable output path."
        )
    )
    parser.add_argument(
        "--tensorboard-dir",
        required=True,
        help="TensorBoard root directory produced by train_model.py.",
    )
    parser.add_argument(
        "--model-prefix",
        required=True,
        help="Checkpoint prefix, for example gm_training_demo/models/model.trained.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to which the selected checkpoint will be copied.",
    )
    parser.add_argument(
        "--validation-subdir",
        default="nll_avg_validation",
        help="Validation-log subdirectory under --tensorboard-dir.",
    )
    parser.add_argument(
        "--scalar-tag",
        default="nll/avg",
        help="TensorBoard scalar tag used for validation NLL.",
    )
    parser.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help=(
            "Value added to each TensorBoard step when resolving checkpoint "
            "names. Keep 0 for a normal uninterrupted training run."
        ),
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional CSV path. Defaults to validation_nll.csv beside --output.",
    )
    return parser.parse_args()


def read_validation_nll(log_dir, scalar_tag):
    values_by_step = defaultdict(list)
    event_files = sorted(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {log_dir}")

    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        if scalar_tag not in accumulator.Tags().get("scalars", []):
            continue
        for item in accumulator.Scalars(scalar_tag):
            values_by_step[item.step].append(item.value)

    if not values_by_step:
        raise ValueError(
            f"Scalar tag '{scalar_tag}' was not found in event files under: {log_dir}"
        )

    return [
        (step, sum(values) / len(values))
        for step, values in sorted(values_by_step.items())
    ]


def main():
    args = parse_args()
    tensorboard_dir = Path(args.tensorboard_dir)
    validation_dir = tensorboard_dir / args.validation_subdir
    output_path = Path(args.output)
    csv_path = (
        Path(args.csv_output)
        if args.csv_output
        else output_path.parent / "validation_nll.csv"
    )

    records = read_validation_nll(validation_dir, args.scalar_tag)
    resolved = []
    for step, nll in records:
        epoch = step + args.epoch_offset
        checkpoint = Path(f"{args.model_prefix}.{epoch}")
        resolved.append((step, epoch, nll, checkpoint, checkpoint.is_file()))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tensorboard_step", "checkpoint_epoch", "validation_nll", "checkpoint_exists"])
        for step, epoch, nll, _, exists in resolved:
            writer.writerow([step, epoch, f"{nll:.10g}", str(exists).lower()])

    available = [record for record in resolved if record[4]]
    if not available:
        raise FileNotFoundError(
            "No validation step matched a saved checkpoint. Check --model-prefix "
            "and, for resumed training, set the appropriate --epoch-offset. "
            f"Validation summary: {csv_path}"
        )

    step, epoch, best_nll, best_checkpoint, _ = min(
        available, key=lambda record: record[2]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_checkpoint.resolve() != output_path.resolve():
        shutil.copy2(best_checkpoint, output_path)

    print(f"Best TensorBoard step: {step}")
    print(f"Best checkpoint epoch: {epoch}")
    print(f"Validation NLL: {best_nll:.10g}")
    print(f"Source checkpoint: {best_checkpoint}")
    print(f"Selected checkpoint: {output_path}")
    print(f"Validation summary: {csv_path}")


if __name__ == "__main__":
    main()