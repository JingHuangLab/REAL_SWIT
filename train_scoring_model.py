from mpn_models.dmpnn import MPNN
from mpn_models.chemprop import *
from mpn_models import utils
from pathlib import Path
import argparse
import sys
import logging
from joblib import dump
import numpy as np
import math
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def latest_subdir(path: Path):
    """Return latest (by mtime) subdirectory name under path, or None if none."""
    if not path.exists() or not path.is_dir():
        return None
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    return latest.name


def main(argv=None):
    configure_logging()
    parser = argparse.ArgumentParser(description='Train a target-specific scoring model.')
    parser.add_argument('training_dataset_path', type=str,
                        help='a path to a file in csv format. The first column should be SMILES and second column should be score.')
    parser.add_argument('task_name', type=str, default="task1",
                        help='the name of the output folder located under examples/')
    parser.add_argument('--testing_dataset_path', type=str, default=None,
                        help='a path to a file in csv format for testing. Default: None')
    parser.add_argument('--ncpu', type=int, default=1,
                        help='number of cores to make available. default: 1')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of iterations for model training. default: 50')
    args = parser.parse_args(argv)

    current_time = datetime.now()
    suffix = f"{current_time.year}_{current_time.month}_{current_time.day}"

    examples_dir = Path("./examples") / args.task_name
    preds_dir = examples_dir / "preds"
    try:
        preds_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create preds directory %s: %s", preds_dir, e)

    # Read training data
    scores_csv = args.training_dataset_path
    scores, failures = utils._read_scores(scores_csv)
    xs, ys = zip(*scores.items())
    logger.info("Training input size: %d", len(xs))

    # Train model
    my_model = MPNN(ncpu=args.ncpu, epochs=args.epochs)
    my_model.train(xs, ys, args.task_name)
    logger.info("Trained model saved in ./examples/%s ...", args.task_name)

    # Save scaler if lightning logs exist
    logs_dir = examples_dir / "lightning_logs"
    folder_name = latest_subdir(logs_dir)
    if folder_name:
        scaler_path = logs_dir / folder_name / 'std_scaler.bin'
        try:
            dump(my_model.scaler, scaler_path, compress=True)
            logger.info("The scaler is stored in %s", scaler_path)
        except Exception as e:
            logger.warning("Failed to dump scaler to %s: %s", scaler_path, e)
    else:
        logger.warning("No lightning_logs subdirectory found under %s; skipping scaler save.", logs_dir)

    # Prediction and evaluation
    if args.testing_dataset_path:
        scores_csv = args.testing_dataset_path
        scores, failures = utils._read_scores(scores_csv)
        xs, ys = zip(*scores.items())
        logger.info("Testing input size: %d", len(xs))

        preds_chunks = my_model.predict(xs)
        pred_scores = []
        for chunk in preds_chunks:
            for pred in chunk:
                pred_scores.append(pred[0])
        logger.info("Prediction output size: %d", len(pred_scores))

        if len(pred_scores) != len(ys):
            logger.warning("The length of prediction and target are not the same (%d vs %d).", len(pred_scores), len(ys))
            return

        # Filter items: original code skipped targets >= 0
        se_lst = []
        new_pred = []
        new_ys = []
        pred_file = preds_dir / f"predictions_{suffix}.csv"
        with pred_file.open('w') as writer:
            writer.write("prediction,target\n")
            for idx, target_score in enumerate(ys):
                if target_score >= 0:
                    continue
                s_err = (target_score - pred_scores[idx]) ** 2
                se_lst.append(s_err)
                new_pred.append(pred_scores[idx])
                new_ys.append(target_score)
                writer.write(f"{round(pred_scores[idx],1)},{target_score}\n")

        if not se_lst:
            logger.warning("No valid (negative) targets found after filtering; skipping evaluation and plotting.")
            return

        mse = np.average(se_lst)
        rmse = math.sqrt(mse)
        PCC, p = stats.pearsonr(new_pred, new_ys)
        logger.info("length of score list: %d", len(se_lst))
        logger.info("RMSE: %.2f", rmse)
        logger.info("PCC : %.2f", PCC)

        # Plot: use the filtered new_pred/new_ys (these correspond to evaluated points)
        fig = plt.figure(figsize=(15, 15))
        x_vals = np.array(new_pred)
        y_vals = np.array(new_ys)
        all_vals = np.concatenate([x_vals, y_vals])
        min_val = float(all_vals.min())
        max_val = float(all_vals.max())
        span = max_val - min_val
        pad = 1.0 if span == 0 else span * 0.05
        x_diag = np.linspace(min_val - pad, max_val + pad, 100)
        plt.scatter(x_vals, y_vals, color='orange', s=6)
        plt.xlim(min_val - pad, max_val + pad)
        plt.ylim(min_val - pad, max_val + pad)
        plt.tick_params(labelsize=20)
        plt.plot(x_diag, x_diag, "b--")
        plt.xlabel("prediction", size=30)
        plt.ylabel("ground truth", size=30)
        plot_path = preds_dir / f"pred_target_scatter_{suffix}.png"
        try:
            plt.savefig(plot_path)
            logger.info("Saved scatter plot to %s", plot_path)
        except Exception as e:
            logger.warning("Failed to save plot to %s: %s", plot_path, e)


if __name__ == "__main__":
    main()
