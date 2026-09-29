from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genetic_disorder_ml.data import DEFAULT_DATA_PATH
from genetic_disorder_ml.train import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train genetic disorder dataset classifiers.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to the input CSV file.")
    parser.add_argument("--target", default="status", help="Target column to predict.")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Path for the saved best model.")
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_PATH, help="Path for metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        data_path=Path(args.data),
        target=args.target,
        model_path=Path(args.model_out),
        metrics_path=Path(args.metrics_out),
    )
    print(f"Best model: {metrics['best_model']}")
    print(f"Best accuracy: {metrics['best_accuracy']:.4f}")
    print(f"Saved model to: {args.model_out}")
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
