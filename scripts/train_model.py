#!/usr/bin/env python3
"""Train a model using the ML platform."""

# At the very top of train_model.py
import _path_setup  # This will setup paths

# Now import other standard libraries
import argparse

from typing import Any
import sys
import os
from pathlib import Path  # ADD THIS LINE

import yaml

from src.training.orchestrator import TrainingOrchestrator
from src.utils.logging import configure_logging
import logging


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        Configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def train_model(config_path: Path, experiment_name: str = None) -> dict[str, Any]:
    """
    Train a model using the specified configuration.

    Args:
        config_path: Path to training configuration.
        experiment_name: Name of the experiment.

    Returns:
        Training results.
    """
    logger = logging.getLogger(__name__)  # create logger

    logger.info(f"Training model with config: {config_path}")

    # Create orchestrator
    orchestrator = TrainingOrchestrator(experiment_name=experiment_name)

    # Run training pipeline
    result = orchestrator.run_pipeline(pipeline_type="train", config_path=config_path)

    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to training configuration YAML"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: from config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(log_level=args.log_level)

    try:
        # Load config to get experiment name if not provided
        if args.experiment is None:
            config = load_config(args.config)
            args.experiment = config.get("experiment_name", "default_experiment")

        # Train model
        result = train_model(args.config, args.experiment)

        if result["success"]:
            print("✓ Model training completed successfully")
            print(f"  Run ID: {result['run_id']}")
            print(f"  Metrics: {result['result'].metrics}")
        else:
            print("✗ Model training failed")
            if "error" in result:
                print(f"  Error: {result['error']}")

    except Exception as e:
        print(f"✗ Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()
