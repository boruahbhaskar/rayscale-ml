#!/usr/bin/env python3
"""Generate synthetic data for the ML platform."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from src.config import settings
from src.utils.logging import configure_logging
from src.config.constants import FEATURE_COLUMNS


def generate_synthetic_data(
    num_rows: int = 100000,
    num_features: int = 4,
    seed: int = 42,
    output_path: Path = None
) -> Path:
    """
    Generate synthetic data for testing and development.
    
    Args:
        num_rows: Number of rows to generate.
        num_features: Number of features to generate.
        seed: Random seed for reproducibility.
        output_path: Output path for the data.
        
    Returns:
        Path to generated data.
    """
    logger.info(
        f"Generating synthetic data: {num_rows} rows, {num_features} features"
    )
    
    if output_path is None:
        output_path = settings.data_dir / "raw" / "synthetic_data.parquet"
    
    # Set random seed
    np.random.seed(seed)
    
    # Generate features
    features = np.random.randn(num_rows, num_features)
    
    # Generate target with some pattern
    # Simple linear combination with noise
    coefficients = np.random.randn(num_features)
    target = features @ coefficients + np.random.randn(num_rows) * 0.1
    
    # Add some nonlinear relationships
    target += 0.5 * features[:, 0] * features[:, 1]  # Interaction
    target += 0.3 * np.sin(features[:, 2] * 2)      # Nonlinear
    
    # Convert to integer for classification (optional)
    target_binary = (target > target.mean()).astype(np.int64)
    
    # Create DataFrame
    df = pd.DataFrame(
        features,
        columns=FEATURE_COLUMNS # [f"feature_{i+1}" for i in range(num_features)]
        #columns = [f"feature_{i+1}_norm" for i in range(num_features-2)] + ["feature_3", "feature_4"]
    )
    df["target"] = target_binary
    df["id"] = range(num_rows)
    
    # Add timestamp (simulating time series data)
    df["timestamp"] = pd.date_range(
        start="2024-01-01",
        periods=num_rows,
        freq="1min"
    )
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    
    logger.info(f"Generated synthetic data saved to: {output_path}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['target'].value_counts()}")
    
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument(
        "--rows",
        type=int,
        default=100000,
        help="Number of rows to generate"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=4,
        help="Number of features to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the data"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(log_level=args.log_level)
    
    try:
        output_path = generate_synthetic_data(
            num_rows=args.rows,
            num_features=args.features,
            seed=args.seed,
            output_path=args.output
        )
        print(f"✓ Synthetic data generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {e}")
        raise


if __name__ == "__main__":
    main()