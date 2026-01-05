"""Data preprocessing module with stateless transforms."""

from typing import Dict, Any, Optional, List
from pathlib import Path

import ray.data as rd
import pandas as pd
import numpy as np
from loguru import logger

from src.config import settings
from src.data.sources import create_data_source
from src.data.schemas import validate_dataset_schema


class Preprocessor:
    """Data preprocessor with stateless transforms."""
    
    def __init__(
        self,
        scaling_method: str = "standard",
        clip_outliers: bool = True,
        clip_threshold: float = 3.0
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: Scaling method ("standard", "minmax", "robust").
            clip_outliers: Whether to clip outliers.
            clip_threshold: Z-score threshold for clipping.
        """
        self.scaling_method = scaling_method
        self.clip_outliers = clip_outliers
        self.clip_threshold = clip_threshold
        self._stats: Dict[str, Dict[str, float]] = {}
        
    def fit(self, dataset: rd.Dataset) -> "Preprocessor":
        """
        Fit preprocessor on dataset.
        
        Args:
            dataset: Dataset to fit on.
            
        Returns:
            Self for chaining.
        """
        logger.info("Fitting preprocessor on dataset")
        
        # Compute statistics for each numerical column
        numerical_cols = [
            col for col in dataset.columns()
            if col.startswith("feature_") and col != "target"
        ]
        
        for col in numerical_cols:
            # Compute statistics using Ray Data
            stats = dataset.map_batches(
                lambda df: pd.DataFrame({
                    "mean": [df[col].mean()],
                    "std": [df[col].std()],
                    "min": [df[col].min()],
                    "max": [df[col].max()],
                    "q25": [df[col].quantile(0.25)],
                    "q75": [df[col].quantile(0.75)],
                }),
                batch_format="pandas"
            ).to_pandas().iloc[0].to_dict()
            
            self._stats[col] = stats
            logger.debug(f"Computed stats for {col}: {stats}")
        
        logger.info(f"Fitted preprocessor on {len(numerical_cols)} columns")
        return self
    
    def transform(self, dataset: rd.Dataset) -> rd.Dataset:
        """
        Transform dataset using fitted preprocessor.
        
        Args:
            dataset: Dataset to transform.
            
        Returns:
            Transformed dataset.
            
        Raises:
            ValueError: If preprocessor is not fitted.
        """
        if not self._stats:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming dataset")
        
        # Extract local variables to avoid capturing self
        stats = self._stats
        clip_outliers = self.clip_outliers
        clip_threshold = self.clip_threshold
        scaling_method = self.scaling_method
        
        def apply_transforms(batch: pd.DataFrame) -> pd.DataFrame:
            """Apply preprocessing transforms to a batch."""
            batch = batch.copy()
            
            # Track which original columns to drop
            columns_to_drop = []
            
            for col, col_stats in stats.items():
                if col in batch.columns:
                    # Clip outliers
                    if clip_outliers:
                        z_scores = np.abs(
                            (batch[col] - col_stats["mean"]) / max(col_stats["std"], 1e-8)
                        )
                        batch.loc[z_scores > clip_threshold, col] = np.nan
                        batch[col] = batch[col].fillna(
                            np.sign(batch[col] - col_stats["mean"]) * 
                            clip_threshold * col_stats["std"] + col_stats["mean"]
                        )
                    
                    # Apply scaling - create new column only if it doesn't already have _norm suffix
                    output_col = col if col.endswith("_norm") else f"{col}_norm"
                    
                    # Only mark original column for dropping if we're creating a new normalized column
                    if output_col != col:
                        columns_to_drop.append(col)
                    
                    if scaling_method == "standard":
                        batch[output_col] = (
                            (batch[col] - col_stats["mean"]) / max(col_stats["std"], 1e-8)
                        )
                    elif scaling_method == "minmax":
                        batch[output_col] = (
                            (batch[col] - col_stats["min"]) / 
                            max(col_stats["max"] - col_stats["min"], 1e-8)
                        )
                    elif scaling_method == "robust":
                        batch[output_col] = (
                            (batch[col] - col_stats["q25"]) / 
                            max(col_stats["q75"] - col_stats["q25"], 1e-8)
                        )
            
            # Drop original columns (only if they're in the batch and not already normalized)
            columns_to_drop = [col for col in columns_to_drop if col in batch.columns]
            batch = batch.drop(columns=columns_to_drop)
            
            return batch
        
        transformed = dataset.map_batches(
            apply_transforms,
            batch_format="pandas"
        )
        
        logger.info("Dataset transformation completed")
        count = transformed.count()
        logger.info(f"Transformed dataset with {count} rows")
        return transformed
        
    def fit_transform(self, dataset: rd.Dataset) -> rd.Dataset:
        """
        Fit preprocessor and transform dataset.
        
        Args:
            dataset: Dataset to fit and transform.
            
        Returns:
            Transformed dataset.
        """
        return self.fit(dataset).transform(dataset)
    
    def save(self, path: Path) -> None:
        """
        Save preprocessor state to disk.
        
        Args:
            path: Path to save preprocessor state.
        """
        import pickle
        
        with open(path, "wb") as f:
            pickle.dump({
                "scaling_method": self.scaling_method,
                "clip_outliers": self.clip_outliers,
                "clip_threshold": self.clip_threshold,
                "stats": self._stats
            }, f)
        
        logger.info(f"Saved preprocessor to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        """
        Load preprocessor state from disk.
        
        Args:
            path: Path to load preprocessor state from.
            
        Returns:
            Loaded Preprocessor instance.
        """
        import pickle
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            scaling_method=state["scaling_method"],
            clip_outliers=state["clip_outliers"],
            clip_threshold=state["clip_threshold"]
        )
        preprocessor._stats = state["stats"]
        
        logger.info(f"Loaded preprocessor from {path}")
        return preprocessor


def preprocess_pipeline(
    input_path: Path,
    output_path: Path,
    source_type: str = "parquet",
    **kwargs
) -> Path:
    """
    Run preprocessing pipeline.
    
    Args:
        input_path: Path to input data.
        output_path: Path to save processed data.
        source_type: Type of data source.
        **kwargs: Additional arguments for preprocessor.
        
    Returns:
        Path to processed data.
    """
    logger.info("Starting preprocessing pipeline")
    
    # Create data source
    data_source = create_data_source(
        source_type=source_type,
        path=input_path
    )
    
    # Load data
    dataset = data_source.load_data()
    
    # Validate raw data schema
    validate_dataset_schema(dataset, is_processed=False)
    
    # Validate data
    if not data_source.validate(dataset):
        raise ValueError("Data validation failed")
    
    # Create and fit preprocessor
    preprocessor = Preprocessor(**kwargs)
    processed_dataset = preprocessor.fit_transform(dataset)

    # Validate processed schema with is_processed=True
    validate_dataset_schema(processed_dataset, is_processed=True)
    
    
    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_dataset.write_parquet(str(output_path))
    
    # Save preprocessor state
    preprocessor_path = output_path.parent / "preprocessor.pkl"
    preprocessor.save(preprocessor_path)
    
    logger.info(
        f"Preprocessing pipeline completed. "
        f"Processed data saved to {output_path}"
    )
    
    return output_path


if __name__ == "__main__":
    """Main entry point for preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--input", 
        type=Path, 
        default=settings.data_dir / "raw" / "synthetic_data.parquet",
        help="Input data path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.data_dir / "processed" / "preprocessed_data.parquet",
        help="Output data path"
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="parquet",
        choices=["parquet", "synthetic"],
        help="Data source type"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = preprocess_pipeline(
            input_path=args.input,
            output_path=args.output,
            source_type=args.source_type,
            scaling_method=settings.feature_scaling_method
        )
        print(f"Successfully processed data: {output_path}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise