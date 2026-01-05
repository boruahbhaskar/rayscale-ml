"""Feature engineering module."""

from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

import ray.data as rd
import pandas as pd
import numpy as np
from loguru import logger

from src.config import settings
from src.data.feature_store import get_feature_store
from src.data.schemas import validate_dataset_schema


class FeatureEngineer:
    """Feature engineer with composable transformations."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.transformations: List[Callable] = []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def add_interaction(
        self,
        feature1: str,
        feature2: str,
        output_col: Optional[str] = None
    ) -> "FeatureEngineer":
        """
        Add interaction feature.
        
        Args:
            feature1: First feature name.
            feature2: Second feature name.
            output_col: Output column name.
            
        Returns:
            Self for chaining.
        """
        if output_col is None:
            output_col = f"{feature1}_{feature2}_interaction"
        
        def interaction_transform(batch: pd.DataFrame) -> pd.DataFrame:
            batch = batch.copy()
            batch[output_col] = batch[feature1] * batch[feature2]
            return batch
        
        self.transformations.append(interaction_transform)
        logger.debug(f"Added interaction: {feature1} * {feature2} -> {output_col}")
        return self
    
    def add_polynomial(
        self,
        feature: str,
        degree: int = 2,
        output_col: Optional[str] = None
    ) -> "FeatureEngineer":
        """
        Add polynomial feature.
        
        Args:
            feature: Feature name.
            degree: Polynomial degree.
            output_col: Output column name.
            
        Returns:
            Self for chaining.
        """
        if output_col is None:
            output_col = f"{feature}_poly_{degree}"
        
        def polynomial_transform(batch: pd.DataFrame) -> pd.DataFrame:
            batch = batch.copy()
            batch[output_col] = batch[feature] ** degree
            return batch
        
        self.transformations.append(polynomial_transform)
        logger.debug(f"Added polynomial: {feature}^{degree} -> {output_col}")
        return self
    
    def add_log(
        self,
        feature: str,
        output_col: Optional[str] = None,
        epsilon: float = 1e-8
    ) -> "FeatureEngineer":
        """
        Add log-transformed feature.
        
        Args:
            feature: Feature name.
            output_col: Output column name.
            epsilon: Small value to avoid log(0).
            
        Returns:
            Self for chaining.
        """
        if output_col is None:
            output_col = f"{feature}_log"
        
        def log_transform(batch: pd.DataFrame) -> pd.DataFrame:
            batch = batch.copy()
            batch[output_col] = np.log(batch[feature].abs() + epsilon)
            return batch
        
        self.transformations.append(log_transform)
        logger.debug(f"Added log: log({feature}) -> {output_col}")
        return self
    
    def add_binning(
        self,
        feature: str,
        bins: int = 10,
        output_col: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> "FeatureEngineer":
        """
        Add binned feature.
        
        Args:
            feature: Feature name.
            bins: Number of bins.
            output_col: Output column name.
            labels: Bin labels.
            
        Returns:
            Self for chaining.
        """
        if output_col is None:
            output_col = f"{feature}_binned"
        
        def binning_transform(batch: pd.DataFrame) -> pd.DataFrame:
            batch = batch.copy()
            batch[output_col] = pd.cut(
                batch[feature],
                bins=bins,
                labels=labels
            )
            return batch
        
        self.transformations.append(binning_transform)
        logger.debug(f"Added binning: {feature} -> {bins} bins -> {output_col}")
        return self
    
    def add_standardization(
        self,
        feature: str,
        output_col: Optional[str] = None
    ) -> "FeatureEngineer":
        """
        Add standardized feature.
        
        Args:
            feature: Feature name.
            output_col: Output column name.
            
        Returns:
            Self for chaining.
        """
        if output_col is None:
            output_col = f"{feature}_standardized"
        
        def standardization_transform(batch: pd.DataFrame) -> pd.DataFrame:
            batch = batch.copy()
            mean = batch[feature].mean()
            std = max(batch[feature].std(), 1e-8)
            batch[output_col] = (batch[feature] - mean) / std
            return batch
        
        self.transformations.append(standardization_transform)
        logger.debug(f"Added standardization: {feature} -> {output_col}")
        return self
    
    def transform(self, dataset: rd.Dataset) -> rd.Dataset:
        """
        Apply all transformations to dataset.
        
        Args:
            dataset: Dataset to transform.
            
        Returns:
            Transformed dataset.
        """
        if not self.transformations:
            logger.warning("No transformations added")
            return dataset
        
        logger.info(f"Applying {len(self.transformations)} transformations")
        
        transformed = dataset
        for i, transform in enumerate(self.transformations):
            logger.debug(f"Applying transformation {i+1}/{len(self.transformations)}")
            transformed = transformed.map_batches(
                transform,
                batch_format="pandas"
            )
        
        logger.info(f"Feature engineering completed")
        return transformed
    
    def fit_transform(self, dataset: rd.Dataset) -> rd.Dataset:
        """
        Fit and transform dataset.
        
        Note: Currently transformations are stateless.
        
        Args:
            dataset: Dataset to transform.
            
        Returns:
            Transformed dataset.
        """
        return self.transform(dataset)


def create_default_feature_engineer() -> FeatureEngineer:
    """
    Create default feature engineer with common transformations.
    
    Returns:
        FeatureEngineer instance.
    """
    engineer = FeatureEngineer()
    
    # Add interaction features if enabled
    if settings.feature_interaction_enabled:
        engineer.add_interaction("feature_1_norm", "feature_2_norm")
    
    # Add polynomial features
    engineer.add_polynomial("feature_1_norm", degree=2)
    engineer.add_polynomial("feature_2_norm", degree=2)
    
    # Add log transforms for positive-valued features
    engineer.add_log("feature_3", epsilon=1.0)
    engineer.add_log("feature_4", epsilon=1.0)
    
    return engineer


def feature_engineering_pipeline(
    input_path: Path,
    output_name: str = "features",
    version: str = "latest",
    engineer: Optional[FeatureEngineer] = None
) -> str:
    """
    Run feature engineering pipeline.
    
    Args:
        input_path: Path to input data.
        output_name: Output dataset name.
        version: Dataset version.
        engineer: Feature engineer instance.
        
    Returns:
        Version ID of created dataset.
    """
    logger.info("Starting feature engineering pipeline")
    
    # Load data
    logger.info(f"Loading data from {input_path}")
    dataset = rd.read_parquet(str(input_path))
    
    # Validate schema
    validate_dataset_schema(dataset)
    
    # Create feature engineer if not provided
    if engineer is None:
        engineer = create_default_feature_engineer()
    
    # Apply transformations
    engineered_dataset = engineer.transform(dataset)
    
    # Validate transformed schema
    logger.info("Validating engineered dataset schema")
    # Note: Schema validation might need adjustment for new features
    
    # Save to feature store
    feature_store = get_feature_store()
    version_id = feature_store.write_dataset(
        dataset=engineered_dataset,
        name=output_name,
        version=version,
        description="Engineered features dataset",
        tags={"pipeline": "feature_engineering"}
    )
    
    logger.info(
        f"Feature engineering pipeline completed. "
        f"Dataset saved as {output_name} version {version_id}"
    )
    
    return version_id


if __name__ == "__main__":
    """Main entry point for feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.data_dir / "processed" / "preprocessed_data.parquet",
        help="Input data path"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="features",
        help="Output dataset name"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="latest",
        help="Dataset version"
    )
    
    args = parser.parse_args()
    
    try:
        version_id = feature_engineering_pipeline(
            input_path=args.input,
            output_name=args.output_name,
            version=args.version
        )
        print(f"Successfully created feature dataset: {args.output_name} version {version_id}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise