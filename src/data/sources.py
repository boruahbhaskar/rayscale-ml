"""Data sources abstraction for the ML platform."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

import ray.data as rd
from loguru import logger

from src.config import settings
from src.data.schemas import validate_dataset_schema


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def load_data(self) -> rd.Dataset:
        """Load data from source."""
        pass
    
    @abstractmethod
    def validate(self, dataset: rd.Dataset) -> bool:
        """Validate loaded data."""
        pass


class ParquetDataSource(DataSource):
    """Parquet file data source."""
    
    def __init__(
        self,
        path: Path,
        schema_validation: bool = True,
        **kwargs
    ):
        """
        Initialize parquet data source.
        
        Args:
            path: Path to parquet file or directory.
            schema_validation: Whether to validate schema.
            **kwargs: Additional arguments for read_parquet.
        """
        self.path = Path(path)
        self.schema_validation = schema_validation
        self.kwargs = kwargs
        
    def load_data(self) -> rd.Dataset:
        """
        Load data from parquet files.
        
        Returns:
            Ray dataset containing the loaded data.
        """
        logger.info(f"Loading data from {self.path}")
        
        if not self.path.exists():
            raise FileNotFoundError(f"Data path not found: {self.path}")
        
        try:
            dataset = rd.read_parquet(
                str(self.path),
                **self.kwargs
            )
            logger.info(f"Loaded dataset with {dataset.count()} rows")
            
            if self.schema_validation:
                validate_dataset_schema(dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load data from {self.path}: {str(e)}")
            raise
    
    def validate(self, dataset: rd.Dataset) -> bool:
        """
        Validate loaded dataset.
        
        Args:
            dataset: Dataset to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            # Check dataset is not empty
            if dataset.count() == 0:
                logger.warning("Dataset is empty")
                return False
            
            # Check schema if validation is enabled
            if self.schema_validation:
                validate_dataset_schema(dataset)
            
            # Check for null values in features
            for column in dataset.columns():
                null_count = dataset.map_batches(
                    lambda df: df[column].isnull().sum(),
                    batch_format="pandas"
                ).sum()
                
                if null_count > 0:
                    logger.warning(
                        f"Column {column} has {null_count} null values"
                    )
            
            logger.info("Dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False


class SyntheticDataSource(DataSource):
    """Synthetic data source for testing and development."""
    
    def __init__(
        self,
        num_samples: int = 10000,
        num_features: int = 4,
        seed: int = 42
    ):
        """
        Initialize synthetic data source.
        
        Args:
            num_samples: Number of samples to generate.
            num_features: Number of features to generate.
            seed: Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.num_features = num_features
        self.seed = seed
        
    def load_data(self) -> rd.Dataset:
        """
        Generate synthetic data.
        
        Returns:
            Ray dataset containing synthetic data.
        """
        import numpy as np
        import pandas as pd
        
        logger.info(
            f"Generating synthetic data: {self.num_samples} samples, "
            f"{self.num_features} features"
        )
        
        np.random.seed(self.seed)
        
        # Generate features
        features = np.random.randn(self.num_samples, self.num_features)
        
        # Generate target with some pattern
        # Simple linear combination with noise
        coefficients = np.random.randn(self.num_features)
        target = features @ coefficients + np.random.randn(self.num_samples) * 0.1
        
        # Create DataFrame
        df = pd.DataFrame(
            features,
            columns=[f"feature_{i+1}" for i in range(self.num_features)]
        )
        df["target"] = target.astype(np.int64)
        df["id"] = range(self.num_samples)
        
        # Convert to Ray dataset
        dataset = rd.from_pandas(df)
        logger.info(f"Generated synthetic dataset with {dataset.count()} rows")
        
        return dataset
    
    def validate(self, dataset: rd.Dataset) -> bool:
        """
        Validate synthetic dataset.
        
        Args:
            dataset: Dataset to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        # Basic validation for synthetic data
        if dataset.count() != self.num_samples:
            logger.warning(
                f"Expected {self.num_samples} samples, "
                f"got {dataset.count()}"
            )
            return False
        
        if len(dataset.columns()) != self.num_features + 2:  # +2 for target and id
            logger.warning(
                f"Expected {self.num_features + 2} columns, "
                f"got {len(dataset.columns())}"
            )
            return False
        
        logger.info("Synthetic dataset validation passed")
        return True


def create_data_source(
    source_type: str,
    **kwargs
) -> DataSource:
    """
    Factory function to create data sources.
    
    Args:
        source_type: Type of data source ("parquet", "synthetic").
        **kwargs: Arguments for the data source.
        
    Returns:
        DataSource instance.
        
    Raises:
        ValueError: If source_type is invalid.
    """
    source_map = {
        "parquet": ParquetDataSource,
        "synthetic": SyntheticDataSource,
    }
    
    if source_type not in source_map:
        raise ValueError(
            f"Invalid source_type: {source_type}. "
            f"Must be one of: {list(source_map.keys())}"
        )
    
    return source_map[source_type](**kwargs)