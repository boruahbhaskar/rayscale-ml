"""Feature store with versioning and metadata management."""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

import ray.data as rd
import pyarrow as pa
import pandas as pd
from loguru import logger


class FeatureStoreError(Exception):
    """Feature store specific errors."""
    pass


class DatasetVersion(str, Enum):
    """Dataset versioning strategies."""
    LATEST = "latest"
    TIMESTAMP = "timestamp"
    UUID = "uuid"
    SEMVER = "semver"


@dataclass
class DatasetMetadata:
    """Metadata for a dataset version."""
    
    version: str
    schema: Dict[str, Any]
    feature_stats: Dict[str, Dict[str, float]]
    created_at: str
    num_rows: int
    partitions: List[str]
    description: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "DatasetMetadata":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class FeatureStore:
    """Production-ready feature store with versioning."""
    
    def __init__(self, base_path: Path):
        """
        Initialize feature store.
        
        Args:
            base_path: Base path for feature store.
        """
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        self._metadata_cache: Dict[str, DatasetMetadata] = {}
        self._load_metadata_cache()
    
    def _load_metadata_cache(self) -> None:
        """Load metadata cache from disk."""
        for metadata_file in self.metadata_path.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = DatasetMetadata.from_json(f.read())
                self._metadata_cache[metadata.version] = metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata {metadata_file}: {e}")
    
    def _save_metadata(self, name: str, version: str, metadata: DatasetMetadata) -> None:
        """
        Save metadata to disk.
        
        Args:
            name: Dataset name.
            version: Dataset version.
            metadata: Metadata to save.
        """
        metadata_file = self.metadata_path / f"{name}_{version}.json"
        with open(metadata_file, "w") as f:
            f.write(metadata.to_json())
        
        self._metadata_cache[version] = metadata
        logger.debug(f"Saved metadata for {name} version {version}")
    
    def _compute_stats(self, dataset: rd.Dataset) -> Dict[str, Dict[str, float]]:
        """
        Compute feature statistics for serving.
        
        Args:
            dataset: Dataset to compute stats for.
            
        Returns:
            Dictionary of feature statistics.
        """
        stats = {}
        
        for column in dataset.columns():
            if column.endswith("_norm") or column.startswith("feature_"):
                # Compute statistics for numerical columns
                column_stats = dataset.map_batches(
                    lambda df: pd.DataFrame({
                        "mean": [df[column].mean()],
                        "std": [df[column].std()],
                        "min": [df[column].min()],
                        "max": [df[column].max()],
                        "count": [df[column].count()],
                        "null_count": [df[column].isnull().sum()],
                    }),
                    batch_format="pandas"
                ).to_pandas().iloc[0].to_dict()
                
                stats[column] = column_stats
        
        return stats
    
    def _get_partitions(self, path: Path) -> List[str]:
        """
        Get partition columns from path.
        
        Args:
            path: Path to check for partitions.
            
        Returns:
            List of partition columns.
        """
        partitions = []
        for part in path.parts:
            if "=" in part:
                partition_col = part.split("=")[0]
                # Version directories (v=...) are metadata, not data partitions
                # Also exclude other single-letter directories that might be metadata
                if partition_col not in ["v", "version"]:
                    partitions.append(partition_col)
        return partitions
    
    # def write_dataset(
    #     self,
    #     dataset: rd.Dataset,
    #     name: str,
    #     version: str = "latest",
    #     description: Optional[str] = None,
    #     tags: Optional[Dict[str, str]] = None,
    #     partition_cols: Optional[List[str]] = None,
    #     compute_stats: bool = True
    # ) -> str:
    #     """
    #     Write dataset with versioning and metadata.
        
    #     Args:
    #         dataset: Dataset to write.
    #         name: Dataset name.
    #         version: Dataset version.
    #         description: Dataset description.
    #         tags: Dataset tags.
    #         partition_cols: Columns to partition by.
    #         compute_stats: Whether to compute feature statistics.
            
    #     Returns:
    #         Version ID of written dataset.
            
    #     Raises:
    #         FeatureStoreError: If writing fails.
    #     """
    #     try:
    #         logger.info(f"Writing dataset {name} version {version}")
            
    #         # Generate version ID
    #         if version == DatasetVersion.LATEST:
    #             version_id = str(uuid.uuid4())[:8]
    #         elif version == DatasetVersion.TIMESTAMP:
    #             version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         elif version == DatasetVersion.UUID:
    #             version_id = str(uuid.uuid4())
    #         else:
    #             version_id = version
            
    #         # Create output path
    #         output_path = self.base_path / name / f"v={version_id}"
    #         output_path.mkdir(parents=True, exist_ok=True)
            
    #         # Write dataset
    #         if partition_cols:
    #             # Check partition columns exist
    #             missing_cols = [
    #                 col for col in partition_cols 
    #                 if col not in dataset.columns()
    #             ]
    #             if missing_cols:
    #                 raise FeatureStoreError(
    #                     f"Partition columns not found: {missing_cols}"
    #                 )
                
    #             dataset.write_parquet(
    #                 str(output_path),
    #                 partition_cols=partition_cols
    #             )
    #         else:
    #             dataset.write_parquet(str(output_path))
            
    #         # Compute metadata
    #         schema_dict = {
    #             field.name: str(field.type)
    #             for field in dataset.schema()
    #         }
            
    #         partitions = self._get_partitions(output_path)
            
    #         metadata = DatasetMetadata(
    #             version=version_id,
    #             schema=schema_dict,
    #             feature_stats=self._compute_stats(dataset) if compute_stats else {},
    #             created_at=datetime.now().isoformat(),
    #             num_rows=dataset.count(),
    #             partitions=partitions,
    #             description=description,
    #             tags=tags or {}
    #         )
            
    #         # Save metadata
    #         self._save_metadata(name, version_id, metadata)
            
    #         logger.info(
    #             f"Successfully wrote dataset {name} version {version_id} "
    #             f"with {metadata.num_rows} rows"
    #         )
            
    #         return version_id
            
    #     except Exception as e:
    #         logger.error(f"Failed to write dataset {name}: {str(e)}")
    #         raise FeatureStoreError(f"Failed to write dataset: {str(e)}")
        
    def write_dataset(
        self,
        dataset: rd.Dataset,
        name: str,
        version: str = "latest",
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        partition_cols: Optional[List[str]] = None,
        compute_stats: bool = True
    ) -> str:
        """
        Write dataset with versioning and metadata.
        
        Args:
            dataset: Dataset to write.
            name: Dataset name.
            version: Dataset version.
            description: Dataset description.
            tags: Dataset tags.
            partition_cols: Columns to partition by.
            compute_stats: Whether to compute feature statistics.
            
        Returns:
            Version ID of written dataset.
            
        Raises:
            FeatureStoreError: If writing fails.
        """
        try:
            logger.info(f"Writing dataset {name} version {version}")
            
            # Generate version ID
            if version == DatasetVersion.LATEST:
                version_id = str(uuid.uuid4())[:8]
            elif version == DatasetVersion.TIMESTAMP:
                version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            elif version == DatasetVersion.UUID:
                version_id = str(uuid.uuid4())
            else:
                version_id = version
            
            # Create output path
            output_path = self.base_path / name / f"v={version_id}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Write dataset
            if partition_cols:
                # Check partition columns exist
                missing_cols = [
                    col for col in partition_cols 
                    if col not in dataset.columns()
                ]
                if missing_cols:
                    raise FeatureStoreError(
                        f"Partition columns not found: {missing_cols}"
                    )
                
                dataset.write_parquet(
                    str(output_path),
                    partition_cols=partition_cols
                )
            else:
                dataset.write_parquet(str(output_path))
            
            # FIX: Properly extract schema from Ray Dataset
            schema = dataset.schema()
            
            # Handle different schema representations in Ray
            if hasattr(schema, 'names') and hasattr(schema, 'types'):
                # Ray 2.6+ format: schema has names and types attributes
                schema_dict = dict(zip(schema.names, [str(t) for t in schema.types]))
            elif hasattr(schema, 'to_arrow_schema'):
                # Convert to PyArrow schema
                arrow_schema = schema.to_arrow_schema()
                schema_dict = {field.name: str(field.type) for field in arrow_schema}
            elif hasattr(schema, 'arrow_schema'):
                # Access PyArrow schema directly
                arrow_schema = schema.arrow_schema
                schema_dict = {field.name: str(field.type) for field in arrow_schema}
            else:
                # Fallback: use column names with unknown types
                schema_dict = {col: "unknown" for col in dataset.columns()}
            
            partitions = self._get_partitions(output_path)
            
            metadata = DatasetMetadata(
                version=version_id,
                schema=schema_dict,
                feature_stats=self._compute_stats(dataset) if compute_stats else {},
                created_at=datetime.now().isoformat(),
                num_rows=dataset.count(),
                partitions=partitions,
                description=description,
                tags=tags or {}
            )
            
            # Save metadata
            self._save_metadata(name, version_id, metadata)
            
            logger.info(
                f"Successfully wrote dataset {name} version {version_id} "
                f"with {metadata.num_rows} rows"
            )
            
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to write dataset {name}: {str(e)}")
            raise FeatureStoreError(f"Failed to write dataset: {str(e)}")    
    
    def read_dataset(
        self,
        name: str,
        version: str = "latest"
    ) -> Tuple[rd.Dataset, DatasetMetadata]:
        """
        Read dataset with metadata.
        
        Args:
            name: Dataset name.
            version: Dataset version.
            
        Returns:
            Tuple of (dataset, metadata).
            
        Raises:
            FeatureStoreError: If reading fails.
        """
        try:
            # Resolve version
            if version == DatasetVersion.LATEST:
                # Find latest version
                dataset_dir = self.base_path / name
                if not dataset_dir.exists():
                    raise FeatureStoreError(f"Dataset {name} not found")
                
                versions = [
                    d.name.split("=")[1]
                    for d in dataset_dir.iterdir()
                    if d.name.startswith("v=")
                ]
                
                if not versions:
                    raise FeatureStoreError(f"No versions found for dataset {name}")
                
                version_id = sorted(versions)[-1]
            else:
                version_id = version
            
            # Load dataset
            dataset_path = self.base_path / name / f"v={version_id}"
            if not dataset_path.exists():
                raise FeatureStoreError(
                    f"Dataset {name} version {version_id} not found"
                )
            
            dataset = rd.read_parquet(str(dataset_path))
            
            # Load metadata
            metadata = self.get_metadata(name, version_id)
            
            logger.info(
                f"Successfully read dataset {name} version {version_id} "
                f"with {metadata.num_rows} rows"
            )
            
            return dataset, metadata
            
        except Exception as e:
            logger.error(f"Failed to read dataset {name}: {str(e)}")
            raise FeatureStoreError(f"Failed to read dataset: {str(e)}")
    
    def get_metadata(self, name: str, version: str) -> DatasetMetadata:
        """
        Get metadata for a dataset version.
        
        Args:
            name: Dataset name.
            version: Dataset version.
            
        Returns:
            Dataset metadata.
            
        Raises:
            FeatureStoreError: If metadata not found.
        """
        if version in self._metadata_cache:
            return self._metadata_cache[version]
        
        metadata_file = self.metadata_path / f"{name}_{version}.json"
        if not metadata_file.exists():
            raise FeatureStoreError(
                f"Metadata not found for {name} version {version}"
            )
        
        with open(metadata_file, "r") as f:
            metadata = DatasetMetadata.from_json(f.read())
        
        self._metadata_cache[version] = metadata
        return metadata
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all datasets in feature store.
        
        Returns:
            List of dataset information.
        """
        datasets = []
        
        for dataset_dir in self.base_path.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith("."):
                versions = [
                    d.name.split("=")[1]
                    for d in dataset_dir.iterdir()
                    if d.name.startswith("v=")
                ]
                
                if versions:
                    latest_version = sorted(versions)[-1]
                    try:
                        metadata = self.get_metadata(
                            dataset_dir.name,
                            latest_version
                        )
                        datasets.append({
                            "name": dataset_dir.name,
                            "latest_version": latest_version,
                            "num_rows": metadata.num_rows,
                            "created_at": metadata.created_at,
                            "num_versions": len(versions)
                        })
                    except FeatureStoreError:
                        # Skip datasets with missing metadata
                        continue
        
        return datasets
    
    def delete_dataset(self, name: str, version: Optional[str] = None) -> None:
        """
        Delete dataset or version.
        
        Args:
            name: Dataset name.
            version: Version to delete (None for all versions).
            
        Raises:
            FeatureStoreError: If deletion fails.
        """
        try:
            if version is None:
                # Delete all versions
                dataset_dir = self.base_path / name
                if dataset_dir.exists():
                    import shutil
                    shutil.rmtree(dataset_dir)
                
                # Delete metadata files
                for metadata_file in self.metadata_path.glob(f"{name}_*.json"):
                    metadata_file.unlink()
                
                # Clear cache
                self._metadata_cache = {
                    k: v for k, v in self._metadata_cache.items()
                    if not k.startswith(f"{name}_")
                }
                
                logger.info(f"Deleted all versions of dataset {name}")
            else:
                # Delete specific version
                dataset_path = self.base_path / name / f"v={version}"
                if dataset_path.exists():
                    import shutil
                    shutil.rmtree(dataset_path)
                
                # Delete metadata
                metadata_file = self.metadata_path / f"{name}_{version}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                # Clear cache
                self._metadata_cache.pop(version, None)
                
                logger.info(f"Deleted dataset {name} version {version}")
                
        except Exception as e:
            logger.error(f"Failed to delete dataset {name}: {str(e)}")
            raise FeatureStoreError(f"Failed to delete dataset: {str(e)}")


# Global feature store instance
def get_feature_store(base_path: Optional[Path] = None) -> FeatureStore:
    """
    Get or create global feature store instance.
    
    Args:
        base_path: Base path for feature store.
        
    Returns:
        FeatureStore instance.
    """
    from src.config import settings
    
    if base_path is None:
        base_path = settings.data_dir / "feature_store"
    
    return FeatureStore(base_path)