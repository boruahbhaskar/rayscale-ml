"""Tests for feature store functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

import ray
import ray.data as rd
import pandas as pd
import pyarrow as pa

from src.data.feature_store import (
    FeatureStore,
    FeatureStoreError,
    DatasetMetadata,
    DatasetVersion,
)
from src.data.schemas import get_feature_schema


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp(prefix="feature_store_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_dataset():
    """Create a sample Ray dataset for testing."""
    data = {
        "id": list(range(100)),
        "feature_1_norm": [float(i) for i in range(100)],
        "feature_2_norm": [float(i * 2) for i in range(100)],
        "feature_3": [float(i * 3) for i in range(100)],
        "feature_4": [float(i * 4) for i in range(100)],
        "target": [i % 2 for i in range(100)],
    }
    df = pd.DataFrame(data)
    return rd.from_pandas(df)


@pytest.fixture
def feature_store(temp_dir):
    """Create a FeatureStore instance for testing."""
    return FeatureStore(temp_dir)


def test_feature_store_initialization(temp_dir):
    """Test FeatureStore initialization."""
    store = FeatureStore(temp_dir)
    
    assert store.base_path == temp_dir
    assert store.metadata_path == temp_dir / "metadata"
    assert store.metadata_path.exists()
    assert isinstance(store._metadata_cache, dict)


def test_dataset_metadata_creation():
    """Test DatasetMetadata creation."""
    schema = {"col1": "float64", "col2": "int64"}
    feature_stats = {"col1": {"mean": 0.5, "std": 0.1}}
    
    metadata = DatasetMetadata(
        version="v1",
        schema=schema,
        feature_stats=feature_stats,
        created_at="2024-01-01T00:00:00",
        num_rows=100,
        partitions=["split=train"],
        description="Test dataset",
        tags={"source": "synthetic"}
    )
    
    assert metadata.version == "v1"
    assert metadata.schema == schema
    assert metadata.feature_stats == feature_stats
    assert metadata.created_at == "2024-01-01T00:00:00"
    assert metadata.num_rows == 100
    assert metadata.partitions == ["split=train"]
    assert metadata.description == "Test dataset"
    assert metadata.tags == {"source": "synthetic"}


def test_dataset_metadata_to_from_json():
    """Test DatasetMetadata JSON serialization/deserialization."""
    metadata = DatasetMetadata(
        version="v1",
        schema={"col": "float64"},
        feature_stats={"col": {"mean": 1.0}},
        created_at="2024-01-01T00:00:00",
        num_rows=50,
        partitions=[]
    )
    
    # Convert to JSON
    json_str = metadata.to_json()
    assert isinstance(json_str, str)
    
    # Parse JSON
    data = json.loads(json_str)
    assert data["version"] == "v1"
    assert data["num_rows"] == 50
    
    # Create from JSON
    metadata2 = DatasetMetadata.from_json(json_str)
    assert metadata2.version == metadata.version
    assert metadata2.num_rows == metadata.num_rows


def test_feature_store_write_dataset(feature_store, sample_dataset, temp_dir):
    """Test writing dataset to feature store."""
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="test_dataset",
        version="latest",
        description="Test dataset",
        tags={"test": "true"},
        compute_stats=True
    )
    
    # Check version ID was generated
    assert isinstance(version_id, str)
    assert len(version_id) > 0
    
    # Check dataset was written
    dataset_path = feature_store.base_path / "test_dataset" / f"v={version_id}"
    assert dataset_path.exists()
    
    # Check metadata was saved
    metadata_files = list(feature_store.metadata_path.glob(f"test_dataset_{version_id}.json"))
    assert len(metadata_files) == 1
    
    # Load and check metadata
    with open(metadata_files[0], "r") as f:
        metadata_data = json.load(f)
    
    assert metadata_data["version"] == version_id
    assert metadata_data["num_rows"] == 100
    assert metadata_data["description"] == "Test dataset"
    assert metadata_data["tags"]["test"] == "true"


def test_feature_store_write_dataset_with_partitioning(feature_store, sample_dataset):
    """Test writing dataset with partitioning."""
    # Add a split column for partitioning
    def add_split(batch):
        batch = batch.copy()
        batch["split"] = ["train"] * len(batch)
        batch.loc[50:, "split"] = "test"
        return batch
    
    dataset_with_split = sample_dataset.map_batches(
        add_split,
        batch_format="pandas"
    )
    
    version_id = feature_store.write_dataset(
        dataset=dataset_with_split,
        name="partitioned_dataset",
        partition_cols=["split"],
        compute_stats=False
    )
    
    # Check partitions were created
    dataset_path = feature_store.base_path / "partitioned_dataset" / f"v={version_id}"
    assert dataset_path.exists()
    
    # Should have train and test partitions
    partitions = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    assert "split=train" in partitions
    assert "split=test" in partitions


def test_feature_store_read_dataset(feature_store, sample_dataset):
    """Test reading dataset from feature store."""
    # First write a dataset
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="read_test",
        version="v1.0.0"
    )

    # Read it back
    loaded_dataset, metadata = feature_store.read_dataset(
        name="read_test",
        version=version_id
    )

    # Get columns, excluding any single-letter partition columns
    loaded_columns = set(loaded_dataset.columns())
    original_columns = set(sample_dataset.columns())
    
    # Filter out potential partition columns (like 'v' from version directory)
    loaded_columns_filtered = {
        col for col in loaded_columns 
        if not (len(col) == 1 and col.isalpha())
    }
    
    assert loaded_columns_filtered == original_columns
    assert loaded_dataset.count() == 100


def test_feature_store_read_latest_version(feature_store, sample_dataset):
    """Test reading latest version of dataset."""
    # Write multiple versions
    for i in range(3):
        # Modify dataset slightly
        modified_dataset = sample_dataset.map_batches(
            lambda batch: batch.assign(new_col=i),
            batch_format="pandas"
        )
        
        feature_store.write_dataset(
            dataset=modified_dataset,
            name="multi_version",
            version=f"v{i+1}.0.0"
        )
    
    # Read latest
    loaded_dataset, metadata = feature_store.read_dataset(
        name="multi_version",
        version="latest"
    )
    
    # Should have the last version's data
    df = loaded_dataset.to_pandas()
    assert "new_col" in df.columns
    assert df["new_col"].iloc[0] == 2  # Last version added 2


def test_feature_store_get_metadata(feature_store, sample_dataset):
    """Test getting metadata for a dataset version."""
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="metadata_test",
        version="v1.0.0"
    )
    
    metadata = feature_store.get_metadata("metadata_test", version_id)
    
    assert metadata.version == version_id
    assert metadata.num_rows == 100
    assert "feature_1_norm" in metadata.feature_stats
    assert "mean" in metadata.feature_stats["feature_1_norm"]


def test_feature_store_list_datasets(feature_store, sample_dataset):
    """Test listing datasets in feature store."""
    # Write multiple datasets
    for name in ["dataset_a", "dataset_b", "dataset_c"]:
        feature_store.write_dataset(
            dataset=sample_dataset,
            name=name,
            version="v1.0.0"
        )
    
    datasets = feature_store.list_datasets()
    
    assert len(datasets) == 3
    dataset_names = {d["name"] for d in datasets}
    assert dataset_names == {"dataset_a", "dataset_b", "dataset_c"}
    
    # Check each dataset has required fields
    for dataset_info in datasets:
        assert "name" in dataset_info
        assert "latest_version" in dataset_info
        assert "num_rows" in dataset_info
        assert "created_at" in dataset_info
        assert "num_versions" in dataset_info


def test_feature_store_delete_dataset_version(feature_store, sample_dataset):
    """Test deleting a specific dataset version."""
    # Write multiple versions
    versions = []
    for i in range(3):
        version_id = feature_store.write_dataset(
            dataset=sample_dataset,
            name="delete_test",
            version=f"v{i+1}.0.0"
        )
        versions.append(version_id)
    
    # Delete middle version
    feature_store.delete_dataset("delete_test", version=versions[1])
    
    # Check it's gone
    with pytest.raises(FeatureStoreError, match="not found"):
        feature_store.get_metadata("delete_test", versions[1])
    
    # Other versions should still exist
    assert feature_store.get_metadata("delete_test", versions[0]) is not None
    assert feature_store.get_metadata("delete_test", versions[2]) is not None


def test_feature_store_delete_all_versions(feature_store, sample_dataset):
    """Test deleting all versions of a dataset."""
    # Write multiple versions
    for i in range(3):
        feature_store.write_dataset(
            dataset=sample_dataset,
            name="delete_all_test",
            version=f"v{i+1}.0.0"
        )
    
    # Delete all versions
    feature_store.delete_dataset("delete_all_test", version=None)
    
    # Dataset should no longer be listed
    datasets = feature_store.list_datasets()
    dataset_names = {d["name"] for d in datasets}
    assert "delete_all_test" not in dataset_names
    
    # Metadata files should be gone
    metadata_files = list(feature_store.metadata_path.glob("delete_all_test_*.json"))
    assert len(metadata_files) == 0


def test_feature_store_compute_stats(feature_store, sample_dataset):
    """Test feature statistics computation."""
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="stats_test",
        compute_stats=True
    )
    
    metadata = feature_store.get_metadata("stats_test", version_id)
    
    # Check stats were computed for numerical columns
    assert "feature_1_norm" in metadata.feature_stats
    stats = metadata.feature_stats["feature_1_norm"]
    
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "count" in stats
    assert "null_count" in stats
    
    # Check values are reasonable
    assert stats["mean"] == pytest.approx(49.5, rel=0.1)  # mean of 0..99
    assert stats["min"] == 0.0
    assert stats["max"] == 99.0
    assert stats["count"] == 100
    assert stats["null_count"] == 0


def test_feature_store_error_handling(feature_store):
    """Test error handling in feature store."""
    # Reading non-existent dataset
    with pytest.raises(FeatureStoreError, match="not found"):
        feature_store.read_dataset("non_existent", "v1.0.0")
    
    # Reading non-existent version
    with pytest.raises(FeatureStoreError, match="version.*not found"):
        feature_store.read_dataset("test", "non_existent_version")
    
    # Getting metadata for non-existent dataset
    with pytest.raises(FeatureStoreError, match="Metadata not found"):
        feature_store.get_metadata("non_existent", "v1.0.0")


def test_feature_store_with_empty_dataset(feature_store):
    """Test feature store with empty dataset."""
    empty_df = pd.DataFrame({
        "id": [],
        "feature_1_norm": [],
        "feature_2_norm": [],
        "feature_3": [],
        "feature_4": [],
        "target": []
    })
    empty_dataset = rd.from_pandas(empty_df)
    
    version_id = feature_store.write_dataset(
        dataset=empty_dataset,
        name="empty_dataset",
        compute_stats=False
    )
    
    # Should still work
    loaded_dataset, metadata = feature_store.read_dataset(
        name="empty_dataset",
        version=version_id
    )
    
    assert loaded_dataset.count() == 0
    assert metadata.num_rows == 0


def test_feature_store_version_strategies(feature_store, sample_dataset):
    """Test different versioning strategies."""
    # Test UUID versioning
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="uuid_version",
        version=DatasetVersion.UUID
    )
    assert len(version_id) == 36  # UUID length
    
    # Test timestamp versioning
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="timestamp_version",
        version=DatasetVersion.TIMESTAMP
    )
    assert "_" in version_id  # Timestamp format: YYYYMMDD_HHMMSS
    
    # Test custom version
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="custom_version",
        version="my-custom-version-1.2.3"
    )
    assert version_id == "my-custom-version-1.2.3"


def test_feature_store_cache_invalidation(feature_store, sample_dataset):
    """Test metadata cache invalidation."""
    # Write dataset
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="cache_test",
        version="v1.0.0"
    )
    
    # Should be in cache
    assert version_id in feature_store._metadata_cache
    
    # Delete dataset
    feature_store.delete_dataset("cache_test", version=version_id)
    
    # Should be removed from cache
    assert version_id not in feature_store._metadata_cache


def test_feature_store_concurrent_access(temp_dir, sample_dataset):
    """Test feature store with concurrent-like access."""
    # Create two stores pointing to same directory
    store1 = FeatureStore(temp_dir)
    store2 = FeatureStore(temp_dir)
    
    # Write from store1
    version_id = store1.write_dataset(
        dataset=sample_dataset,
        name="concurrent_test",
        version="v1.0.0"
    )
    
    # Read from store2 (should see the data)
    dataset, metadata = store2.read_dataset("concurrent_test", version_id)
    assert dataset.count() == 100
    
    # Metadata should be in both caches
    assert version_id in store1._metadata_cache
    assert version_id in store2._metadata_cache


def test_feature_store_get_partitions(feature_store):
    """Test partition detection from path."""
    # Test with partitioned path
    partitioned_path = Path("/data/dataset/v=1.0.0/split=train/date=2024-01-01")
    partitions = feature_store._get_partitions(partitioned_path)
    
    assert partitions == ["split", "date"]
    
    # Test with non-partitioned path
    simple_path = Path("/data/dataset/v=1.0.0")
    partitions = feature_store._get_partitions(simple_path)
    
    assert partitions == []


def test_feature_store_validation_errors(feature_store, sample_dataset):
    """Test validation errors in feature store."""
    # Try to partition by non-existent column
    with pytest.raises(FeatureStoreError, match="Partition columns not found"):
        feature_store.write_dataset(
            dataset=sample_dataset,
            name="bad_partition",
            partition_cols=["non_existent_column"]
        )


def test_feature_store_metadata_persistence(temp_dir, sample_dataset):
    """Test that metadata persists across FeatureStore instances."""
    # Write dataset with first instance
    store1 = FeatureStore(temp_dir)
    version_id = store1.write_dataset(
        dataset=sample_dataset,
        name="persistence_test",
        version="v1.0.0"
    )
    
    # Create new instance
    store2 = FeatureStore(temp_dir)
    
    # Should be able to read metadata
    metadata = store2.get_metadata("persistence_test", version_id)
    assert metadata.version == version_id


def test_feature_store_with_large_dataset(feature_store):
    """Test feature store with larger dataset."""
    # Create larger dataset
    n_rows = 10000
    data = {
        "id": list(range(n_rows)),
        "feature_1_norm": [float(i) for i in range(n_rows)],
        "feature_2_norm": [float(i * 2) for i in range(n_rows)],
        "target": [i % 3 for i in range(n_rows)]
    }
    df = pd.DataFrame(data)
    large_dataset = rd.from_pandas(df)
    
    version_id = feature_store.write_dataset(
        dataset=large_dataset,
        name="large_dataset",
        compute_stats=True
    )
    
    # Check stats were computed
    metadata = feature_store.get_metadata("large_dataset", version_id)
    assert metadata.num_rows == n_rows
    assert "feature_1_norm" in metadata.feature_stats


def test_feature_store_append_metadata(feature_store, sample_dataset):
    """Test adding additional metadata fields."""
    version_id = feature_store.write_dataset(
        dataset=sample_dataset,
        name="metadata_rich",
        version="v1.0.0",
        description="Dataset with rich metadata",
        tags={
            "source": "synthetic",
            "quality": "high",
            "purpose": "testing"
        }
    )
    
    metadata = feature_store.get_metadata("metadata_rich", version_id)
    
    assert metadata.description == "Dataset with rich metadata"
    assert metadata.tags["source"] == "synthetic"
    assert metadata.tags["quality"] == "high"
    assert metadata.tags["purpose"] == "testing"


@pytest.mark.parametrize("num_samples", [1, 10, 100, 1000])
def test_feature_store_varying_sizes(feature_store, num_samples):
    """Test feature store with varying dataset sizes."""
    data = {
        "id": list(range(num_samples)),
        "feature_1_norm": [float(i) for i in range(num_samples)],
        "target": [i % 2 for i in range(num_samples)]
    }
    df = pd.DataFrame(data)
    dataset = rd.from_pandas(df)
    
    version_id = feature_store.write_dataset(
        dataset=dataset,
        name=f"size_test_{num_samples}",
        compute_stats=True
    )
    
    # Verify
    loaded_dataset, metadata = feature_store.read_dataset(
        f"size_test_{num_samples}",
        version_id
    )
    
    assert loaded_dataset.count() == num_samples
    assert metadata.num_rows == num_samples


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v"])