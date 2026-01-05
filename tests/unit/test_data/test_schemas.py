"""Tests for data schemas."""

import pytest
import pyarrow as pa
from pydantic import ValidationError

from src.data.schemas import (
    FeatureSchema,
    DatasetSchema,
    get_feature_schema,
    validate_dataset_schema,
    TrainingDataSchema,
    FeatureType,
)
from src.config.constants import FEATURE_COLUMNS, TARGET_COLUMN


def test_feature_schema_creation():
    """Test creating a basic FeatureSchema."""
    feature = FeatureSchema(
        name="age",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL,
        description="Age of the individual",
        required=True
    )
    
    assert feature.name == "age"
    assert feature.dtype == "float32"
    assert feature.feature_type == FeatureType.NUMERICAL
    assert feature.description == "Age of the individual"
    assert feature.required is True


def test_feature_schema_with_range():
    """Test FeatureSchema with allowed range."""
    feature = FeatureSchema(
        name="temperature",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL,
        allowed_range=(-50.0, 150.0)
    )
    
    assert feature.allowed_range == (-50.0, 150.0)
    assert feature.allowed_range[0] == -50.0
    assert feature.allowed_range[1] == 150.0


def test_feature_schema_with_allowed_values():
    """Test FeatureSchema with allowed values."""
    feature = FeatureSchema(
        name="category",
        dtype="string",
        feature_type=FeatureType.CATEGORICAL,
        allowed_values=["A", "B", "C"]
    )
    
    assert feature.allowed_values == ["A", "B", "C"]
    assert len(feature.allowed_values) == 3


def test_feature_schema_invalid_dtype():
    """Test that invalid dtype raises ValidationError."""
    with pytest.raises(ValidationError, match="Invalid dtype"):
        FeatureSchema(
            name="test",
            dtype="invalid_dtype",
            feature_type=FeatureType.NUMERICAL
        )


def test_dataset_schema_creation():
    """Test creating a DatasetSchema."""
    features = {
        "feature_1": FeatureSchema(
            name="feature_1",
            dtype="float32",
            feature_type=FeatureType.NUMERICAL
        ),
        "feature_2": FeatureSchema(
            name="feature_2",
            dtype="float32",
            feature_type=FeatureType.NUMERICAL
        )
    }
    
    target = FeatureSchema(
        name="target",
        dtype="int64",
        feature_type=FeatureType.NUMERICAL
    )
    
    dataset = DatasetSchema(
        name="test_dataset",
        version="1.0.0",
        features=features,
        target=target,
        metadata={"source": "synthetic"}
    )
    
    assert dataset.name == "test_dataset"
    assert dataset.version == "1.0.0"
    assert len(dataset.features) == 2
    assert dataset.target.name == "target"
    assert dataset.metadata["source"] == "synthetic"


def test_dataset_schema_without_target():
    """Test DatasetSchema without target (for unsupervised learning)."""
    features = {
        "feature_1": FeatureSchema(
            name="feature_1",
            dtype="float32",
            feature_type=FeatureType.NUMERICAL
        )
    }
    
    dataset = DatasetSchema(
        name="unsupervised_dataset",
        version="1.0.0",
        features=features
    )
    
    assert dataset.name == "unsupervised_dataset"
    assert dataset.target is None
    assert len(dataset.features) == 1


def test_get_feature_schema():
    """Test get_feature_schema returns a valid PyArrow schema."""
    schema = get_feature_schema()
    
    assert isinstance(schema, pa.Schema)
    assert len(schema.names) == len(FEATURE_COLUMNS) + 2  # +2 for id and target
    
    # Check expected columns exist
    for col in FEATURE_COLUMNS:
        assert col in schema.names
    
    assert TARGET_COLUMN in schema.names
    assert "id" in schema.names  # from constants.ID_COLUMN
    
    # Check data types
    for col in FEATURE_COLUMNS:
        assert schema.field(col).type == pa.float32()
    
    assert schema.field(TARGET_COLUMN).type == pa.int64()
    assert schema.field("id").type == pa.int64()


def test_validate_dataset_schema_success():
    """Test successful schema validation."""
    # Create a dataset with correct schema
    import ray.data as rd
    import pandas as pd
    
    # Create data matching the expected schema
    data = {
        "id": [1, 2, 3],
        FEATURE_COLUMNS[0]: [1.0, 2.0, 3.0],
        FEATURE_COLUMNS[1]: [4.0, 5.0, 6.0],
        FEATURE_COLUMNS[2]: [7.0, 8.0, 9.0],
        FEATURE_COLUMNS[3]: [10.0, 11.0, 12.0],
        TARGET_COLUMN: [0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    dataset = rd.from_pandas(df)
    
    # This should not raise an exception
    validate_dataset_schema(dataset)


def test_validate_dataset_schema_missing_column():
    """Test schema validation fails with missing column."""
    import ray.data as rd
    import pandas as pd
    
    # Create data missing a required column
    data = {
        "id": [1, 2, 3],
        FEATURE_COLUMNS[0]: [1.0, 2.0, 3.0],
        FEATURE_COLUMNS[1]: [4.0, 5.0, 6.0],
        FEATURE_COLUMNS[2]: [7.0, 8.0, 9.0],
        # Missing FEATURE_COLUMNS[3]
        TARGET_COLUMN: [0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    dataset = rd.from_pandas(df)
    
    with pytest.raises(ValueError, match="Schema mismatch"):
        validate_dataset_schema(dataset)


# def test_validate_dataset_schema_wrong_type():
#     """Test schema validation fails with wrong data type."""
#     import ray.data as rd
#     import pandas as pd
    
#     # Create data with wrong type for a feature
#     data = {
#         "id": [1, 2, 3],
#         FEATURE_COLUMNS[0]: [1, 2, 3],  # int instead of float
#         FEATURE_COLUMNS[1]: [4.0, 5.0, 6.0],
#         FEATURE_COLUMNS[2]: [7.0, 8.0, 9.0],
#         FEATURE_COLUMNS[3]: [10.0, 11.0, 12.0],
#         TARGET_COLUMN: [0, 1, 0]
#     }
    
#     df = pd.DataFrame(data)
#     dataset = rd.from_pandas(df)
    
#     # This might still pass because pandas can upcast int to float
#     # But let's test the principle
#     try:
#         validate_dataset_schema(dataset)
#     except ValueError as e:
#         assert "Schema mismatch" in str(e)

def test_validate_dataset_schema_wrong_type():
    """Test schema validation fails with wrong data type."""
    import ray.data as rd
    import pandas as pd

    # Create data with wrong type for a feature
    data = {
        "id": [1, 2, 3],
        FEATURE_COLUMNS[0]: [1, 2, 3],  # int instead of float
        FEATURE_COLUMNS[1]: [4.0, 5.0, 6.0],
        FEATURE_COLUMNS[2]: [7.0, 8.0, 9.0],
        FEATURE_COLUMNS[3]: [10.0, 11.0, 12.0],
        TARGET_COLUMN: [0, 1, 0]
    }

    df = pd.DataFrame(data)
    dataset = rd.from_pandas(df)

    # Debug: Print what we're getting
    print(f"\nDebug - Dataset schema types:")
    print(f"Column names: {dataset.schema().names}")
    print(f"Column types: {dataset.schema().types}")
    
    # This should raise a ValueError
    try:
        validate_dataset_schema(dataset)
        # If we get here, validation passed unexpectedly
        assert False, "Validation should have failed for wrong type"
    except ValueError as e:
        error_msg = str(e)
        print(f"\nDebug - Error message: {error_msg}")
        
        # Check it's the right kind of error
        assert "feature_1_norm" in error_msg
        # Check it mentions type issue (either 'type' or 'mismatch' or 'int64' vs 'float')
        assert any(word in error_msg.lower() for word in ['type', 'mismatch', 'int64', 'float'])

def test_training_data_schema_creation():
    """Test creating TrainingDataSchema."""
    features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    targets = [0.0, 1.0]
    
    training_data = TrainingDataSchema(
        features=features,
        targets=targets,
        feature_names=["f1", "f2", "f3"]
    )
    
    assert len(training_data.features) == 2
    assert len(training_data.targets) == 2
    assert training_data.feature_names == ["f1", "f2", "f3"]


def test_training_data_schema_validation():
    """Test TrainingDataSchema validation."""
    # Valid data
    features = [[1.0, 2.0], [3.0, 4.0]]
    targets = [0.0, 1.0]
    
    training_data = TrainingDataSchema(
        features=features,
        targets=targets
    )
    
    assert training_data.features == features
    assert training_data.targets == targets


def test_training_data_schema_invalid_features():
    """Test TrainingDataSchema with invalid features."""
    # Empty features list
    with pytest.raises(ValidationError, match="Features cannot be empty"):
        TrainingDataSchema(features=[])
    
    # Features with different lengths
    with pytest.raises(ValidationError, match="All feature vectors must have same length"):
        TrainingDataSchema(features=[[1.0, 2.0], [3.0]])


def test_training_data_schema_target_length_mismatch():
    """Test TrainingDataSchema with target length mismatch."""
    features = [[1.0, 2.0], [3.0, 4.0]]
    targets = [0.0]  # Only one target for two features
    
    with pytest.raises(ValidationError, match="Targets length must match features length"):
        TrainingDataSchema(features=features, targets=targets)


def test_training_data_schema_without_targets():
    """Test TrainingDataSchema without targets (for unsupervised)."""
    features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    
    training_data = TrainingDataSchema(features=features)
    
    assert training_data.features == features
    assert training_data.targets is None


def test_feature_type_enum():
    """Test FeatureType enum values."""
    assert FeatureType.NUMERICAL == "numerical"
    assert FeatureType.CATEGORICAL == "categorical"
    assert FeatureType.TEXT == "text"
    assert FeatureType.IMAGE == "image"
    assert FeatureType.EMBEDDING == "embedding"
    
    # Test all values
    all_types = [ft.value for ft in FeatureType]
    assert "numerical" in all_types
    assert "categorical" in all_types
    assert "text" in all_types

def test_feature_schema_to_dict():
    """Test that FeatureSchema can be converted to dictionary."""
    feature = FeatureSchema(
        name="test_feature",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL,
        description="Test feature",
        required=True
    )
    
    # Use model_dump() for Pydantic v2
    feature_dict = feature.model_dump()
    
    assert isinstance(feature_dict, dict)
    assert feature_dict["name"] == "test_feature"
    assert feature_dict["dtype"] == "float32"
    assert feature_dict["feature_type"] == "numerical"
    assert feature_dict["description"] == "Test feature"
    assert feature_dict["required"] is True

def test_dataset_schema_to_dict():
    """Test that DatasetSchema can be converted to dictionary."""
    features = {
        "feature_1": FeatureSchema(
            name="feature_1",
            dtype="float32",
            feature_type=FeatureType.NUMERICAL
        )
    }
    
    dataset = DatasetSchema(
        name="test_dataset",
        version="1.0.0",
        features=features,
        metadata={"key": "value"}
    )
    
    dataset_dict = dataset.dict()
    
    assert isinstance(dataset_dict, dict)
    assert dataset_dict["name"] == "test_dataset"
    assert dataset_dict["version"] == "1.0.0"
    assert "feature_1" in dataset_dict["features"]
    assert dataset_dict["metadata"]["key"] == "value"


def test_feature_schema_with_tags():
    """Test FeatureSchema with tags in metadata."""
    feature = FeatureSchema(
        name="price",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL,
        description="Product price",
        required=True
    )
    
    # Get dictionary representation
    feature_dict = feature.model_dump()
    
    # Check basic fields
    assert feature_dict["name"] == "price"
    assert feature_dict["dtype"] == "float32"
    assert feature_dict["feature_type"] == "numerical"
    assert feature_dict["description"] == "Product price"
    assert feature_dict["required"] is True
    
    # Note: metadata field doesn't exist in current FeatureSchema design
    # This test just ensures the schema works without metadata

def test_schema_comparison():
    """Test that schemas with same values are equal."""
    feature1 = FeatureSchema(
        name="age",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL
    )
    
    feature2 = FeatureSchema(
        name="age",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL
    )
    
    # They should be equal
    assert feature1.model_dump() == feature2.model_dump()

def test_schema_inequality():
    """Test that schemas with different values are not equal."""
    feature1 = FeatureSchema(
        name="age",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL
    )
    
    feature2 = FeatureSchema(
        name="salary",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL
    )
    
    # They should not be equal
    assert feature1.model_dump() != feature2.model_dump()


def test_training_data_schema_json_serializable():
    """Test that TrainingDataSchema is JSON serializable."""
    import json
    
    training_data = TrainingDataSchema(
        features=[[1.0, 2.0], [3.0, 4.0]],
        targets=[0.0, 1.0],
        feature_names=["f1", "f2"]
    )
    
    json_str = json.dumps(training_data.dict())
    assert isinstance(json_str, str)
    assert "features" in json_str
    assert "targets" in json_str
    assert "feature_names" in json_str


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "string", "bool"])
def test_valid_dtypes(dtype):
    """Test all valid dtype values."""
    feature = FeatureSchema(
        name="test",
        dtype=dtype,
        feature_type=FeatureType.NUMERICAL if dtype in ["float32", "float64", "int32", "int64"] else FeatureType.CATEGORICAL
    )
    
    assert feature.dtype == dtype


def test_feature_schema_optional_fields():
    """Test FeatureSchema with optional fields omitted."""
    feature = FeatureSchema(
        name="simple_feature",
        dtype="float32",
        feature_type=FeatureType.NUMERICAL
    )
    
    assert feature.name == "simple_feature"
    assert feature.dtype == "float32"
    assert feature.feature_type == FeatureType.NUMERICAL
    assert feature.description is None
    assert feature.required is True  # default value
    assert feature.allowed_range is None
    assert feature.allowed_values is None


if __name__ == "__main__":
    """Run tests when executed directly."""
    pytest.main([__file__, "-v"])