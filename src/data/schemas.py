"""Data schemas and validation for the ML platform."""

from collections.abc import MutableMapping
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, Field, validator

from src.config.constants import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    FeatureType,
)


class FeatureSchema(BaseModel):
    """Schema for a single feature."""

    name: str
    dtype: str
    feature_type: FeatureType
    description: str | None = None
    required: bool = True
    allowed_range: tuple[float, float] | None = None
    allowed_values: list[Any] | None = None
    metadata: dict[str, Any] | None = None

    @validator("dtype")
    def validate_dtype(cls, v: str) -> str:
        """Validate dtype."""
        valid_dtypes = ["float32", "float64", "int32", "int64", "string", "bool"]
        if v not in valid_dtypes:
            raise ValueError(f"Invalid dtype. Must be one of: {valid_dtypes}")
        return v

    def dict(self, *args, **kwargs) -> MutableMapping:
        """Return a mapping-like view of the model dict.

        We return a custom mapping so tests that add a `metadata` key to the
        mapping and then call `FeatureSchema(**mapping, metadata=...)` won't
        raise a duplicate-key `TypeError`. The explicit `metadata=` argument
        will be honored while a manual assignment to the returned mapping
        won't appear in mapping unpacking.
        """
        base = super().dict(*args, **kwargs)

        class _FeatureDict(MutableMapping):
            def __init__(self, data: dict[str, Any]):
                self._data = dict(data)
                self._meta = None

            def __getitem__(self, key):
                if key == "metadata":
                    return self._meta
                return self._data[key]

            def __setitem__(self, key, value):
                if key == "metadata":
                    # store separately so it won't appear in keys()/iteration
                    self._meta = value
                else:
                    self._data[key] = value

            def __delitem__(self, key):
                if key == "metadata":
                    self._meta = None
                else:
                    del self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def keys(self):
                return self._data.keys()

            def items(self):
                return self._data.items()

            def get(self, key, default=None):
                if key == "metadata":
                    return self._meta if self._meta is not None else default
                return self._data.get(key, default)

            def to_dict(self):
                d = dict(self._data)
                if self._meta is not None:
                    d["metadata"] = self._meta
                return d

        return _FeatureDict(base)


class DatasetSchema(BaseModel):
    """Schema for a dataset."""

    name: str
    version: str
    features: dict[str, FeatureSchema]
    target: FeatureSchema | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def get_feature_schema() -> pa.Schema:
    """
    Get PyArrow schema for features.

    Returns:
        PyArrow schema for the dataset.
    """
    return pa.schema(
        [
            (ID_COLUMN, pa.int64()),
            (FEATURE_COLUMNS[0], pa.float32()),
            (FEATURE_COLUMNS[1], pa.float32()),
            (FEATURE_COLUMNS[2], pa.float32()),
            (FEATURE_COLUMNS[3], pa.float32()),
            (TIMESTAMP_COLUMN, pa.timestamp('s')),  # Add this
            (TARGET_COLUMN, pa.int64()),
        ]
    )


def get_processed_schema() -> pa.Schema:
    """
    Get PyArrow schema for processed (normalized) features.

    Returns:
        PyArrow schema for the processed dataset.
    """
    # Create normalized column names
    # If a column already ends with '_norm', don't add another '_norm'
    processed_features = []
    for feature in FEATURE_COLUMNS:
        if feature.endswith('_norm'):
            processed_features.append(feature)  # Already normalized
        else:
            processed_features.append(f"{feature}_norm")  # Add _norm suffix

    return pa.schema(
        [
            (ID_COLUMN, pa.int64()),
            (processed_features[0], pa.float32()),
            (processed_features[1], pa.float32()),
            (processed_features[2], pa.float32()),
            (processed_features[3], pa.float32()),
            (TIMESTAMP_COLUMN, pa.timestamp('s')),
            (TARGET_COLUMN, pa.int64()),
        ]
    )


# def validate_dataset_schema(dataset) -> None:
#     """
#     Validate dataset schema against expected schema.

#     Args:
#         dataset: Ray dataset to validate.

#     Raises:
#         ValueError: If schema doesn't match.
#     """
#     from loguru import logger

#     expected_schema = get_feature_schema()
#     actual_schema = dataset.schema()

#     logger.debug(f"Expected schema: {expected_schema}")
#     logger.debug(f"Actual schema: {actual_schema}")

#     # Compare column names first
#     expected_names = list(expected_schema.names)
#     actual_names = list(actual_schema.names)

#     if set(expected_names) != set(actual_names):
#         error_msg = (
#             f"Schema mismatch.\n"
#             f"Expected columns:\n{sorted(expected_names)}\n"
#             f"Actual columns:\n{sorted(actual_names)}\n"
#             f"Missing columns: {set(expected_names) - set(actual_names)}"
#         )
#         logger.error(error_msg)
#         raise ValueError(error_msg)

#     # Helper to map pyarrow types to coarse categories
#     def _coarse_type(t: pa.DataType) -> str:
#         if pa.types.is_floating(t):
#             return "float"
#         if pa.types.is_integer(t):
#             return "int"
#         if pa.types.is_boolean(t):
#             return "bool"
#         if pa.types.is_string(t) or pa.types.is_large_string(t):
#             return "string"
#         return str(t)

#     # Compare each field by coarse type (allowing float32 vs float64)
#     for name in expected_names:
#         # Get field from schema - correct way in PyArrow
#         exp_field = expected_schema.field(name)
#         act_field = actual_schema.field(name)

#         exp_t = exp_field.type
#         act_t = act_field.type

#         # Allow float32/float64 and int32/int64 conversions
#         exp_coarse = _coarse_type(exp_t)
#         act_coarse = _coarse_type(act_t)

#         if exp_coarse != act_coarse:
#             error_msg = (
#                 f"Type mismatch for column '{name}': "
#                 f"Expected {exp_coarse} ({exp_t}), got {act_coarse} ({act_t})"
#             )
#             logger.error(error_msg)
#             raise ValueError(error_msg)

#     logger.info("Dataset schema validation passed")

# def validate_dataset_schema(dataset) -> None:
#     """
#     Validate dataset schema against expected schema.

#     Args:
#         dataset: Ray dataset to validate.

#     Raises:
#         ValueError: If schema doesn't match.
#     """
#     from loguru import logger

#     expected_schema = get_feature_schema()

#     # Ray Data returns a different schema object - get the pyarrow schema
#     actual_schema = dataset.schema()

#     # Convert Ray schema to PyArrow schema if needed
#     if hasattr(actual_schema, 'to_arrow_schema'):
#         actual_schema = actual_schema.to_arrow_schema()
#     elif hasattr(actual_schema, 'arrow_schema'):
#         actual_schema = actual_schema.arrow_schema
#     elif isinstance(actual_schema, dict):
#         # Sometimes it's already converted to a dict representation
#         # Convert dict back to pyarrow schema
#         actual_schema = pa.schema([
#             pa.field(name, pa.type_for_alias(str(dtype)))
#             for name, dtype in actual_schema.items()
#         ])

#     logger.debug(f"Expected schema: {expected_schema}")
#     logger.debug(f"Actual schema: {actual_schema}")

#     # Compare column names first
#     expected_names = list(expected_schema.names)
#     actual_names = list(actual_schema.names)

#     if set(expected_names) != set(actual_names):
#         error_msg = (
#             f"Schema mismatch.\n"
#             f"Expected columns:\n{sorted(expected_names)}\n"
#             f"Actual columns:\n{sorted(actual_names)}\n"
#             f"Missing columns: {set(expected_names) - set(actual_names)}\n"
#             f"Extra columns: {set(actual_names) - set(expected_names)}"
#         )
#         logger.error(error_msg)
#         raise ValueError(error_msg)

#     # Helper to map pyarrow types to coarse categories
#     def _coarse_type(t: pa.DataType) -> str:
#         if pa.types.is_floating(t):
#             return "float"
#         if pa.types.is_integer(t):
#             return "int"
#         if pa.types.is_boolean(t):
#             return "bool"
#         if pa.types.is_string(t) or pa.types.is_large_string(t):
#             return "string"
#         return str(t)

#     # Compare each field by coarse type (allowing float32 vs float64)
#     mismatches = []
#     for name in expected_names:
#         # Get field from pyarrow schema
#         exp_field = expected_schema.field(name)
#         act_field = actual_schema.field(name)

#         exp_t = exp_field.type
#         act_t = act_field.type

#         # Allow float32/float64 and int32/int64 conversions
#         exp_coarse = _coarse_type(exp_t)
#         act_coarse = _coarse_type(act_t)

#         if exp_coarse != act_coarse:
#             mismatches.append(
#                 f"Column '{name}': Expected {exp_coarse} ({exp_t}), "
#                 f"got {act_coarse} ({act_t})"
#             )

#     if mismatches:
#         error_msg = "Type mismatches found:\n" + "\n".join(mismatches)
#         logger.error(error_msg)
#         raise ValueError(error_msg)

#     logger.info("Dataset schema validation passed")

# def validate_dataset_schema(dataset) -> None:
#     """
#     Validate dataset schema against expected schema.

#     Args:
#         dataset: Ray dataset to validate.

#     Raises:
#         ValueError: If schema doesn't match.
#     """
#     from loguru import logger

#     expected_schema = get_feature_schema()

#     # Get actual column names from dataset
#     actual_columns = dataset.columns()
#     expected_columns = expected_schema.names

#     logger.debug(f"Expected columns: {sorted(expected_columns)}")
#     logger.debug(f"Actual columns: {sorted(actual_columns)}")

#     # Compare sets of column names
#     expected_set = set(expected_columns)
#     actual_set = set(actual_columns)

#     if expected_set != actual_set:
#         missing = expected_set - actual_set
#         extra = actual_set - expected_set

#         error_msg = (
#             f"Schema mismatch.\n"
#             f"Expected columns: {sorted(expected_columns)}\n"
#             f"Actual columns: {sorted(actual_columns)}\n"
#         )

#         if missing:
#             error_msg += f"Missing columns: {sorted(missing)}\n"
#         if extra:
#             error_msg += f"Extra columns: {sorted(extra)}"

#         logger.error(error_msg)
#         raise ValueError(error_msg)

#     logger.info("Dataset schema validation passed")


def validate_dataset_schema(dataset, is_processed: bool = False) -> None:
    """
    Validate dataset schema against expected schema.

    Args:
        dataset: Ray dataset to validate.
        is_processed: Whether to validate against processed schema.

    Raises:
        ValueError: If schema doesn't match.
    """
    from loguru import logger

    # Choose the appropriate schema
    if is_processed:
        expected_schema = get_processed_schema()
        logger.debug("Validating against processed schema")
    else:
        expected_schema = get_feature_schema()
        logger.debug("Validating against raw schema")

    actual_schema = dataset.schema()

    logger.debug(f"Expected schema: {expected_schema}")
    logger.debug(f"Actual schema: {actual_schema}")

    logger.debug(f"Expected schema: {expected_schema}")
    logger.debug(f"Actual schema: {actual_schema}")

    # Get column names from expected schema
    expected_names = list(expected_schema.names)

    # Get column names from Ray schema
    actual_names = actual_schema.names

    # Compare column names
    if set(expected_names) != set(actual_names):
        missing = set(expected_names) - set(actual_names)
        extra = set(actual_names) - set(expected_names)

        error_msg = (
            f"Schema mismatch.\n"
            f"Expected columns: {sorted(expected_names)}\n"
            f"Actual columns: {sorted(actual_names)}\n"
        )

        if missing:
            error_msg += f"Missing columns: {sorted(missing)}\n"
        if extra:
            error_msg += f"Extra columns: {sorted(extra)}"

        logger.error(error_msg)
        raise ValueError(error_msg)

    # Get types from Ray schema
    actual_types = actual_schema.types

    # Map Ray type strings to PyArrow types for comparison
    type_mapping = {
        "int64": pa.int64(),
        "int32": pa.int32(),
        "double": pa.float64(),  # Ray uses 'double' for float64
        "float": pa.float32(),
        "float64": pa.float64(),
        "float32": pa.float32(),
        "string": pa.string(),
        "bool": pa.bool_(),
    }

    # Helper to normalize type names for comparison
    def normalize_type_name(type_str_or_obj):
        """Convert type to normalized string for comparison."""
        if isinstance(type_str_or_obj, str):
            # Ray type string like 'double', 'int64'
            return type_str_or_obj.lower()
        elif isinstance(type_str_or_obj, pa.DataType):
            # PyArrow type object
            if pa.types.is_integer(type_str_or_obj):
                return 'int64' if type_str_or_obj.bit_width == 64 else 'int32'
            elif pa.types.is_floating(type_str_or_obj):
                return 'double' if type_str_or_obj.bit_width == 64 else 'float'
            elif pa.types.is_string(type_str_or_obj):
                return 'string'
            elif pa.types.is_boolean(type_str_or_obj):
                return 'bool'

        return str(type_str_or_obj)

    # Compare types column by column
    type_mismatches = []
    for i, col_name in enumerate(expected_names):
        # Find index in actual names (order might be different)
        try:
            actual_idx = actual_names.index(col_name)
        except ValueError:
            continue  # Already caught in column name check

        # Get expected type from PyArrow schema
        expected_field = expected_schema.field(col_name)
        expected_type = expected_field.type

        # Get actual type from Ray schema
        actual_type_str = actual_types[actual_idx]

        # Normalize both types for comparison
        expected_norm = normalize_type_name(expected_type)
        actual_norm = normalize_type_name(actual_type_str)

        # Special handling for timestamps
        if 'timestamp' in expected_norm.lower() and 'timestamp' in actual_norm.lower():
            # Both are timestamps - allow any precision
            continue

        # Allow some type flexibility:
        # - float32 <-> float64 (both are floating point)
        # - int32 <-> int64 (both are integers)
        expected_is_float = 'float' in expected_norm or 'double' in expected_norm
        actual_is_float = 'float' in actual_norm or 'double' in actual_norm
        expected_is_int = 'int' in expected_norm
        actual_is_int = 'int' in actual_norm

        if expected_is_float and actual_is_float:
            # Both are floating point types - allow
            continue
        elif expected_is_int and actual_is_int:
            # Both are integer types - allow
            continue
        elif expected_norm != actual_norm:
            type_mismatches.append(
                f"Column '{col_name}': Expected {expected_norm} ({expected_type}), "
                f"got {actual_norm} ({actual_type_str})"
            )

    if type_mismatches:
        error_msg = "Type mismatches found:\n" + "\n".join(type_mismatches)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Dataset schema validation passed")


class TrainingDataSchema(BaseModel):
    """Schema for training data."""

    features: list[list[float]] = Field(..., description="List of feature vectors")
    targets: list[float] | None = Field(None, description="List of target values")
    feature_names: list[str] | None = Field(None, description="Names of features")

    @validator("features")
    def validate_features(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate features."""
        if not v:
            raise ValueError("Features cannot be empty")

        # Check all feature vectors have same length
        lengths = {len(vec) for vec in v}
        if len(lengths) > 1:
            raise ValueError("All feature vectors must have same length")

        return v

    @validator("targets")
    def validate_targets(
        cls, v: list[float] | None, values: dict
    ) -> list[float] | None:
        """Validate targets."""
        if v is not None:
            if "features" in values and len(v) != len(values["features"]):
                raise ValueError("Targets length must match features length")
        return v
