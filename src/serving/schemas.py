"""API schemas for model serving."""

from typing import Any

from pydantic import BaseModel, Field, validator


class FeatureRequest(BaseModel):
    """Request schema for individual features."""

    feature_1: float = Field(..., description="First normalized feature")
    feature_2: float = Field(..., description="Second normalized feature")
    feature_3: float = Field(..., description="Third feature")
    feature_4: float = Field(..., description="Fourth feature")
    feature_interaction: float | None = Field(
        None, description="Interaction feature (will be computed if not provided)"
    )

    @validator('feature_interaction', pre=True, always=True)
    def compute_interaction(cls, v, values):
        """Compute interaction feature if not provided."""
        if v is None:
            if 'feature_1' in values and 'feature_2' in values:
                return values['feature_1'] * values['feature_2']
        return v


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    features: list[FeatureRequest] = Field(
        ...,
        description="List of feature sets for prediction",
        min_items=1,
        max_items=1000,  # Limit batch size for safety
    )

    model_version: str | None = Field(
        "latest", description="Model version to use for prediction"
    )
    return_features: bool = Field(
        False, description="Whether to return input features in response"
    )
    include_confidence: bool = Field(
        False, description="Whether to include confidence scores"
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: float = Field(..., description="Predicted value")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction ID")
    timestamp: str = Field(..., description="Prediction timestamp")

    features: dict[str, float] | None = Field(
        None, description="Input features (if requested)"
    )
    confidence: float | None = Field(
        None, description="Confidence score (if available)"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    batch_id: str = Field(..., description="Batch ID")
    processing_time: float = Field(..., description="Total processing time in seconds")
    model_version: str = Field(..., description="Model version used")
    num_predictions: int = Field(..., description="Number of predictions in batch")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    framework: str = Field(..., description="Model framework")
    created_at: str = Field(..., description="Model creation timestamp")
    input_schema: dict[str, Any] = Field(..., description="Expected input schema")
    output_schema: dict[str, Any] = Field(..., description="Output schema")
    metrics: dict[str, float] = Field(..., description="Model performance metrics")
    status: str = Field(..., description="Model status")


class HealthCheckResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str | None = Field(None, description="Loaded model version")
    uptime: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: str | None = Field(None, description="Request ID (if available)")
    details: dict[str, Any] | None = Field(
        None, description="Additional error details"
    )
