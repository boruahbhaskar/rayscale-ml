"""FastAPI application for model serving."""

import uuid
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from src.serving.schemas import (
    FeatureRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthCheckResponse,
    ErrorResponse
)
from src.serving.models import get_model_manager, ModelLoadingError
from src import  __version__
from src.config import settings



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    logger.info("Starting FastAPI application")
    
    # Load default model
    try:
        model_manager = get_model_manager()
        model_manager.load_model(
            model_name="tabular_nn",
            version="latest"
        )
        logger.info("Default model loaded successfully")
    except ModelLoadingError as e:
        logger.error(f"Failed to load default model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application")
    # Cleanup resources if needed


# Create FastAPI app
app = FastAPI(
    title="RayScale ML Platform API",
    description="Machine Learning Model Serving API",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment == "development" else None,
    redoc_url="/redoc" if settings.environment == "development" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
model_manager = get_model_manager()


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors."""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    
    error_response = ErrorResponse(
        error="Validation Error",
        error_code="VALIDATION_ERROR",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        request_id=request.headers.get("X-Request-ID"),
        details={"errors": error_details}
    )
    
    return JSONResponse(
        status_code=422,
        content=error_response.dict()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    error_response = ErrorResponse(
        error=exc.detail,
        error_code="HTTP_ERROR",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error="Internal Server Error",
        error_code="INTERNAL_ERROR",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        request_id=request.headers.get("X-Request-ID")
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    request_id = str(uuid.uuid4())
    request.headers.__dict__["_list"].append(
        (b"x-request-id", request_id.encode())
    )
    
    start_time = time.time()
    
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            f"Response: {response.status_code} "
            f"Time: {process_time:.3f}s "
            f"Request-ID: {request_id}"
        )
        
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"Error: {str(e)}"
        )
        raise


# Dependency for model manager
async def get_model_manager_dep():
    """Dependency to get model manager."""
    return model_manager


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check service health and status"
)
async def health_check(
    model_manager: get_model_manager_dep = Depends()
) -> HealthCheckResponse:
    """Health check endpoint."""
    model_loaded = len(model_manager.get_loaded_models()) > 0
    
    return HealthCheckResponse(
        status="healthy",
        version=__version__,
        model_loaded=model_loaded,
        model_version=model_manager.current_version if model_loaded else None,
        uptime=model_manager.get_uptime(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
    )


# Model info endpoint
@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Get information about the loaded model"
)
async def get_model_info(
    model_name: str = "tabular_nn",
    version: Optional[str] = None,
    model_manager: get_model_manager_dep = Depends()
) -> ModelInfoResponse:
    """Get model information."""
    try:
        model_info = model_manager.get_model_info(model_name, version)
        
        return ModelInfoResponse(
            model_name=model_name,
            model_version=model_info["model_version"],
            framework=model_info["framework"],
            created_at=model_info["metadata"].get("created_at", ""),
            input_schema={
                "feature_1_norm": "float",
                "feature_2_norm": "float",
                "feature_3": "float",
                "feature_4": "float"
            },
            output_schema={"prediction": "float"},
            metrics=model_info["metadata"].get("metrics", {}),
            status="loaded"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Single prediction endpoint
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make Prediction",
    description="Make a single prediction with the model"
)
async def predict(
    request: FeatureRequest,
    model_name: str = "tabular_nn",
    version: Optional[str] = None,
    include_features: bool = False,
    model_manager: get_model_manager_dep = Depends()
) -> PredictionResponse:
    """Make a single prediction."""
    start_time = time.time()
    
    try:
        # Convert request to features dict
        features = request.dict()
        
        # Make prediction
        prediction = model_manager.predict(
            features=features,
            model_name=model_name,
            version=version
        )
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response = PredictionResponse(
            prediction=prediction,
            model_version=version or model_manager.current_version or "unknown",
            prediction_id=str(uuid.uuid4()),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
        if include_features:
            response.features = features
        
        logger.info(
            f"Prediction made: {prediction:.4f} "
            f"Model: {model_name} "
            f"Time: {processing_time:.3f}s"
        )
        
        return response
        
    except ModelLoadingError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


# Batch prediction endpoint
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Prediction",
    description="Make multiple predictions in a batch"
)
async def predict_batch(
    request: BatchPredictionRequest,
    model_name: str = "tabular_nn",
    model_manager: get_model_manager_dep = Depends()
) -> BatchPredictionResponse:
    """Make batch predictions."""
    start_time = time.time()
    
    try:
        # Prepare features list
        features_list = [feature.dict() for feature in request.features]
        
        # Make predictions
        predictions = model_manager.batch_predict(
            features_list=features_list,
            model_name=model_name,
            version=request.model_version
        )
        
        processing_time = time.time() - start_time
        
        # Prepare responses
        prediction_responses = []
        for features, prediction in zip(request.features, predictions):
            response = PredictionResponse(
                prediction=prediction,
                model_version=request.model_version or model_manager.current_version or "unknown",
                prediction_id=str(uuid.uuid4()),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            
            if request.return_features:
                response.features = features.dict()
            
            prediction_responses.append(response)
        
        logger.info(
            f"Batch prediction made: {len(predictions)} predictions "
            f"Model: {model_name} "
            f"Time: {processing_time:.3f}s"
        )
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            batch_id=str(uuid.uuid4()),
            processing_time=processing_time,
            model_version=request.model_version or model_manager.current_version or "unknown",
            num_predictions=len(predictions)
        )
        
    except ModelLoadingError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal batch prediction error")


# Model management endpoints
@app.post(
    "/model/load",
    summary="Load Model",
    description="Load a model into memory"
)
async def load_model(
    model_name: str = "tabular_nn",
    version: str = "latest",
    force: bool = False,
    model_manager: get_model_manager_dep = Depends()
) -> dict:
    """Load a model."""
    try:
        model_manager.load_model(model_name, version, force_reload=force)
        
        return {
            "message": f"Model {model_name} version {version} loaded successfully",
            "model_name": model_name,
            "version": version,
            "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
    except ModelLoadingError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post(
    "/model/unload",
    summary="Unload Model",
    description="Unload a model from memory"
)
async def unload_model(
    model_name: str = "tabular_nn",
    version: Optional[str] = None,
    model_manager: get_model_manager_dep = Depends()
) -> dict:
    """Unload a model."""
    try:
        unloaded = model_manager.unload_model(model_name, version)
        
        if unloaded:
            message = f"Model {model_name}"
            if version:
                message += f" version {version}"
            message += " unloaded successfully"
            
            return {
                "message": message,
                "success": True
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/model/loaded",
    summary="Get Loaded Models",
    description="Get list of currently loaded models"
)
async def get_loaded_models(
    model_manager: get_model_manager_dep = Depends()
) -> dict:
    """Get loaded models."""
    loaded_models = model_manager.get_loaded_models()
    
    return {
        "loaded_models": loaded_models,
        "count": len(loaded_models),
        "current_version": model_manager.current_version
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RayScale ML Platform API",
        "version": __version__,
        "documentation": "/docs" if settings.environment == "development" else None,
        "health_check": "/health"
    }