"""
MLflow client for artifact management.
"""
import mlflow
import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MLflowClient:
    """MLflow client for managing experiments and models."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow client.
        
        Args:
            tracking_uri: MLflow tracking URI. If None, uses default or environment variable.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        
        self.client = mlflow.tracking.MlflowClient()
    
    def log_model(self, model, artifact_path: str, **kwargs):
        """Log a model to MLflow."""
        import mlflow.pytorch
        
        mlflow.pytorch.log_model(model, artifact_path, **kwargs)
    
    def load_model(self, model_uri: str):
        """Load a model from MLflow."""
        import mlflow.pytorch
        
        return mlflow.pytorch.load_model(model_uri)
    
    def create_experiment(self, name: str) -> str:
        """Create a new experiment."""
        try:
            experiment_id = mlflow.create_experiment(name)
            logger.info(f"Created experiment {name} with ID {experiment_id}")
            return experiment_id
        except Exception as e:
            logger.warning(f"Experiment {name} may already exist: {e}")
            return mlflow.get_experiment_by_name(name).experiment_id
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None, **kwargs):
        """Start an MLflow run."""
        experiment_id = self.create_experiment(experiment_name)
        return mlflow.start_run(experiment_id=experiment_id, run_name=run_name, **kwargs)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact to current run."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def get_latest_model(self, experiment_name: str, model_name: str):
        """Get the latest version of a model from MLflow Model Registry."""
        try:
            model_uri = f"models:/{model_name}/latest"
            return self.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None