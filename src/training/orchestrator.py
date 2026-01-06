"""Training orchestration and workflow management."""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from datetime import datetime

from loguru import logger

from src.training.pipelines.base_pipeline import BaseTrainingPipeline, PipelineConfig
from src.training.pipelines.ray_pipeline import RayTrainingPipeline
from src.training.pipelines.tune_pipeline import TunePipeline
from src.artifacts.mlflow_client import MLflowClient
from src.config import settings


class TrainingOrchestrator:
    """Orchestrates training pipelines and experiments."""
    
    def __init__(self, experiment_name: Optional[str] = None):
        """
        Initialize orchestrator.
        
        Args:
            experiment_name: Name of the experiment.
        """
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.mlflow_client = MLflowClient()
        self.current_run = None
        
    def run_pipeline(
        self,
        pipeline_type: str,
        config_path: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a training pipeline.
        
        Args:
            pipeline_type: Type of pipeline ("train", "tune").
            config_path: Path to pipeline configuration.
            **kwargs: Additional pipeline arguments.
            
        Returns:
            Pipeline results.
            
        Raises:
            ValueError: If pipeline type is invalid.
        """
        logger.info(f"Running {pipeline_type} pipeline with config: {config_path}")
        
        # Load configuration
        config = self._load_config(config_path)
        pipeline_config = PipelineConfig.from_dict(config)
        
        # Create pipeline
        pipeline = self._create_pipeline(pipeline_type, pipeline_config)
        
        # Start MLflow run
        self.current_run = self.mlflow_client.start_run(
            experiment_name=self.experiment_name,
            run_name=f"{pipeline_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "pipeline_type": pipeline_type,
                "config_file": config_path.name
            }
        )
        
        try:
            # Log configuration
            self.mlflow_client.log_params(config)
            
            # Run pipeline
            result = pipeline.run()
            
            # Log results
            self._log_results(result)
            
            # Save artifacts
            self._save_artifacts(result, pipeline_type)
            
            logger.info(f"{pipeline_type} pipeline completed successfully")
            return {
                "success": True,
                "result": result,
                "run_id": self.current_run.info.run_id
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            
            # Log failure
            self.mlflow_client.log_param("status", "failed")
            self.mlflow_client.log_param("error", str(e))
            
            raise
            
        finally:
            # End MLflow run
            if self.current_run:
                self.mlflow_client.end_run()
                self.current_run = None
            
            # Cleanup pipeline resources
            if hasattr(pipeline, 'cleanup'):
                pipeline.cleanup()
    
    def _create_pipeline(
        self,
        pipeline_type: str,
        config: PipelineConfig
    ) -> BaseTrainingPipeline:
        """Create pipeline instance."""
        pipeline_map = {
            "train": RayTrainingPipeline,
            "tune": TunePipeline
        }
        
        if pipeline_type not in pipeline_map:
            raise ValueError(
                f"Invalid pipeline type: {pipeline_type}. "
                f"Must be one of: {list(pipeline_map.keys())}"
            )
        
        return pipeline_map[pipeline_type](config)
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _log_results(self, result: Any) -> None:
        """Log pipeline results to MLflow."""
        if hasattr(result, 'metrics'):
            self.mlflow_client.log_metrics(result.metrics)
        
        if hasattr(result, 'model') and hasattr(result.model, 'hyperparameters'):
            self.mlflow_client.log_params(result.model.hyperparameters)
    
    def _save_artifacts(self, result: Any, pipeline_type: str) -> None:
        """Save pipeline artifacts."""
        if hasattr(result, 'model'):
            # Save model
            model_path = Path("models") / f"{result.model.name}_{pipeline_type}.pth"
            result.model.save(model_path)
            
            # Log model as artifact
            self.mlflow_client.log_artifact(str(model_path), "models")
        
        # Save metrics
        if hasattr(result, 'metrics'):
            import json
            metrics_path = Path("metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(result.metrics, f, indent=2)
            
            self.mlflow_client.log_artifact(str(metrics_path))
    
    def run_experiment_series(
        self,
        experiment_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run a series of experiments.
        
        Args:
            experiment_configs: List of experiment configurations.
            
        Returns:
            List of experiment results.
        """
        results = []
        
        for exp_config in experiment_configs:
            logger.info(f"Running experiment: {exp_config.get('name', 'unnamed')}")
            
            try:
                pipeline_type = exp_config.get("pipeline_type", "train")
                config_path = Path(exp_config["config_path"])
                
                result = self.run_pipeline(pipeline_type, config_path)
                results.append({
                    "experiment": exp_config.get("name"),
                    "success": True,
                    **result
                })
                
            except Exception as e:
                logger.error(f"Experiment failed: {str(e)}")
                results.append({
                    "experiment": exp_config.get("name"),
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def compare_experiments(
        self,
        experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of MLflow run IDs.
            
        Returns:
            Comparison results.
        """
        comparison = {
            "experiments": [],
            "best_experiment": None,
            "comparison_metrics": {}
        }
        
        best_metric = float('inf')
        
        for run_id in experiment_ids:
            run_info = self.mlflow_client.get_run(run_id)
            metrics = self.mlflow_client.get_run_metrics(run_id)
            
            experiment_data = {
                "run_id": run_id,
                "metrics": metrics,
                "params": run_info.data.params,
                "tags": run_info.data.tags
            }
            
            comparison["experiments"].append(experiment_data)
            
            # Find best experiment based on validation loss
            val_loss = metrics.get("val_loss", float('inf'))
            if val_loss < best_metric:
                best_metric = val_loss
                comparison["best_experiment"] = run_id
        
        return comparison