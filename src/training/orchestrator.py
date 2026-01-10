"""Training orchestration and workflow management."""

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.artifacts.mlflow_client import MLflowClient
from src.config import settings
from src.training.pipelines.base_pipeline import BaseTrainingPipeline, PipelineConfig
from src.training.pipelines.ray_pipeline import RayTrainingPipeline
from src.training.pipelines.tune_pipeline import TunePipeline


class TrainingOrchestrator:
    """Orchestrates training pipelines and experiments."""

    def __init__(self, experiment_name: str | None = None):
        """
        Initialize orchestrator.

        Args:
            experiment_name: Name of the experiment.
        """
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.mlflow_client = MLflowClient()
        self.current_run = None

    # def run_pipeline(
    #     self, pipeline_type: str, config_path: Path, **kwargs
    # ) -> dict[str, Any]:
    #     """
    #     Run a training pipeline.

    #     Args:
    #         pipeline_type: Type of pipeline ("train", "tune").
    #         config_path: Path to pipeline configuration.
    #         **kwargs: Additional pipeline arguments.

    #     Returns:
    #         Pipeline results.

    #     Raises:
    #         ValueError: If pipeline type is invalid.
    #     """
    #     logger.info(f"Running {pipeline_type} pipeline with config: {config_path}")

    #     # Load configuration
    #     config = self._load_config(config_path)
    #     pipeline_config = PipelineConfig.from_dict(config)

    #     # Create pipeline
    #     pipeline = self._create_pipeline(pipeline_type, pipeline_config)

    #     # Start MLflow run
    #     self.current_run = self.mlflow_client.start_run(
    #         experiment_name=self.experiment_name,
    #         run_name=f"{pipeline_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    #         tags={"pipeline_type": pipeline_type, "config_file": config_path.name},
    #     )

    #     try:
    #         # Log configuration
    #         self.mlflow_client.log_params(config)

    #         # Run pipeline
    #         result = pipeline.run()

    #         # Log results
    #         self._log_results(result)

    #         # Save artifacts
    #         self._save_artifacts(result, pipeline_type)

    #         logger.info(f"{pipeline_type} pipeline completed successfully")
    #         return {
    #             "success": True,
    #             "result": result,
    #             "run_id": self.current_run.info.run_id,
    #         }

    #     except Exception as e:
    #         logger.error(f"Pipeline failed: {str(e)}")

    #         # Log failure
    #         self.mlflow_client.log_params({"status":"failed"})
    #         self.mlflow_client.log_params({"error": str(e)})

    #         raise

    #     finally:
    #         # End MLflow run
    #         if self.current_run:
    #             import mlflow
    #             mlflow.end_run()
    #             #self.mlflow_client.end_run()
    #             self.current_run = None

    #         # Cleanup pipeline resources
    #         if hasattr(pipeline, 'cleanup'):
    #             pipeline.cleanup()

    def run_pipeline(self, pipeline_type: str, config_path: Path, **kwargs) -> dict[str, Any]:
        """
        Run a training pipeline with safe MLflow handling.
        """
        logger.info(f"Running {pipeline_type} pipeline with config: {config_path}")

        import mlflow
        
        # Load configuration
        config = self._load_config(config_path)
        pipeline_config = PipelineConfig.from_dict(config)
        
        # Initialize MLflow with safe defaults
        mlflow_enabled = True
        mlflow_run = None
        run_info = {}
        
        try:
            # Try to setup MLflow
            tracking_uri = getattr(settings, 'mlflow_tracking_uri', 'sqlite:///mlflow.db')
            mlflow.set_tracking_uri(tracking_uri)
            
            # Get or create experiment
            experiment_name = getattr(settings, 'mlflow_experiment_name', self.experiment_name)
            
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created MLflow experiment: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")
                
                # Start MLflow run
                mlflow_run = mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=f"{pipeline_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                # Log metadata
                mlflow.set_tags({
                    "pipeline_type": pipeline_type,
                    "config_file": config_path.name,
                    "framework": "ray",
                    "environment": settings.environment,
                })
                
                # Log configuration
                mlflow.log_params(self._flatten_dict(config))
                mlflow.log_artifact(str(config_path))
                
            except Exception as mlflow_error:
                logger.warning(f"MLflow setup failed: {mlflow_error}")
                logger.info("Continuing without MLflow tracking")
                mlflow_enabled = True
            
            # Create pipeline
            pipeline = self._create_pipeline(pipeline_type, pipeline_config)
            
            # Pass MLflow info if available
            if mlflow_enabled and mlflow_run:
                pipeline.mlflow_run_id = mlflow_run.info.run_id
            
            # IMPORTANT: Disable MLflow in Ray callbacks
            # FIX: Check if pipeline has config attribute and handle it properly
            if hasattr(pipeline, 'config'):
                # Check what type of config we have
                if hasattr(pipeline.config, 'dict'):  # Pydantic model
                    # Convert to dictionary to modify
                    config_dict = pipeline.config.dict()
                    if 'training' not in config_dict:
                        config_dict['training'] = {}
                    config_dict['training']['enable_mlflow'] = False
                    # Convert back if possible, otherwise keep as dict
                    try:
                        pipeline.config = pipeline.config.__class__(**config_dict)
                    except:
                        pipeline.config = config_dict
                elif isinstance(pipeline.config, dict):  # Already a dict
                    if 'training' not in pipeline.config:
                        pipeline.config['training'] = {}
                    pipeline.config['training']['enable_mlflow'] = False
                else:  # Some other object
                    # Try to access training attribute
                    try:
                        # Check if config has training attribute
                        if not hasattr(pipeline.config, 'training'):
                            # Create training attribute
                            setattr(pipeline.config, 'training', {})
                        # Get training config
                        training_config = getattr(pipeline.config, 'training')
                        if not isinstance(training_config, dict):
                            training_config = {}
                        training_config['enable_mlflow'] = False
                        setattr(pipeline.config, 'training', training_config)
                    except Exception as e:
                        logger.warning(f"Could not disable MLflow in pipeline config: {e}")
                        # Continue anyway - it's not critical
            
            # Run pipeline
            result = pipeline.run()
            
            # Log results to MLflow if enabled
            if mlflow_enabled and mlflow_run:
                try:
                    if result and hasattr(result, 'metrics') and result.metrics:
                        metrics_to_log = {}
                        for key, value in result.metrics.items():
                            if isinstance(value, (int, float)):
                                metrics_to_log[key] = value
                        if metrics_to_log:
                            mlflow.log_metrics(metrics_to_log)
                    
                    # Store run info for return
                    run_info = {
                        "run_id": mlflow_run.info.run_id,
                        "experiment_id": mlflow_run.info.experiment_id,
                        "mlflow_uri": mlflow.get_tracking_uri(),
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to log results to MLflow: {e}")
            
            logger.info(f"{pipeline_type} pipeline completed successfully")
            
            return {
                "success": True,
                "result": result,
                "mlflow_enabled": mlflow_enabled,
                **run_info,
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            
            # Log failure to MLflow if enabled
            if mlflow_enabled and mlflow_run:
                try:
                    mlflow.log_params({"status": "failed"})
                    mlflow.log_param("error", str(e)[:500])
                except:
                    pass
            
            # Re-raise the exception
            raise

        finally:
            # End MLflow run if started
            if mlflow_enabled and mlflow_run:
                try:
                    mlflow.end_run()
                except:
                    pass
            
            # Cleanup pipeline resources
            if hasattr(pipeline, 'cleanup'):
                pipeline.cleanup()
                    
    def _create_pipeline(
        self, pipeline_type: str, config: PipelineConfig
    ) -> BaseTrainingPipeline:
        """Create pipeline instance."""
        pipeline_map = {"train": RayTrainingPipeline, "tune": TunePipeline}

        if pipeline_type not in pipeline_map:
            raise ValueError(
                f"Invalid pipeline type: {pipeline_type}. "
                f"Must be one of: {list(pipeline_map.keys())}"
            )

        return pipeline_map[pipeline_type](config)

    def _load_config(self, config_path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
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
        self, experiment_configs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                results.append(
                    {"experiment": exp_config.get("name"), "success": True, **result}
                )

            except Exception as e:
                logger.error(f"Experiment failed: {str(e)}")
                results.append(
                    {
                        "experiment": exp_config.get("name"),
                        "success": False,
                        "error": str(e),
                    }
                )

        return results
    

    def _flatten_dict(self, d, parent_key='', sep='.'):
        """
        Flatten a nested dictionary for MLflow parameter logging.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested dictionaries
            sep: Separator for nested keys
        
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert non-serializable values to strings
                if not isinstance(v, (str, int, float, bool)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)

    def compare_experiments(self, experiment_ids: list[str]) -> dict[str, Any]:
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
            "comparison_metrics": {},
        }

        best_metric = float('inf')

        for run_id in experiment_ids:
            run_info = self.mlflow_client.get_run(run_id)
            metrics = self.mlflow_client.get_run_metrics(run_id)

            experiment_data = {
                "run_id": run_id,
                "metrics": metrics,
                "params": run_info.data.params,
                "tags": run_info.data.tags,
            }

            comparison["experiments"].append(experiment_data)

            # Find best experiment based on validation loss
            val_loss = metrics.get("val_loss", float('inf'))
            if val_loss < best_metric:
                best_metric = val_loss
                comparison["best_experiment"] = run_id

        return comparison
