"""Base training pipeline interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.models.base import BaseModel
from src.models.trainers.base_trainer import TrainingResult


@dataclass
class PipelineConfig:
    """Configuration for training pipeline."""
    
    data_config: Dict[str, Any]
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    output_config: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(
            data_config=config_dict.get("data", {}),
            model_config=config_dict.get("model", {}),
            training_config=config_dict.get("training", {}),
            output_config=config_dict.get("output", {})
        )


class BaseTrainingPipeline(ABC):
    """Abstract base class for training pipelines."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.model: Optional[BaseModel] = None
        self.trainer = None
        self.results: Optional[TrainingResult] = None
        
    @abstractmethod
    def load_data(self) -> Any:
        """
        Load and prepare data.
        
        Returns:
            Prepared data.
        """
        pass
    
    @abstractmethod
    def create_model(self) -> BaseModel:
        """
        Create model instance.
        
        Returns:
            Model instance.
        """
        pass
    
    @abstractmethod
    def create_trainer(self) -> Any:
        """
        Create trainer instance.
        
        Returns:
            Trainer instance.
        """
        pass
    
    @abstractmethod
    def train(self) -> TrainingResult:
        """
        Execute training.
        
        Returns:
            Training result.
        """
        pass
    
    def run(self) -> TrainingResult:
        """
        Execute full training pipeline.
        
        Returns:
            Training result.
        """
        logger.info("Starting training pipeline")
        
        try:
            # Load data
            logger.info("Loading data")
            data = self.load_data()
            
            # Create model
            logger.info("Creating model")
            self.model = self.create_model()
            
            # Create trainer
            logger.info("Creating trainer")
            self.trainer = self.create_trainer()
            
            # Train
            logger.info("Starting training")
            self.results = self.train()
            
            # Save results
            self._save_results()
            
            logger.info("Training pipeline completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def _save_results(self) -> None:
        """Save training results."""
        if self.results is None:
            return
        
        output_dir = Path(self.config.output_config.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / f"{self.model.name}_model.pth"
        self.results.model.save(model_path)
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        import json
        with open(metrics_path, "w") as f:
            json.dump(self.results.metrics, f, indent=2)
        
        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.results.metadata, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate pipeline on test data.
        
        Args:
            test_data: Test data.
            
        Returns:
            Evaluation metrics.
        """
        if self.results is None:
            raise ValueError("Pipeline must be run before evaluation")
        
        return self.trainer.evaluate(test_data)