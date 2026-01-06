"""Base trainer interface for the ML platform."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.models.base import BaseModel


@dataclass
class TrainingResult:
    """Results from model training."""

    model: BaseModel
    metrics: dict[str, float]
    training_time: float
    artifacts: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model.name,
            "metrics": self.metrics,
            "training_time": self.training_time,
            "artifacts": list(self.artifacts.keys()),
            "metadata": self.metadata,
        }


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""

    def __init__(self, model: BaseModel, **kwargs):
        """
        Initialize trainer.

        Args:
            model: Model to train.
            **kwargs: Trainer configuration.
        """
        self.model = model
        self.config = kwargs
        self.training_result: TrainingResult | None = None
        self._callbacks = []

    def add_callback(self, callback: Any) -> None:
        """
        Add training callback.

        Args:
            callback: Callback instance.
        """
        self._callbacks.append(callback)
        logger.debug(f"Added callback: {callback.__class__.__name__}")

    def _execute_callbacks(self, event: str, **kwargs) -> None:
        """
        Execute all callbacks for an event.

        Args:
            event: Event name.
            **kwargs: Event data.
        """
        for callback in self._callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)

    @abstractmethod
    def prepare_data(self, data: Any) -> tuple[Any, Any | None]:
        """
        Prepare data for training.

        Args:
            data: Raw data.

        Returns:
            Tuple of (train_data, validation_data).
        """
        pass

    @abstractmethod
    def train(self, data: Any, **kwargs) -> TrainingResult:
        """
        Train model.

        Args:
            data: Training data.
            **kwargs: Additional training arguments.

        Returns:
            Training result.
        """
        pass

    def evaluate(self, test_data: Any) -> dict[str, float]:
        """
        Evaluate trained model.

        Args:
            test_data: Test data.

        Returns:
            Evaluation metrics.

        Raises:
            ValueError: If model is not trained.
        """
        if self.training_result is None:
            raise ValueError("Model must be trained before evaluation")

        return self.model.evaluate(test_data)

    def save(self, path: Path) -> None:
        """
        Save trainer state.

        Args:
            path: Path to save trainer.
        """
        raise NotImplementedError("Subclasses must implement save method")

    @classmethod
    def load(cls, path: Path) -> "BaseTrainer":
        """
        Load trainer state.

        Args:
            path: Path to load trainer from.

        Returns:
            Loaded trainer instance.
        """
        raise NotImplementedError("Subclasses must implement load method")


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, loss: float, model: Any) -> bool:
        """
        Callback for epoch end.

        Args:
            epoch: Current epoch.
            loss: Current loss.
            model: Model instance.

        Returns:
            True if training should stop.
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_weights = self._get_model_weights(model)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    self._set_model_weights(model, self.best_weights)
                return True

        return False

    def _get_model_weights(self, model: Any) -> Any:
        """Get model weights."""

        if hasattr(model, 'state_dict'):
            return model.state_dict().copy()
        elif hasattr(model, 'get_weights'):
            return model.get_weights()
        else:
            return None

    def _set_model_weights(self, model: Any, weights: Any) -> None:
        """Set model weights."""

        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(weights)
        elif hasattr(model, 'set_weights'):
            model.set_weights(weights)


class LearningRateScheduler:
    """Learning rate scheduler callback."""

    def __init__(
        self,
        initial_lr: float = 0.001,
        decay_factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
    ):
        """
        Initialize learning rate scheduler.

        Args:
            initial_lr: Initial learning rate.
            decay_factor: Factor to decay learning rate.
            patience: Number of epochs to wait before decaying.
            min_lr: Minimum learning rate.
        """
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.min_lr = min_lr

        self.current_lr = initial_lr
        self.wait = 0
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch: int, loss: float, optimizer: Any) -> float:
        """
        Callback for epoch end.

        Args:
            epoch: Current epoch.
            loss: Current loss.
            optimizer: Optimizer instance.

        Returns:
            Updated learning rate.
        """
        if loss < self.best_loss - 1e-4:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                self._update_optimizer_lr(optimizer, self.current_lr)
                self.wait = 0
                logger.info(f"Learning rate reduced to {self.current_lr}")

        return self.current_lr

    def _update_optimizer_lr(self, optimizer: Any, lr: float) -> None:
        """Update optimizer learning rate."""
        import torch

        if isinstance(optimizer, torch.optim.Optimizer):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif hasattr(optimizer, 'learning_rate'):
            optimizer.learning_rate = lr
