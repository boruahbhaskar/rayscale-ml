"""Tabular neural network architectures."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import PyTorchModel


class TabularMLP(nn.Module):
    """
    Multi-layer perceptron for tabular data.

    This architecture is suitable for structured/tabular data
    and includes features like batch normalization and dropout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = None,
        output_size: int = 1,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize MLP.

        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Number of output units.
            dropout_rate: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
            activation: Activation function.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32, 16]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        # Activation function
        activation_map = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }

        if activation not in activation_map:
            raise ValueError(
                f"Unknown activation: {activation}. "
                f"Must be one of: {list(activation_map.keys())}"
            )

        self.activation = activation_map[activation]

        # Build layers
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(self.activation)

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor.
        """
        return self.network(x)

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation": str(self.activation),
        }


class ResidualMLP(TabularMLP):
    """
    MLP with residual connections.

    Residual connections can help with gradient flow
    in deeper networks.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = None,
        output_size: int = 1,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        """Initialize residual MLP."""
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            activation=activation,
        )

        # Create residual blocks
        self.blocks = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            block = self._create_residual_block(prev_size, hidden_size)
            self.blocks.append(block)
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

    def _create_residual_block(self, input_size: int, output_size: int) -> nn.Module:
        """Create a residual block."""
        layers = []

        # First linear layer
        layers.append(nn.Linear(input_size, output_size))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(output_size))
        layers.append(self.activation)

        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))

        # Second linear layer (optional)
        if output_size == input_size:
            # Skip connection with same dimension
            layers.append(nn.Linear(output_size, output_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(output_size))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        residual = x

        for block in self.blocks:
            out = block(residual)

            # Skip connection if dimensions match
            if out.shape == residual.shape:
                out = out + residual

            residual = out

        return self.output_layer(residual)


class FeatureEmbeddingMLP(nn.Module):
    """
    MLP with feature embeddings for categorical features.

    This architecture is useful when you have a mix of
    numerical and categorical features.
    """

    def __init__(
        self,
        num_numerical: int,
        categorical_dims: dict[str, int],
        embedding_dims: dict[str, int] = None,
        hidden_sizes: list[int] = None,
        output_size: int = 1,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize feature embedding MLP.

        Args:
            num_numerical: Number of numerical features.
            categorical_dims: Dictionary mapping categorical feature names
                            to their cardinality (number of unique values).
            embedding_dims: Dictionary mapping categorical feature names
                          to their embedding dimensions.
            hidden_sizes: Hidden layer sizes.
            output_size: Output size.
            dropout_rate: Dropout rate.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        if embedding_dims is None:
            # Default embedding dimension heuristic
            embedding_dims = {
                name: min(50, (dim + 1) // 2) for name, dim in categorical_dims.items()
            }

        self.num_numerical = num_numerical
        self.categorical_dims = categorical_dims
        self.embedding_dims = embedding_dims

        # Create embedding layers
        self.embeddings = nn.ModuleDict()
        total_embedding_size = 0

        for name, dim in categorical_dims.items():
            embedding_dim = embedding_dims.get(name, min(50, (dim + 1) // 2))
            self.embeddings[name] = nn.Embedding(dim, embedding_dim)
            total_embedding_size += embedding_dim

        # Total input size = numerical + all embeddings
        total_input_size = num_numerical + total_embedding_size

        # Build MLP
        layers = []
        prev_size = total_input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, numerical: torch.Tensor, categorical: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            numerical: Numerical features tensor.
            categorical: Dictionary of categorical feature tensors.

        Returns:
            Output tensor.
        """
        # Embed categorical features
        embedded = []

        for name, tensor in categorical.items():
            if name in self.embeddings:
                embedding = self.embeddings[name](tensor)
                embedded.append(embedding)

        # Concatenate all features
        if embedded:
            all_embedded = torch.cat(embedded, dim=1)
            x = torch.cat([numerical, all_embedded], dim=1)
        else:
            x = numerical

        return self.mlp(x)


class TabularNNModel(PyTorchModel):
    """Tabular neural network model wrapper."""

    def __init__(self, name: str = "tabular_nn", architecture: str = "mlp", **kwargs):
        """
        Initialize tabular NN model.

        Args:
            name: Model name.
            architecture: Architecture type ("mlp", "residual", "embedding").
            **kwargs: Model hyperparameters.
        """
        super().__init__(name, **kwargs)
        self.architecture = architecture
        self.model = None

    def build(self) -> nn.Module:
        """
        Build tabular neural network.

        Returns:
            PyTorch model.
        """
        architecture_map = {
            "mlp": TabularMLP,
            "residual": ResidualMLP,
            "embedding": FeatureEmbeddingMLP,
        }

        if self.architecture not in architecture_map:
            raise ValueError(
                f"Unknown architecture: {self.architecture}. "
                f"Must be one of: {list(architecture_map.keys())}"
            )

        model_class = architecture_map[self.architecture]

        # Get hyperparameters
        input_size = self.hyperparameters.get("input_size", 4)

        if self.architecture == "embedding":
            # For embedding architecture, we need special handling
            num_numerical = self.hyperparameters.get("num_numerical", input_size)
            categorical_dims = self.hyperparameters.get("categorical_dims", {})

            self.model = model_class(
                num_numerical=num_numerical,
                categorical_dims=categorical_dims,
                **{
                    k: v
                    for k, v in self.hyperparameters.items()
                    if k not in ["input_size", "num_numerical", "categorical_dims"]
                },
            )
        else:
            # For MLP and residual architectures
            self.model = model_class(
                input_size=input_size,
                **{k: v for k, v in self.hyperparameters.items() if k != "input_size"},
            )

        return self.model.to(self.device)

    def train(
        self, train_data: Any, val_data: Any | None = None, **kwargs
    ) -> dict[str, float]:
        """
        Train model.

        Args:
            train_data: Training data.
            val_data: Validation data.
            **kwargs: Additional training arguments.

        Returns:
            Training metrics.
        """
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # Extract training arguments
        learning_rate = kwargs.get("learning_rate", 0.001)
        batch_size = kwargs.get("batch_size", 1024)
        num_epochs = kwargs.get("num_epochs", 10)
        weight_decay = kwargs.get("weight_decay", 0.0)

        # Prepare data
        if isinstance(train_data, tuple):
            X_train, y_train = train_data
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            )
        else:
            # Assume it's already a dataset
            train_dataset = train_data

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Prepare validation data if provided
        val_loader = None
        if val_data is not None:
            if isinstance(val_data, tuple):
                X_val, y_val = val_data
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val), torch.FloatTensor(y_val)
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )

        # Build model if not already built
        if self.model is None:
            self.model = self.build()

        # Setup optimizer and loss function
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        metrics_history = {"train_loss": [], "val_loss": []}

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / max(num_batches, 1)
            metrics_history["train_loss"].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._evaluate_on_loader(val_loader, criterion)
                metrics_history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss: {avg_train_loss:.4f}"
                )

        self._is_trained = True

        # Create final metrics
        final_metrics = {"final_train_loss": metrics_history["train_loss"][-1]}

        if val_loader is not None:
            final_metrics["final_val_loss"] = metrics_history["val_loss"][-1]

        # Update metadata
        self.update_metadata(
            metrics=final_metrics,
            hyperparameters={
                **self.hyperparameters,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "weight_decay": weight_decay,
            },
        )

        return final_metrics

    def _evaluate_on_loader(
        self, data_loader: DataLoader, criterion: nn.Module
    ) -> float:
        """
        Evaluate model on data loader.

        Args:
            data_loader: Data loader.
            criterion: Loss function.

        Returns:
            Average loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def predict(self, data: Any) -> np.ndarray:
        """
        Make predictions.

        Args:
            data: Input data.

        Returns:
            Predictions.

        Raises:
            ValueError: If model is not trained.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data_tensor = torch.FloatTensor(data).to(self.device)
            elif isinstance(data, torch.Tensor):
                data_tensor = data.to(self.device)
            else:
                raise TypeError(f"Unsupported data type: {type(data)}")

            predictions = self.model(data_tensor)

        return predictions.cpu().numpy()

    def evaluate(self, data: Any, labels: Any) -> dict[str, float]:
        """
        Evaluate model.

        Args:
            data: Test data.
            labels: True labels.

        Returns:
            Evaluation metrics.
        """
        import torch
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        predictions = self.predict(data)

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        metrics = {
            "mse": float(mean_squared_error(labels, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
            "mae": float(mean_absolute_error(labels, predictions)),
            "r2": float(r2_score(labels, predictions)),
        }

        return metrics

    def save(self, path: Path) -> None:
        """
        Save model.

        Args:
            path: Path to save model.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), path)

        # Save metadata
        metadata_path = path.with_suffix(".json")
        if self.metadata:
            with open(metadata_path, "w") as f:
                f.write(self.metadata.to_json())

        # Save hyperparameters
        config_path = path.with_suffix(".config.json")
        with open(config_path, "w") as f:
            json.dump(self.hyperparameters, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TabularNNModel":
        """
        Load model.

        Args:
            path: Path to load model from.

        Returns:
            Loaded model instance.
        """
        import json

        # Load hyperparameters
        config_path = path.with_suffix(".config.json")
        with open(config_path) as f:
            hyperparameters = json.load(f)

        # Create model instance
        architecture = hyperparameters.pop("architecture", "mlp")
        name = hyperparameters.pop("name", "loaded_tabular_nn")

        instance = cls(name=name, architecture=architecture, **hyperparameters)
        instance.model = instance.build()

        # Load model state
        instance.model.load_state_dict(torch.load(path, map_location=instance.device))
        instance._is_trained = True

        # Load metadata if available
        metadata_path = path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_dict = json.load(f)

            instance.metadata = ModelMetadata(**metadata_dict)

        return instance
