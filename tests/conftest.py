"""Test configuration and fixtures."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray

from src.utils.logging import configure_logging


@pytest.fixture(scope="session")
def test_dir():
    """Create temporary test directory."""
    test_dir = Path(tempfile.mkdtemp(prefix="rayscale_test_"))
    yield test_dir
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def ray_init():
    """Initialize Ray for testing."""
    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)
    yield
    ray.shutdown()


@pytest.fixture
def synthetic_data():
    """Generate synthetic test data."""
    np.random.seed(42)

    n_samples = 1000
    n_features = 4

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate target
    coefficients = np.random.randn(n_features)
    y = X @ coefficients + np.random.randn(n_samples) * 0.1

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["target"] = y
    df["id"] = range(n_samples)

    return df


@pytest.fixture
def preprocessed_data(synthetic_data):
    """Create preprocessed test data."""
    df = synthetic_data.copy()

    # Normalize features
    for i in range(4):
        col = f"feature_{i+1}"
        df[f"{col}_norm"] = (df[col] - df[col].mean()) / df[col].std()

    return df


@pytest.fixture
def ray_dataset(preprocessed_data, ray_init):
    """Create Ray dataset for testing."""
    import ray.data as rd

    return rd.from_pandas(preprocessed_data)


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    configure_logging(log_level="WARNING")


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow for testing."""
    mock_runs = []
    mock_metrics = {}
    mock_params = {}

    class MockRun:
        def __init__(self, run_id="test_run"):
            self.info = type('obj', (object,), {'run_id': run_id})()
            self.data = type(
                'obj',
                (object,),
                {
                    'metrics': mock_metrics.get(run_id, {}),
                    'params': mock_params.get(run_id, {}),
                    'tags': {},
                },
            )()

    class MockMlflowClient:
        def create_run(self, *args, **kwargs):
            run_id = f"run_{len(mock_runs)}"
            mock_runs.append(run_id)
            return MockRun(run_id)

        def log_metric(self, run_id, key, value, *args, **kwargs):
            if run_id not in mock_metrics:
                mock_metrics[run_id] = {}
            mock_metrics[run_id][key] = value

        def log_param(self, run_id, key, value):
            if run_id not in mock_params:
                mock_params[run_id] = {}
            mock_params[run_id][key] = value

        def set_terminated(self, run_id, *args, **kwargs):
            pass

    monkeypatch.setattr("src.artifacts.mlflow_client.MLflowClient", MockMlflowClient)

    return MockMlflowClient()
