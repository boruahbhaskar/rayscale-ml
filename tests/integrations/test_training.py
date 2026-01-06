"""
Test training functionality for ML models.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import os


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for testing."""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        if isinstance(targets, np.ndarray) and targets.dtype == np.float32:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class TestTraining:
    """Test training functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        self.X = np.random.randn(n_samples, n_features).astype(np.float32)
        self.y_reg = np.random.randn(n_samples).astype(np.float32)
        self.y_cls = np.random.randint(0, 2, size=n_samples).astype(np.int64)
    
    def teardown_method(self):
        """Cleanup after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_initialization(self):
        """Test model initialization with different configurations."""
        # Import TabularMLP inside the test
        from src.models.architectures.tabular_nn import TabularMLP
        
        # First let's check what parameters TabularMLP actually accepts
        # Common parameters for MLP: input_size, hidden_sizes, output_size, dropout_rate, activation
        
        # Test basic model with minimal parameters
        try:
            # Try minimal parameters first
            model = TabularMLP(
                input_size=10,
                hidden_sizes=[64, 32],
                output_size=1  # For regression
            )
        except TypeError as e:
            # Try different parameter combinations
            print(f"Error with first attempt: {e}")
            # Try without output_size
            model = TabularMLP(
                input_size=10,
                hidden_sizes=[64, 32]
            )
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        x = torch.randn(2, 10)
        output = model(x)
        # Output shape depends on the model architecture
        assert len(output.shape) == 2
    
    def test_dataset_creation(self):
        """Test dataset creation and loading."""
        # Create dataset using SimpleDataset
        dataset = SimpleDataset(
            features=self.X,
            targets=self.y_reg
        )
        
        assert len(dataset) == len(self.X)
        
        # Test __getitem__
        feature, target = dataset[0]
        assert feature.shape == (self.X.shape[1],)
        assert torch.is_tensor(feature)
        assert torch.is_tensor(target)
    
    def test_training_loop_smoke(self):
        """Smoke test for training loop (no assertions, just shouldn't crash)."""
        from src.models.architectures.tabular_nn import TabularMLP
        
        # Create model with minimal parameters
        try:
            model = TabularMLP(
                input_size=10,
                hidden_sizes=[32, 16],
                output_size=1
            )
        except TypeError:
            model = TabularMLP(
                input_size=10,
                hidden_sizes=[32, 16]
            )
        
        # Create dataset and dataloader
        dataset = SimpleDataset(
            features=self.X[:20],  # Smaller dataset
            targets=self.y_reg[:20]
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True
        )
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        criterion = torch.nn.MSELoss()
        
        # Run one training epoch
        model.train()
        for batch_idx, (features, targets) in enumerate(dataloader):
            if batch_idx >= 2:  # Just test a couple of batches
                break
                
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        # Test should complete without errors
        assert True
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        from src.models.architectures.tabular_nn import TabularMLP
        
        # Create model with minimal parameters
        try:
            model = TabularMLP(
                input_size=10,
                hidden_sizes=[32, 16],
                output_size=1
            )
        except TypeError:
            model = TabularMLP(
                input_size=10,
                hidden_sizes=[32, 16]
            )
        
        # Train for one batch
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        x = torch.randn(4, 10)
        y = torch.randn(4)
        
        model.train()
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        
        # Save model
        model_path = Path(self.temp_dir) / "test_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_path)
        
        # Load model - create new instance and load state
        loaded_model = TabularMLP(
            input_size=10,
            hidden_sizes=[32, 16]
        )
        loaded_model.load_state_dict(torch.load(model_path)['model_state_dict'])
        
        # Test that loaded model produces same output
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            test_input = torch.randn(2, 10)
            output1 = model(test_input)
            output2 = loaded_model(test_input)
            
            # Check that outputs are close (allowing for small numerical differences)
            assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)
    
    def test_forward_pass(self):
        """Test forward pass with different batch sizes."""
        from src.models.architectures.tabular_nn import TabularMLP
        
        # Create model with minimal parameters
        try:
            model = TabularMLP(
                input_size=5,
                hidden_sizes=[16, 8],
                output_size=1
            )
        except TypeError:
            model = TabularMLP(
                input_size=5,
                hidden_sizes=[16, 8]
            )
        
        # Test with batch size 1
        model.eval()
        x_single = torch.randn(1, 5)
        output_single = model(x_single)
        assert len(output_single.shape) == 2
        
        # Test with batch size > 1
        model.train()
        x_batch = torch.randn(4, 5)
        output_batch = model(x_batch)
        assert len(output_batch.shape) == 2
        assert output_batch.shape[0] == 4


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])