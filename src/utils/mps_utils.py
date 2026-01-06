"""Apple Silicon (MPS) utilities for PyTorch."""

import platform
import torch
from loguru import logger


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon.
    
    Returns:
        True if running on Apple Silicon.
    """
    return platform.processor() == "arm"


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.
    
    Returns:
        True if MPS is available.
    """
    if not is_apple_silicon():
        return False
    
    return torch.backends.mps.is_available()


def configure_for_mps() -> bool:
    """
    Configure PyTorch for MPS on Apple Silicon.
    
    Returns:
        True if MPS was configured successfully.
    """
    if not is_apple_silicon():
        logger.debug("Not running on Apple Silicon, skipping MPS configuration")
        return False
    
    if not is_mps_available():
        logger.warning(
            "Apple Silicon detected but MPS not available. "
            "Make sure you have PyTorch with MPS support."
        )
        return False
    
    # Configure MPS
    try:
        # Set cache limit (4GB)
        torch.backends.mps.set_cache_limit(4 * 1024**3)
        
        # Set memory fraction if needed
        # torch.backends.mps.set_memory_fraction(0.8)
        
        logger.info("MPS configured successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to configure MPS: {e}")
        return False


def get_device() -> torch.device:
    """
    Get appropriate device for Apple Silicon.
    
    Returns:
        Torch device.
    """
    from src.config import settings
    
    if settings.use_mps and is_mps_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def optimize_for_mps(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model for MPS.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Optimized model.
    """
    device = get_device()
    
    if device.type == "mps":
        # Apply MPS-specific optimizations
        logger.debug("Applying MPS optimizations")
        
        # Move model to MPS device
        model = model.to(device)
        
        # Set eval mode if not training (can improve performance)
        if not model.training:
            model.eval()
    
    return model


def mps_memory_info() -> dict:
    """
    Get MPS memory information.
    
    Returns:
        Dictionary with memory information.
    """
    if not is_mps_available():
        return {"available": False}
    
    try:
        import torch.mps
        
        return {
            "available": True,
            "allocated_memory": torch.mps.current_allocated_memory(),
            "driver_allocated_memory": torch.mps.driver_allocated_memory(),
            "has_unified_memory": torch.mps.has_unified_memory(),
            "recommended_max_memory": torch.mps.recommended_max_memory()
        }
    except Exception as e:
        logger.warning(f"Could not get MPS memory info: {e}")
        return {"available": True, "error": str(e)}


def print_device_info() -> None:
    """Print device information."""
    device = get_device()
    
    logger.info(f"Using device: {device}")
    
    if device.type == "mps":
        memory_info = mps_memory_info()
        logger.info(f"MPS Memory Info: {memory_info}")
    
    elif device.type == "cuda":
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA Memory: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated")
    
    else:
        logger.info("Using CPU")


def benchmark_mps(model: torch.nn.Module, input_shape: tuple, num_iterations: int = 100) -> dict:
    """
    Benchmark model performance on MPS vs CPU.
    
    Args:
        model: PyTorch model.
        input_shape: Input tensor shape.
        num_iterations: Number of iterations for benchmarking.
        
    Returns:
        Benchmark results.
    """
    import time
    
    results = {}
    
    # Benchmark on CPU
    device_cpu = torch.device("cpu")
    model_cpu = model.to(device_cpu)
    input_cpu = torch.randn(input_shape, device=device_cpu)
    
    # Warmup
    for _ in range(10):
        _ = model_cpu(input_cpu)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model_cpu(input_cpu)
    cpu_time = time.time() - start_time
    
    results["cpu"] = {
        "time_per_iteration": cpu_time / num_iterations,
        "total_time": cpu_time
    }
    
    # Benchmark on MPS if available
    if is_mps_available():
        device_mps = torch.device("mps")
        model_mps = model.to(device_mps)
        input_mps = input_cpu.to(device_mps)
        
        # Warmup
        for _ in range(10):
            _ = model_mps(input_mps)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model_mps(input_mps)
        mps_time = time.time() - start_time
        
        results["mps"] = {
            "time_per_iteration": mps_time / num_iterations,
            "total_time": mps_time,
            "speedup": cpu_time / mps_time
        }
    
    return results