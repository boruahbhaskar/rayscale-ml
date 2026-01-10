#!/bin/bash
# cleanup.sh - Clean up ML resources before leaving

echo "🧹 Cleaning up ML resources..."

# Stop Ray
echo "Stopping Ray..."
ray stop 2>/dev/null || true
pkill -f "ray" 2>/dev/null || true

# Stop MLflow server
echo "Stopping MLflow server..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true

# Kill any remaining Python ML processes
echo "Killing remaining ML processes..."
pkill -f "uvicorn.*serving" 2>/dev/null || true
pkill -f "mlflow" 2>/dev/null || true
pkill -f "tensorboard" 2>/dev/null || true

# Optional: Clear cache
echo "Clearing cache..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
python -c "import gc; gc.collect()"

echo "✅ Cleanup complete!"