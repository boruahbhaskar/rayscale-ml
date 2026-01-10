# test_data_loading.py
import sys
sys.path.append('.')
import ray
from src.data.feature_store import FeatureStore

ray.init()

feature_store = FeatureStore()
dataset = feature_store.read_dataset(name="features", version=None, split=None)

print(f"Dataset type: {type(dataset)}")
print(f"Dataset: {dataset}")

if hasattr(dataset, 'take'):
    print("Sample rows:")
    for row in dataset.take(2):
        print(row)