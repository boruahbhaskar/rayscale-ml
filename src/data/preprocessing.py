#Stateless transforms

import ray
import ray.data as rd
import os

ray.init(ignore_reinit_error=True)

RAW_PATH = "data/raw/synthetic_data.parquet"
PROCESSED_PATH = "data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

#Load data with Ray Data
ds = rd.read_parquet(RAW_PATH)

def preprocess(batch):
    batch["feature_1_norm"] = (batch["feature_1"] - batch["feature_1"].mean()) / batch["feature_1"].std()
    batch["feature_2_norm"] = (batch["feature_2"] - batch["feature_2"].mean()) / batch["feature_2"].std()
    return batch

ds = ds.map_batches(preprocess, batch_format="pandas")
output_file = os.path.join(PROCESSED_PATH,"preprocessed_data.parquet")
ds.write_parquet(output_file)
print(f"Preprocessed data saved to {output_file}")
