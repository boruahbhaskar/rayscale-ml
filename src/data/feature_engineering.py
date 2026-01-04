import ray
import ray.data as rd
import os

ray.init(ignore_reinit_error=True)

INPUT_PATH = "data/processed/preprocessed_data.parquet"
OUTPUT_PATH = "data/processed/features"
os.makedirs(OUTPUT_PATH,exist_ok=True)

ds = rd.read_parquet(INPUT_PATH)


def feature_engineer(batch):
    # Example : feature interaction
    batch["feature_interaction"] = batch["feature_1_norm"] * batch["feature_2_norm"]
    return batch

ds = ds.map_batches(feature_engineer, batch_format="pandas")
output_file = os.path.join(OUTPUT_PATH, "features.parquet")
ds.write_parquet(output_file)
print(f"Features saved to {output_file}")



