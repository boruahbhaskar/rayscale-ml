import pandas as pd
import numpy as np
import argparse
import os
import ray

ray.init(ignore_reinit_error=True)

@ray.remote
def generate_partition(n_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature_1": rng.normal(size=n_rows),
        "feature_2": rng.uniform(size=n_rows),
        "target": rng.integers(0, 2, size=n_rows)
    })

def main(rows: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    partitions = 10
    futures = [generate_partition.remote(rows // partitions , i) for i in range(partitions)]
    df = pd.concat(ray.get(futures))
    output_path = os.path.join(output_dir,"synthetic_data.parquet")
    df.to_parquet(output_path,index=False)
    print(f"Synthetic data saved to {output_path},  shape={df.shape}")


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows",type=int, default=1_000_000)
    parser.add_argument("--output_dir", type=str, default="data/raw")
    args = parser.parse_args()
    main(args.rows, args.output_dir)
