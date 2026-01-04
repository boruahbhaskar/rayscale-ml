import ray
import ray.data as rd
import torch
from torch import nn as nn
import torch.optim as optim

from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.air.config import RunConfig

# from ray.air import session,CheckpointConfig
# import tempfile
# import os

from src.config.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_schema
)


FEATURES_DIR = "data/processed/features"


# -------------------------
# Torch model
# -------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net == nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )

    def forward(self,x):
        return self.net(x)


# -------------------------
# Training Loop
# -------------------------


def train_loop_per_worker(config):
    dataset = rd.read_parquet(FEATURES_DIR)
    validate_schema(dataset)

    train_ds, val_ds = dataset.train_test_split(test_size=0.2)

    model = MLP()
    optimizer = optim.Adam(model.parameters(), lr = config["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(config["epochs"]):
        for batch in train_ds.iter_torch_batches(
            batch_size=1024,
            dtypes=torch.float32,
        ):
            x = torch.stack([batch[c] for c in FEATURE_COLUMNS], dim=1)
            y = batch[TARGET_COLUMN].unsqueeze(1)

            preds = model(x)
            loss = loss_fn(preds,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Loss {loss.item():.4f}")  

        # swap Torch <--> TensorFlow

        # for batch in train_ds.iter_tf_batches(
        #     batch_size=1024,
        #     dtypes=tf.float32,
        # ):
        #     x = tf.stack([batch[c] for c in FEATURE_COLUMNS], axis=1)
        #     y = batch[TARGET_COLUMN]
 


# -------------------------
# Trainer entry point
# -------------------------

def main():
    ray.init()

    trainer = TorchTrainer(
        train_loop_per_worker= train_loop_per_worker,
        train_loop_config={"lr": 1e-3, "epochs": 5},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        run_config=RunConfig(name="torch-train"),
    )

    trainer.fit()



if __name__ == "__manin__":
    main()
