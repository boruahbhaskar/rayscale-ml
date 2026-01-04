import ray
import ray.data as rd
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from ray import tune
from ray.air import session
from ray.air.config import RunConfig,ScalingConfig
from ray.tune.search.optuna import OptunaSearch
from ray.train.torch import TorchTrainer

ray.init(ignore_reinit_error=True)

search_space = {
    "lr" : tune.loguniform(1e-4,1e-2),
    "batchsize": tune.choice([64,128, 256])
}


from src.config.schema import(
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_schema,
)

FEATURES_DIR = "data/processed/features"

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1),
        )

    def forward(self,x):
        return self.net(x)
    

def train_loop(config):
    dataset = rd.read_parquet(FEATURES_DIR)
    validate_schema(dataset)

    train_ds, val_ds = dataset.train_test_split(test_size=0.2)

    model = MLP(config["hidden_dim"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    mlflow.start_run()

    for epoch in range(5):
        epoch_loss = 0.0

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

            epoch_loss += loss.item()


        mlflow.log_metric("train_loss", epoch_loss, step = epoch)
        session.report({"loss": epoch_loss})

    mlflow.log_params(config)
    mlflow.end_run()        


def main():
    ray.init()

    tuner = tune.Tuner(
        TorchTrainer(
            train_loop_per_worker=train_loop,
            scaling_config= ScalingConfig(num_workers=2, use_gpu=False),
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=OptunaSearch(),
            num_samples=5,
        ),
        param_space={
            "lr": tune.loguniform(1e-4,1e-2),
            "hidden_dim": tune.choice([32,64,128]),
        },
        run_config=RunConfig(name="ray-tune-mlflow"),
    )

    tuner.fit()    


if __name__ =="__main__":
    main()