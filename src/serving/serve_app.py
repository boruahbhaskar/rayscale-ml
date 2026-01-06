import ray
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

ray.init(ignore_reinit_error=True)
app = FastAPI()

# Simple Linear model matching train_ray


class PredictorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 1)
        # Normally load state_dict

    def forward(self, x):
        return self.layer(x)


model = PredictorModel()
model.eval()


class InputData(BaseModel):
    feature_1: float
    feature_2: float
    feature_interaction: float = None


@app.post("/predict")
async def predict(data: InputData):
    # Compute interaction if not provided
    if data.feature_interaction is None:
        data.feature_interaction = data.feature_1 * data.feature_2

    x = torch.tensor(
        [[data.feature_1, data.feature_2, data.feature_interaction]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        pred = model(x).item()
        return {"prediction": float(pred)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
