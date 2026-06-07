from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.hardware import count_model_parameters, detect_device
from src.utils.io import ensure_dir, project_root


class ForecastLSTM(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int, num_classes: int = 0) -> None:
        super().__init__()
        self.class_embedding = nn.Embedding(num_classes, 8) if num_classes > 0 else None
        effective_input = input_size + (8 if self.class_embedding is not None else 0)
        self.lstm = nn.LSTM(effective_input, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.class_embedding is not None and labels is not None:
            embedded = self.class_embedding(labels).unsqueeze(1).repeat(1, x.size(1), 1)
            x = torch.cat([x, embedded], dim=2)
        outputs, _ = self.lstm(x)
        return self.head(outputs[:, -1])


@dataclass
class ForecastOutput:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    model_path: Path
    parameter_count: int
    mean_inference_time: float


def _loader(split: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.tensor(split["x"], dtype=torch.float32)
    y = torch.tensor(split["targets"], dtype=torch.float32)
    labels = torch.tensor(split["y"], dtype=torch.long)
    return DataLoader(TensorDataset(x, y, labels), batch_size=batch_size, shuffle=shuffle)


def _train_single_model(
    name: str,
    model: ForecastLSTM,
    use_class_labels: bool,
    train_split: dict[str, np.ndarray],
    val_split: dict[str, np.ndarray],
    test_split: dict[str, np.ndarray],
    config: dict[str, Any],
) -> ForecastOutput:
    device = torch.device(detect_device(prefer_gpu=True))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = nn.MSELoss()
    train_loader = _loader(train_split, int(config["batch_size"]), shuffle=True)
    val_loader = _loader(val_split, int(config["batch_size"]), shuffle=False)
    test_loader = _loader(test_split, int(config["batch_size"]), shuffle=False)
    best_state = None
    best_val = float("inf")
    patience_left = int(config["patience"])

    for _ in range(int(config["epochs"])):
        model.train()
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad()
            predictions = model(xb, lb if use_class_labels else None)
            loss = criterion(predictions, yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, lb in val_loader:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                predictions = model(xb, lb if use_class_labels else None)
                val_loss += float(criterion(predictions, yb).item()) * len(xb)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_left = int(config["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    total_time = 0.0
    with torch.no_grad():
        for xb, yb, lb in test_loader:
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            xb = xb.to(device)
            lb = lb.to(device)
            predictions = model(xb, lb if use_class_labels else None)
            if device.type == "cuda":
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end) / 1000.0
            y_pred.append(predictions.cpu().numpy())
            y_true.append(yb.numpy())

    model_dir = ensure_dir(project_root() / "experiments" / "runs")
    model_path = model_dir / f"{name}.pt"
    torch.save(model.state_dict(), model_path)
    mean_inference_time = total_time / max(len(test_loader.dataset), 1) if total_time > 0 else 0.0
    return ForecastOutput(
        name=name,
        y_true=np.concatenate(y_true),
        y_pred=np.concatenate(y_pred),
        model_path=model_path,
        parameter_count=count_model_parameters(model) or 0,
        mean_inference_time=mean_inference_time,
    )


def train_forecasting_models(
    train_split: dict[str, np.ndarray],
    val_split: dict[str, np.ndarray],
    test_split: dict[str, np.ndarray],
    config: dict[str, Any],
) -> list[ForecastOutput]:
    input_size = train_split["x"].shape[2]
    output_size = train_split["targets"].shape[1]
    num_classes = int(np.max(train_split["y"])) + 1
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    baseline = ForecastLSTM(input_size, output_size, hidden_size, num_layers, num_classes=0)
    class_aware = ForecastLSTM(input_size, output_size, hidden_size, num_layers, num_classes=num_classes)
    return [
        _train_single_model("lstm_baseline", baseline, False, train_split, val_split, test_split, config),
        _train_single_model("lstm_class_aware", class_aware, True, train_split, val_split, test_split, config),
    ]
