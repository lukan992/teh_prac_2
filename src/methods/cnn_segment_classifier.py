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


class SegmentCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, hidden_channels: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for hidden in hidden_channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                ]
            )
            in_channels = hidden
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@dataclass
class ClassificationOutput:
    y_true: np.ndarray
    y_pred: np.ndarray
    history: list[dict[str, float]]
    model_path: Path
    parameter_count: int
    mean_inference_time: float


def _make_loader(split: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    x = torch.tensor(split["x"], dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(split["y"], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def train_cnn_classifier(
    train_split: dict[str, np.ndarray],
    val_split: dict[str, np.ndarray],
    test_split: dict[str, np.ndarray],
    config: dict[str, Any],
) -> ClassificationOutput:
    device = torch.device(detect_device(prefer_gpu=True))
    num_classes = int(np.max(train_split["y"])) + 1
    model = SegmentCNN(train_split["x"].shape[2], num_classes, list(config["hidden_channels"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = _make_loader(train_split, int(config["batch_size"]), shuffle=True)
    val_loader = _make_loader(val_split, int(config["batch_size"]), shuffle=False)
    test_loader = _make_loader(test_split, int(config["batch_size"]), shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    patience_left = int(config["patience"])
    history: list[dict[str, float]] = []

    for epoch in range(int(config["epochs"])):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * len(xb)
        val_loss /= len(val_loader.dataset)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
        for xb, yb in test_loader:
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            xb = xb.to(device)
            if start is not None and end is not None:
                start.record()
            logits = model(xb)
            if start is not None and end is not None:
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end) / 1000.0
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.append(predictions)
            y_true.append(yb.numpy())

    model_dir = ensure_dir(project_root() / "experiments" / "runs")
    model_path = model_dir / "cnn_segment_classifier.pt"
    torch.save(model.state_dict(), model_path)
    num_samples = len(test_loader.dataset)
    mean_inference_time = total_time / max(num_samples, 1) if total_time > 0 else 0.0
    return ClassificationOutput(
        y_true=np.concatenate(y_true),
        y_pred=np.concatenate(y_pred),
        history=history,
        model_path=model_path,
        parameter_count=count_model_parameters(model) or 0,
        mean_inference_time=mean_inference_time,
    )
