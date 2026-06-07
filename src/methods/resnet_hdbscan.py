from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.methods.common import run_density_clustering
from src.utils.hardware import count_model_parameters, detect_device
from src.utils.io import ensure_dir, project_root


def trajectory_to_image(frame: pd.DataFrame, image_size: int) -> np.ndarray:
    coords = frame[["x", "y", "z"]].to_numpy()
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    normalized = (coords - mins) / span
    image = np.zeros((3, image_size, image_size), dtype=np.float32)
    xy = np.clip((normalized[:, :2] * (image_size - 1)).astype(int), 0, image_size - 1)
    for index, (ix, iy) in enumerate(xy):
        image[0, iy, ix] += 1.0
        image[1, iy, ix] = max(image[1, iy, ix], normalized[index, 2])
        image[2, iy, ix] = max(image[2, iy, ix], frame["speed"].iloc[index] / max(frame["speed"].max(), 1e-6))
    image /= max(image.max(), 1.0)
    return image


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + x)


class ResNetLikeEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer1 = ResidualBlock(32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).flatten(1)
        embedding = self.embedding(x)
        logits = self.classifier(torch.relu(embedding))
        return embedding, logits


@dataclass
class EmbeddingClusteringOutput:
    embeddings: np.ndarray
    trajectory_ids: np.ndarray
    y_true: np.ndarray
    results: dict[str, np.ndarray]
    model_path: Path
    parameter_count: int
    mean_inference_time: float
    notes: list[str]


def _trajectory_feature_baseline(frame: pd.DataFrame) -> np.ndarray:
    coords = frame[["x", "y", "z"]].to_numpy()
    speeds = frame["speed"].to_numpy()
    return np.array(
        [
            coords[:, 0].mean(),
            coords[:, 1].mean(),
            coords[:, 2].mean(),
            coords[:, 0].std(),
            coords[:, 1].std(),
            coords[:, 2].std(),
            np.linalg.norm(np.diff(coords, axis=0), axis=1).sum(),
            speeds.mean(),
            speeds.std(),
            frame["yaw"].std(),
        ],
        dtype=np.float32,
    )


def train_resnet_embedding_clusterer(trajectories: pd.DataFrame, labels_df: pd.DataFrame, config: dict[str, Any]) -> EmbeddingClusteringOutput:
    device = torch.device(detect_device(prefer_gpu=True))
    image_size = int(config["image_size"])
    rows = []
    images = []
    labels = []
    trajectory_ids = []
    handcrafted = []
    for trajectory_id, frame in trajectories.groupby("trajectory_id"):
        ordered = frame.sort_values("t").reset_index(drop=True)
        images.append(trajectory_to_image(ordered, image_size))
        labels.append(int(ordered["pattern_id"].iloc[0]))
        trajectory_ids.append(int(trajectory_id))
        handcrafted.append(_trajectory_feature_baseline(ordered))
        rows.append({"trajectory_id": int(trajectory_id)})
    x = torch.tensor(np.stack(images), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=int(config["batch_size"]), shuffle=True)

    model = ResNetLikeEncoder(int(config["embedding_dim"]), int(np.max(labels)) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = nn.CrossEntropyLoss()
    for _ in range(int(config["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            _, logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    embeddings = []
    total_time = 0.0
    eval_loader = DataLoader(TensorDataset(x, y), batch_size=int(config["batch_size"]), shuffle=False)
    with torch.no_grad():
        for xb, _ in eval_loader:
            xb = xb.to(device)
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            embedding, _ = model(xb)
            if device.type == "cuda":
                end.record()
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end) / 1000.0
            embeddings.append(embedding.cpu().numpy())
    embedding_matrix = np.vstack(embeddings)
    handcrafted_matrix = np.vstack(handcrafted)
    density_labels, density_notes = run_density_clustering(
        embedding_matrix,
        min_cluster_size=max(5, len(labels) // 120),
        eps=0.8,
        prefer_hdbscan=bool(config.get("use_hdbscan", True)),
    )
    handcrafted_labels, handcrafted_notes = run_density_clustering(
        handcrafted_matrix,
        min_cluster_size=max(5, len(labels) // 120),
        eps=0.8,
        prefer_hdbscan=bool(config.get("use_hdbscan", True)),
    )
    kmeans_labels = KMeans(n_clusters=len(np.unique(labels)), random_state=int(config["seed"]), n_init=10).fit_predict(
        embedding_matrix
    )
    model_dir = ensure_dir(project_root() / "experiments" / "runs")
    model_path = model_dir / "resnet_like_encoder.pt"
    torch.save(model.state_dict(), model_path)
    notes = [
        "Adapted ResNet-like encoder trained on synthetic trajectory images.",
        "Embeddings are produced from a supervised synthetic pretraining stage.",
    ]
    notes.extend(density_notes)
    notes.extend(handcrafted_notes)
    mean_inference = total_time / max(len(labels), 1) if total_time > 0 else 0.0
    return EmbeddingClusteringOutput(
        embeddings=embedding_matrix,
        trajectory_ids=np.array(trajectory_ids, dtype=int),
        y_true=np.array(labels, dtype=int),
        results={
            "handcrafted_hdbscan": handcrafted_labels,
            "resnet_kmeans": kmeans_labels.astype(int),
            "resnet_hdbscan": density_labels,
        },
        model_path=model_path,
        parameter_count=count_model_parameters(model) or 0,
        mean_inference_time=mean_inference,
        notes=notes,
    )
