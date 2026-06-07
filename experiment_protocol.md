# Experiment Protocol

## Executed experiments

1. Trajectory clustering comparison on medium-noise trajectories.
2. Segment classification with a 1D CNN.
3. Forecasting with baseline and class-aware LSTM variants.
4. Point cloud trajectory recovery on clean, medium, and hard data.
5. Embedding clustering with ResNet-like encoder ablations.

## Compared methods

- Clustering: TRACLUS-like, ST-DBSCAN, Vector Field k-Means, spatio-temporal clustering, ResNet embeddings variants.
- Classification: CNN segment classifier.
- Forecasting: baseline LSTM and class-aware LSTM.
- Point cloud: sparse estimation, Cluster Filter, CL-Det-like tracking.

## Metrics

- Clustering: ARI, NMI, purity, cluster accuracy, macro-F1, silhouette, noise ratio.
- Classification: accuracy, precision, recall, macro-F1, confusion matrix.
- Forecasting: MAE, MSE, RMSE, ADE, FDE, per-horizon error.
- Tracking: position RMSE, detection rate, false positive rate, fragmentation, FPS.
- Resources: runtime, peak RAM, peak VRAM, parameter count, mean inference time.

## Reproducibility

- Random seed is loaded from YAML configs and applied to Python, NumPy, and PyTorch.
- Generated configs are saved alongside dataset metadata and run logs.

## Artifact handling

- Metrics are saved to CSV and summarized in JSON logs.
- Figures are written to `results/figures/` and copied to `report_assets/figures/`.
- Comparison tables are mirrored into `report_assets/tables/`.
