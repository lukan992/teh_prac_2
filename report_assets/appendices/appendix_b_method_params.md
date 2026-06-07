# Appendix B. Method parameters

## Clustering methods
- TRACLUS-like: `min_samples=6`, `eps=1.35`, `curvature_threshold=0.35`
- ST-DBSCAN: `eps_spatial=4.0`, `eps_temporal=10.0`, `min_samples=6`
- Vector Field k-Means: `grid_size=12`, `n_clusters=9`
- Spatio-temporal clustering: `n_clusters=9`

## CNN parameters
- epochs=8, batch_size=64, learning_rate=0.001, weight_decay=0.0001, patience=3, hidden_channels=[32, 64, 128]

## LSTM parameters
- epochs=8, batch_size=64, learning_rate=0.001, hidden_size=96, num_layers=2, patience=3

## Point cloud / tracking parameters
- dbscan_eps=1.2, dbscan_min_samples=4, cluster_filter_voxel_size=1.0, smoothing_alpha=0.65

## ResNet-like parameters
- epochs=6, batch_size=64, learning_rate=0.001, embedding_dim=64, image_size=64, use_hdbscan=True
