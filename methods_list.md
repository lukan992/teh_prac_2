# Methods List

| Method | Source Idea | Input | Output | Metrics | Adaptation | Expected Limitations |
| --- | --- | --- | --- | --- | --- | --- |
| TRACLUS-like | Lee et al., 2007 | Full trajectories | Cluster labels | ARI, NMI, purity, macro-F1 | Simplified segmentation plus density clustering | Not a full partition-and-group implementation |
| ST-DBSCAN | Birant and Kut, 2007 | Spatio-temporal points | Cluster labels | ARI, NMI, silhouette, noise ratio | Direct adaptation to synthetic trajectory points | Sensitive to eps scaling |
| Vector Field k-Means | Ferreira et al., 2013 | Trajectory velocity fields | Cluster labels | ARI, NMI, macro-F1 | Grid-based velocity descriptor approximation | Loses fine local dynamics |
| Spatio-temporal clustering | Zhong et al., 2022 | Segment windows | Segment clusters | ARI, NMI, macro-F1 | Feature clustering instead of original full pipeline | Depends on hand-crafted features |
| CNN segment classifier | Adapted 1D CNN baseline | Segment tensors | Behavior class | Accuracy, macro-F1, confusion matrix | Lightweight architecture for synthetic windows | Generalization tied to synthetic distribution |
| LSTM predictor | Adapted LSTM forecasting | Segment sequences | Future coordinates | MAE, RMSE, ADE, FDE | Includes baseline and class-aware variants | Horizon quality degrades on hard motion changes |
| Sparse pointcloud estimation | Liang et al., 2025 | Point cloud frames | Estimated UAV centers | RMSE, detection rate, FPS | Simplified global-local clustering and smoothing | No end-to-end learned 3D model |
| Cluster Filter | Liang et al., 2024 | Point cloud frames | Estimated UAV centers | RMSE, detection rate, FPR | Heuristic score-based cluster selection | Simplified challenge setting |
| CL-Det-like | Xiao et al., 2024 | Point cloud frames | Estimated UAV centers | RMSE, detection rate, fragmentation | DBSCAN-based tracker with smoothing | Pose estimation is out of scope |
| ResNet embeddings + HDBSCAN | Adapted ResNet-style encoder | Trajectory images | Cluster labels | ARI, NMI, macro-F1, silhouette | Local 2D representation plus embedding clustering | Uses 2D rendering rather than full 3D ResNet |
