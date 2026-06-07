| method | runtime_seconds | peak_ram_mb | peak_vram_mb | parameter_count | mean_inference_time | peak_vram_gpu0_mb | peak_vram_gpu1_mb | peak_vram_total_mb | estimated_inference_time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TRACLUS-like | 0.24 | 1141.0 | 7215.883;308.0 | — | — | 7215.9 | 308.0 | 7523.9 | 0.0 |
| ST-DBSCAN | 0.25 | 1155.7 | 7215.883;308.0 | — | — | 7215.9 | 308.0 | 7523.9 | 0.0 |
| Vector Field k-Means | 1.7 | 1162.0 | 7220.57;308.0 | — | — | 7220.6 | 308.0 | 7528.6 | 0.01 |
| Spatio-temporal trajectory clustering | 0.12 | 1162.3 | 7221.07;308.0 | — | — | 7221.1 | 308.0 | 7529.1 | 0.0 |
| CNN-классификатор сегментов | 3.08 | 651.3 | 7220.57;308.0 | 33321.0 | 0.0 | 7220.6 | 308.0 | 7528.6 | 0.01 |
| LSTM baseline | 12.52 | 703.2 | 7221.07;308.0 | 116364.0 | 0.0 | 7221.1 | 308.0 | 7529.1 | 0.02 |
| Class-aware LSTM | 12.52 | 703.2 | 7221.07;308.0 | 119508.0 | 0.0 | 7221.1 | 308.0 | 7529.1 | 0.02 |
| Sparse point cloud trajectory estimation (clean) | 2.85 | 975.1 | 7215.945;308.0 | — | — | 7215.9 | 308.0 | 7523.9 | 0.0 |
| Sparse point cloud trajectory estimation (medium) | 2.96 | 978.0 | 7221.07;308.0 | — | — | 7221.1 | 308.0 | 7529.1 | 0.0 |
| Sparse point cloud trajectory estimation (hard) | 2.58 | 979.0 | 7220.32;308.0 | — | — | 7220.3 | 308.0 | 7528.3 | 0.0 |
| Cluster Filter (clean) | 4.28 | 981.3 | 7220.633;308.0 | — | — | 7220.6 | 308.0 | 7528.6 | 0.0 |
| Cluster Filter (medium) | 3.93 | 982.2 | 7220.57;308.0 | — | — | 7220.6 | 308.0 | 7528.6 | 0.0 |
| Cluster Filter (hard) | 3.06 | 980.9 | 7215.883;308.0 | — | — | 7215.9 | 308.0 | 7523.9 | 0.0 |
| CL-Det / DBSCAN LiDAR tracking (clean) | 3.59 | 980.9 | 7215.945;308.0 | — | — | 7215.9 | 308.0 | 7523.9 | 0.0 |
| CL-Det / DBSCAN LiDAR tracking (medium) | 3.72 | 980.9 | 7215.883;308.0 | — | — | 7215.9 | 308.0 | 7523.9 | 0.0 |
| CL-Det / DBSCAN LiDAR tracking (hard) | 3.31 | 980.9 | 7221.07;308.0 | — | — | 7221.1 | 308.0 | 7529.1 | 0.0 |
| ResNet-like embedding family | 11.09 | 1025.1 | 7221.07;308.0 | 117065.0 | 0.0 | 7221.1 | 308.0 | 7529.1 | 0.01 |