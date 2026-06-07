# Dataset Specification

## Trajectories

- Minimum default size: 1000 trajectories for the quick profile.
- Each trajectory contains 80 to 200 points.
- Channels: `x, y, z, vx, vy, vz, speed, yaw, pitch, roll`.
- Labels: `pattern_id`, `pattern_name`, `noise_level`.

## Segments

- Fixed-length windows of 50 points with stride 25.
- Feature channels: `x, y, z, vx, vy, vz, speed, yaw`.
- Labels: behavior class and future trajectory targets.

## Point Cloud

- Per-frame synthetic point clouds at `clean`, `medium`, and `hard` difficulty.
- Each frame contains UAV points, background points, random noise, and false clusters.
- Ground truth UAV center is stored for tracking evaluation.

## Pattern Classes

1. `straight_flight`
2. `circular_orbit`
3. `rectangular_patrol`
4. `hover`
5. `spiral_climb`
6. `zigzag`
7. `sharp_turn`
8. `descent_approach`
9. `random_anomalous`

## Noise Profiles

- `clean`: low positional noise, dense UAV observations.
- `medium`: moderate noise, background clusters, occasional frame drops.
- `hard`: sparse UAV returns, stronger clutter, false targets, track gaps.

## File Formats

- CSV for trajectories, labels, point cloud frames, metrics, and tables.
- NPZ for segment arrays and targets.
- JSON for configs, metadata, and run logs.
- PNG for figures.
