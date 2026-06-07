# Appendix A. Dataset parameters

## Trajectory generation
- Dataset name: `synthetic_v1`
- Random seed: `42`
- Number of trajectories: `1000`
- Trajectory length range: `80..200` points
- Sample rate: `10` Hz

## Pattern list
- `straight_flight`
- `circular_orbit`
- `rectangular_patrol`
- `hover`
- `spiral_climb`
- `zigzag`
- `sharp_turn`
- `descent_approach`
- `random_anomalous`

## Segment dataset
- Segment length: `50`
- Segment stride: `25`
- Forecast horizons (steps): `[5, 10, 20, 30]`

## Noise profiles
- `clean`: position_std=0.12, drop_probability=0.01, false_clusters=1
- `medium`: position_std=0.35, drop_probability=0.05, false_clusters=2
- `hard`: position_std=0.7, drop_probability=0.12, false_clusters=3

## Point cloud generation
- Frames per trajectory: `12`
- Background points: `24`
- Missing-point probabilities: `{'clean': 0.02, 'medium': 0.08, 'hard': 0.18}`