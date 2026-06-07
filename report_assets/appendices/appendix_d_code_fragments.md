# Appendix D. Code fragments

## Trajectory generation

```python
bundle = generate_trajectory_dataset(config)
save_trajectory_bundle(bundle)
```

## Metric calculation

```python
metrics = evaluate_clustering(y_true, y_pred, features)
resource_usage = monitor.stop()
```

## Experiment run

```python
uv run python -m src.experiments.run_all --dataset synthetic_v1 --output results
```

## Example method logic

```text
1. Build trajectory descriptor
2. Normalize features
3. Run DBSCAN/HDBSCAN or K-Means
4. Compare labels against ground truth
```
