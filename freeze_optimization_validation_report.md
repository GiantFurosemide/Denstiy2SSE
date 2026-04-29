# Freeze Optimization Validation Report

## Summary
- Exact optimized kernel keeps metric outputs identical to legacy baseline on the benchmark slice.
- CPU benchmark speedup (legacy:numpy vs optimized:numpy): **10.38x**.
- Torch backend equivalence vs optimized numpy: all tracked metric deltas are zero on benchmark run.
- Adaptive metric schedule and final exact evaluation smoke run completed successfully.

## Benchmark Evidence
- Source: `outputs/metrics_backend_benchmark_baseline.json`
  - `legacy:numpy` elapsed: 0.157161s
  - `optimized:numpy` elapsed: 0.015140s
  - speedup: 10.38x
- Source: `outputs/metrics_backend_benchmark_torch.json`
  - `optimized:numpy` elapsed: 0.041419s
  - `optimized:torch` elapsed: 0.046181s
  - metric deltas: {'center_error': 0.0, 'angle_error': 0.0, 'length_error': 0.0, 'coverage_ratio': 0.0, 'clash_voxels': 0.0, 'loss_total': 0.0}

## Runtime Policy Smoke Run
- Run directory: `/Users/muwang/Documents/science/draft/test_001/Denstiy2SSE/outputs/train/20260420_140021_baseline_cnn_1e0128`
- Metrics rows written: 5
- Last row: {'model_name': 'baseline_cnn', 'run_id': '20260420_140021_baseline_cnn_1e0128', 'epoch': '2', 'split': 'val', 'center_error': '12.58576250076294', 'angle_error': '60.196297022723385', 'length_error': '14.539627224206924', 'coverage_ratio': '0.0', 'clash_voxels': '0.0', 'loss_total': '72.46778869628906'}
- Behavior observed from logs:
  - stage-level timing logs present
  - adaptive metrics cadence adjustment triggered
  - final exact evaluation executed at end of training

## Recommended Production Config (cluster)
```yaml
training:
  device: auto
  num_workers: 4
  metrics_kernel_impl: optimized
  metrics_backend: auto
  metrics_profile_components: true
  adaptive_metrics_schedule: true
  metrics_target_seconds: 180
  metrics_every_n_epochs: 1
  metrics_train_max_batches: 4
  val_metrics_max_batches: 8
  metrics_compute_coverage: true
  metrics_compute_clash: true
  metrics_log_every_n_batches: 10
  viz_enabled: false
  final_exact_eval: true
```

## Notes
- On CPU-only systems, `metrics_backend: auto` resolves to numpy backend.
- On CUDA systems, `metrics_backend: auto` will use torch backend for heavy occupancy/count kernels.
- Keep `final_exact_eval: true` to preserve final benchmark semantics while allowing adaptive in-epoch scheduling.
