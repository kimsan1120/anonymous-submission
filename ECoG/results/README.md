# Results

This directory is reserved for aggregate result CSVs released with the paper.

Running `scripts/run_ablation.py --run` writes fresh raw benchmark outputs under
`outputs/runs/benchmarks` and normalized benchmark performance artifacts under
`outputs/runs/benchmark_performance`.

`dry_run_ablation_benchmark_seed10.json` is the normalized command-resolution
baseline for all exported ablations. Verify it with:

```bash
python scripts/benchmark_ablation_dry_run.py --check --seed 10
```
