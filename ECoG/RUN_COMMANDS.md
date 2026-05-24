# Run Commands

All commands assume the exported repo root and `decoder311`.

Install runtime dependencies and the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

`scripts/run_ablation.py` uses the data shipped inside this export package. It
also passes `--skip-prepare-data` and `--skip-length-profiles`, so these commands
do not regenerate `data/length_profiles.json/csv`; they use static batch
planning unless a lower-level suite runner is invoked directly without
`--skip-length-profiles`.

Dry-run benchmark verification in a fresh environment:

```bash
python scripts/benchmark_ablation_dry_run.py --check --seed 10
```

Dry-run command resolution:

```bash
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec --rec-lambda 0.1 --seed 10 --dry-run
```

Generate configs only:

```bash
conda run -n decoder311 python scripts/run_ablation.py --method label_evidence --seed 10
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale --seed 10
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec --rec-lambdas 0,0.05,0.1,0.2 --seed 10
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec_last_pooling --rec-lambda 0.1 --seed 10
conda run -n decoder311 python scripts/run_ablation.py --method last_label_ablation --seed 10
```

Run train/eval:

```bash
conda run -n decoder311 python scripts/run_ablation.py --method label_evidence --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec --rec-lambdas 0,0.05,0.1,0.2 --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec_last_pooling --rec-lambda 0.1 --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method last_label_ablation --seed 10 --run
```
