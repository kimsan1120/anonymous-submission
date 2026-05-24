# VP Decoder Ablation Reproduction Package

This export contains the code, prompts, and de-identified data needed to run the
paper ablations for:

- `label_evidence`
- `label_rationale`
- `label_rationale_rec`
- `label_rationale_rec_last_pooling`
- `last_label_ablation`

Trained model checkpoints and previous raw run outputs are intentionally
excluded. Running the scripts regenerates checkpoints and benchmark outputs under
`outputs/runs/...`.

## Layout

```text
.
├─ configs/benchmarks/main_ablation.yaml
├─ data/
│  ├─ sms/in_domain
│  ├─ sms/ood
│  ├─ sms/evidence/keep
│  ├─ voice/in_domain
│  ├─ voice/ood
│  └─ voice/evidence/keep
├─ results/
│  └─ dry_run_ablation_benchmark_seed10.json
├─ requirements.txt
├─ scripts/
│  ├─ benchmark_ablation_dry_run.py
│  ├─ run_ablation.py
│  ├─ run_evidence_reason_suite.py
│  ├─ run_sms_evidence_reason_suite.py
│  ├─ run_voice_evidence_reason_suite.py
│  ├─ run_noevidence_reason_suite.py
│  ├─ write_benchmark_performance.py
│  ├─ extract_benchmark_metrics.py
│  ├─ compute_consistency_metrics.py
│  └─ compute_evidence_explanation_metrics.py
└─ src/phishdec/
```

## Quick Start

Use the `decoder311` environment from the exported repo root.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec --rec-lambda 0.1 --seed 10 --dry-run
```

Generate configs without launching training:

```bash
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec --rec-lambda 0.1 --seed 10
```

Launch the actual train/eval run:

```bash
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec --rec-lambda 0.1 --seed 10 --run
```

The wrapper defaults to seed `10`, HyperCLOVAX 0.5B, scenarios `A` through `G`,
and CUDA devices `0,1,2`. Override these with `--seed`, `--model`,
`--scenario-letters`, and `--cuda-visible-devices`. Use `--skip-bertscore` for
faster benchmark performance extraction when BERTScore is not needed.

## Data and Length Profiles

Run from the exported repo root. Generated configs use relative paths under the
export package, for example `data/sms/evidence/keep/A_train.csv`,
`data/sms/in_domain/test.csv`, and `data/voice/evidence/keep/F_train.csv`.

The top-level `scripts/run_ablation.py` wrapper passes `--skip-prepare-data` and
`--skip-length-profiles` to keep reproduction runs deterministic and lightweight.
It therefore uses the exported data and conservative static batch planning rather
than regenerating length profiles. The released `data/length_profiles.json` and
`data/length_profiles.csv` are included for inspection/reference. Lower-level
suite runners can regenerate profiles only when invoked directly without
`--skip-length-profiles`.

## Dry-Run Benchmark

The dry-run benchmark checks whether a fresh environment resolves the same
ablation commands as the released artifact. It does not import training
dependencies, train models, generate configs, or read private checkpoints.

```bash
python -m pip install -e .
python scripts/benchmark_ablation_dry_run.py --check --seed 10
```

The expected seed-10 digest is stored in
`results/dry_run_ablation_benchmark_seed10.json`. To refresh it intentionally:

```bash
python scripts/benchmark_ablation_dry_run.py --write-expected --seed 10
```

## Methods

```bash
conda run -n decoder311 python scripts/run_ablation.py --method label_evidence --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec --rec-lambdas 0,0.05,0.1,0.2 --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method label_rationale_rec_last_pooling --rec-lambda 0.1 --seed 10 --run
conda run -n decoder311 python scripts/run_ablation.py --method last_label_ablation --seed 10 --run
```

`last_label_ablation` uses the `span_explanation_label` target format:
evidence spans and rationale are generated first, and the binary label is placed
last.

## Outputs

Raw benchmark outputs are written to:

```text
outputs/runs/benchmarks/{suite}/{model_alias}/...
```

Normalized benchmark performance artifacts are written to:

```text
outputs/runs/benchmark_performance/{benchmark_type}/{model_alias}/...
```

The `results/` directory is reserved for released aggregate CSVs, if needed.
