# Configs

`configs/benchmarks/main_ablation.yaml` records the released ablation matrix.
The runnable interface is `scripts/run_ablation.py`; method-specific generated
train/eval YAML files are written under `configs/generated/...` at runtime.

The export does not keep one static YAML per experiment because the public
ablation axes are argparse-controlled:

- method
- reconstruction lambda
- reconstruction pooling
- scenario letters
- seed
- model
- CUDA devices
