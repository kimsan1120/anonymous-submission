import argparse
import os
from pathlib import Path

import yaml

from phishdec.train.decoder import evaluate_decoder_classifier_from_config
from phishdec.train.sft_runner import _get_cfg, load_yaml_config, timestamp_run_dir
from phishdec.utils.env import setup_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--out_dir", type=str, default="", help="Optional run directory override")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Optional checkpoint/run dir override")
    parser.add_argument("--eval_csv", type=str, default=None, help="Optional eval CSV override")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional test CSV override")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg.setdefault("model", {})
    data_cfg = cfg.setdefault("data", {})
    if args.checkpoint_dir:
        model_cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.eval_csv:
        data_cfg["eval_csv"] = args.eval_csv
    if args.test_csv:
        data_cfg["test_csv"] = args.test_csv

    hf_token = setup_env()
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    cuda_devices = _get_cfg(cfg, "run.cuda_visible_devices")
    if cuda_devices is not None and str(cuda_devices).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    if args.out_dir:
        run_dir = args.out_dir
    else:
        exp_name = _get_cfg(cfg, "exp_name", "decoder_classifier_eval")
        out_root = _get_cfg(cfg, "run.out_root", "outputs/runs/results/decoder_eval")
        run_dir = timestamp_run_dir(exp_name, out_root)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(run_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    evaluate_decoder_classifier_from_config(cfg=cfg, run_dir=run_dir, hf_token=hf_token)
    print(f"[done] run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
