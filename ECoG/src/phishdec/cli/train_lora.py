import argparse
import os
import shutil
from pathlib import Path
from typing import Optional
import yaml

from phishdec.utils.env import setup_env
from phishdec.utils.seed import set_seed
from phishdec.train.sft_runner import (
    _get_cfg,
    load_yaml_config,
    run_training_from_config,
    timestamp_run_dir,
)


def _is_main_process() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except Exception:
        return True


def _prepare_run_dirs(cfg, exp_name: str, out_dir_override: Optional[str]):
    out_root = _get_cfg(cfg, "run.out_root", "outputs/runs")
    use_running = bool(_get_cfg(cfg, "run.use_running_dir", True))
    running_root = _get_cfg(cfg, "run.running_root")

    if not running_root:
        running_root = os.path.join("outputs", "runs", "running")

    if out_dir_override:
        try:
            running_root_path = Path(running_root).resolve()
            override_path = Path(out_dir_override).resolve()
        except Exception:
            running_root_path = Path(running_root)
            override_path = Path(out_dir_override)

        if use_running and (running_root_path == override_path or running_root_path in override_path.parents):
            run_dir = str(override_path)
            final_run_dir = os.path.join(out_root, override_path.name)
            return run_dir, final_run_dir, running_root, use_running

        final_run_dir = out_dir_override
    else:
        final_run_dir = timestamp_run_dir(exp_name, out_root)

    if use_running:
        run_dir = os.path.join(running_root, os.path.basename(final_run_dir))
    else:
        run_dir = final_run_dir
    return run_dir, final_run_dir, running_root, use_running


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,

    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    train_cfg = cfg.setdefault("train", {})
    train_cfg["peft"] = "lora"
    if args.lora_r is not None:
        train_cfg["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        train_cfg["lora_alpha"] = args.lora_alpha
    if args.lora_dropout is not None:
        train_cfg["lora_dropout"] = args.lora_dropout
    if args.lora_target_modules:
        train_cfg["lora_target_modules"] = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    hf_token = setup_env()

    cuda_devices = _get_cfg(cfg, "run.cuda_visible_devices")
    if cuda_devices is not None and str(cuda_devices).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    seed = int(_get_cfg(cfg, "run.seed", 10))
    deterministic = bool(_get_cfg(cfg, "run.deterministic", True))
    benchmark = bool(_get_cfg(cfg, "run.benchmark", False))
    set_seed(seed, deterministic=deterministic, benchmark=benchmark)

    exp_name = _get_cfg(cfg, "exp_name", "lora_sft")
    run_dir, final_run_dir, running_root, use_running = _prepare_run_dirs(cfg, exp_name, args.out_dir)
    os.makedirs(run_dir, exist_ok=True)

    train_cfg = cfg.setdefault("train", {})
    final_logging_dir = train_cfg.get("logging_dir")
    running_logging_root = None
    if use_running and final_logging_dir:
        running_logging_root = _get_cfg(cfg, "run.running_tb_root")
        if not running_logging_root:
            running_logging_root = os.path.join("outputs", "runs", "tb_logs", "running")
        train_cfg["logging_dir"] = running_logging_root

    cfg_copy_path = os.path.join(run_dir, "config_used.yaml")
    with open(cfg_copy_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    deepspeed_cfg = args.deepspeed if args.deepspeed is not None else _get_cfg(cfg, "train.deepspeed")
    success = False
    try:
        run_training_from_config(
            cfg=cfg,
            run_dir=run_dir,
            trainer_choice="trl",
            deepspeed_cfg=deepspeed_cfg,
            hf_token=hf_token,
        )
        success = True
    finally:
        if success and use_running and _is_main_process():
            Path(final_run_dir).parent.mkdir(parents=True, exist_ok=True)
            if os.path.exists(final_run_dir):
                raise RuntimeError(f"Final run_dir already exists: {final_run_dir}")
            shutil.move(run_dir, final_run_dir)

            if final_logging_dir and running_logging_root:
                src_log_dir = os.path.join(running_logging_root, os.path.basename(final_run_dir))
                if os.path.exists(src_log_dir):
                    Path(final_logging_dir).mkdir(parents=True, exist_ok=True)
                    dst_log_dir = os.path.join(final_logging_dir, os.path.basename(final_run_dir))
                    if not os.path.exists(dst_log_dir):
                        shutil.move(src_log_dir, dst_log_dir)

    if _is_main_process():
        print(f"[done] run_dir={final_run_dir if (success and use_running) else run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
