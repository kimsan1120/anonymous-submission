#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_evidence_reason_suite.py"
DEFAULT_MODEL = "hyperclovax0p5b=naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
REC_SWEEP = (0.0, 0.05, 0.1, 0.2)


def safe_float_suffix(value: float) -> str:
    text = f"{float(value):.4g}"
    return text.replace("-", "m").replace(".", "p")


def parse_float_list(raw_value: str | None) -> list[float]:
    if raw_value is None or not str(raw_value).strip():
        return []
    out: list[float] = []
    for part in str(raw_value).replace(",", " ").split():
        out.append(float(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or render the paper ablation suite: label+evidence, "
            "label+rationale, rationale reconstruction, last-pooling, and last-label ablations."
        )
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "label_evidence",
            "label_rationale",
            "label_rationale_rec",
            "label_rationale_rec_last_pooling",
            "last_label_ablation",
            "all",
        ),
    )
    parser.add_argument("--model", action="append", default=[], help="alias=model_ref. Defaults to HyperCLOVAX 0.5B.")
    parser.add_argument("--scenario-letters", default="ABCDEFG")
    parser.add_argument("--eval-target-ids", default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cuda-visible-devices", default="0,1,2")
    parser.add_argument("--sms-cuda-visible-devices", default=None)
    parser.add_argument("--voice-cuda-visible-devices", default=None)
    parser.add_argument("--bench-name", default=None)
    parser.add_argument("--rec-lambda", type=float, default=0.1)
    parser.add_argument(
        "--rec-lambdas",
        default=None,
        help="Comma/space-separated lambda sweep for --method label_rationale_rec. Default for --method all: 0,0.05,0.1,0.2.",
    )
    parser.add_argument("--rec-pooling", choices=("mean", "last"), default="mean")
    parser.add_argument("--decode-max-new-tokens", type=int, default=200)
    parser.add_argument("--sms-hf-batch-size", type=int, default=None)
    parser.add_argument("--voice-hf-batch-size", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Actually train/evaluate. Without this, configs are generated only.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands without generating configs or running.")
    parser.add_argument("--no-eval-full-generation", dest="eval_full_generation", action="store_false")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore in benchmark performance reports.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional argument forwarded to run_evidence_reason_suite.py.")
    parser.set_defaults(eval_full_generation=True)
    return parser.parse_args()


def model_args(args: argparse.Namespace) -> list[str]:
    specs = args.model or [DEFAULT_MODEL]
    out: list[str] = []
    for spec in specs:
        out.extend(["--model", str(spec)])
    return out


def common_args(args: argparse.Namespace, *, suite_name: str, benchmark_type: str, bench_name: str) -> list[str]:
    cmd = [
        "--suite-name",
        suite_name,
        "--benchmark-layout",
        "flat",
        "--benchmark-type",
        benchmark_type,
        "--bench-name",
        bench_name,
        "--scenario-letters",
        str(args.scenario_letters),
        "--seed",
        str(int(args.seed)),
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--skip-prepare-data",
        "--skip-length-profiles",
    ]
    if args.skip_bertscore:
        cmd.append("--skip-benchmark-performance-bertscore")
    if args.eval_target_ids:
        cmd.extend(["--eval-target-ids", str(args.eval_target_ids)])
    if args.sms_cuda_visible_devices is not None:
        cmd.extend(["--sms-cuda-visible-devices", str(args.sms_cuda_visible_devices)])
    if args.voice_cuda_visible_devices is not None:
        cmd.extend(["--voice-cuda-visible-devices", str(args.voice_cuda_visible_devices)])
    if args.eval_full_generation:
        cmd.extend(["--eval-full-generation", "--decode-max-new-tokens", str(int(args.decode_max_new_tokens))])
    if args.sms_hf_batch_size is not None:
        cmd.extend(["--sms-hf-batch-size", str(int(args.sms_hf_batch_size))])
    if args.voice_hf_batch_size is not None:
        cmd.extend(["--voice-hf-batch-size", str(int(args.voice_hf_batch_size))])
    if args.run:
        cmd.append("--run")
    for item in args.extra_arg:
        cmd.extend(str(item).split())
    return cmd + model_args(args)


def default_bench_name(method: str, args: argparse.Namespace, *, rec_lambda: float | None = None, pooling: str | None = None) -> str:
    if args.bench_name:
        if rec_lambda is None:
            return args.bench_name
        suffix = f"rec_{safe_float_suffix(rec_lambda)}"
        if pooling and pooling != "mean":
            suffix = f"{suffix}_{pooling}"
        return f"{args.bench_name}_{suffix}"
    if method == "label_evidence":
        return "label_evidence"
    if method == "label_rationale":
        return "label_rationale"
    if method == "label_rationale_rec":
        suffix = f"label_rationale_rec_{safe_float_suffix(float(rec_lambda or 0.0))}"
        if pooling and pooling != "mean":
            suffix = f"{suffix}_{pooling}"
        return suffix
    if method == "label_rationale_rec_last_pooling":
        return f"label_rationale_rec_{safe_float_suffix(float(rec_lambda or args.rec_lambda))}_last"
    if method == "last_label_ablation":
        return "last_label_ablation"
    return method


def build_method_command(args: argparse.Namespace, method: str, *, rec_lambda: float | None = None, pooling: str | None = None) -> list[str]:
    pooling = pooling or args.rec_pooling
    if method == "label_evidence":
        child = common_args(
            args,
            suite_name="label_evidence",
            benchmark_type="label_evidence",
            bench_name=default_bench_name(method, args),
        )
        child.extend(
            [
                "--joint-stage12",
                "--joint-reconstruction-loss-weight",
                "0",
                "--stage2-target-format",
                "label_span",
            ]
        )
        if args.sms_hf_batch_size is None:
            child.extend(["--sms-hf-batch-size", "60"])
        if args.voice_hf_batch_size is None:
            child.extend(["--voice-hf-batch-size", "12"])
        return [sys.executable, str(RUNNER), *child]

    if method == "label_rationale":
        child = common_args(
            args,
            suite_name="label_explanation",
            benchmark_type="label_explanation",
            bench_name=default_bench_name(method, args),
        )
        child.extend(
            [
                "--stages",
                "stage2",
                "--stage2-from-base",
                "--stage2-target-format",
                "label_first_explanation",
                "--stage2-reconstruction-loss-weight",
                "0",
            ]
        )
        if args.sms_hf_batch_size is None:
            child.extend(["--sms-hf-batch-size", "48"])
        if args.voice_hf_batch_size is None:
            child.extend(["--voice-hf-batch-size", "8"])
        return [sys.executable, str(RUNNER), *child]

    if method in {"label_rationale_rec", "label_rationale_rec_last_pooling"}:
        weight = float(args.rec_lambda if rec_lambda is None else rec_lambda)
        if method == "label_rationale_rec_last_pooling":
            pooling = "last"
        child = common_args(
            args,
            suite_name="label_explanation",
            benchmark_type="label_explanation",
            bench_name=default_bench_name(method, args, rec_lambda=weight, pooling=pooling),
        )
        child.extend(
            [
                "--stages",
                "stage2",
                "--stage2-from-base",
                "--stage2-target-format",
                "label_first_explanation",
                "--stage2-reconstruction-loss-weight",
                str(weight),
                "--stage2-reconstruction-scope",
                "explanation",
                "--stage2-reconstruction-pooling",
                pooling,
            ]
        )
        if args.sms_hf_batch_size is None:
            child.extend(["--sms-hf-batch-size", "48"])
        if args.voice_hf_batch_size is None:
            child.extend(["--voice-hf-batch-size", "8"])
        return [sys.executable, str(RUNNER), *child]

    if method == "last_label_ablation":
        child = common_args(
            args,
            suite_name="label_evidence_explanation",
            benchmark_type="label_evidence_explanation",
            bench_name=default_bench_name(method, args),
        )
        child.extend(
            [
                "--joint-stage12",
                "--joint-reconstruction-loss-weight",
                "0",
                "--stage2-target-format",
                "span_explanation_label",
            ]
        )
        if args.sms_hf_batch_size is None:
            child.extend(["--sms-hf-batch-size", "48"])
        if args.voice_hf_batch_size is None:
            child.extend(["--voice-hf-batch-size", "8"])
        return [sys.executable, str(RUNNER), *child]

    raise ValueError(f"Unsupported method: {method}")


def expand_jobs(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if args.method == "all":
        rec_values = parse_float_list(args.rec_lambdas) or list(REC_SWEEP)
        jobs = [
            ("label_evidence", build_method_command(args, "label_evidence")),
            ("label_rationale", build_method_command(args, "label_rationale")),
        ]
        for value in rec_values:
            jobs.append((f"label_rationale_rec:{value}", build_method_command(args, "label_rationale_rec", rec_lambda=value)))
        jobs.append(
            (
                f"label_rationale_rec_last_pooling:{args.rec_lambda}",
                build_method_command(args, "label_rationale_rec_last_pooling", rec_lambda=args.rec_lambda, pooling="last"),
            )
        )
        jobs.append(("last_label_ablation", build_method_command(args, "last_label_ablation")))
        return jobs

    if args.method == "label_rationale_rec":
        rec_values = parse_float_list(args.rec_lambdas) or [float(args.rec_lambda)]
        return [
            (f"label_rationale_rec:{value}", build_method_command(args, "label_rationale_rec", rec_lambda=value))
            for value in rec_values
        ]
    return [(args.method, build_method_command(args, args.method))]


def main() -> int:
    args = parse_args()
    env = dict(os.environ)
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    jobs = expand_jobs(args)
    for label, cmd in jobs:
        print(f"[ablation] {label}")
        print("[run]", " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
