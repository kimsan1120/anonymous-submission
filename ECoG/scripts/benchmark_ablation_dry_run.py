#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ABLATION = ROOT / "scripts" / "run_ablation.py"
DEFAULT_EXPECTED = ROOT / "results" / "dry_run_ablation_benchmark_seed10.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or verify the normalized dry-run benchmark for every exported "
            "paper ablation. This does not train models or generate configs."
        )
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--scenario-letters", default="ABCDEFG")
    parser.add_argument("--cuda-visible-devices", default="0,1,2")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--rec-lambdas", default="0,0.05,0.1,0.2")
    parser.add_argument("--rec-lambda", type=float, default=0.1, help="Lambda used by the last-pooling rec ablation.")
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help="Expected benchmark JSON for --check, or output path for --write-expected.",
    )
    parser.add_argument("--write-expected", action="store_true", help="Write the current normalized benchmark JSON.")
    parser.add_argument("--check", action="store_true", help="Compare the current dry-run benchmark with --expected.")
    parser.add_argument(
        "--observed-output",
        type=Path,
        default=None,
        help="Optional path to write the observed JSON during --check.",
    )
    parser.add_argument(
        "--install-local-package",
        action="store_true",
        help="Run `python -m pip install -e .` before the benchmark. Dry-run itself only needs the standard library.",
    )
    args = parser.parse_args()
    if not args.write_expected and not args.check:
        args.check = args.expected.exists()
        args.write_expected = not args.expected.exists()
    return args


def install_local_package() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(ROOT)], cwd=str(ROOT), check=True)


def normalize_token(token: str) -> str:
    root_text = str(ROOT)
    python_text = str(Path(sys.executable).resolve())
    token_path = Path(token).resolve() if token.startswith("/") else None
    if token == sys.executable or token == python_text or token_path == Path(sys.executable).resolve():
        return "<PYTHON>"
    if token == root_text:
        return "<ROOT>"
    if token.startswith(root_text + os.sep):
        return "<ROOT>" + token[len(root_text) :]
    return token


def parse_dry_run_stdout(stdout: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    current_label: str | None = None
    for line in stdout.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("[ablation] "):
            current_label = line[len("[ablation] ") :]
            continue
        if line.startswith("[run] "):
            if current_label is None:
                raise RuntimeError(f"Found [run] before [ablation]: {line}")
            argv = [normalize_token(token) for token in shlex.split(line[len("[run] ") :])]
            jobs.append({"label": current_label, "argv": argv})
            current_label = None
            continue
        raise RuntimeError(f"Unexpected dry-run output line: {line}")
    return jobs


def canonical_payload(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark": "vp_decoder_repro_ablation_dry_run",
        "schema_version": 1,
        "seed": int(args.seed),
        "scenario_letters": str(args.scenario_letters),
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "rec_lambdas": str(args.rec_lambdas),
        "rec_lambda": float(args.rec_lambda),
        "models": list(args.model),
        "jobs": jobs,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "digest": f"sha256:{digest}"}


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(RUN_ABLATION),
        "--method",
        "all",
        "--seed",
        str(int(args.seed)),
        "--scenario-letters",
        str(args.scenario_letters),
        "--cuda-visible-devices",
        str(args.cuda_visible_devices),
        "--rec-lambdas",
        str(args.rec_lambdas),
        "--rec-lambda",
        str(float(args.rec_lambda)),
        "--dry-run",
    ]
    for model in args.model:
        command.extend(["--model", str(model)])

    env = dict(os.environ)
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    if completed.stderr.strip():
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    return canonical_payload(args, parse_dry_run_stdout(completed.stdout))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def diff_payload(expected: dict[str, Any], observed: dict[str, Any]) -> str:
    expected_text = json.dumps(expected, ensure_ascii=True, indent=2, sort_keys=True).splitlines()
    observed_text = json.dumps(observed, ensure_ascii=True, indent=2, sort_keys=True).splitlines()
    return "\n".join(
        difflib.unified_diff(expected_text, observed_text, fromfile="expected", tofile="observed", lineterm="")
    )


def main() -> int:
    args = parse_args()
    if args.install_local_package:
        install_local_package()

    observed = run_dry_run(args)

    if args.write_expected:
        write_json(args.expected, observed)
        print(f"[write] {args.expected}")
        print(f"[digest] {observed['digest']}")
        print(f"[jobs] {len(observed['jobs'])}")

    if args.check:
        expected = load_json(args.expected)
        if args.observed_output:
            write_json(args.observed_output, observed)
        if expected == observed:
            print(f"[ok] dry-run benchmark matches {args.expected}")
            print(f"[digest] {observed['digest']}")
            print(f"[jobs] {len(observed['jobs'])}")
            return 0
        print(f"[mismatch] dry-run benchmark differs from {args.expected}", file=sys.stderr)
        print(diff_payload(expected, observed), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
