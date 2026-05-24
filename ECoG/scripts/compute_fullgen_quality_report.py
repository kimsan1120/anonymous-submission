#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


EVAL_ORDER = [f"{letter}{idx}" for letter in "ABCDEFG" for idx in (1, 2)]
SMS_LETTERS = set("ABCD")
QUALITY_COMPONENTS = {"consistency", "evidence", "explanation"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_results_csv(run_dir: Path) -> Path | None:
    preferred = run_dir / "results_data_parallel_merged.csv"
    if preferred.exists():
        return preferred
    matches = sorted(run_dir.glob("results_*hf_decode_*.csv"))
    if matches:
        return matches[0]
    matches = sorted(run_dir.glob("results*.csv"))
    return matches[0] if matches else None


def _eval_id_from_run_dir(run_dir: Path) -> str | None:
    parts = run_dir.parts
    for idx, part in enumerate(parts):
        if part == "eval" and idx + 2 < len(parts):
            eval_id = parts[idx + 2]
            if eval_id in EVAL_ORDER:
                return eval_id
    return None


def _collect_latest_run_dirs(args: argparse.Namespace) -> list[Path]:
    root = _repo_root()
    candidates: list[Path] = []
    layout = str(getattr(args, "benchmark_layout", "nested") or "nested").strip().lower()
    suites = (
        [str(args.suite_name).strip()]
        if str(getattr(args, "suite_name", "") or "").strip()
        else ["evidence_reason_suite", "sms_evidence_reason_suite", "voice_evidence_reason_suite"]
    )
    for suite in suites:
        if layout == "flat":
            base = root / "outputs" / "runs" / "benchmarks" / suite / args.model_alias / "eval"
        else:
            base = root / "outputs" / "runs" / "benchmarks" / suite / args.bench_name / args.model_alias / "eval"
        if not base.exists():
            continue
        for run_dir in base.glob("*/*/*"):
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            if args.run_name_contains and args.run_name_contains not in name:
                continue
            if args.require_name_contains:
                if any(token not in name for token in args.require_name_contains):
                    continue
            if _eval_id_from_run_dir(run_dir) is None:
                continue
            if _find_results_csv(run_dir) is None:
                continue
            candidates.append(run_dir)

    by_eval: dict[str, Path] = {}
    for run_dir in sorted(candidates, key=lambda p: p.name):
        eval_id = _eval_id_from_run_dir(run_dir)
        if eval_id is None:
            continue
        by_eval[eval_id] = run_dir

    missing = [eval_id for eval_id in EVAL_ORDER if eval_id not in by_eval]
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            "Missing fullgen run dirs for eval ids: "
            + ", ".join(missing)
            + ". Use --allow-missing to continue."
        )
    return [by_eval[eval_id] for eval_id in EVAL_ORDER if eval_id in by_eval]


def _run_metric_script(script: str, run_dirs: list[Path], out_prefix: Path, extra: list[str]) -> None:
    cmd = [sys.executable, str(_repo_root() / "scripts" / script)]
    for run_dir in run_dirs:
        cmd.extend(["--run-dir", str(run_dir)])
    cmd.extend(["--out-prefix", str(out_prefix)])
    cmd.extend(extra)
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_repo_root()), check=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_components(raw_value: str) -> set[str]:
    raw = str(raw_value or "all").strip().lower()
    if not raw or raw == "all":
        return set(QUALITY_COMPONENTS)
    parts = [part.strip().lower() for part in raw.replace(",", " ").split() if part.strip()]
    bad = [part for part in parts if part not in QUALITY_COMPONENTS]
    if bad:
        raise ValueError(f"Unknown quality component(s): {bad}. Choose from all, consistency, evidence, explanation.")
    return set(parts)


def _scenario_label(eval_id: str) -> str:
    names = {
        "A": "sms_in_domain",
        "B": "sms_ood_credit",
        "C": "sms_ood_finance",
        "D": "sms_ood_parcel",
        "E": "voice_in_domain",
        "F": "voice_ood_finance",
        "G": "voice_ood_government",
    }
    return names.get(eval_id[:1], "")


def _format_summary(
    *,
    run_dirs: list[Path],
    consistency_prefix: Path,
    evidence_prefix: Path,
    final_csv: Path,
    components: set[str],
    format_percent: bool,
    include_run_dir: bool,
) -> None:
    consistency_path = consistency_prefix.with_name(consistency_prefix.name + "_summary.csv")
    evidence_path = evidence_prefix.with_name(evidence_prefix.name + "_summary.csv")
    consistency = pd.read_csv(consistency_path) if consistency_path.exists() else None
    evidence = pd.read_csv(evidence_path) if evidence_path.exists() else None

    rows: list[dict[str, Any]] = []
    run_dir_by_eval = {_eval_id_from_run_dir(run_dir): str(run_dir) for run_dir in run_dirs}
    for eval_id in EVAL_ORDER:
        run_dir = run_dir_by_eval.get(eval_id)
        if not run_dir:
            continue
        evidence_row: dict[str, Any] = {}
        consistency_row: dict[str, Any] = {}
        if evidence is not None:
            row_df = evidence[evidence["run_dir"] == run_dir]
            if not row_df.empty:
                evidence_row = row_df.iloc[0].to_dict()
        if consistency is not None:
            row_df = consistency[consistency["run_dir"] == run_dir]
            if not row_df.empty:
                consistency_row = row_df.iloc[0].to_dict()
        if not evidence_row and not consistency_row:
            continue

        row = {
            "Eval": eval_id,
            "Scenario": _scenario_label(eval_id),
            "Modality": "sms" if eval_id[:1] in SMS_LETTERS else "voice",
            "N": int((evidence_row or consistency_row).get("n", 0)),
        }
        if "consistency" in components:
            row["Inconsistency strict rate"] = consistency_row.get("strict_inconsistency_rate")
        if "evidence" in components:
            row.update(
                {
                    "Gold": int(evidence_row.get("evidence_total_gold_spans", 0)),
                    "Pred": int(evidence_row.get("evidence_total_pred_spans", 0)),
                    "Matched": int(evidence_row.get("evidence_total_matched_spans", 0)),
                    "Span P": evidence_row.get("evidence_span_precision"),
                    "Span R": evidence_row.get("evidence_span_recall"),
                    "Span F1": evidence_row.get("evidence_span_f1"),
                    "Exact Match": evidence_row.get("evidence_exact_match_rate"),
                }
            )
        if "explanation" in components:
            row.update(
                {
                    "ROUGE-L F1": evidence_row.get("rouge_l_f1"),
                    "BERTScore F1": evidence_row.get("bertscore_f1", ""),
                }
            )
        row["Run Dir"] = run_dir
        rows.append(row)

    c_agg: dict[str, Any] = {}
    e_agg: dict[str, Any] = {}
    if consistency_prefix.with_suffix(".json").exists():
        consistency_json = _read_json(consistency_prefix.with_suffix(".json"))
        c_agg = consistency_json.get("aggregate", consistency_json)
    if evidence_prefix.with_suffix(".json").exists():
        evidence_json = _read_json(evidence_prefix.with_suffix(".json"))
        e_agg = evidence_json.get("aggregate", evidence_json)
    agg_source = e_agg or c_agg
    agg = {"Eval": "ALL", "Scenario": "aggregate", "Modality": "all", "N": int(agg_source.get("n", 0))}
    if "consistency" in components:
        agg["Inconsistency strict rate"] = c_agg.get("strict_inconsistency_rate")
    if "evidence" in components:
        agg.update(
            {
                "Gold": int(e_agg.get("evidence_total_gold_spans", 0)),
                "Pred": int(e_agg.get("evidence_total_pred_spans", 0)),
                "Matched": int(e_agg.get("evidence_total_matched_spans", 0)),
                "Span P": e_agg.get("evidence_span_precision"),
                "Span R": e_agg.get("evidence_span_recall"),
                "Span F1": e_agg.get("evidence_span_f1"),
                "Exact Match": e_agg.get("evidence_exact_match_rate"),
            }
        )
    if "explanation" in components:
        agg.update({"ROUGE-L F1": e_agg.get("rouge_l_f1"), "BERTScore F1": e_agg.get("bertscore_f1", "")})
    agg["Run Dir"] = ""
    rows.append(agg)

    final_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    if not include_run_dir and "Run Dir" in out.columns:
        out = out.drop(columns=["Run Dir"])
    if format_percent:
        percent_cols = [
            "Inconsistency strict rate",
            "Span P",
            "Span R",
            "Span F1",
            "Exact Match",
            "ROUGE-L F1",
            "BERTScore F1",
        ]
        for col in percent_cols:
            if col not in out.columns:
                continue
            out[col] = out[col].map(
                lambda value: ""
                if pd.isna(value) or value == ""
                else f"{float(value) * 100.0:.2f}"
            )
    out.to_csv(final_csv, index=False, encoding="utf-8-sig")
    print(f"[write] {final_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute consistency, evidence span, ROUGE-L, and BERTScore summaries for full-generation runs."
    )
    parser.add_argument("--bench-name", default="hyperclovax0p5b_evidence_joint12_span_noidx_rec0p2_v1")
    parser.add_argument(
        "--suite-name",
        default=None,
        help=(
            "Benchmark suite namespace under outputs/runs/benchmarks. "
            "When omitted, search evidence_reason_suite plus the legacy SMS/Voice suite roots."
        ),
    )
    parser.add_argument("--model-alias", default="hyperclovax0p5b")
    parser.add_argument(
        "--benchmark-layout",
        choices=("nested", "flat"),
        default="nested",
        help=(
            "Directory layout to search. nested uses {suite}/{bench}/{model}/eval; "
            "flat uses {suite}/{model}/eval."
        ),
    )
    parser.add_argument("--run-name-contains", default="fullgen200")
    parser.add_argument(
        "--components",
        default="all",
        help="Comma/space-separated quality components: consistency, evidence, explanation, or all.",
    )
    parser.add_argument(
        "--require-name-contains",
        action="append",
        default=[],
        help="Require token in run directory name. Can be passed multiple times.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--out-prefix",
        default="reports/evidence_explanation_metrics/hyperclovax0p5b_evidence_joint12_span_noidx_rec0p2_v1_maxnew200_balanced_20260429",
    )
    parser.add_argument("--compute-bertscore", action="store_true")
    parser.add_argument(
        "--bertscore-only",
        action="store_true",
        help="Run only BERTScore for generated explanations; skip consistency, span, and ROUGE recomputation.",
    )
    parser.add_argument("--bertscore-model", default="klue/roberta-base")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-max-length", type=int, default=512)
    parser.add_argument("--bertscore-device", default="auto")
    parser.add_argument("--skip-existing-metrics", action="store_true")
    parser.add_argument("--format-percent", action="store_true")
    parser.add_argument("--include-run-dir", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    components = _parse_components(args.components)
    if args.bertscore_only:
        components = {"explanation"}
        args.compute_bertscore = True
    run_dirs = _collect_latest_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No matching run dirs found.")
    print("[runs]")
    for run_dir in run_dirs:
        print(f"  {_eval_id_from_run_dir(run_dir)} {run_dir}")

    base = Path(args.out_prefix)
    consistency_prefix = base.with_name(base.name + "_consistency")
    evidence_prefix = base.with_name(base.name + "_evidence")
    final_csv = base.with_name(base.name + "_quality_summary.csv")

    if args.bertscore_only:
        _run_metric_script(
            "compute_evidence_explanation_metrics.py",
            run_dirs,
            evidence_prefix,
            [
                "--bertscore-only",
                "--compute-bertscore",
                "--bertscore-model",
                args.bertscore_model,
                "--bertscore-batch-size",
                str(args.bertscore_batch_size),
                "--bertscore-max-length",
                str(args.bertscore_max_length),
                "--bertscore-device",
                args.bertscore_device,
            ],
        )
        _format_summary(
            run_dirs=run_dirs,
            consistency_prefix=consistency_prefix,
            evidence_prefix=evidence_prefix,
            final_csv=final_csv,
            components=components,
            format_percent=args.format_percent,
            include_run_dir=args.include_run_dir,
        )
        return 0

    if "consistency" in components and (
        not args.skip_existing_metrics or not consistency_prefix.with_suffix(".json").exists()
    ):
        _run_metric_script(
            "compute_consistency_metrics.py",
            run_dirs,
            consistency_prefix,
            ["--text-scope", "explanation"],
        )

    evidence_extra = []
    if args.compute_bertscore and "explanation" in components:
        evidence_extra.extend(
            [
                "--compute-bertscore",
                "--bertscore-model",
                args.bertscore_model,
                "--bertscore-batch-size",
                str(args.bertscore_batch_size),
                "--bertscore-max-length",
                str(args.bertscore_max_length),
                "--bertscore-device",
                args.bertscore_device,
            ]
        )
    if {"evidence", "explanation"} & components and (
        not args.skip_existing_metrics or not evidence_prefix.with_suffix(".json").exists()
    ):
        _run_metric_script(
            "compute_evidence_explanation_metrics.py",
            run_dirs,
            evidence_prefix,
            evidence_extra,
        )

    _format_summary(
        run_dirs=run_dirs,
        consistency_prefix=consistency_prefix,
        evidence_prefix=evidence_prefix,
        final_csv=final_csv,
        components=components,
        format_percent=args.format_percent,
        include_run_dir=args.include_run_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
