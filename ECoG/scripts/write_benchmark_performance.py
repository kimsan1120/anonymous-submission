#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TYPES = {
    "label_decoder",
    "label_dlml",
    "label_encoder",
    "label_evidence",
    "label_evidence_explanation",
    "label_explanation",
}
QUALITY_COMPONENTS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "label_decoder": (),
    "label_dlml": (),
    "label_encoder": (),
    "label_evidence": ("evidence",),
    "label_explanation": ("consistency", "explanation"),
    "label_evidence_explanation": ("consistency", "evidence", "explanation"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write benchmark performance artifacts under outputs/runs/benchmark_performance. "
            "Classification is extracted with extract_benchmark_metrics.py; optional fullgen quality "
            "is extracted with compute_fullgen_quality_report.py."
        )
    )
    parser.add_argument("--benchmark-type", choices=sorted(BENCHMARK_TYPES), required=True)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--bench-name", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--benchmark-layout",
        choices=("auto", "nested", "flat"),
        default="auto",
        help=(

        ),
    )
    parser.add_argument("--out-root", default="outputs/runs/benchmark_performance")
    parser.add_argument("--run-name-contains", default="fullgen200")
    parser.add_argument("--decode-max-new-tokens", type=int, default=200)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--include-run-dir", action="store_true")
    parser.add_argument("--format-percent", action="store_true", default=True)
    parser.add_argument("--raw-rates", dest="format_percent", action="store_false")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument(
        "--quality-mode",
        choices=("all", "cpu", "bertscore-only"),
        default="all",
        help=(

        ),
    )
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="klue/roberta-base")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--bertscore-max-length", type=int, default=512)
    parser.add_argument("--bertscore-device", default="auto")
    return parser.parse_args()


def repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def resolved_benchmark_layout(args: argparse.Namespace) -> str:
    layout = str(args.benchmark_layout or "auto").strip().lower()
    if layout in {"nested", "flat"}:
        return layout

    base = ROOT / "outputs" / "runs" / "benchmarks" / args.suite_name
    nested = base / args.bench_name / args.model_alias
    flat = base / args.model_alias
    if nested.exists():
        return "nested"
    if flat.exists():
        return "flat"
    return "nested"


def raw_model_root(args: argparse.Namespace) -> Path:
    base = ROOT / "outputs" / "runs" / "benchmarks" / args.suite_name
    if resolved_benchmark_layout(args) == "flat":
        return base / args.model_alias
    return base / args.bench_name / args.model_alias


def classification_root(args: argparse.Namespace) -> Path:
    model_root = raw_model_root(args)
    eval_root = model_root / "eval"
    return eval_root if eval_root.exists() else model_root


def output_dir(args: argparse.Namespace) -> Path:
    return repo_path(args.out_root) / args.benchmark_type / args.model_alias


def output_prefix(args: argparse.Namespace) -> Path:
    name = str(args.bench_name or args.model_alias).strip() or args.model_alias
    return output_dir(args) / name


def main_macro_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + "_main_macro_f1.csv")


def generated_quality_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + "_generated_quality.csv")


def classification_detail_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + "_classification_detail.csv")


def cleanup_existing_outputs(prefix: Path, keep: set[Path]) -> None:
    if not prefix.parent.exists():
        return
    keep_resolved = {path.resolve() for path in keep}
    for path in prefix.parent.glob(prefix.name + "_*"):
        if not path.is_file():
            continue
        if path.resolve() in keep_resolved:
            continue
        path.unlink()
        print(f"[remove] {path}")


def run_subprocess(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("[run]", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def _write_classification_csv(
    args: argparse.Namespace,
    *,
    output_format: str,
    out_file: Path,
) -> Path | None:
    if args.skip_classification:
        return None
    root = classification_root(args)
    if not root.exists():
        if args.allow_missing:
            print(f"[warn] classification root missing: {root}")
            return None
        raise FileNotFoundError(root)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "extract_benchmark_metrics.py"),
        "--include-200",
        "--output-format",
        output_format,
        str(root),
    ]
    if args.allow_missing:
        cmd.insert(-1, "--allow-partial-lines")
    result = run_subprocess(cmd, capture=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(result.stdout, encoding="utf-8")
    print(f"[write] {out_file}")
    return out_file


def write_classification(args: argparse.Namespace, prefix: Path) -> tuple[Path | None, Path | None]:
    if args.skip_classification:
        return None, None
    main_path = _write_classification_csv(
        args,
        output_format="main-macro-f1-csv",
        out_file=main_macro_path(prefix),
    )
    detail_path = _write_classification_csv(
        args,
        output_format="classification-detail-csv",
        out_file=classification_detail_path(prefix),
    )
    return main_path, detail_path


QUALITY_FIELD_MAP_BY_TYPE: dict[str, list[tuple[str, str]]] = {
    "label_evidence": [
        ("Span F1", "Span F1"),
    ],
    "label_explanation": [
        ("Inconsistency strict rate", "Inconsistency"),
        ("BERTScore F1", "BERTScore"),
        ("ROUGE-L F1", "ROUGE-L"),
    ],
    "label_evidence_explanation": [
        ("Inconsistency strict rate", "Inconsistency"),
        ("Span F1", "Span F1"),
        ("BERTScore F1", "BERTScore"),
        ("ROUGE-L F1", "ROUGE-L"),
    ],
}


def write_compact_quality_csv(
    *,
    source_csv: Path,
    out_file: Path,
    benchmark_type: str,
) -> Path:
    field_map = QUALITY_FIELD_MAP_BY_TYPE.get(benchmark_type, [])
    fieldnames = ["Eval", *[out_name for _in_name, out_name in field_map]]
    with source_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Eval": row.get("Eval", ""),
                    **{out_name: row.get(in_name, "") for in_name, out_name in field_map},
                }
            )
    print(f"[write] {out_file}")
    return out_file


def merge_bertscore_quality_csv(
    *,
    source_csv: Path,
    out_file: Path,
    benchmark_type: str,
) -> Path:
    field_map = QUALITY_FIELD_MAP_BY_TYPE.get(benchmark_type, [])
    fieldnames = ["Eval", *[out_name for _in_name, out_name in field_map]]
    if "BERTScore" not in fieldnames:
        print(f"[skip] {benchmark_type} has no BERTScore column.")
        return out_file

    existing_rows: list[dict[str, str]] = []
    if out_file.exists():
        with out_file.open("r", encoding="utf-8-sig", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        for row in existing_rows:
            if "BERTScore F1" in row and "BERTScore" not in row:
                row["BERTScore"] = row.get("BERTScore F1", "")

    by_eval = {str(row.get("Eval", "")): dict(row) for row in existing_rows if row.get("Eval")}
    order = [str(row.get("Eval", "")) for row in existing_rows if row.get("Eval")]
    with source_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for source_row in csv.DictReader(f):
            eval_id = str(source_row.get("Eval", ""))
            if not eval_id:
                continue
            row = by_eval.setdefault(eval_id, {"Eval": eval_id})
            if eval_id not in order:
                order.append(eval_id)
            row["BERTScore"] = source_row.get("BERTScore F1", source_row.get("BERTScore", ""))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for eval_id in order:
            row = by_eval[eval_id]
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    print(f"[merge] {out_file} <- BERTScore from {source_csv}")
    return out_file


def run_quality_report(
    args: argparse.Namespace,
    *,
    tmp_prefix: Path,
    components: tuple[str, ...],
    compute_bertscore: bool,
    bertscore_only: bool,
) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "compute_fullgen_quality_report.py"),
        "--suite-name",
        args.suite_name,
        "--bench-name",
        args.bench_name,
        "--benchmark-layout",
        resolved_benchmark_layout(args),
        "--model-alias",
        args.model_alias,
        "--run-name-contains",
        args.run_name_contains,
        "--components",
        ",".join(components),
        "--out-prefix",
        str(tmp_prefix),
    ]
    if args.allow_missing:
        cmd.append("--allow-missing")
    if args.format_percent:
        cmd.append("--format-percent")
    if bertscore_only:
        cmd.append("--bertscore-only")
    if compute_bertscore:
        cmd.extend(
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

    run_subprocess(cmd)
    return tmp_prefix.with_name(tmp_prefix.name + "_quality_summary.csv")


def write_quality(args: argparse.Namespace, prefix: Path) -> Path | None:
    components = QUALITY_COMPONENTS_BY_TYPE[args.benchmark_type]
    if args.skip_quality or not components:
        return None

    out_file = generated_quality_path(prefix)
    if args.quality_mode == "bertscore-only":
        if "explanation" not in components:
            print(f"[skip] {args.benchmark_type} has no explanation/BERTScore quality metric.")
            return out_file
        with tempfile.TemporaryDirectory(prefix="benchmark_bertscore_") as tmp_dir_raw:
            tmp_prefix = Path(tmp_dir_raw) / f"{prefix.name}_maxnew{int(args.decode_max_new_tokens)}"
            source_csv = run_quality_report(
                args,
                tmp_prefix=tmp_prefix,
                components=("explanation",),
                compute_bertscore=True,
                bertscore_only=True,
            )
            return merge_bertscore_quality_csv(
                source_csv=source_csv,
                out_file=out_file,
                benchmark_type=args.benchmark_type,
            )

    with tempfile.TemporaryDirectory(prefix="benchmark_quality_") as tmp_dir_raw:
        tmp_prefix = Path(tmp_dir_raw) / f"{prefix.name}_maxnew{int(args.decode_max_new_tokens)}"
        source_csv = run_quality_report(
            args,
            tmp_prefix=tmp_prefix,
            components=components,
            compute_bertscore=(
                "explanation" in components
                and args.quality_mode == "all"
                and not args.skip_bertscore
            ),
            bertscore_only=False,
        )
        return write_compact_quality_csv(
            source_csv=source_csv,
            out_file=out_file,
            benchmark_type=args.benchmark_type,
        )


def main() -> int:
    args = parse_args()
    prefix = output_prefix(args)
    if args.quality_mode == "bertscore-only":
        write_quality(args, prefix)
        return 0

    main_classification_path, detail_classification_path = write_classification(args, prefix)
    quality_path = write_quality(args, prefix)
    final_paths = {
        path
        for path in (main_classification_path, quality_path, detail_classification_path)
        if path is not None
    }
    cleanup_existing_outputs(prefix, final_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
