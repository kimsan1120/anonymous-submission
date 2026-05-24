#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

LINE_SPECS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("A1", [("A", "A1", "credit"), ("A", "A1", "finance"), ("A", "A1", "parcel")]),
    ("A2", [("A", "A2", "credit"), ("A", "A2", "finance"), ("A", "A2", "parcel")]),
    ("B1, C1, D1", [("B", "B1", "credit"), ("C", "C1", "finance"), ("D", "D1", "parcel")]),
    ("B2, C2, D2", [("B", "B2", "credit"), ("C", "C2", "finance"), ("D", "D2", "parcel")]),
    ("E1", [("E", "E1", "government"), ("E", "E1", "finance")]),
    ("E2", [("E", "E2", "government"), ("E", "E2", "finance")]),
    ("G1, F1", [("G", "G1", "government"), ("F", "F1", "finance")]),
    ("G2, F2", [("G", "G2", "government"), ("F", "F2", "finance")]),
]

STAGE1_SPLIT_MAP: dict[str, tuple[str, str]] = {
    "A1": ("A", "eval"),
    "A2": ("A", "test"),
    "B1": ("B", "eval"),
    "B2": ("B", "test"),
    "C1": ("C", "eval"),
    "C2": ("C", "test"),
    "D1": ("D", "eval"),
    "D2": ("D", "test"),
    "E1": ("E", "eval"),
    "E2": ("E", "test"),
    "F1": ("F", "eval"),
    "F2": ("F", "test"),
    "G1": ("G", "eval"),
    "G2": ("G", "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract benchmark metrics in the A1/A2/B1,C1,D1/... tab-separated format."
    )
    parser.add_argument("roots", nargs="+")
    parser.add_argument(
        "--include-200",
        action="store_true",

    )
    parser.add_argument(
        "--include-root-label",
        action="store_true",

    )
    parser.add_argument(
        "--output-format",
        choices=("legacy-tsv", "main-macro-f1-csv", "classification-detail-csv"),
        default="legacy-tsv",
        help=(

        ),
    )
    parser.add_argument(
        "--allow-partial-lines",
        action="store_true",
        help=(

        ),
    )
    return parser.parse_args()


def latest_run_dir(path: Path, include_200: bool = False) -> Path | None:
    candidates = (
        [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]
        if path.exists()
        else []
    )

    if not include_200:
        candidates = [p for p in candidates if "200" not in p.name]

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def has_only_200_standard_runs(root: Path) -> bool:
    for letter_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for eval_id_dir in sorted(p for p in letter_dir.iterdir() if p.is_dir()):
            candidates = [
                p
                for p in eval_id_dir.iterdir()
                if p.is_dir() and not p.name.startswith(("_", "."))
            ]
            if candidates and all("200" in p.name for p in candidates):
                return True
    return False


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_binary_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    ys = [int(row["label"]) for row in rows]
    ps = [int(row.get("predicted", row.get("pred", 0))) for row in rows]

    n = len(rows)
    acc = sum(int(y == p) for y, p in zip(ys, ps)) / n if n else 0.0

    out: dict[str, float] = {"accuracy": acc}

    for cls in (0, 1):
        tp = sum(int(y == cls and p == cls) for y, p in zip(ys, ps))
        fp = sum(int(y != cls and p == cls) for y, p in zip(ys, ps))
        fn = sum(int(y == cls and p != cls) for y, p in zip(ys, ps))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        out[f"class_{cls}_recall"] = recall
        out[f"class_{cls}_f1"] = f1

    out["macro_f1"] = (out["class_0_f1"] + out["class_1_f1"]) / 2.0
    return out


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def metrics_from_results_csv(run_dir: Path) -> dict[str, dict[str, float]] | None:
    csv_paths = sorted(run_dir.glob("results_*.csv"))

    if not csv_paths:
        return None

    rows = load_csv_rows(csv_paths[-1])

    by_category: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_category.setdefault(str(row.get("category", "")), []).append(row)

    return {cat: compute_binary_metrics(cat_rows) for cat, cat_rows in by_category.items()}


def normalize_category_metrics(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}

    for category, raw in (payload.get("by_category") or {}).items():
        class_1 = raw.get("class_1") or {}

        out[str(category)] = {
            "accuracy": float(raw.get("accuracy", 0.0)),
            "class_1_recall": float(class_1.get("recall", raw.get("positive_recall", 0.0))),
            "macro_f1": float(raw.get("macro_f1", 0.0)),
        }

    return out


def extract_standard_root(
    root: Path,
    include_200: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}

    for letter_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for eval_id_dir in sorted(p for p in letter_dir.iterdir() if p.is_dir()):
            eval_id = eval_id_dir.name
            run_dir = latest_run_dir(eval_id_dir, include_200=include_200)

            if run_dir is None:
                continue

            metrics_path = run_dir / "metrics.json"
            metrics_by_cat: dict[str, dict[str, float]] = {}

            if metrics_path.exists():
                metrics_by_cat = normalize_category_metrics(read_json(metrics_path))

            if not metrics_by_cat:
                metrics_by_cat = metrics_from_results_csv(run_dir) or {}

            if metrics_by_cat:
                out[eval_id] = metrics_by_cat

    return out


def extract_stage1_root(
    root: Path,
    include_200: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}

    for eval_id, (letter, split) in STAGE1_SPLIT_MAP.items():
        letter_dir = root / letter
        run_dir = latest_run_dir(letter_dir, include_200=include_200)

        if run_dir is None:
            continue

        csv_path = run_dir / f"{split}_predictions.csv"

        if not csv_path.exists():
            continue

        rows = load_csv_rows(csv_path)

        by_category: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_category.setdefault(str(row.get("category", "")), []).append(row)

        out[eval_id] = {
            cat: compute_binary_metrics(cat_rows)
            for cat, cat_rows in by_category.items()
        }

    return out


def extract_direct_metrics_root(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Read label_dlml-style roots: {model}/{letter}/{eval_id}_metrics.json."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for eval_id, (letter, _split) in STAGE1_SPLIT_MAP.items():
        metrics_path = root / letter / f"{eval_id}_metrics.json"
        if not metrics_path.exists():
            continue
        metrics_by_cat = normalize_category_metrics(read_json(metrics_path))
        if metrics_by_cat:
            out[eval_id] = metrics_by_cat
    return out


def extract_encoder_root(
    root: Path,
    include_200: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """Read label_encoder-style roots: {model}/{letter}/{run}/metrics.json + post_eval/{eval_id}/metrics.json."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for letter in "ABCDEFG":
        letter_dir = root / letter
        run_dir = latest_run_dir(letter_dir, include_200=include_200)
        if run_dir is None:
            continue

        eval1 = f"{letter}1"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics_by_cat = normalize_category_metrics(read_json(metrics_path))
            if metrics_by_cat:
                out[eval1] = metrics_by_cat

        eval2 = f"{letter}2"
        post_metrics_path = run_dir / "post_eval" / eval2 / "metrics.json"
        if post_metrics_path.exists():
            metrics_by_cat = normalize_category_metrics(read_json(post_metrics_path))
            if metrics_by_cat:
                out[eval2] = metrics_by_cat
    return out


def detect_direct_metrics_root(root: Path) -> bool:
    return any((root / letter / f"{letter}{idx}_metrics.json").exists() for letter in "ABCDEFG" for idx in (1, 2))


def detect_encoder_root(root: Path, include_200: bool = False) -> bool:
    for letter in "ABCDEFG":
        run_dir = latest_run_dir(root / letter, include_200=include_200)
        if run_dir is not None and (run_dir / "metrics.json").exists() and (run_dir / "post_eval").exists():
            return True
    return False


def detect_stage1_root(root: Path, include_200: bool = False) -> bool:
    letter_dir = root / "A"
    run_dir = latest_run_dir(letter_dir, include_200=include_200) if letter_dir.exists() else None

    if run_dir is None:
        return False

    return (run_dir / "eval_predictions.csv").exists() or (
        run_dir / "test_predictions.csv"
    ).exists()


def format_block(extracted: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    lines: list[str] = []

    for line_label, specs in LINE_SPECS:
        vals: list[str] = []
        ok = True

        for _letter, eval_id, category in specs:
            category_metrics = extracted.get(eval_id, {}).get(category)

            if category_metrics is None:
                ok = False
                break

            vals.extend(
                [
                    f"{category_metrics['accuracy'] * 100:.2f}",
                    f"{category_metrics['class_1_recall'] * 100:.2f}",
                    f"{category_metrics['macro_f1'] * 100:.2f}",
                ]
            )

        if ok:
            lines.append(line_label + "\t" + "\t".join(vals))

    return lines


def _percent(value: float) -> str:
    return f"{float(value) * 100.0:.2f}"


MAIN_MACRO_FIELDNAMES = [
    "Eval",
    "Credit/Government Macro-F1",
    "Finance Macro-F1",
    "Parcel Macro-F1",
]


DETAIL_FIELDNAMES = [
    "Eval",
    "Credit/Government Accuracy",
    "Credit/Government Recall-1",
    "Credit/Government Macro-F1",
    "Finance Accuracy",
    "Finance Recall-1",
    "Finance Macro-F1",
    "Parcel Accuracy",
    "Parcel Recall-1",
    "Parcel Macro-F1",
]


def format_main_macro_rows(
    extracted: dict[str, dict[str, dict[str, float]]],
    *,
    allow_partial_lines: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for line_label, specs in LINE_SPECS:
        values = ["", "", ""]
        found = False

        for idx, (_letter, eval_id, category) in enumerate(specs):
            category_metrics = extracted.get(eval_id, {}).get(category)
            if category_metrics is None:
                continue
            found = True
            values[idx] = _percent(category_metrics["macro_f1"])

        if found and (allow_partial_lines or all(values)):
            rows.append(
                {
                    "Eval": line_label,
                    "Credit/Government Macro-F1": values[0],
                    "Finance Macro-F1": values[1],
                    "Parcel Macro-F1": values[2],
                }
            )

    return rows


def format_classification_detail_rows(
    extracted: dict[str, dict[str, dict[str, float]]],
    *,
    allow_partial_lines: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for line_label, specs in LINE_SPECS:
        row = {field: "" for field in DETAIL_FIELDNAMES}
        row["Eval"] = line_label
        found = False

        for idx, (_letter, eval_id, category) in enumerate(specs):
            category_metrics = extracted.get(eval_id, {}).get(category)
            if category_metrics is None:
                continue
            found = True

            if idx == 0:
                prefix = "Credit/Government"
            elif idx == 1:
                prefix = "Finance"
            else:
                prefix = "Parcel"

            row[f"{prefix} Accuracy"] = _percent(category_metrics["accuracy"])
            row[f"{prefix} Recall-1"] = _percent(category_metrics["class_1_recall"])
            row[f"{prefix} Macro-F1"] = _percent(category_metrics["macro_f1"])

        if found and (
            allow_partial_lines
            or all(row[field] for field in DETAIL_FIELDNAMES if field != "Eval")
        ):
            rows.append(row)

    return rows


def write_csv_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main() -> int:
    args = parse_args()
    csv_mode = args.output_format != "legacy-tsv"

    if csv_mode and len(args.roots) != 1:
        raise SystemExit("--output-format *-csv supports exactly one root.")

    for idx, raw_root in enumerate(args.roots):
        root = Path(raw_root)

        if not root.is_absolute():
            root = ROOT / root

        if not root.exists():
            raise FileNotFoundError(f"Root not found: {root}")

        if detect_direct_metrics_root(root):
            extracted = extract_direct_metrics_root(root)
        elif detect_encoder_root(root, include_200=args.include_200):
            extracted = extract_encoder_root(root, include_200=args.include_200)
        elif detect_stage1_root(root, include_200=args.include_200):
            extracted = extract_stage1_root(root, include_200=args.include_200)
        else:
            extracted = extract_standard_root(root, include_200=args.include_200)

        if args.output_format == "main-macro-f1-csv":
            rows = format_main_macro_rows(extracted, allow_partial_lines=args.allow_partial_lines)
            if not rows:
                raise SystemExit(f"No metrics extracted from {root}.")
            write_csv_rows(rows, MAIN_MACRO_FIELDNAMES)
            continue

        if args.output_format == "classification-detail-csv":
            rows = format_classification_detail_rows(
                extracted,
                allow_partial_lines=args.allow_partial_lines,
            )
            if not rows:
                raise SystemExit(f"No metrics extracted from {root}.")
            write_csv_rows(rows, DETAIL_FIELDNAMES)
            continue

        lines = format_block(extracted)
        if not lines:
            if not args.include_200 and has_only_200_standard_runs(root):
                raise SystemExit(
                    f"No metrics extracted from {root}: all available run directories contain "
                    '"200". Re-run with --include-200.'
                )
            raise SystemExit(f"No metrics extracted from {root}.")

        if idx > 0:
            print()

        if args.include_root_label:
            print(root.relative_to(ROOT) if root.is_relative_to(ROOT) else root)

        for line in lines:
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
