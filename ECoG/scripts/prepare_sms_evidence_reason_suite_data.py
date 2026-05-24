#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REASON_ROOT = ROOT / "data" / "sms" / "reason"
FUNCTIONAL_ROOT = ROOT / "data" / "arc" / "teacher_gpt_4o_mini" / "sms" / "functional_label01_batch"
FUNCTIONAL_EVAL_ROOT = ROOT / "data" / "arc" / "teacher_gpt_4o_mini" / "sms" / "functional_label01_evalsets_batch"
DEFAULT_OUT_ROOT = ROOT / "data" / "sms" / "evidence"
ARC_DROPPED_ROOT = ROOT / "data" / "arc" / "dropped_reason_rows" / "sms"


def repo_rel(path_like: str | Path) -> str:
    path = Path(path_like)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SMS reason-suite A/B/C/D CSVs with SMS functional evidence CSVs."
    )
    parser.add_argument(
        "--variant",
        choices=("keep",),
        default="keep",
        help="Functional evidence variant to materialize.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="Output root. Variant subdirectories are created under this root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a functional merge drops any reason rows.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults: dict[str, Any] = {
        "spans": "[]",
        "evidences": "[]",
        "use_evidence_loss": 0,
        "functional_valid": "",
        "label0_has_evidence": "",
        "functional_tags": "[]",
        "functional_reason": "",
        "functional_category": "",
        "functional_row_idx": "",
        "evidence_reason_merge": "reason_only",
    }
    for col, value in defaults.items():
        if col not in out.columns:
            out[col] = value
    return out


def _functional_split_path(*, split: str, variant: str) -> Path:
    run_name = f"gpt-4o-mini_sms_{split}_batch"
    if variant != "keep":
        raise ValueError(f"Unsupported evidence variant: {variant}")
    return FUNCTIONAL_ROOT / run_name / "functional_keep.csv"


def _functional_eval_path(*, letter: str, split: str, variant: str) -> Path:
    run_name = f"gpt-4o-mini_sms_{letter}_{split}_batch"
    if variant != "keep":
        raise ValueError(f"Unsupported evidence variant: {variant}")
    return FUNCTIONAL_EVAL_ROOT / run_name / "functional_keep.csv"


def _dropped_reason_rows_path(out_path: Path) -> Path:
    try:
        rel = out_path.relative_to(DEFAULT_OUT_ROOT)
    except ValueError:
        return out_path.with_suffix(".dropped_reason_rows.csv")
    return (ARC_DROPPED_ROOT / rel).with_suffix(".dropped_reason_rows.csv")


def _raw_split_path(*, letter: str, split: str) -> Path:
    sms_paths = {
        ("A", "train"): ROOT / "data" / "sms" / "in_domain" / "train.csv",
        ("A", "validation"): ROOT / "data" / "sms" / "in_domain" / "validation.csv",
        ("A", "test"): ROOT / "data" / "sms" / "in_domain" / "test.csv",
        ("A", "challenge"): ROOT / "data" / "sms" / "in_domain" / "challenging.csv",
        ("B", "train"): ROOT / "data" / "sms" / "ood" / "train" / "credit_train.csv",
        ("B", "validation"): ROOT / "data" / "sms" / "ood" / "validation" / "credit_validation.csv",
        ("B", "test"): ROOT / "data" / "sms" / "ood" / "test" / "credit_test.csv",
        ("B", "challenge"): ROOT / "data" / "sms" / "ood" / "challenging" / "credit_challenging.csv",
        ("C", "train"): ROOT / "data" / "sms" / "ood" / "train" / "finance_train.csv",
        ("C", "validation"): ROOT / "data" / "sms" / "ood" / "validation" / "finance_validation.csv",
        ("C", "test"): ROOT / "data" / "sms" / "ood" / "test" / "finance_test.csv",
        ("C", "challenge"): ROOT / "data" / "sms" / "ood" / "challenging" / "finance_challenging.csv",
        ("D", "train"): ROOT / "data" / "sms" / "ood" / "train" / "parcel_train.csv",
        ("D", "validation"): ROOT / "data" / "sms" / "ood" / "validation" / "parcel_validation.csv",
        ("D", "test"): ROOT / "data" / "sms" / "ood" / "test" / "parcel_test.csv",
        ("D", "challenge"): ROOT / "data" / "sms" / "ood" / "challenging" / "parcel_challenging.csv",
    }
    try:
        return sms_paths[(letter, split)]
    except KeyError as exc:
        raise ValueError(f"Unknown SMS raw split mapping for {letter}_{split}") from exc


def _merge_csv(
    *,
    reason_path: Path,
    functional_path: Path,
    out_path: Path,
    strict: bool,
) -> dict[str, Any]:
    reason = _read_csv(reason_path)
    functional = _read_csv(functional_path)
    required_reason = {"text", "category", "label", "reason_value"}
    required_functional = {"text", "category", "label", "spans", "evidences", "use_evidence_loss"}
    reason_missing = required_reason.difference(reason.columns)
    functional_missing = required_functional.difference(functional.columns)
    if reason_missing:
        raise ValueError(f"{reason_path} is missing required columns: {sorted(reason_missing)}")
    if functional_missing:
        raise ValueError(f"{functional_path} is missing required columns: {sorted(functional_missing)}")

    join_keys = ["text", "label", "category"]
    reason_dup = int(reason.duplicated(join_keys).sum())
    functional_dup = int(functional.duplicated(join_keys).sum())
    if reason_dup or functional_dup:
        raise ValueError(
            "Expected one-to-one text+label+category keys. "
            f"reason_dup={reason_dup}, functional_dup={functional_dup}"
        )

    functional_work = functional.copy()
    if "row_idx" in functional_work.columns:
        functional_work = functional_work.rename(columns={"row_idx": "functional_row_idx"})
    functional_work["functional_category"] = functional_work["category"]
    merged = reason.merge(functional_work, on=join_keys, how="inner", validate="one_to_one")
    dropped = int(len(reason) - len(merged))
    if strict and dropped:
        raise ValueError(
            f"{reason_path.name} dropped {dropped} reason rows when merged with {functional_path.name}"
        )

    merged["evidence_reason_merge"] = "text_label_inner"
    merged = _ensure_columns(merged)
    preferred = [
        "text",
        "category",
        "label",
        "reason_value",
        "spans",
        "evidences",
        "use_evidence_loss",
        "functional_valid",
        "label0_has_evidence",
        "functional_tags",
        "functional_reason",
        "functional_category",
        "functional_row_idx",
        "evidence_reason_merge",
    ]
    cols = [col for col in preferred if col in merged.columns] + [
        col for col in merged.columns if col not in preferred
    ]
    merged = merged[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    mismatch_path = None
    if dropped:
        missing = reason.merge(
            functional_work[join_keys],
            on=join_keys,
            how="left",
            indicator=True,
        )
        missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
        mismatch_path = _dropped_reason_rows_path(out_path)
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(mismatch_path, index=False, encoding="utf-8-sig")

    return {
        "output_csv": repo_rel(out_path),
        "reason_csv": repo_rel(reason_path),
        "functional_csv": repo_rel(functional_path),
        "reason_rows": int(len(reason)),
        "functional_rows": int(len(functional)),
        "output_rows": int(len(merged)),
        "dropped_reason_rows": dropped,
        "dropped_reason_rows_csv": repo_rel(mismatch_path) if mismatch_path else None,
        "use_evidence_loss_sum": int(merged["use_evidence_loss"].fillna(0).astype(int).sum()),
        "merge_type": "text_label_inner",
    }


def _merge_via_raw_csv(
    *,
    raw_path: Path,
    reason_path: Path,
    functional_path: Path,
    out_path: Path,
    strict: bool,
) -> dict[str, Any]:
    raw = _read_csv(raw_path)
    reason = _read_csv(reason_path)
    functional = _read_csv(functional_path)

    required_raw = {"text", "category", "label"}
    required_reason = {"text", "category", "label", "reason_value"}
    required_functional = {"text", "category", "label", "spans", "evidences", "use_evidence_loss"}
    raw_missing = required_raw.difference(raw.columns)
    reason_missing = required_reason.difference(reason.columns)
    functional_missing = required_functional.difference(functional.columns)
    if raw_missing:
        raise ValueError(f"{raw_path} is missing required columns: {sorted(raw_missing)}")
    if reason_missing:
        raise ValueError(f"{reason_path} is missing required columns: {sorted(reason_missing)}")
    if functional_missing:
        raise ValueError(f"{functional_path} is missing required columns: {sorted(functional_missing)}")

    join_keys = ["text", "label", "category"]
    raw_work = raw.copy()
    raw_work["text"] = raw_work["text"].fillna("").astype(str)
    raw_work["label"] = raw_work["label"].astype(int)
    raw_work["category"] = raw_work["category"].fillna("").astype(str)

    reason_work = reason.copy()
    reason_work["text"] = reason_work["text"].fillna("").astype(str)
    reason_work["label"] = reason_work["label"].astype(int)
    reason_work["category"] = reason_work["category"].fillna("").astype(str)
    reason_dup = int(reason_work.duplicated(join_keys).sum())
    reason_map = reason_work.drop_duplicates(subset=join_keys, keep="first")

    functional_work = functional.copy()
    functional_work["text"] = functional_work["text"].fillna("").astype(str)
    functional_work["label"] = functional_work["label"].astype(int)
    functional_work["category"] = functional_work["category"].fillna("").astype(str)
    functional_dup = int(functional_work.duplicated(join_keys).sum())
    functional_map = functional_work.drop_duplicates(subset=join_keys, keep="first")
    if "row_idx" in functional_work.columns:
        functional_work = functional_work.rename(columns={"row_idx": "functional_row_idx"})
        functional_map = functional_map.rename(columns={"row_idx": "functional_row_idx"})
    functional_work["functional_category"] = functional_work["category"]
    functional_map["functional_category"] = functional_map["category"]

    merged = raw_work.merge(
        reason_map[["text", "label", "category", "reason_value"]],
        on=join_keys,
        how="left",
        validate="m:1",
    )
    missing_reason_count = int(merged["reason_value"].isna().sum())
    if missing_reason_count:
        raise ValueError(
            f"{out_path.name}: {missing_reason_count} rows missing reason targets after raw merge"
        )

    merged = merged.merge(
        functional_map,
        on=join_keys,
        how="left",
        validate="m:1",
        indicator="_functional_merge",
    )
    dropped = int((merged["_functional_merge"] == "left_only").sum())
    if strict and dropped:
        raise ValueError(
            f"{out_path.name}: {dropped} raw rows missing functional evidence after merge"
        )

    mismatch_path = None
    if dropped:
        missing = merged.loc[merged["_functional_merge"] == "left_only", raw.columns.tolist()].copy()
        mismatch_path = _dropped_reason_rows_path(out_path)
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(mismatch_path, index=False, encoding="utf-8-sig")

    merged = merged.drop(columns=["_functional_merge"])
    merged["evidence_reason_merge"] = "raw_reason_functional_left"
    merged = _ensure_columns(merged)
    preferred = [
        "text",
        "category",
        "label",
        "reason_value",
        "spans",
        "evidences",
        "use_evidence_loss",
        "functional_valid",
        "label0_has_evidence",
        "functional_tags",
        "functional_reason",
        "functional_category",
        "functional_row_idx",
        "evidence_reason_merge",
    ]
    cols = [col for col in preferred if col in merged.columns] + [
        col for col in merged.columns if col not in preferred
    ]
    merged = merged[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")

    return {
        "output_csv": repo_rel(out_path),
        "raw_csv": repo_rel(raw_path),
        "reason_csv": repo_rel(reason_path),
        "functional_csv": repo_rel(functional_path),
        "raw_rows": int(len(raw_work)),
        "reason_rows": int(len(reason)),
        "reason_unique_rows": int(len(reason_map)),
        "reason_dup_rows": reason_dup,
        "functional_rows": int(len(functional)),
        "functional_unique_rows": int(len(functional_map)),
        "functional_dup_rows": functional_dup,
        "output_rows": int(len(merged)),
        "dropped_reason_rows": dropped,
        "dropped_reason_rows_csv": repo_rel(mismatch_path) if mismatch_path else None,
        "use_evidence_loss_sum": int(merged["use_evidence_loss"].fillna(0).astype(int).sum()),
        "merge_type": "raw_reason_functional_left",
    }


def prepare_variant(*, variant: str, out_root: Path, strict: bool) -> dict[str, Any]:
    variant_root = out_root / variant
    rows: list[dict[str, Any]] = []

    split_map = {
        "train": _functional_split_path(split="train", variant=variant),
        "validation": _functional_split_path(split="validation", variant=variant),
    }

    for letter in ("A", "B", "C", "D"):
        for split in ("train", "validation"):
            rows.append(
                {
                    "scenario": letter,
                    "split": split,
                    **_merge_csv(
                        reason_path=REASON_ROOT / f"{letter}_{split}.csv",
                        functional_path=split_map[split],
                        out_path=variant_root / f"{letter}_{split}.csv",
                        strict=strict,
                    ),
                }
            )

    for letter in ("A", "B", "C", "D"):
        for split in ("test", "challenge"):
            rows.append(
                {
                    "scenario": letter,
                    "split": split,
                    **_merge_via_raw_csv(
                        raw_path=_raw_split_path(letter=letter, split=split),
                        reason_path=REASON_ROOT / f"{letter}_{split}.csv",
                        functional_path=_functional_eval_path(letter=letter, split=split, variant=variant),
                        out_path=variant_root / f"{letter}_{split}.csv",
                        strict=strict,
                    ),
                }
            )

    manifest = {
        "variant": variant,
        "generated_at": datetime.now().isoformat(),
        "out_root": repo_rel(variant_root),
        "rows": rows,
    }
    variant_root.mkdir(parents=True, exist_ok=True)
    manifest_path = variant_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(variant_root / "manifest.csv", index=False, encoding="utf-8-sig")
    return manifest


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    manifests = [prepare_variant(variant=str(args.variant), out_root=out_root, strict=bool(args.strict))]
    print(json.dumps({"prepared": [m["out_root"] for m in manifests]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
