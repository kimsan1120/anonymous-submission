#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REASON_ROOT = ROOT / "data" / "voice" / "reason"
DEFAULT_OUT_ROOT = ROOT / "data" / "voice" / "evidence"
ARC_DROPPED_ROOT = ROOT / "data" / "arc" / "dropped_reason_rows" / "voice"
FUNCTIONAL_EVAL_ROOT = ROOT / "data" / "arc" / "teacher_gpt_4o_mini" / "voice" / "teacher" / "functional_label01_evalsets_batch"


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
        description="Merge voice reason-suite E/F/G CSVs with voice functional evidence CSVs."
    )
    parser.add_argument(
        "--variant",
        choices=("keep",),
        default="keep",
        help="Functional train evidence variant to materialize. Validation gold is shared.",
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
    return pd.read_csv(path)


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


def _dropped_reason_rows_path(out_path: Path) -> Path:
    try:
        rel = out_path.relative_to(DEFAULT_OUT_ROOT)
    except ValueError:
        return out_path.with_suffix(".dropped_reason_rows.csv")
    return (ARC_DROPPED_ROOT / rel).with_suffix(".dropped_reason_rows.csv")


def _reason_only_csv(reason_path: Path, out_path: Path) -> dict[str, Any]:
    reason = _read_csv(reason_path)
    required = {"text", "category", "label", "reason_value"}
    missing = required.difference(reason.columns)
    if missing:
        raise ValueError(f"{reason_path} is missing required columns: {sorted(missing)}")
    out = reason.copy()
    out["spans"] = "[]"
    out["evidences"] = "[]"
    out["use_evidence_loss"] = 0
    out["functional_valid"] = ""
    out["label0_has_evidence"] = ""
    out["functional_tags"] = "[]"
    out["functional_reason"] = ""
    out["functional_category"] = ""
    out["functional_row_idx"] = ""
    out["evidence_reason_merge"] = "reason_only"
    out = _ensure_columns(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return {
        "output_csv": repo_rel(out_path),
        "reason_csv": repo_rel(reason_path),
        "functional_csv": None,
        "reason_rows": int(len(reason)),
        "functional_rows": 0,
        "output_rows": int(len(out)),
        "dropped_reason_rows": 0,
        "use_evidence_loss_sum": 0,
        "merge_type": "reason_only",
    }


def _concat_merged_csvs(
    *,
    letter: str,
    split: str,
    reason_path: Path,
    source_paths: list[Path],
    out_path: Path,
) -> dict[str, Any]:
    reason = _read_csv(reason_path)
    frames = [_read_csv(path) for path in source_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = _ensure_columns(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    reason_keys = set(zip(reason["text"].astype(str), reason["label"].astype(int)))
    merged_keys = set(zip(merged["text"].astype(str), merged["label"].astype(int)))
    dropped = max(0, int(len(reason) - len(merged)))
    key_missing = reason_keys - merged_keys
    mismatch_path = None
    if key_missing:
        missing = reason[
            [
                (str(row["text"]), int(row["label"])) in key_missing
                for _, row in reason.iterrows()
            ]
        ].copy()
        mismatch_path = _dropped_reason_rows_path(out_path)
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(mismatch_path, index=False)

    return {
        "output_csv": repo_rel(out_path),
        "reason_csv": repo_rel(reason_path),
        "functional_csv": "+".join(repo_rel(path) for path in source_paths),
        "reason_rows": int(len(reason)),
        "functional_rows": int(sum(len(frame) for frame in frames)),
        "output_rows": int(len(merged)),
        "dropped_reason_rows": dropped,
        "dropped_reason_key_count": int(len(key_missing)),
        "dropped_reason_rows_csv": repo_rel(mismatch_path) if mismatch_path else None,
        "use_evidence_loss_sum": int(merged["use_evidence_loss"].fillna(0).astype(int).sum()),
        "merge_type": f"{letter}_{split}_concat_F_G_merged",
    }


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

    reason_dup = int(reason.duplicated(["text", "label"]).sum())
    functional_dup = int(functional.duplicated(["text", "label"]).sum())
    if reason_dup or functional_dup:
        raise ValueError(
            "Expected one-to-one text+label keys. "
            f"reason_dup={reason_dup}, functional_dup={functional_dup}"
        )

    functional_work = functional.copy()
    rename_map = {"category": "functional_category", "row_idx": "functional_row_idx"}
    functional_work = functional_work.rename(
        columns={k: v for k, v in rename_map.items() if k in functional_work.columns}
    )
    merged = reason.merge(functional_work, on=["text", "label"], how="inner", validate="one_to_one")
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
    merged.to_csv(out_path, index=False)

    mismatch_path = None
    if dropped:
        missing = reason.merge(
            functional_work[["text", "label"]],
            on=["text", "label"],
            how="left",
            indicator=True,
        )
        missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
        mismatch_path = _dropped_reason_rows_path(out_path)
        mismatch_path.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(mismatch_path, index=False)

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


def _functional_train_path(letter: str, variant: str) -> Path:
    if variant != "keep":
        raise ValueError(f"Unsupported evidence variant: {variant}")
    if letter == "F":
        return ROOT / "data" / "voice" / "ood" / "train" / f"ood_train_government_functional_{variant}_pilot.csv"
    if letter == "G":
        return ROOT / "data" / "voice" / "ood" / "train" / f"ood_train_finance_functional_{variant}_pilot.csv"
    raise ValueError(f"No functional train path for scenario {letter}")


def _functional_validation_path(letter: str) -> Path:
    if letter == "F":
        return ROOT / "data" / "voice" / "ood" / "validation" / "ood_validation_finance_functional_gold.csv"
    if letter == "G":
        return ROOT / "data" / "voice" / "ood" / "validation" / "ood_validation_government_functional_gold.csv"
    raise ValueError(f"No functional validation path for scenario {letter}")


def _functional_eval_path(*, letter: str, split: str, variant: str) -> Path:
    run_name = f"gpt-4o-mini_voice_{letter}_{split}_batch"
    if variant != "keep":
        raise ValueError(f"Unsupported evidence variant: {variant}")
    return FUNCTIONAL_EVAL_ROOT / run_name / "functional_keep.csv"


def prepare_variant(*, variant: str, out_root: Path, strict: bool) -> dict[str, Any]:
    variant_root = out_root / variant
    rows: list[dict[str, Any]] = []

    for letter in ("F", "G"):
        rows.append(
            {
                "scenario": letter,
                "split": "train",
                **_merge_csv(
                    reason_path=REASON_ROOT / f"{letter}_train.csv",
                    functional_path=_functional_train_path(letter, variant),
                    out_path=variant_root / f"{letter}_train.csv",
                    strict=strict,
                ),
            }
        )
        rows.append(
            {
                "scenario": letter,
                "split": "validation",
                **_merge_csv(
                    reason_path=REASON_ROOT / f"{letter}_validation.csv",
                    functional_path=_functional_validation_path(letter),
                    out_path=variant_root / f"{letter}_validation.csv",
                    strict=strict,
                ),
            }
        )

    for split in ("train", "validation"):
        e_row = {
            "scenario": "E",
            "split": split,
            **_concat_merged_csvs(
                letter="E",
                split=split,
                reason_path=REASON_ROOT / f"E_{split}.csv",
                source_paths=[
                    variant_root / f"F_{split}.csv",
                    variant_root / f"G_{split}.csv",
                ],
                out_path=variant_root / f"E_{split}.csv",
            ),
        }
        rows.insert(0 if split == "train" else 1, e_row)

    for letter in ("E", "F", "G"):
        for split in ("test", "challenge"):
            rows.append(
                {
                    "scenario": letter,
                    "split": split,
                    **_merge_csv(
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
    manifest_path = variant_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(variant_root / "manifest.csv", index=False)
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
