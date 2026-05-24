#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
SMS_REASON_ROOT = DATA_ROOT / "sms" / "reason"
VOICE_REASON_ROOT = DATA_ROOT / "voice" / "reason"
MANIFEST_PATH = DATA_ROOT / "reason_manifest.json"


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_reason_map(
    reason_csv: Path,
    *,
    text_col: str,
    label_col: str,
    reason_col: str,
) -> pd.DataFrame:
    df = _read_csv(reason_csv)
    missing = [c for c in (text_col, label_col, reason_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{reason_csv} missing columns: {missing}")
    work = df[[text_col, label_col, reason_col]].copy()
    work[text_col] = work[text_col].map(_norm_text)
    work[label_col] = work[label_col].astype(int)
    work[reason_col] = work[reason_col].fillna("").astype(str).str.strip()
    work = work[(work[text_col] != "") & (work[reason_col] != "")].copy()
    work["key"] = work[text_col] + "\u241f" + work[label_col].astype(str)
    work = work.drop_duplicates(subset=["key"], keep="first")
    return work.rename(
        columns={
            text_col: "reason_text",
            label_col: "reason_label",
            reason_col: "reason_value",
        }
    )


def _build_joined_train_split(
    *,
    out_path: Path,
    raw_train_csv: Path,
    raw_text_col: str,
    raw_label_col: str,
    reason_csv: Path,
    reason_text_col: str,
    reason_label_col: str,
    reason_col: str,
    keep_date: bool,
    metadata: dict[str, Any],
) -> None:
    raw_df = _read_csv(raw_train_csv)
    missing = [c for c in (raw_text_col, raw_label_col) if c not in raw_df.columns]
    if missing:
        raise ValueError(f"{raw_train_csv} missing columns: {missing}")
    raw_df = raw_df.copy()
    raw_df[raw_text_col] = raw_df[raw_text_col].fillna("").astype(str)
    raw_df[raw_label_col] = raw_df[raw_label_col].astype(int)
    raw_df["norm_text"] = raw_df[raw_text_col].map(_norm_text)
    raw_df["key"] = raw_df["norm_text"] + "\u241f" + raw_df[raw_label_col].astype(str)

    reason_map = _load_reason_map(
        reason_csv,
        text_col=reason_text_col,
        label_col=reason_label_col,
        reason_col=reason_col,
    )[["key", "reason_value"]]

    merged = raw_df.merge(reason_map, on="key", how="left", validate="m:1")
    missing_count = int(merged["reason_value"].isna().sum())
    if missing_count:
        sample = merged.loc[merged["reason_value"].isna(), [raw_text_col, raw_label_col]].head(3).to_dict(orient="records")
        raise ValueError(
            f"{out_path.name}: {missing_count} rows missing reason targets after join. sample={sample}"
        )

    out = pd.DataFrame(
        {
            "text": merged[raw_text_col].fillna("").astype(str),
            "category": (
                merged["category"].fillna("").astype(str)
                if "category" in merged.columns
                else pd.Series([""] * len(merged), dtype="object")
            ),
            "label": merged[raw_label_col].astype(int),
            "reason_value": merged["reason_value"].fillna("").astype(str).str.strip(),
        }
    )
    if keep_date:
        out.insert(
            3,
            "date",
            (
                merged["date"].fillna("").astype(str)
                if "date" in merged.columns
                else pd.Series([""] * len(merged), dtype="object")
            ),
        )
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    metadata[out_path.stem] = {
        "raw_train_csv": str(raw_train_csv.relative_to(ROOT)),
        "reason_csv": str(reason_csv.relative_to(ROOT)),
        "rows": int(len(merged)),
        "missing_reason_rows": missing_count,
    }


def _build_joined_validation_split(
    *,
    out_path: Path,
    raw_validation_csv: Path,
    raw_text_col: str,
    raw_label_col: str,
    reason_csv: Path,
    reason_text_col: str,
    reason_label_col: str,
    reason_col: str,
    keep_date: bool,
    metadata: dict[str, Any],
) -> None:
    _build_joined_train_split(
        out_path=out_path,
        raw_train_csv=raw_validation_csv,
        raw_text_col=raw_text_col,
        raw_label_col=raw_label_col,
        reason_csv=reason_csv,
        reason_text_col=reason_text_col,
        reason_label_col=reason_label_col,
        reason_col=reason_col,
        keep_date=keep_date,
        metadata=metadata,
    )
    row = metadata.get(out_path.stem, {}) or {}
    raw_train_csv = row.pop("raw_train_csv", None)
    if raw_train_csv is not None:
        row["raw_validation_csv"] = raw_train_csv
    metadata[out_path.stem] = row


def _build_direct_reason_split(
    *,
    out_path: Path,
    reason_csv: Path,
    text_col: str,
    label_col: str,
    reason_col: str,
    keep_date: bool,
    metadata: dict[str, Any],
) -> None:
    df = _read_csv(reason_csv)
    missing = [c for c in (text_col, label_col, reason_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{reason_csv} missing columns: {missing}")

    work = df.copy()
    work[text_col] = work[text_col].fillna("").astype(str)
    work[label_col] = work[label_col].astype(int)
    work[reason_col] = work[reason_col].fillna("").astype(str).str.strip()
    work = work[(work[text_col].map(_norm_text) != "") & (work[reason_col] != "")].copy()
    out = pd.DataFrame(
        {
            "text": work[text_col].fillna("").astype(str),
            "category": (
                work["category"].fillna("").astype(str)
                if "category" in work.columns
                else pd.Series([""] * len(work), dtype="object")
            ),
            "label": work[label_col].astype(int),
            "reason_value": work[reason_col].fillna("").astype(str).str.strip(),
        }
    )
    if keep_date:
        out.insert(
            3,
            "date",
            (
                work["date"].fillna("").astype(str)
                if "date" in work.columns
                else pd.Series([""] * len(work), dtype="object")
            ),
        )
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    metadata[out_path.stem] = {
        "reason_csv": str(reason_csv.relative_to(ROOT)),
        "rows": int(len(work)),
    }


def main() -> int:
    SMS_REASON_ROOT.mkdir(parents=True, exist_ok=True)
    VOICE_REASON_ROOT.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}

    sms_reason_train = ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_full_sms_train_batch_merged" / "results.csv"
    voice_reason_train = ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_full_voice_train_batch" / "results.csv"
    sms_reason_validation = ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_full_sms_validation_batch" / "results.csv"
    voice_reason_validation = ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_full_voice_validation_batch" / "results.csv"

    _build_joined_train_split(
        out_path=SMS_REASON_ROOT / "A_train.csv",
        raw_train_csv=ROOT / "data" / "sms" / "in_domain" / "train.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_train,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_validation_split(
        out_path=SMS_REASON_ROOT / "A_validation.csv",
        raw_validation_csv=ROOT / "data" / "sms" / "in_domain" / "validation.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_validation,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_train_split(
        out_path=SMS_REASON_ROOT / "B_train.csv",
        raw_train_csv=ROOT / "data" / "sms" / "ood" / "train" / "credit_train.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_train,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_validation_split(
        out_path=SMS_REASON_ROOT / "B_validation.csv",
        raw_validation_csv=ROOT / "data" / "sms" / "ood" / "validation" / "credit_validation.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_validation,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_train_split(
        out_path=SMS_REASON_ROOT / "C_train.csv",
        raw_train_csv=ROOT / "data" / "sms" / "ood" / "train" / "finance_train.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_train,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_validation_split(
        out_path=SMS_REASON_ROOT / "C_validation.csv",
        raw_validation_csv=ROOT / "data" / "sms" / "ood" / "validation" / "finance_validation.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_validation,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_train_split(
        out_path=SMS_REASON_ROOT / "D_train.csv",
        raw_train_csv=ROOT / "data" / "sms" / "ood" / "train" / "parcel_train.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_train,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_validation_split(
        out_path=SMS_REASON_ROOT / "D_validation.csv",
        raw_validation_csv=ROOT / "data" / "sms" / "ood" / "validation" / "parcel_validation.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=sms_reason_validation,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=True,
        metadata=metadata,
    )
    _build_joined_train_split(
        out_path=VOICE_REASON_ROOT / "E_train.csv",
        raw_train_csv=ROOT / "data" / "voice" / "in_domain" / "train.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=voice_reason_train,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=False,
        metadata=metadata,
    )
    _build_joined_validation_split(
        out_path=VOICE_REASON_ROOT / "E_validation.csv",
        raw_validation_csv=ROOT / "data" / "voice" / "in_domain" / "validation.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=voice_reason_validation,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=False,
        metadata=metadata,
    )
    _build_joined_train_split(
        out_path=VOICE_REASON_ROOT / "F_train.csv",
        raw_train_csv=ROOT / "data" / "voice" / "ood" / "train" / "ood_train_government.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=voice_reason_train,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=False,
        metadata=metadata,
    )
    _build_joined_validation_split(
        out_path=VOICE_REASON_ROOT / "F_validation.csv",
        raw_validation_csv=ROOT / "data" / "voice" / "ood" / "validation" / "ood_validation_finance.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=voice_reason_validation,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=False,
        metadata=metadata,
    )
    _build_joined_train_split(
        out_path=VOICE_REASON_ROOT / "G_train.csv",
        raw_train_csv=ROOT / "data" / "voice" / "ood" / "train" / "ood_train_finance.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=voice_reason_train,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=False,
        metadata=metadata,
    )
    _build_joined_validation_split(
        out_path=VOICE_REASON_ROOT / "G_validation.csv",
        raw_validation_csv=ROOT / "data" / "voice" / "ood" / "validation" / "ood_validation_government.csv",
        raw_text_col="text",
        raw_label_col="label",
        reason_csv=voice_reason_validation,
        reason_text_col="text",
        reason_label_col="label",
        reason_col="explanation",
        keep_date=False,
        metadata=metadata,
    )

    sms_eval_reason_runs = [
        ("A_test", ROOT / "data" / "sms" / "in_domain" / "test.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_A_test_batch" / "results.csv"),
        ("A_challenge", ROOT / "data" / "sms" / "in_domain" / "challenging.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_A_challenge_batch" / "results.csv"),
        ("B_test", ROOT / "data" / "sms" / "ood" / "test" / "credit_test.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_B_test_batch" / "results.csv"),
        ("B_challenge", ROOT / "data" / "sms" / "ood" / "challenging" / "credit_challenging.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_B_challenge_batch" / "results.csv"),
        ("C_test", ROOT / "data" / "sms" / "ood" / "test" / "finance_test.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_C_test_batch" / "results.csv"),
        ("C_challenge", ROOT / "data" / "sms" / "ood" / "challenging" / "finance_challenging.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_C_challenge_batch" / "results.csv"),
        ("D_test", ROOT / "data" / "sms" / "ood" / "test" / "parcel_test.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_D_test_batch" / "results.csv"),
        ("D_challenge", ROOT / "data" / "sms" / "ood" / "challenging" / "parcel_challenging.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_D_challenge_batch" / "results.csv"),
    ]
    for stem, raw_csv, reason_csv in sms_eval_reason_runs:
        _build_joined_train_split(
            out_path=SMS_REASON_ROOT / f"{stem}.csv",
            raw_train_csv=raw_csv,
            raw_text_col="text",
            raw_label_col="label",
            reason_csv=reason_csv,
            reason_text_col="text",
            reason_label_col="label",
            reason_col="explanation",
            keep_date=True,
            metadata=metadata,
        )

    voice_eval_reason_runs = [
        ("E_test", ROOT / "data" / "voice" / "in_domain" / "test.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_E_test_batch" / "results.csv"),
        ("E_challenge", ROOT / "data" / "voice" / "in_domain" / "challenge.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_E_challenge_batch" / "results.csv"),
        ("F_test", ROOT / "data" / "voice" / "ood" / "test" / "ood_test_finance.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_F_test_batch" / "results.csv"),
        ("F_challenge", ROOT / "data" / "voice" / "ood" / "challenging" / "finance_ood_challenge.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_F_challenge_batch" / "results.csv"),
        ("G_test", ROOT / "data" / "voice" / "ood" / "test" / "ood_test_government.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_G_test_batch" / "results.csv"),
        ("G_challenge", ROOT / "data" / "voice" / "ood" / "challenging" / "government_ood_challenge.csv", ROOT / "outputs" / "analysis" / "korsmishing_explainer_batch" / "gpt-4o-mini_korsmishing_G_challenge_batch" / "results.csv"),
    ]
    for stem, raw_csv, reason_csv in voice_eval_reason_runs:
        _build_joined_train_split(
            out_path=VOICE_REASON_ROOT / f"{stem}.csv",
            raw_train_csv=raw_csv,
            raw_text_col="text",
            raw_label_col="label",
            reason_csv=reason_csv,
            reason_text_col="text",
            reason_label_col="label",
            reason_col="explanation",
            keep_date=False,
            metadata=metadata,
        )

    MANIFEST_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {MANIFEST_PATH}")
    for name, item in metadata.items():
        print(f"[saved] {name} rows={item['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
