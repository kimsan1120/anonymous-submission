import os
import json
import re
import pandas as pd


_GEN_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_INDEXED_GEN_COL_RE = re.compile(r"^gen_(evidence|explanation)_(\d+)$")


def _parse_gen_json(gen_text):
    raw = str(gen_text or "").strip()
    if not raw:
        return None, "empty"

    candidates = []
    candidates.append(raw)

    fenced = _GEN_JSON_BLOCK_RE.search(raw)
    if fenced:
        candidates.append(fenced.group(1).strip())

    first_brace = raw.find("{")
    if first_brace >= 0:
        candidates.append(raw[first_brace:].strip())

    seen = set()
    decoder = json.JSONDecoder()
    last_error = "no_json_candidate_found"

    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        for candidate in (cand, cand.lstrip()):
            try:
                obj, _ = decoder.raw_decode(candidate)
            except Exception as exc:
                last_error = str(exc)
                continue
            if isinstance(obj, dict):
                return obj, None
            last_error = f"json_not_object(type={type(obj).__name__})"

    return None, last_error


def _normalize_list_field(value):
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str("" if item is None else item).strip()
        if text:
            out.append(text)
    return out


def _build_evidence_w_explanation(parsed_obj):
    if not isinstance(parsed_obj, dict):
        return ""

    evidences = _normalize_list_field(parsed_obj.get("evidence"))
    explanations = _normalize_list_field(parsed_obj.get("explanation"))
    n_items = min(2, max(len(evidences), len(explanations)))
    if n_items <= 0:
        return ""

    blocks = []
    for idx in range(n_items):
        evidence = evidences[idx] if idx < len(evidences) else ""
        explanation = explanations[idx] if idx < len(explanations) else ""
        parts = []
        if evidence:
            parts.append(f"근거: {evidence}")
        if explanation:
            parts.append(f"설명: {explanation}")
        if parts:
            blocks.append(f"{idx + 1}) " + " | ".join(parts))
    return "\n".join(blocks).strip()


def _build_evidence_w_explanation_from_row(row):
    indexed = {}
    for col, value in row.items():
        match = _INDEXED_GEN_COL_RE.match(str(col))
        if not match:
            continue
        field_name = match.group(1)
        idx = int(match.group(2))
        text = str("" if value is None else value).strip()
        if not text or text.lower() == "nan":
            continue
        indexed.setdefault(idx, {})[field_name] = text

    if not indexed:
        return ""

    blocks = []
    for idx in sorted(indexed)[:2]:
        evidence = indexed[idx].get("evidence", "")
        explanation = indexed[idx].get("explanation", "")
        parts = []
        if evidence:
            parts.append(f"근거: {evidence}")
        if explanation:
            parts.append(f"설명: {explanation}")
        if parts:
            blocks.append(f"{idx}) " + " | ".join(parts))
    return "\n".join(blocks).strip()


def _to_json_list_text(value):
    if not isinstance(value, (list, tuple)):
        return ""
    return json.dumps(list(value), ensure_ascii=False)


def _normalize_logprobs(value):
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if item is None:
            out.append(None)
            continue
        try:
            out.append(float(item))
        except Exception:
            out.append(None)
    return out


def save_results_csv(
    df,
    preds,
    gens,
    out_file,
    *,
    gen_tokens=None,
    gen_token_logprobs=None,
    save_parsed_json_columns: bool = False,
    expand_parsed_json_fields: bool = False,
    drop_columns: list[str] | None = None,
):
    if len(df) != len(preds) or len(preds) != len(gens):
        raise ValueError(
            f"Length mismatch: len(df)={len(df)}, len(preds)={len(preds)}, len(gens)={len(gens)}"
        )
    if gen_tokens is not None and len(gen_tokens) != len(gens):
        raise ValueError(
            f"Length mismatch: len(gens)={len(gens)}, len(gen_tokens)={len(gen_tokens)}"
        )
    if gen_token_logprobs is not None and len(gen_token_logprobs) != len(gens):
        raise ValueError(
            "Length mismatch: "
            f"len(gens)={len(gens)}, len(gen_token_logprobs)={len(gen_token_logprobs)}"
        )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    out = df.reset_index(drop=True).copy()
    if drop_columns:
        cols_to_drop = [str(col) for col in drop_columns if str(col) in out.columns]
        if cols_to_drop:
            out = out.drop(columns=cols_to_drop)

    if "text" in out.columns:
        if "input" not in out.columns:
            out.insert(0, "input", out["text"].astype(str))
        out = out.drop(columns=["text"])
    elif "input" not in out.columns:
        out.insert(0, "input", "")

    if "category" not in out.columns:
        out.insert(1, "category", "")

    raw_gens = [str(gen) for gen in gens]
    out["gen"] = raw_gens
    if gen_tokens is not None:
        out["gen_tokens"] = [_to_json_list_text(tokens) for tokens in gen_tokens]
    if gen_token_logprobs is not None:
        out["gen_token_logprobs"] = [
            _to_json_list_text(_normalize_logprobs(token_logprobs))
            for token_logprobs in gen_token_logprobs
        ]
    out["pred"] = pd.to_numeric(pd.Series(preds), errors="coerce").fillna(0).astype(int)
    if "label" in out.columns:
        out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)

    should_parse_json = bool(save_parsed_json_columns or expand_parsed_json_fields)
    if should_parse_json:
        parsed_objs = []
        parse_errors = []
        for gen_text in raw_gens:
            parsed_obj, parse_error = _parse_gen_json(gen_text)
            parsed_objs.append(parsed_obj)
            parse_errors.append("" if parse_error is None else str(parse_error))

        if any(obj is not None for obj in parsed_objs):
            if save_parsed_json_columns:
                out["gen_raw"] = raw_gens
                out["gen_json_valid"] = [obj is not None for obj in parsed_objs]
                out["gen_json_error"] = parse_errors
                out["gen_json"] = [
                    json.dumps(obj, ensure_ascii=False) if obj is not None else ""
                    for obj in parsed_objs
                ]

            if expand_parsed_json_fields:
                scalar_keys = sorted(
                    {
                        key
                        for obj in parsed_objs
                        if isinstance(obj, dict)
                        for key, value in obj.items()
                        if not isinstance(value, (list, dict))
                    }
                )
                list_keys = sorted(
                    {
                        key
                        for obj in parsed_objs
                        if isinstance(obj, dict)
                        for key, value in obj.items()
                        if isinstance(value, list)
                    }
                )

                for key in scalar_keys:
                    out[f"gen_{key}"] = [
                        ("" if obj is None or obj.get(key) is None else str(obj.get(key)))
                        for obj in parsed_objs
                    ]

                for key in list_keys:
                    max_len = max(
                        len(obj.get(key) or [])
                        for obj in parsed_objs
                        if isinstance(obj, dict)
                    )
                    for idx in range(max_len):
                        out[f"gen_{key}_{idx + 1}"] = [
                            (
                                ""
                                if obj is None
                                or not isinstance(obj.get(key), list)
                                or len(obj.get(key)) <= idx
                                or obj.get(key)[idx] is None
                                else str(obj.get(key)[idx])
                            )
                            for obj in parsed_objs
                        ]

    preferred = ["input", "category", "gen", "gen_tokens", "gen_token_logprobs", "pred", "label"]
    remain = [c for c in out.columns if c not in preferred]
    out = out[[c for c in preferred if c in out.columns] + remain]

    out.to_csv(out_file, index=False, encoding="utf-8-sig")


def save_evidence_w_explanation_csv(df, gens, out_file):
    if len(df) != len(gens):
        raise ValueError(f"Length mismatch: len(df)={len(df)}, len(gens)={len(gens)}")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    work = df.reset_index(drop=True).copy()

    if "text" in work.columns:
        text_series = work["text"].fillna("").astype(str)
    elif "input" in work.columns:
        text_series = work["input"].fillna("").astype(str)
    else:
        text_series = pd.Series([""] * len(work))

    if "date" in work.columns:
        date_series = work["date"].fillna("").astype(str)
    else:
        date_series = pd.Series([""] * len(work))

    if "category" in work.columns:
        category_series = work["category"].fillna("").astype(str)
    else:
        category_series = pd.Series([""] * len(work))

    if "label" in work.columns:
        label_series = pd.to_numeric(work["label"], errors="coerce").fillna(0).astype(int)
    else:
        label_series = pd.Series([0] * len(work), dtype=int)

    evidence_w_explanations = []
    for row, gen_text in zip(work.to_dict(orient="records"), gens):
        combined = _build_evidence_w_explanation_from_row(row)
        if not combined:
            parsed_obj, _ = _parse_gen_json(gen_text)
            combined = _build_evidence_w_explanation(parsed_obj)
        evidence_w_explanations.append(combined)

    out = pd.DataFrame(
        {
            "date": date_series.tolist(),
            "text": text_series.tolist(),
            "category": category_series.tolist(),
            "evidence_w_explanation": evidence_w_explanations,
            "label": label_series.tolist(),
        }
    )
    out.to_csv(out_file, index=False, encoding="utf-8-sig")
