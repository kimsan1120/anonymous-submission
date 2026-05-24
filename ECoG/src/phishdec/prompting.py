from __future__ import annotations

import re
from string import Formatter
from typing import Any, Dict


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _to_int_or_none(value: Any):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def render_row_template(df, row_template: str) -> list[str]:
    field_names = {
        field_name
        for _, field_name, _, _ in Formatter().parse(row_template)
        if field_name is not None and field_name != ""
    }
    missing = sorted(name for name in field_names if name not in df.columns)
    if missing:
        raise ValueError(
            f"prompt.row_template references missing columns: {missing}. "
            f"available={list(df.columns)}"
        )

    rendered: list[str] = []
    for row in df.to_dict(orient="records"):
        safe_row: Dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                safe_row[key] = ""
            elif isinstance(value, float) and value != value:
                safe_row[key] = ""
            else:
                safe_row[key] = value
        rendered.append(row_template.format(**safe_row))
    return rendered


def _match_rule_value(row_value: Any, expected_value: Any) -> bool:
    row_int = _to_int_or_none(row_value)
    expected_int = _to_int_or_none(expected_value)
    if row_int is not None and expected_int is not None:
        return row_int == expected_int
    return str("" if row_value is None else row_value).strip().lower() == str(
        "" if expected_value is None else expected_value
    ).strip().lower()


def _row_matches_instruction_rule(row: Dict[str, Any], when: Dict[str, Any]) -> bool:
    for key, expected_value in when.items():
        if key not in row:
            return False
        if not _match_rule_value(row.get(key), expected_value):
            return False
    return True


_DOUBLE_BRACE_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_SINGLE_BRACE_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _render_instruction_template(template: str, row: Dict[str, Any], text_col: str) -> tuple[str, bool]:
    safe_row: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            safe_row[key] = ""
        elif isinstance(value, float) and value != value:
            safe_row[key] = ""
        else:
            safe_row[key] = value

    text_value = str(safe_row.get(text_col, ""))
    safe_row.setdefault("text", text_value)
    safe_row.setdefault("transcript", text_value)
    safe_row.setdefault("input_text", text_value)

    used_keys: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in safe_row:
            return match.group(0)
        used_keys.add(key)
        return str(safe_row.get(key, ""))

    rendered = _DOUBLE_BRACE_PLACEHOLDER_RE.sub(_replace, template)
    rendered = _SINGLE_BRACE_PLACEHOLDER_RE.sub(_replace, rendered).strip()
    embeds_input = any(key in used_keys for key in {text_col, "text", "transcript", "input_text"})
    return rendered, embeds_input


def resolve_instruction_texts(
    df, prompt_cfg: Dict[str, Any], text_col: str = "text"
) -> tuple[list[str], list[str], list[bool]]:
    instruction_path = str(prompt_cfg.get("instruction_path", "")).strip()
    instruction_rules = prompt_cfg.get("instruction_rules") or []

    if not instruction_rules:
        if not instruction_path:
            return [""] * len(df), [""] * len(df), [False] * len(df)
        instr = _read_text(instruction_path)
        texts: list[str] = []
        embedded_flags: list[bool] = []
        for row in df.to_dict(orient="records"):
            rendered, embeds_input = _render_instruction_template(instr, row=row, text_col=text_col)
            texts.append(rendered)
            embedded_flags.append(embeds_input)
        return texts, [instruction_path] * len(df), embedded_flags

    if not isinstance(instruction_rules, list):
        raise ValueError("prompt.instruction_rules must be a list")

    cached_texts: Dict[str, str] = {}
    normalized_rules: list[dict[str, Any]] = []
    for idx, rule in enumerate(instruction_rules):
        if not isinstance(rule, dict):
            raise ValueError(f"prompt.instruction_rules[{idx}] must be a dict")
        when = rule.get("when", {}) or {}
        if not isinstance(when, dict):
            raise ValueError(f"prompt.instruction_rules[{idx}].when must be a dict")
        rule_path = str(rule.get("instruction_path", "")).strip()
        if not rule_path:
            raise ValueError(f"prompt.instruction_rules[{idx}].instruction_path is empty")
        if rule_path not in cached_texts:
            cached_texts[rule_path] = _read_text(rule_path)
        normalized_rules.append({"when": when, "instruction_path": rule_path})

    if instruction_path and instruction_path not in cached_texts:
        cached_texts[instruction_path] = _read_text(instruction_path)

    texts: list[str] = []
    paths: list[str] = []
    embedded_flags: list[bool] = []
    for row_idx, row in enumerate(df.to_dict(orient="records")):
        matched_path = ""
        for rule in normalized_rules:
            if _row_matches_instruction_rule(row=row, when=rule["when"]):
                matched_path = str(rule["instruction_path"])
                break
        if not matched_path:
            if instruction_path:
                matched_path = instruction_path
            else:
                raise ValueError(
                    f"No prompt.instruction_rules matched row_idx={row_idx}. "
                    f"available_columns={list(row.keys())}"
                )
        rendered, embeds_input = _render_instruction_template(
            cached_texts[matched_path], row=row, text_col=text_col
        )
        texts.append(rendered)
        paths.append(matched_path)
        embedded_flags.append(embeds_input)
    return texts, paths, embedded_flags
