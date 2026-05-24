import re
from typing import Optional, Dict




_RE_ANCHOR_LINE = re.compile(
    r'(?im)^\s*(?:정답|answer|label)\s*[:：]\s*([01])\s*$'
)
_RE_ANCHOR_ANY = re.compile(r'(?i)(?:정답|answer|label)\s*[:：]\s*')




_RE01 = re.compile(r'(?<!\d)([01])(?!\d|\.\d)')
_RE_LEADING_BINARY = re.compile(r'^\s*([01])')
_TAIL_CHARS = 256
_AFTER_ANCHOR_CHARS = 64

_DEFAULT_MAP: Dict[str, int] = {
    "true": 1, "false": 0,
    "yes": 1, "no": 0,
    "y": 1, "n": 0,
    "positive": 1, "negative": 0,
    "pos": 1, "neg": 0,
    "phish": 1, "phishing": 1,
    "spam": 1, "ham": 0,
    "benign": 0, "clean": 0, "safe": 0,
    "malicious": 1, "bad": 1,
    "예": 1, "네": 1, "맞": 1, "참": 1, "진짜": 1,
    "아니": 0, "아님": 0, "거짓": 0, "틀": 0,
    "피싱": 1, "정상": 0,
    "보이스피싱": 1, "논피싱": 0,
}

def parse_pred(
    text: Optional[object],
    default: int = 0,
    *,
    strict: bool = False,
    extra_map: Optional[Dict[str, int]] = None,
) -> int:
    if text is None:
        if strict:
            raise ValueError("parse_pred: text is None")
        return default

    if isinstance(text, int):
        if text in (0, 1):
            return int(text)
        if strict:
            raise ValueError(f"parse_pred: int not in {{0,1}}: {text}")
        return default

    s = str(text).strip()
    if not s:
        if strict:
            raise ValueError("parse_pred: empty string")
        return default

    low = s.lower().strip()

    merged = dict(_DEFAULT_MAP)
    if extra_map:
        merged.update({k.lower(): v for k, v in extra_map.items()})

    
    ms = list(_RE_ANCHOR_LINE.finditer(low))
    if ms:
        return int(ms[-1].group(1))

    
    
    anchors = list(_RE_ANCHOR_ANY.finditer(low))
    if anchors:
        start = anchors[-1].end()
        window = low[start : start + _AFTER_ANCHOR_CHARS].strip()
        toks_after_anchor = _RE01.findall(window)
        if toks_after_anchor:
            return int(toks_after_anchor[0])

    
    tail = low[-_TAIL_CHARS:]
    toks = _RE01.findall(tail)
    if toks:
        return int(toks[-1])

    
    if low in merged:
        return merged[low]

    if len(low) <= 32:
        for k, v in merged.items():
            if k and k in low:
                return v

    if strict:
        raise ValueError(f"parse_pred: cannot parse -> {s[:200]!r}")
    return default


def parse_leading_binary_token(
    text: Optional[object],
    default: int = 0,
    *,
    strict: bool = False,
) -> int:
    if text is None:
        if strict:
            raise ValueError("parse_leading_binary_token: text is None")
        return default

    if isinstance(text, int):
        if text in (0, 1):
            return int(text)
        if strict:
            raise ValueError(f"parse_leading_binary_token: int not in {{0,1}}: {text}")
        return default

    s = str(text).strip()
    if not s:
        if strict:
            raise ValueError("parse_leading_binary_token: empty string")
        return default

    match = _RE_LEADING_BINARY.match(s)
    if match:
        return int(match.group(1))

    if strict:
        raise ValueError(f"parse_leading_binary_token: cannot parse -> {s[:200]!r}")
    return default


def parse_trailing_binary_token(
    text: Optional[object],
    default: int = 0,
    *,
    strict: bool = False,
) -> int:
    if text is None:
        if strict:
            raise ValueError("parse_trailing_binary_token: text is None")
        return default

    if isinstance(text, int):
        if text in (0, 1):
            return int(text)
        if strict:
            raise ValueError(f"parse_trailing_binary_token: int not in {{0,1}}: {text}")
        return default

    s = str(text).strip()
    if not s:
        if strict:
            raise ValueError("parse_trailing_binary_token: empty string")
        return default

    low = s.lower().strip()
    ms = list(_RE_ANCHOR_LINE.finditer(low))
    if ms:
        return int(ms[-1].group(1))

    anchors = list(_RE_ANCHOR_ANY.finditer(low))
    if anchors:
        start = anchors[-1].end()
        window = low[start : start + _AFTER_ANCHOR_CHARS].strip()
        toks_after_anchor = _RE01.findall(window)
        if toks_after_anchor:
            return int(toks_after_anchor[0])

    tail = low[-_TAIL_CHARS:]
    toks = _RE01.findall(tail)
    if toks:
        return int(toks[-1])

    if strict:
        raise ValueError(f"parse_trailing_binary_token: cannot parse -> {s[:200]!r}")
    return default
