import re
from typing import Optional, Dict

# 라인 단위 앵커: "정답: 0" / "answer: 1" / "label: 0"
# - 라인 끝이 0/1로 끝나야 함 (정의 문장 "정답: 0 또는 1" 방지)
# - 멀티라인 텍스트에서 매치되도록 (?m)
_RE_ANCHOR_LINE = re.compile(
    r'(?im)^\s*(?:정답|answer|label)\s*[:：]\s*([01])\s*$'
)
_RE_ANCHOR_ANY = re.compile(r'(?i)(?:정답|answer|label)\s*[:：]\s*')

# 뒤에서 가까운 0/1 후보:
# - float(0.95, 1.0) 첫 자리 0/1 제외
# - 숫자열(10, 101 등) 중간의 0/1 제외
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

    # 0) 앵커(정답/answer/label)는 "라인 단위"로만 인정 + 마지막 매치 우선
    ms = list(_RE_ANCHOR_LINE.finditer(low))
    if ms:
        return int(ms[-1].group(1))

    # 0.5) 라인 앵커는 아니어도, 마지막 앵커 이후의 가까운 숫자를 우선
    # (앵커 뒤에 답을 쓰고 그 뒤에 잡다한 숫자를 더 쓰는 경우 방지)
    anchors = list(_RE_ANCHOR_ANY.finditer(low))
    if anchors:
        start = anchors[-1].end()
        window = low[start : start + _AFTER_ANCHOR_CHARS].strip()
        toks_after_anchor = _RE01.findall(window)
        if toks_after_anchor:
            return int(toks_after_anchor[0])

    # 1) 뒤에서 가까운 0/1: 전체가 아니라 마지막 구간만 본다
    tail = low[-_TAIL_CHARS:]
    toks = _RE01.findall(tail)
    if toks:
        return int(toks[-1])

    # 2) 매핑은 "짧은 출력"에만 적용
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
