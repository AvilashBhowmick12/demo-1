"""Estimate a candidate's total years of professional experience from resume text.

Two signals, in order of trust:
  1. "stated": the resume says it, e.g. "8+ years of experience". We take the largest such claim.
  2. "dates":  we add up the job date ranges ("Jan 2019 - Present"), merging overlaps and
               skipping lines that look like education.
The result is an estimate; callers should treat it as approximate.
"""
import re
from datetime import date
from typing import List, Optional, Tuple

MAX_REASONABLE_YEARS = 45

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

_STATED = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b", re.I)

_EDU_WORDS = re.compile(
    r"universit|college|institute|school|bachelor|master|b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|"
    r"\bmba\b|degree|diploma|gpa|cgpa|certificat|course|graduat|\bphd\b|education|award|medal|honou?r|scholarship", re.I)

_MONTH_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?|\d{1,2}/"
_RANGE = re.compile(
    rf"(?:(?P<m1>{_MONTH_RE})\s*,?\s*)?(?P<y1>(?:19|20)\d{{2}})\s*(?:-|–|—|to)\s*"
    rf"(?:(?P<m2>{_MONTH_RE})\s*,?\s*)?(?P<y2>(?:19|20)\d{{2}}|present|current|now|till date|to date)",
    re.I,
)


def _stated_years(text: str) -> Optional[float]:
    best: Optional[float] = None
    for m in _STATED.finditer(text):
        n = float(m.group(1))
        if not 0 < n <= MAX_REASONABLE_YEARS:
            continue
        around = text[max(0, m.start() - 60): m.end() + 60].lower()
        if "experience" in around or " exp" in around:
            best = n if best is None else max(best, n)
    return best


def _month_index(month: Optional[str], year: int, end: bool) -> float:
    """Position on a timeline in years; unknown month = start/end of the year."""
    if month:
        month = month.strip().lower()
        m = int(month.rstrip("/")) if month[0].isdigit() else _MONTHS.get(month[:3])
        if m and 1 <= m <= 12:
            return year + (m - 1) / 12 + (1 / 12 if end else 0)
    return year + (1.0 if end else 0.0)


def _date_years(text: str, today: Optional[date] = None) -> Optional[float]:
    today = today or date.today()
    now = today.year + (today.month - 1) / 12
    spans: List[Tuple[float, float]] = []
    for line in text.splitlines():
        if _EDU_WORDS.search(line):
            continue
        for m in _RANGE.finditer(line):
            start = _month_index(m.group("m1"), int(m.group("y1")), end=False)
            y2 = m.group("y2").lower()
            end = now if not y2[0].isdigit() else _month_index(m.group("m2"), int(y2), end=True)
            end = min(end, now)
            if end > start and end - start <= MAX_REASONABLE_YEARS:
                spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    total, cur_s, cur_e = 0.0, spans[0][0], spans[0][1]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total if total > 0 else None


def estimate_experience(text: str, today: Optional[date] = None) -> dict:
    """Return {"years", "source", "stated", "dates"}.

    "years" is the resume's own stated figure if it has one, otherwise the date-range estimate
    ("source" says which: "stated", "dates" or None). Both raw signals are kept for reference.
    """
    stated = _stated_years(text)
    dated = _date_years(text, today)
    dated = round(dated, 1) if dated is not None else None
    if stated is not None:
        years, source = stated, "stated"
    elif dated is not None:
        years, source = dated, "dates"
    else:
        years, source = None, None
    return {"years": years, "source": source, "stated": stated, "dates": dated}