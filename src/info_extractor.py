"""Step 3: pull structured info (contact, skills, education, experience) out of a resume."""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from .preprocessing import clean_text, split_sections

SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{8,}\d)")
YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\+|(?:-|–|to)\s*\d+(?:\.\d+)?)?\s*(?:years?|yrs?)\b", re.I
)
RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|till date|now)", re.I
)

EDU_LEVELS = {"none": 0, "diploma": 1, "bachelor": 2, "master": 3, "phd": 4}
_EDU_PATTERNS = [
    ("phd", r"\bph\.?\s?d\b|\bdoctorate\b"),
    ("master", r"\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmaster'?s?\b|\bmba\b|\bmca\b|\bm\.\s?e\b"),
    ("bachelor", r"\bb\.?\s?tech\b|\bb\.?\s?sc\b|\bbachelor'?s?\b|\bbca\b|\bbba\b|\bb\.?\s?com\b|\bb\.\s?e\b"),
    ("diploma", r"\bdiploma\b"),
]


# ---------- skills ----------
@lru_cache(maxsize=1)
def load_skills(path: str | None = None) -> dict[str, list[str]]:
    return json.loads(Path(path or SKILLS_PATH).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _skill_patterns() -> dict[str, re.Pattern]:
    patterns = {}
    for canonical, aliases in load_skills().items():
        alts = sorted({canonical, *aliases}, key=len, reverse=True)
        body = "|".join(re.escape(a) for a in alts)
        patterns[canonical] = re.compile(rf"(?<![A-Za-z0-9+#.]){'(?:' + body + ')'}(?![A-Za-z0-9+#])", re.I)
    return patterns


def extract_skills(text: str) -> list[str]:
    return sorted(s for s, pat in _skill_patterns().items() if pat.search(text))


# ---------- contact ----------
def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else None


def extract_phone(text: str) -> str | None:
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(1))
        if 10 <= len(digits) <= 13 and not re.fullmatch(r"(19|20)\d{2}(19|20)\d{2}\d*", digits):
            return m.group(1).strip()
    return None


def extract_name(text: str) -> str | None:
    """Heuristic: first short alphabetic line near the top (no digits, '@', or section words)."""
    for line in text.splitlines()[:6]:
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 4 and re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", line) \
                and line.lower() not in {"resume", "curriculum vitae"}:
            return line.title() if line.isupper() else line
    return None


# ---------- education / experience ----------
def extract_education_level(text: str) -> str:
    best = "none"
    for level, pattern in _EDU_PATTERNS:
        if re.search(pattern, text, re.I) and EDU_LEVELS[level] > EDU_LEVELS[best]:
            best = level
    return best


def extract_stated_years(text: str) -> float | None:
    """'3+ years of experience', '3-5 years experience' -> 3.0 (lower bound)."""
    found = []
    for m in YEARS_RE.finditer(text):
        window = text[max(0, m.start() - 60): m.end() + 60].lower()
        if "experience" in window and float(m.group(1)) <= 40:
            found.append(float(m.group(1)))
    return max(found) if found else None


def years_from_date_ranges(text: str) -> float:
    total = 0
    for start, end in RANGE_RE.findall(text):
        end_year = date.today().year if not end.isdigit() else int(end)
        total += max(end_year - int(start), 0)
    return float(total)


def parse_resume(raw_text: str) -> dict:
    text = clean_text(raw_text)
    sections = split_sections(text)
    exp_text = sections.get("experience", "")
    stated = extract_stated_years(text) or 0.0
    computed = years_from_date_ranges(exp_text)
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "education_level": extract_education_level(sections.get("education", text)),
        "experience_years": max(stated, computed),
        "sections": sections,
        "text": text,
    }
