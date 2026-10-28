"""Step 4: analyse a job description -> required/preferred skills, experience, education."""
from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .info_extractor import extract_education_level, extract_skills, extract_stated_years
from .preprocessing import clean_text

PREFERRED_RE = re.compile(
    r"\b(preferred|nice to have|good to have|bonus|a plus|is a plus|desirable|optional)\b", re.I
)


def _is_header(line: str) -> bool:
    return len(line) < 60 and (line.endswith(":") or line.isupper())


def analyze_jd(raw_text: str) -> dict:
    text = clean_text(raw_text)
    required: set[str] = set()
    preferred: set[str] = set()
    mode = "required"

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _is_header(line):
            mode = "preferred" if PREFERRED_RE.search(line) else "required"
            continue
        target = preferred if (mode == "preferred" or PREFERRED_RE.search(line)) else required
        target.update(extract_skills(line))

    preferred -= required  # required always wins

    words = re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", text.lower())
    counts = Counter(w for w in words if w not in ENGLISH_STOP_WORDS)
    return {
        "required_skills": sorted(required),
        "preferred_skills": sorted(preferred),
        "required_years": extract_stated_years(text),
        "education_level": extract_education_level(text),
        "keywords": [w for w, _ in counts.most_common(15)],
        "text": text,
    }
