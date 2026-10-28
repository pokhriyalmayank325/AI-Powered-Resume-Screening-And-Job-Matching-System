"""Step 2: clean the text and split a resume into sections."""
from __future__ import annotations

import re

SECTION_HEADERS = {
    "summary": ["summary", "profile", "objective", "about me", "professional summary"],
    "skills": ["skills", "technical skills", "core competencies", "technologies", "tech stack"],
    "education": ["education", "academic background", "qualifications", "academics"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "internships", "internship", "internship experience"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "certifications": ["certifications", "certificates", "licenses", "courses"],
}
_HEADER_LOOKUP = {h: sec for sec, hs in SECTION_HEADERS.items() for h in hs}


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"[•●▪◦►■]", "-", text)
    text = re.sub(r"[^\x09\x0a\x20-\x7e\u00c0-\u024f\u2013\u2014]", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def split_sections(text: str) -> dict[str, str]:
    """Return {section: text}. Anything before the first header goes in 'header'."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        key = line.strip().rstrip(":").lower()
        if key in _HEADER_LOOKUP and len(line.strip()) < 40:
            current = _HEADER_LOOKUP[key]
            sections.setdefault(current, [])
            continue
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "".join(v).strip()}
