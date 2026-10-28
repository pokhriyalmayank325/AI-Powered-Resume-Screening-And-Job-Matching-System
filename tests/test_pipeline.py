from pathlib import Path

from src.gap_analysis import learning_plan
from src.info_extractor import extract_education_level, extract_skills, parse_resume
from src.jd_analyzer import analyze_jd
from src.matcher import compute_match

DATA = Path(__file__).resolve().parent.parent / "data"
RESUME = (DATA / "sample_resume.txt").read_text()
JD = (DATA / "sample_jd.txt").read_text()

UNRELATED = """Arjun Rao
arjun@example.com

SKILLS
Bookkeeping, Tally, Customer service

EXPERIENCE
Front desk assistant, Sunrise Hotel (2019 - 2023)
- Managed guest check-in and room allocation
- Handled billing queries and daily cash reconciliation
"""


def test_contact_extraction():
    r = parse_resume(RESUME)
    assert r["name"] == "Riya Menon"
    assert r["email"] == "riya.menon@example.com"
    assert r["phone"] and "98765" in r["phone"]


def test_skill_extraction_handles_symbols():
    skills = extract_skills("Worked with C++, C#, Node.js and scikit-learn. Not javascripty.")
    assert {"c++", "c#", "node.js", "scikit-learn"} <= set(skills)
    assert "javascript" not in skills


def test_education_level():
    assert extract_education_level("B.Tech in Computer Science") == "bachelor"
    assert extract_education_level("M.Sc Data Science, PhD pursuing") == "phd"


def test_jd_splits_required_and_preferred():
    jd = analyze_jd(JD)
    assert {"python", "sql", "docker", "aws"} <= set(jd["required_skills"])
    assert {"nlp", "llm", "power bi", "tableau"} <= set(jd["preferred_skills"])
    assert jd["required_years"] == 2
    assert jd["education_level"] == "bachelor"


def test_match_reports_gaps():
    res = compute_match(RESUME, JD, use_embeddings=False)
    assert "python" in res.matched_required
    assert {"docker", "aws"} <= set(res.missing_required)
    assert 0 <= res.overall <= 100


def test_good_resume_beats_unrelated_resume():
    good = compute_match(RESUME, JD, use_embeddings=False).overall
    bad = compute_match(UNRELATED, JD, use_embeddings=False).overall
    assert good > bad + 20


def test_learning_plan_prioritises_required():
    plan = learning_plan(["docker"], ["nlp"])
    assert plan[0]["skill"] == "docker" and plan[0]["priority"].startswith("High")
