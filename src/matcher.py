"""Step 5: score a resume against a job description.

Final score = weighted blend of four components (each 0-1):
  skills      required skills count 1.0, preferred count 0.5
  semantic    how well each JD requirement line is covered by *some* resume line
              (sentence-transformer embeddings, falls back to TF-IDF)
  experience  resume years / required years (capped at 1)
  education   resume degree level / required degree level (capped at 1)
Components that don't apply (e.g. JD states no years) are dropped and the
remaining weights are renormalised.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .info_extractor import EDU_LEVELS, parse_resume
from .jd_analyzer import analyze_jd

DEFAULT_WEIGHTS = {"skills": 0.50, "semantic": 0.30, "experience": 0.10, "education": 0.10}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Raw cosine -> 0..1 calibration. MiniLM cosines for "related" text sit around 0.3-0.6,
# TF-IDF cosines between short lines are much lower, so each method gets its own range.
CALIBRATION = {"embeddings": (0.15, 0.65), "tfidf": (0.0, 0.35)}


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the model once; return None if the library/model isn't available."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        return None


def _chunks(text: str, min_words: int = 3) -> list[str]:
    return [l.strip("-• ").strip() for l in text.splitlines() if len(l.split()) >= min_words]


def similarity_matrix(a: list[str], b: list[str], use_embeddings: bool = True):
    model = get_embedding_model() if use_embeddings else None
    if model is not None:
        ea = model.encode(a, normalize_embeddings=True)
        eb = model.encode(b, normalize_embeddings=True)
        return np.asarray(ea) @ np.asarray(eb).T, "embeddings"
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(a + b)
    return cosine_similarity(vec.transform(a), vec.transform(b)), "tfidf"


def tfidf_document_similarity(resume_text: str, jd_text: str) -> float:
    """Plain whole-document TF-IDF cosine. Used as the baseline in evaluation."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform([resume_text, jd_text])
    return float(cosine_similarity(m[0], m[1])[0, 0])


def semantic_coverage(resume_text: str, jd_text: str, use_embeddings: bool = True):
    """For each JD line find the best-matching resume line; average those best scores."""
    jd_lines, resume_lines = _chunks(jd_text), _chunks(resume_text)
    if not jd_lines or not resume_lines:
        return 0.0, "tfidf", [], []
    sim, method = similarity_matrix(jd_lines, resume_lines, use_embeddings)
    lo, hi = CALIBRATION[method]
    best_idx = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)
    scaled = np.clip((best_sim - lo) / (hi - lo), 0, 1)
    triples = [(jd_lines[i], resume_lines[int(best_idx[i])], float(scaled[i])) for i in range(len(jd_lines))]
    ranked = sorted(triples, key=lambda t: t[2], reverse=True)
    return float(scaled.mean()), method, ranked[:5], ranked[-3:][::-1]


@dataclass
class MatchResult:
    overall: float
    components: dict
    weights_used: dict
    matched_required: list = field(default_factory=list)
    missing_required: list = field(default_factory=list)
    matched_preferred: list = field(default_factory=list)
    missing_preferred: list = field(default_factory=list)
    extra_skills: list = field(default_factory=list)
    method: str = "tfidf"
    evidence: list = field(default_factory=list)
    weak_spots: list = field(default_factory=list)
    resume: dict = field(default_factory=dict)
    jd: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def compute_match(resume_text: str, jd_text: str, weights: dict | None = None,
                  use_embeddings: bool = True) -> MatchResult:
    weights = weights or DEFAULT_WEIGHTS
    resume, jd = parse_resume(resume_text), analyze_jd(jd_text)

    r_sk, req, pref = set(resume["skills"]), set(jd["required_skills"]), set(jd["preferred_skills"])
    matched_req, matched_pref = sorted(req & r_sk), sorted(pref & r_sk)

    sem, method, evidence, weak = semantic_coverage(resume["text"], jd["text"], use_embeddings)

    denom = len(req) + 0.5 * len(pref)
    skills = (len(matched_req) + 0.5 * len(matched_pref)) / denom if denom else None

    exp = None
    if jd["required_years"]:
        exp = min(resume["experience_years"] / jd["required_years"], 1.0)

    edu = None
    jd_level = EDU_LEVELS[jd["education_level"]]
    if jd_level:
        edu = min(EDU_LEVELS[resume["education_level"]] / jd_level, 1.0)

    components = {"skills": skills, "semantic": sem, "experience": exp, "education": edu}
    active = {k: weights[k] for k, v in components.items() if v is not None and weights.get(k, 0) > 0}
    total = sum(active.values()) or 1.0
    overall = sum(components[k] * w for k, w in active.items()) / total * 100

    return MatchResult(
        overall=round(overall, 1),
        components=components,
        weights_used={k: round(w / total, 3) for k, w in active.items()},
        matched_required=matched_req,
        missing_required=sorted(req - r_sk),
        matched_preferred=matched_pref,
        missing_preferred=sorted(pref - r_sk),
        extra_skills=sorted(r_sk - req - pref),
        method=method,
        evidence=evidence,
        weak_spots=weak,
        resume={k: v for k, v in resume.items() if k not in {"text", "sections"}},
        jd={k: v for k, v in jd.items() if k != "text"},
    )
