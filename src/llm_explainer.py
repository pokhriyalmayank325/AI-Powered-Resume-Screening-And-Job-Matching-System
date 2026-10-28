"""OPTIONAL: plain-English explanation of a match using an LLM.

The core app never needs this. It only runs when ANTHROPIC_API_KEY is set.
Only the extracted summary (skills, scores) is sent, not the full resume text.
"""
from __future__ import annotations

import os


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def explain_match(result: dict) -> str:
    import anthropic

    summary = (
        f"Overall match: {result['overall']}%\n"
        f"Component scores (0-1): {result['components']}\n"
        f"Matched required skills: {result['matched_required']}\n"
        f"Missing required skills: {result['missing_required']}\n"
        f"Missing preferred skills: {result['missing_preferred']}\n"
        f"Candidate experience (years): {result['resume'].get('experience_years')}\n"
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-sonnet-5"),
        max_tokens=500,
        system=(
            "You explain resume-to-job match results to a recruiter in under 150 words. "
            "Use only the data given. Do not comment on the candidate's name, gender, age or background."
        ),
        messages=[{"role": "user", "content": summary}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
