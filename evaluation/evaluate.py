"""Compare TF-IDF baseline vs the hybrid matcher against human labels.

Usage:  python -m evaluation.evaluate
Fill evaluation/labeled_pairs.csv with 20-30 rows: resume_path, jd_path, label (0-1 = how good a fit
YOU judge it to be). Reports Spearman rank correlation for each method (higher = better agreement).
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.matcher import compute_match, tfidf_document_similarity

ROOT = Path(__file__).resolve().parent.parent


def main():
    df = pd.read_csv(ROOT / "evaluation" / "labeled_pairs.csv")
    if len(df) < 5:
        print(f"Only {len(df)} labelled pairs; add more (aim for 20-30) for a meaningful result.")
    rows = []
    for _, r in df.iterrows():
        resume = (ROOT / r.resume_path).read_text(encoding="utf-8", errors="ignore")
        jd = (ROOT / r.jd_path).read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "label": r.label,
            "tfidf_baseline": tfidf_document_similarity(resume, jd),
            "hybrid_tfidf": compute_match(resume, jd, use_embeddings=False).overall,
            "hybrid_embeddings": compute_match(resume, jd, use_embeddings=True).overall,
        })
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    if len(res) >= 3:
        print("\nSpearman correlation with human labels:")
        for col in ["tfidf_baseline", "hybrid_tfidf", "hybrid_embeddings"]:
            rho, _ = spearmanr(res["label"], res[col])
            print(f"  {col:20s} {rho:.3f}")


if __name__ == "__main__":
    main()
