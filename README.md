# AI-Powered Resume Screening and Job Matching System

<img width="1917" height="1017" alt="Screenshot 2026-09-20 194524" src="https://github.com/user-attachments/assets/b431ff40-6b87-48c2-96ab-6b850c90ebd0" />


Upload a resume, paste a job description, and get a **transparent** match analysis: score, matched and
missing skills, evidence lines, and a learning plan for the gaps. Built with Python, NLP, sentence
embeddings and Streamlit. **No API key is needed** for the core app.

<!-- Add a screenshot: ![demo](docs/demo.png) and your live link here -->

## How it works

```
Resume (PDF/DOCX/TXT) -> text extraction -> cleaning + section split -> info extraction
                                                                             |
Job description -> required/preferred skills, years, degree ------------------+
                                                                             v
                          skills overlap + semantic coverage + experience + education
                                                                             v
                                       score, gaps, evidence, learning plan (Streamlit)
```

| Module | Purpose |
|---|---|
| `src/extraction.py` | PDF (PyMuPDF), DOCX (python-docx), TXT to text |
| `src/preprocessing.py` | cleaning and resume section detection |
| `src/info_extractor.py` | name, email, phone, skills, education level, experience years |
| `src/jd_analyzer.py` | required vs preferred skills, years, degree, keywords |
| `src/matcher.py` | hybrid scoring (skills, semantic, experience, education) |
| `src/gap_analysis.py` | prioritised learning plan |
| `src/llm_explainer.py` | *optional* LLM explanation (needs `ANTHROPIC_API_KEY`) |
| `evaluation/evaluate.py` | compares TF-IDF baseline vs hybrid against your labels |

## Scoring

`score = 0.50 skills + 0.30 semantic + 0.10 experience + 0.10 education` (adjustable in the sidebar).

- **Skills:** required skills weigh 1.0, preferred 0.5.
- **Semantic:** for every JD line, find the most similar resume line (MiniLM embeddings, TF-IDF fallback)
  and average. This is why "built ML models with Scikit-learn" matches "developed ML solutions".
- Components that don't apply (e.g. the JD states no years) are dropped and weights renormalised.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm                 # only if you add spaCy NER later
streamlit run app.py
```

Try it with `data/sample_resume.txt` and `data/sample_jd.txt`. Run tests with `pytest`.

## Optional: LLM explanation

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Then tick "AI explanation" in the sidebar.
Only extracted skills and scores are sent, not the full resume.

## Evaluation

Add 20-30 rows to `evaluation/labeled_pairs.csv` (`resume_path,jd_path,label` with label 0-1) and run
`python -m evaluation.evaluate`. Paste the correlation table here:

| Method | Spearman vs human labels |
|---|---|
| TF-IDF baseline | _todo_ |
| Hybrid (TF-IDF) | _todo_ |
| Hybrid (embeddings) | _todo_ |

## Limitations and fairness

- Score ignores name, contact details, gender, age and photo.
- Skill detection uses a curated list (`data/skills.json`); extend it for your domain.
- Scanned PDFs need OCR. Section detection depends on standard headings.
- It is a screening aid; a human should make the decision.

## Deploy

Push to GitHub and deploy on Streamlit Community Cloud (or Hugging Face Spaces). Add any keys in the
platform's Secrets settings, never in the repo.

## Roadmap

- spaCy NER for names and organisations, OCR for scanned PDFs
- FAISS/ChromaDB search to rank many resumes for one job
- Batch upload with CSV export
