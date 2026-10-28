"""Step 6: turn missing skills into a prioritised learning plan."""
from __future__ import annotations

FIRST_STEPS = {
    "docker": "Containerise one of your existing projects with a Dockerfile and docker-compose.",
    "aws": "Deploy a small app on EC2 or use S3 for storage; the free tier is enough.",
    "kubernetes": "Run a local cluster with minikube and deploy your Dockerised app.",
    "nlp": "Build a text-classification or keyword-extraction mini project with spaCy.",
    "sql": "Practise joins, window functions and CTEs on a public dataset.",
    "power bi": "Build a dashboard from a public dataset and publish a screenshot in your README.",
    "mlops": "Track experiments with MLflow and add a CI step that runs your tests.",
    "llm": "Build a small app that calls an LLM API with structured output.",
    "rag": "Build a document Q&A app with embeddings plus FAISS or ChromaDB.",
    "spark": "Reimplement a pandas pipeline in PySpark on a larger dataset.",
    "git": "Push every project to GitHub with clear commits and a README.",
}


def learning_plan(missing_required: list[str], missing_preferred: list[str]) -> list[dict]:
    plan = []
    for priority, skills in (("High (required)", missing_required), ("Nice to have", missing_preferred)):
        for skill in skills:
            plan.append({
                "skill": skill,
                "priority": priority,
                "first_step": FIRST_STEPS.get(
                    skill, f"Build a small project using {skill} and list it under Projects."
                ),
            })
    return plan
