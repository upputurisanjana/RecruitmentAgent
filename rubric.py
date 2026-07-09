"""
rubric.py — RubricSchema definition and the default rubric for the Junior AI Engineer role.

The rubric is data, not a prompt string.  It is loaded into agent state and displayed
verbatim in the UI sidebar so reviewers can see exactly what weights were applied.
"""

from schemas import RubricSchema, RubricCriterion

# ---------------------------------------------------------------------------
# Score-level descriptor (shared across all criteria)
# ---------------------------------------------------------------------------

SCORE_SCALE: dict[int, str] = {
    0: "No evidence in résumé",
    1: "Mentioned only, no supporting detail",
    2: "Coursework-level evidence only",
    3: "One project with moderate depth",
    4: "Multiple projects or clear depth in one",
    5: "Strong, repeated, well-documented evidence across multiple contexts",
}

# ---------------------------------------------------------------------------
# Default rubric instance — Junior AI Engineer, TechVest
# Weights must sum to 1.0 (100 %).
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC = RubricSchema(
    title="Junior AI Engineer Evaluation Rubric",
    role="Junior AI Engineer — TechVest",
    criteria=[
        RubricCriterion(
            name="Python / ML Fundamentals",
            weight=0.35,
            descriptor=(
                "Python proficiency (data-science stack: NumPy, pandas) plus understanding of "
                "core ML concepts: regression, classification, basic deep learning "
                "(backprop, CNNs/RNNs, or Transformers).  "
                "Weighted highest because this is the non-negotiable foundation for the role."
            ),
            scale=SCORE_SCALE,
        ),
        RubricCriterion(
            name="Relevant Projects",
            weight=0.30,
            descriptor=(
                "Hands-on project evidence — GitHub repos, internships, Kaggle, or coursework "
                "with measurable outcomes (metrics, competition rankings, production deployments).  "
                "Coursework-only is borderline (score ≤ 2); independent or professional projects "
                "with stated outcomes score higher."
            ),
            scale=SCORE_SCALE,
        ),
        RubricCriterion(
            name="Hands-on Tooling (Frameworks & Libraries)",
            weight=0.20,
            descriptor=(
                "Real usage of at least one ML/LLM framework: scikit-learn, PyTorch, TensorFlow, "
                "LangChain, HuggingFace Transformers, or equivalent.  "
                "Also captures MLOps hygiene: MLflow, experiment tracking, Docker, CI/CD, cloud basics."
            ),
            scale=SCORE_SCALE,
        ),
        RubricCriterion(
            name="Communication",
            weight=0.15,
            descriptor=(
                "Evidence of clear written documentation, technical presentations to non-technical "
                "audiences, or team collaboration.  "
                "Weighted least but required — a Junior who cannot communicate findings cannot "
                "contribute to cross-functional product work."
            ),
            scale=SCORE_SCALE,
        ),
    ],
)


def get_default_rubric() -> RubricSchema:
    """Return the default rubric.  Import this in tools, runner, and tests."""
    return DEFAULT_RUBRIC


def rubric_as_table(rubric: RubricSchema) -> list[dict]:
    """Return the rubric as a list of dicts suitable for st.dataframe / pd.DataFrame."""
    return [
        {
            "Criterion": c.name,
            "Weight": f"{int(c.weight * 100)}%",
            "Description": c.descriptor,
        }
        for c in rubric.criteria
    ]


def scale_as_table() -> list[dict]:
    """Return the 0–5 descriptor scale as a list of dicts."""
    return [{"Score": k, "Descriptor": v} for k, v in SCORE_SCALE.items()]
