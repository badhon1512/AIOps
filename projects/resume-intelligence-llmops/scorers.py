import mlflow
from mlflow.genai import scorer


@scorer
def concise(outputs):
    return len(outputs) < 1200


@scorer
def has_sections(outputs):
    text = outputs.lower()
    return (
        "technical skills" in text
        and "soft skills" in text
        and "potential" in text
        and "summary" in text
    )


@scorer
def mentions_expected_skill(outputs, expectations):
    expected_skill = expectations.get("expected_skill", "").lower()

    if expected_skill == "":
        return True

    return expected_skill in outputs.lower()


def get_scorers():
    return [
        mlflow.genai.scorers.Correctness(),
        mlflow.genai.scorers.Guidelines(
            name="professional_tone",
            guidelines="The response should be professional, clear, and recruiter-friendly.",
        ),
        concise,
        has_sections,
        mentions_expected_skill,
    ]
