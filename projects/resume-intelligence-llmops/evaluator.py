import os

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

from data import get_resume_text
from scorers import get_scorers


load_dotenv()

PROJECT_NAME = "resume-intelligence-llmops"
EXPERIMENT_NAME = "resume-prompt-evaluation"
PROMPT_NAMES = [
    "cv-skill-extraction-basic",
    "cv-skill-extraction-structured",
    "cv-skill-extraction-strict",
]

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(EXPERIMENT_NAME)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
resume_text = get_resume_text()

data = [
    {
        "inputs": {
            "cv_text": resume_text,
        },
        "expectations": {
            "expected_response": "The response should clearly extract technical skills, soft skills, candidate potential, and a short professional summary from the CV.",
            "expected_skill": "python",
        },
    }
]


def build_predict_fn(prompt):
    def predict_fn(cv_text):
        message = prompt.format(cv_text=cv_text)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": message}],
        )
        return response.choices[0].message.content

    return predict_fn


def evaluate_prompt(prompt_name):
    prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}/1")
    predict_fn = build_predict_fn(prompt)

    with mlflow.start_run(run_name=prompt_name):
        mlflow.log_param("project_name", PROJECT_NAME)
        mlflow.log_param("prompt_name", prompt_name)
        mlflow.log_param("model_name", os.getenv("OPENAI_MODEL", "gpt-4o"))

        result = mlflow.genai.evaluate(
            data=data,
            scorers=get_scorers(),
            predict_fn=predict_fn,
        )

        # print(f"Finished evaluation for {prompt_name}")
        # print(result)


if __name__ == "__main__":
    for prompt_name in PROMPT_NAMES:
        evaluate_prompt(prompt_name)
