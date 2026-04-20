import mlflow
from mlflow.genai import scorer
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment("Prompt Evaluation")


prompt = mlflow.genai.load_prompt("prompts:/FAQ-Prompt/5")
def predict_fn(question):

    msg = prompt.format(question=question)

    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": msg}]
    )
    
    
    return response.choices[0].message.content

data = [
    {
        "inputs": {
            "question": "Who is the inventor of Python programming language?",
        },
        "expectations": {
            "expected_response": "Python was created by Guido van Rossum.",
        },
    },
    {
        "inputs": {
            "question": "What is the capital of France?",
        },
        "expectations": {
            "expected_response": "The capital of France is Paris.",
        },
    },
    {
        "inputs": {
            "question": "What is MLOps?",
        },
        "expectations": {
            "expected_response": "MLOps is a set of practices for deploying, monitoring, and maintaining machine learning systems in production.",
        },
    }
]
        

#Custom scorer
@scorer
def concise(inputs, outputs, expectations):
    return len(outputs) < 1000

scorers = [
    mlflow.genai.scorers.Correctness(),
    mlflow.genai.scorers.Guidelines(name="is_professional", guidelines="The response should be professional, and respectful in tone."),
    concise
]


if __name__ == "__main__":

    mlflow.genai.evaluate(
        data=data,
        scorers=scorers,
        predict_fn=predict_fn
    )
