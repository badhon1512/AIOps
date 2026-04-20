import mlflow
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

prompt = mlflow.genai.load_prompt("prompts:/FAQ-Prompt/4")
print("Loaded Prompt Template:", prompt.format(question="What is MLOps?"))

question = prompt.format(question="What is MLOps?")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client_response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": question}]
)
print("Model Response:", client_response.choices[0].message.content)