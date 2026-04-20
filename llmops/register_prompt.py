import mlflow
import os
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Prompt Evaluation")

prompt = """you are a helpful assistant that answers question about any topic. 
Rules:
1. Always provide a concise and accurate answer to the user's question.
2. If the question is ambiguous, ask for clarification. 

Answer the following question: {{question}}"""
mlflow.genai.register_prompt(
    name="FAQ-Prompt",
    template=prompt,
    commit_message="Registering FAQ prompt"
)   
