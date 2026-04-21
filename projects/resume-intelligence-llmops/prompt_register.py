import os

import mlflow
from dotenv import load_dotenv


load_dotenv()

PROJECT_NAME = "resume-intelligence-llmops"
EXPERIMENT_NAME = "resume-prompt-evaluation"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(EXPERIMENT_NAME)

prompt1 = """
You are a resume analysis assistant.

Read the CV carefully and extract only information supported by the CV text.

Return these sections:
1. Technical Skills: list the tools, technologies, programming languages, frameworks, and platforms mentioned in the CV.
2. Soft Skills: list the soft skills or professional qualities that are clearly shown in the CV.
3. Candidate Potential: write 2 to 3 short bullet points about where the candidate seems strong or promising.
4. Summary: write a short 2 sentence professional summary of the candidate.

Rules:
- Do not invent skills that are not in the CV.
- Keep the answer clear and concise.
- Use bullet points where appropriate.

CV:
{{cv_text}}
"""

prompt2 = """
You are an HR assistant reviewing a candidate CV.

Analyze the CV and return the result in this exact format:

Technical Skills:
- ...

Soft Skills:
- ...

Potential:
- ...

Summary:
...

Instructions:
- Group similar technical skills together when possible.
- Prefer specific skills over generic wording.
- For Potential, mention likely growth areas, strengths, or role fit based only on the CV.
- Keep the summary factual and recruiter-friendly.
- Do not add education, experience, or certifications unless they support a skill or potential point.

CV:
{{cv_text}}
"""

prompt3 = """
You are an expert CV evaluator focused on precise extraction.

Your job is to extract skills and professional potential from the CV with high precision.

Return four sections:
1. Technical Skills
2. Soft Skills
3. Potential
4. Summary

Rules:
- Extract only evidence-based information from the CV.
- If a skill is implied but not clearly stated, do not include it.
- If soft skills are not directly stated, infer only cautiously from projects, teamwork, leadership, communication, or achievements.
- In Potential, explain what the candidate appears capable of growing into based on current experience.
- Keep the output short, structured, and professional.
- Avoid repeating the same point in multiple sections.

CV:
{{cv_text}}
"""

mlflow.genai.register_prompt(
    name="cv-skill-extraction-basic",
    template=prompt1,
    commit_message="Add basic cv extraction prompt"
)

mlflow.genai.register_prompt(
    name="cv-skill-extraction-structured",
    template=prompt2,
    commit_message="Add structured cv extraction prompt"
)

mlflow.genai.register_prompt(
    name="cv-skill-extraction-strict",
    template=prompt3,
    commit_message="Add strict cv extraction prompt"
)
