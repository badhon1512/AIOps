import re

import pypdf


def get_resume_text():
    reader = pypdf.PdfReader("./data/CV_MD_BADHON_MIAH.pdf")

    resume_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            resume_text += text + "\n"

    return clean_resume_text(resume_text)


def clean_resume_text(resume_text):
    cleaned_text = resume_text.replace("\x00", " ")
    cleaned_text = cleaned_text.replace("\r", "\n")
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
    cleaned_text = re.sub(r"\s+([,.;:])", r"\1", cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text
