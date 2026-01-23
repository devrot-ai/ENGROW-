import os
import re
import string

import PyPDF2
import spacy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------

RESUME_FOLDER = "resumes"          # folder containing candidate PDFs
JOB_DESC_FILE = "job_description.txt"

# Simple skill dictionary (you can expand this as needed)
TECH_SKILLS = [
    "python", "java", "c++", "javascript", "react", "node", "django",
    "flask", "sql", "mysql", "postgresql", "mongodb", "git", "docker",
    "kubernetes", "aws", "azure", "gcp", "html", "css", "pandas", "numpy",
    "tensorflow", "pytorch", "nlp", "machine learning", "deep learning",
]

SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving",
    "time management", "adaptability", "creativity", "critical thinking",
]

TOOLS = [
    "jira", "confluence", "excel", "powerpoint", "notion", "slack",
]

# Weighting for final score
W_SIMILARITY = 0.6
W_SKILL_MATCH = 0.3
W_EXPERIENCE = 0.1

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ---------- HELPER FUNCTIONS ----------

def read_job_description(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def preprocess_text(text: str) -> str:
    """
    Lowercasing, remove punctuation, stopwords; lemmatization.
    """
    text = text.lower()
    text = text.replace("\n", " ")
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.strip()
        if lemma:
            tokens.append(lemma)
    return " ".join(tokens)


def extract_sections(raw_text: str):
    """
    Very simple section detection heuristics: search for keywords.
    Returns dict with 'skills', 'experience', 'education' as raw text snippets.
    """
    text = raw_text.lower()
    sections = {"skills": "", "experience": "", "education": ""}

    # Simple regex-based splitting by common headers
    patterns = {
        "skills": r"(skills|technical skills|key skills)",
        "experience": r"(experience|work experience|professional experience|employment history)",
        "education": r"(education|academic background|qualifications)",
    }

    for key, pat in patterns.items():
        match = re.search(pat, text)
        if match:
            start = match.start()
            # get chunk from this header to next header or end
            next_headers = [m.start() for k2, p2 in patterns.items()
                            if k2 != key for m in re.finditer(p2, text)]
            next_headers = [h for h in next_headers if h > start]
            end = min(next_headers) if next_headers else len(text)
            sections[key] = text[start:end]

    return sections


def extract_keywords_from_text(text: str, keyword_list):
    found = set()
    for kw in keyword_list:
        # simple word boundary search
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text.lower()):
            found.add(kw.lower())
    return list(found)


def compute_skill_match(resume_skills, jd_skills):
    """
    Skill match = |intersection| / |JD skills| (0 if JD has no skills).
    """
    jd_set = set([s.lower() for s in jd_skills])
    res_set = set([s.lower() for s in resume_skills])
    if not jd_set:
        return 0.0
    return len(jd_set & res_set) / len(jd_set)


def estimate_experience_years(text: str) -> float:
    """
    Very rough heuristic: search for patterns like 'X years' or 'X+ years'.
    """
    matches = re.findall(r"(\d+)\s*\+?\s*year", text.lower())
    years = [int(m) for m in matches]
    return float(max(years)) if years else 0.0


# ---------- MAIN PIPELINE ----------

def main():
    # 1. Ingest JD and resumes
    if not os.path.exists(JOB_DESC_FILE):
        raise FileNotFoundError(f"Job description file '{JOB_DESC_FILE}' not found.")

    jd_raw = read_job_description(JOB_DESC_FILE)

    resume_files = [
        os.path.join(RESUME_FOLDER, f)
        for f in os.listdir(RESUME_FOLDER)
        if f.lower().endswith(".pdf")
    ]
    if not resume_files:
        raise FileNotFoundError(f"No PDF resumes found in folder '{RESUME_FOLDER}'.")

    # 2. Extract text from resumes
    resumes_raw = []
    for path in resume_files:
        text = extract_text_from_pdf(path)
        resumes_raw.append(text)

    # 3. Text preprocessing
    jd_clean = preprocess_text(jd_raw)
    resumes_clean = [preprocess_text(t) for t in resumes_raw]

    # 4. Section detection + skill extraction + experience estimation
    jd_sections = extract_sections(jd_raw)
    jd_skills_all = (
        extract_keywords_from_text(jd_sections.get("skills", jd_raw), TECH_SKILLS)
        + extract_keywords_from_text(jd_sections.get("skills", jd_raw), SOFT_SKILLS)
        + extract_keywords_from_text(jd_sections.get("skills", jd_raw), TOOLS)
    )
    jd_skills_all = list(set(jd_skills_all))

    resume_skill_match_scores = []
    resume_experience_scores = []
    resume_skill_lists = []

    for raw in resumes_raw:
        sections = extract_sections(raw)
        skills_text = sections.get("skills", raw)
        exp_text = sections.get("experience", raw)

        res_skills = (
            extract_keywords_from_text(skills_text, TECH_SKILLS)
            + extract_keywords_from_text(skills_text, SOFT_SKILLS)
            + extract_keywords_from_text(skills_text, TOOLS)
        )
        res_skills = list(set(res_skills))

        skill_match = compute_skill_match(res_skills, jd_skills_all)
        exp_years = estimate_experience_years(exp_text)

        resume_skill_lists.append(res_skills)
        resume_skill_match_scores.append(skill_match)
        resume_experience_scores.append(exp_years)

    # Normalize experience to [0,1] using min-max
    if resume_experience_scores:
        exp_arr = np.array(resume_experience_scores, dtype=float)
        exp_min, exp_max = exp_arr.min(), exp_arr.max()
        if exp_max > exp_min:
            exp_norm = (exp_arr - exp_min) / (exp_max - exp_min)
        else:
            exp_norm = np.zeros_like(exp_arr)
    else:
        exp_norm = np.zeros(len(resume_files))

    # 5. Feature engineering (TF-IDF) and similarity scoring
    corpus = [jd_clean] + resumes_clean
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vec = tfidf_matrix[0:1]
    resume_vecs = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vec, resume_vecs)[0]

    # Normalize similarity and skill match to [0,1] (they already mostly are, but to be safe)
    sim_arr = np.array(similarities, dtype=float)
    sim_min, sim_max = sim_arr.min(), sim_arr.max()
    if sim_max > sim_min:
        sim_norm = (sim_arr - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = np.zeros_like(sim_arr)

    skill_arr = np.array(resume_skill_match_scores, dtype=float)
    skill_min, skill_max = skill_arr.min(), skill_arr.max()
    if skill_max > skill_min:
        skill_norm = (skill_arr - skill_min) / (skill_max - skill_min)
    else:
        skill_norm = np.zeros_like(skill_arr)

    # 6. Candidate ranking logic
    final_scores = (
        W_SIMILARITY * sim_norm
        + W_SKILL_MATCH * skill_norm
        + W_EXPERIENCE * exp_norm
    )

    # 7. Output table
    data = []
    for i, path in enumerate(resume_files):
        data.append({
            "resume_file": os.path.basename(path),
            "similarity_score": round(float(sim_norm[i]), 4),
            "skill_match_score": round(float(skill_norm[i]), 4),
            "experience_score": round(float(exp_norm[i]), 4),
            "final_score": round(float(final_scores[i]), 4),
            "skills_detected": ", ".join(resume_skill_lists[i]),
        })

    df = pd.DataFrame(data)
    df_sorted = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    print("\n=== Ranked Candidates ===\n")
    print(df_sorted)

    # Save to CSV for report/dashboard
    df_sorted.to_csv("ranked_candidates.csv", index=False)
    print("\nResults saved to 'ranked_candidates.csv'.")

if __name__ == "__main__":
    main()
