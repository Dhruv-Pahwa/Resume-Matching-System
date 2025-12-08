import streamlit as st
import os
import docx2txt
import PyPDF2
import matplotlib.pyplot as plt
import base64
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------
# APP CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="wide")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -----------------------------------------------------------
# FILE EXTRACTORS
# -----------------------------------------------------------
def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except:
        return ""
    return text


def extract_text_from_docx(path):
    try:
        return docx2txt.process(path)
    except:
        return ""


def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def extract_text(path):
    if path.endswith(".pdf"):
        return extract_text_from_pdf(path)
    elif path.endswith(".docx"):
        return extract_text_from_docx(path)
    elif path.endswith(".txt"):
        return extract_text_from_txt(path)
    return ""


# -----------------------------------------------------------
# HUMAN STYLE STRENGTHS & WEAKNESSES
# -----------------------------------------------------------
def analyze_resume_strengths_weaknesses(text, job_description):
    text_lower = text.lower()
    strengths = []
    weaknesses = []

    # --- Experience depth ---
    if any(x in text_lower for x in ["intern", "experience", "worked", "project", "developed", "built"]):
        strengths.append("Shows practical project or internship experience.")
    else:
        weaknesses.append("Lacks mention of clear project or work experience.")

    # --- Technical skills detection ---
    skill_keywords = ["python", "java", "sql", "machine learning", "data analysis", "cloud", "api"]
    skill_count = sum(1 for sk in skill_keywords if sk in text_lower)

    if skill_count >= 4:
        strengths.append("Strong technical skillset relevant to the job.")
    elif 2 <= skill_count < 4:
        strengths.append("Has some relevant technical skills.")
        weaknesses.append("Technical skills could be expanded for this role.")
    else:
        weaknesses.append("Technical depth appears limited based on the resume.")

    # --- Achievements & measurable results ---
    if any(x in text_lower for x in ["improved", "increased", "reduced", "achieved", "%", "led"]):
        strengths.append("Includes measurable achievements that show real impact.")
    else:
        weaknesses.append("Does not list measurable achievements or quantifiable impact.")

    # --- Certifications ---
    if "cert" in text_lower:
        strengths.append("Has certifications that enhance professional credibility.")
    else:
        weaknesses.append("No certifications mentioned.")

    # --- Soft skills check ---
    if any(x in text_lower for x in ["team", "lead", "collaborat", "communication"]):
        strengths.append("Shows teamwork or leadership-related soft skills.")
    else:
        weaknesses.append("Soft skills are not clearly highlighted.")

    # --- JD alignment (semantic approximation) ---
    jd_keywords = job_description.lower().split()
    resume_words = text_lower.split()
    overlap = len(set(jd_keywords).intersection(resume_words))

    if overlap > 20:
        strengths.append("Resume content aligns well with job requirements.")
    else:
        weaknesses.append("Alignment with job description appears somewhat limited.")

    return strengths[:5], weaknesses[:5]


# -----------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------
def create_plot(names, scores):
    plt.figure(figsize=(12, 6))
    colors = ["green" if s >= 0.5 else "orange" if s >= 0.3 else "red" for s in scores]

    plt.scatter(names, scores, c=colors, s=120)
    plt.axhline(0.3, color="red", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Similarity Score (0 - 1)")
    plt.title("Resume Matching Scores")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close()

    return encoded


# -----------------------------------------------------------
# FEEDBACK LABELS
# -----------------------------------------------------------
def generate_feedback(score):
    if score >= 0.5:
        return "Excellent match! Strong alignment with job description.", "🟢 Strong"
    elif score >= 0.3:
        return "Moderate match — resume shows partial relevance.", "🟠 Medium"
    else:
        return "Weak match — limited alignment with job requirements.", "🔴 Weak"


# -----------------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------------
st.title("📄 AI Resume Matcher & Candidate Insights")
st.write("Upload multiple resumes and compare them to a job description. Get similarity score, strengths, and weaknesses.")

with st.sidebar:
    st.header("⚙️ Inputs")
    job_description = st.text_area("Job Description", height=200)
    resumes = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    run_btn = st.button("Match Resumes")


# -----------------------------------------------------------
# PROCESSING
# -----------------------------------------------------------
if run_btn:

    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    if not resumes:
        st.error("Please upload at least one resume.")
        st.stop()

    extracted_texts = []
    filenames = []

    for file in resumes:
        save_path = os.path.join(UPLOAD_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
        text = extract_text(save_path)

        if text.strip():
            extracted_texts.append(text)
            filenames.append(file.name)
        else:
            st.warning(f"⚠ Could not extract text from {file.name}")

    if not extracted_texts:
        st.error("No readable resumes found.")
        st.stop()

    # TF-IDF Similarity
    vectorizer = TfidfVectorizer().fit_transform([job_description] + extracted_texts)
    vectors = vectorizer.toarray()
    similarities = cosine_similarity([vectors[0]], vectors[1:])[0]

    sorted_data = sorted(zip(filenames, extracted_texts, similarities), key=lambda x: x[2], reverse=True)

    # -----------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------
    st.subheader("📊 Resume Matching Results")

    plot = create_plot([x[0] for x in sorted_data], [round(x[2], 2) for x in sorted_data])
    st.image("data:image/png;base64," + plot)

    results_csv = []

    for filename, text, score in sorted_data:
        feedback, label = generate_feedback(score)
        strengths, weaknesses = analyze_resume_strengths_weaknesses(text, job_description)

        with st.expander(f"📄 {filename} — Score: {round(score, 2)} ({label})"):
            st.markdown(f"### Match Score: `{round(score, 2)}`")
            st.markdown(f"**Feedback:** {feedback}")

            st.write("### 🟢 Strengths")
            for s in strengths:
                st.write(f"- {s}")

            st.write("### 🔴 Weaknesses")
            for w in weaknesses:
                st.write(f"- {w}")

        results_csv.append(f"{filename},{round(score, 2)},{label}")

    st.download_button(
        "⬇ Download CSV Report",
        "\n".join(results_csv),
        "resume_results.csv",
        "text/csv"
    )
