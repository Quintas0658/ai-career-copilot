import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np
import pandas as pd
import re
from fpdf import FPDF
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(page_title="AI Career Copilot", layout="wide")
st.title("🚀 AI-Ready: Your Career Copilot")
st.markdown("Welcome to your personalized AI career assistant. Upload your resume, get matched to future-proof careers, and build a custom roadmap.")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Config")
model_name = st.sidebar.selectbox("Gemini Model", [
    "models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/chat-bison-001"
])
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Output Tokens", 100, 2048, 800)
language = st.sidebar.selectbox("🌐 Language", ["English", "Hindi", "Spanish", "French"])

uploaded_file = st.sidebar.file_uploader("📁 Upload Resume", type=["pdf"])

# --- GEMINI CONFIG ---
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

# --- FUNCTIONS ---
def extract_text_from_resume(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

def get_gemini_embedding(text, model="models/embedding-001"):
    response = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_pdf_report(name, role, skills, score, roadmap):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("AI Career Report")

    pdf.multi_cell(0, 10, f"AI Career Report for {name}")
    pdf.ln()

    pdf.cell(0, 10, f"🧑 Matched Role: {role}", ln=True)
    pdf.cell(0, 10, f"🎯 Recommended Skills: {skills}", ln=True)
    pdf.cell(0, 10, f"⚠️ AI Automation Risk Score: {score}/10", ln=True)

    pdf.ln(10)
    pdf.multi_cell(0, 10, "🧠 AI-Generated 3-Month Roadmap:")
    pdf.multi_cell(0, 10, roadmap)

    output_path = "Career_Report.pdf"
    pdf.output(output_path)
    return output_path

def plot_skill_gap_chart(user_skills, target_skills):
    skills = list(set(target_skills + user_skills))
    user_scores = [1 if skill in user_skills else 0.3 for skill in skills]
    target_scores = [1 for _ in skills]

    angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
    user_scores += user_scores[:1]
    target_scores += target_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, target_scores, label="Target", linewidth=2)
    ax.plot(angles, user_scores, label="You", linestyle='dashed', linewidth=2)
    ax.fill(angles, user_scores, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), skills)
    ax.set_title("Skill Gap Radar")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("skill_radar_chart.png")
    plt.close()

# --- SAMPLE JOB DATA ---
df_jobs = pd.DataFrame([
    {"job_title": "Software Engineer", "resilient_skills": "Python, API Design, Git"},
    {"job_title": "Data Analyst", "resilient_skills": "SQL, Tableau, Data Cleaning"},
    {"job_title": "Cybersecurity Analyst", "resilient_skills": "Networking, Linux, Threat Modeling"},
])

# --- RESUME UPLOAD + EMBEDDING ---
if uploaded_file:
    resume_text = extract_text_from_resume(uploaded_file)
    resume_summary = resume_text[:600]
    st.subheader("📝 Resume Extracted Text (Preview)")
    st.text_area("Extracted Resume Text", resume_summary, height=300)

    if "job_embedding" not in df_jobs.columns:
        df_jobs["job_embedding"] = df_jobs["job_title"].apply(get_gemini_embedding)

    resume_embedding = get_gemini_embedding(resume_summary)
    df_jobs["similarity"] = df_jobs["job_embedding"].apply(lambda x: cosine_similarity(resume_embedding, x))
    best_match = df_jobs.sort_values(by="similarity", ascending=False).iloc[0]

    prompt = f"""
You are a career coach. Based on the resume below and the job title, create:
1. An AI automation risk score (1–10)
2. A 3-month personalized upskilling roadmap with weekly milestones.

Resume:
{resume_summary}

Matched Job Role:
{best_match['job_title']}

Key Recommended Skills:
{best_match['resilient_skills']}
"""

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )

    response = model.generate_content(prompt)
    roadmap_output = response.text

    risk_match = re.search(r"risk score.*?(\d{1,2})", roadmap_output, re.IGNORECASE)
    risk_score = risk_match.group(1) if risk_match else "N/A"

    known_skills = [
        "Python", "Java", "C++", "JavaScript", "SQL", "Git", "Linux", "Networking", "Bash",
        "Tableau", "Excel", "Power BI", "Burp Suite", "Nmap", "Wireshark", "Prompt Engineering",
        "APIs", "Cloud", "Docker", "Kubernetes", "Ethical Hacking"
    ]
    user_skills = [skill for skill in known_skills if skill.lower() in resume_text.lower()]
    target_skills = [s.strip() for s in best_match['resilient_skills'].split(",")]
    plot_skill_gap_chart(user_skills, target_skills)

    st.success("✅ Resume uploaded. Analysis complete.")
    st.subheader("🌟 Matched Job Role")
    st.markdown(f"**Best Match:** {best_match['job_title']}")
    st.markdown(f"**Recommended Skills:** {best_match['resilient_skills']}")
    st.markdown(f"**Similarity Score:** {round(best_match['similarity'], 3)}")
    st.markdown(f"**⚠️ AI Automation Risk Score:** {risk_score}/10")

    st.markdown("### 📈 Skill Gap Radar")
    st.image("skill_radar_chart.png")

    st.markdown("### 🧠 3-Month AI Roadmap")
    st.text_area("Gemini Output:", value=roadmap_output, height=300)

    if st.button("🗕️ Download Career Report"):
        file_path = generate_pdf_report(
            name="Candidate",
            role=best_match['job_title'],
            skills=best_match['resilient_skills'],
            score=risk_score,
            roadmap=roadmap_output
        )
        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇️ Click to download PDF",
                data=f,
                file_name="AI_Career_Report.pdf",
                mime="application/pdf"
            )

    if st.button("🔍 Suggest Resume Improvements"):
        improve_prompt = f"""
You are a professional resume advisor. Suggest improvements for the following resume content targeting the role of {best_match['job_title']}:

{resume_summary}
"""
        improvements = model.generate_content(improve_prompt).text
        st.markdown("### ✨ Suggested Improvements")
        st.markdown(improvements)

    st.markdown("### 🤖 Career Chatbot (Tutor Mode)")
    example_qs = [
        "How can I learn Python for cybersecurity?",
        "What are good certifications for a data analyst?",
        "What should I master to become a cloud engineer?",
        "What’s a good roadmap for AI in software engineering?"
    ]
    selected_q = st.selectbox("Need inspiration?", ["-- Select --"] + example_qs)
    user_query = st.text_input("Ask a career question:", value=selected_q if selected_q != "-- Select --" else "")

    if user_query:
        tutor_prompt = f"You are a career tutor. Respond in {language}. Answer this question in under 150 words: '{user_query}'"
        tutor_response = model.generate_content(tutor_prompt).text
        st.markdown(f"<div style='background-color:#1e1e1e;padding:10px;border-radius:10px'><b>💡 Career Bot:</b> {tutor_response}</div>", unsafe_allow_html=True)

else:
    st.warning("👆 Upload a resume to get started.")