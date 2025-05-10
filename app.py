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
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    st.sidebar.success("✅ API key configured")
except Exception as e:
    st.sidebar.error(f"⚠️ API key error: {str(e)}")
    st.stop()

# --- FUNCTIONS ---
def extract_text_from_resume(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

def get_gemini_embedding(text, model="models/embedding-001"):
    try:
        response = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        # Return a dummy embedding in case of error
        return [0.0] * 768

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

def extract_job_info_with_llm(job_description):
    try:
        st.info("Extracting job information with Gemini...")
        
        prompt = f"""
        Extract the job title and key skills from the job description below.
        Format your response EXACTLY as follows:
        Job Title: [extracted job title]
        Skills: [comma-separated list of key skills]

        Job Description:
        {job_description}
        """
        
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=max_tokens,
            )
        )
        
        response = model.generate_content(prompt).text
        st.success("Successfully extracted job information")
        
        # Extract job title and skills using regex
        job_title_match = re.search(r"Job Title:\s*(.*?)(?:\n|$)", response)
        skills_match = re.search(r"Skills:\s*(.*?)(?:\n|$)", response)
        
        job_title = job_title_match.group(1) if job_title_match else "Unknown Position"
        skills_text = skills_match.group(1) if skills_match else ""
        skills = [skill.strip() for skill in skills_text.split(",")]
        
        return {
            "job_title": job_title,
            "resilient_skills": ", ".join(skills)
        }
    except Exception as e:
        st.error(f"Error extracting job info: {str(e)}")
        # Return a default job in case of error
        return {
            "job_title": "Error Processing Job",
            "resilient_skills": "Error, AI, Processing"
        }

# --- SAMPLE JOB DATA ---
df_jobs = pd.DataFrame([
    {"job_title": "Software Engineer", "resilient_skills": "Python, API Design, Git"},
    {"job_title": "Data Analyst", "resilient_skills": "SQL, Tableau, Data Cleaning"},
    {"job_title": "Cybersecurity Analyst", "resilient_skills": "Networking, Linux, Threat Modeling"},
])

# --- MODE SELECTION ---
st.markdown("## 📍 Select Mode")
mode = st.radio(
    "Choose how you want to explore career options:",
    ["Mode 1: System Recommendations", "Mode 2: Custom Job Descriptions"]
)

# --- RESUME REQUIREMENT ---
if not uploaded_file:
    st.warning("👆 Upload a resume from the sidebar to get started.")
    st.stop()

resume_text = extract_text_from_resume(uploaded_file)
resume_summary = resume_text[:600]
st.subheader("📝 Resume Extracted Text (Preview)")
st.text_area("Extracted Resume Text", resume_summary, height=300)
resume_embedding = get_gemini_embedding(resume_summary)

# --- MODE 1: SYSTEM RECOMMENDATIONS ---
if mode == "Mode 1: System Recommendations":
    st.markdown("### 🔍 System Analyzing Your Resume Against Predefined Roles")
    
    if "job_embedding" not in df_jobs.columns:
        df_jobs["job_embedding"] = df_jobs["job_title"].apply(get_gemini_embedding)

    df_jobs["similarity"] = df_jobs["job_embedding"].apply(lambda x: cosine_similarity(resume_embedding, x))
    best_match = df_jobs.sort_values(by="similarity", ascending=False).iloc[0]
    
    st.success(f"✅ Best match found: {best_match['job_title']}")
    
    job_title = best_match['job_title']
    resilient_skills = best_match['resilient_skills']

# --- MODE 2: CUSTOM JOB DESCRIPTIONS ---
else:
    st.markdown("### 📝 Paste Job Descriptions")
    st.info("In Mode 2, you can paste one or more job descriptions to analyze. Separate multiple job descriptions with '---'")
    
    # 添加示例按钮
    if st.button("Load Example Job Description"):
        example_job = """
        Job Title: Full Stack Developer
        
        We are looking for a Full Stack Developer who is passionate about building innovative web applications. The ideal candidate should have experience with modern JavaScript frameworks, RESTful APIs, and cloud services.
        
        Requirements:
        - 3+ years of experience with React, Vue, or Angular
        - Strong knowledge of Node.js and Express
        - Experience with SQL and NoSQL databases
        - Familiarity with AWS or Azure cloud services
        - Understanding of CI/CD pipelines
        - Good communication skills
        """
        job_descriptions = st.text_area(
            "Paste one or more job descriptions (separate multiple jobs with '---')",
            value=example_job,
            height=300,
            key="job_input"
        )
    else:
        job_descriptions = st.text_area(
            "Paste one or more job descriptions (separate multiple jobs with '---')",
            height=300,
            placeholder="Paste job description here...\n\n---\n\nNext job description here...",
            key="job_input"
        )
    
    # 添加提交按钮以便明确触发分析
    analyze_clicked = st.button("Analyze Job Descriptions")
    
    if job_descriptions and analyze_clicked:
        st.info("Processing job descriptions... This may take a moment.")
        
        # Split multiple job descriptions
        job_list = [jd.strip() for jd in job_descriptions.split("---") if jd.strip()]
        
        if not job_list:
            st.error("Please provide at least one job description.")
            st.stop()
        
        # Process each job description
        custom_jobs = []
        progress_bar = st.progress(0)
        
        for i, job_desc in enumerate(job_list):
            with st.spinner(f"Processing job {i+1}/{len(job_list)}..."):
                try:
                    job_info = extract_job_info_with_llm(job_desc)
                    job_info["job_embedding"] = get_gemini_embedding(job_desc)
                    job_info["similarity"] = cosine_similarity(resume_embedding, job_info["job_embedding"])
                    custom_jobs.append(job_info)
                    progress_bar.progress((i + 1) / len(job_list))
                except Exception as e:
                    st.error(f"Error processing job {i+1}: {str(e)}")
        
        progress_bar.empty()
        
        if not custom_jobs:
            st.error("Could not process any of the job descriptions. Please try again.")
            st.stop()
        
        # Create DataFrame from custom jobs
        df_custom_jobs = pd.DataFrame(custom_jobs)
        
        # Display job matches
        st.markdown("### 🔍 Job Matches")
        for i, job in df_custom_jobs.iterrows():
            st.markdown(f"**Job {i+1}: {job['job_title']}**")
            st.markdown(f"Skills: {job['resilient_skills']}")
            st.markdown(f"Similarity: {round(job['similarity'], 3)}")
            st.markdown("---")
        
        # Select best match or let user choose
        if len(custom_jobs) > 1:
            selected_job_index = st.selectbox(
                "Select a job to analyze further:",
                range(len(custom_jobs)),
                format_func=lambda i: f"{custom_jobs[i]['job_title']} (Match: {round(custom_jobs[i]['similarity'], 2)})"
            )
            best_match = custom_jobs[selected_job_index]
        else:
            best_match = custom_jobs[0]
            
        job_title = best_match['job_title']
        resilient_skills = best_match['resilient_skills']
        
        st.success(f"✅ Selected job for analysis: {job_title}")
        
        # 添加一个标记，表明分析已完成
        st.session_state['job_analysis_complete'] = True
        st.session_state['job_title'] = job_title
        st.session_state['resilient_skills'] = resilient_skills
    elif job_descriptions:
        st.info("Click 'Analyze Job Descriptions' when you're ready to process.")
    else:
        st.warning("Please paste job descriptions to analyze.")
        st.stop()
        
    # 检查分析是否已完成
    if not job_descriptions or not analyze_clicked and not st.session_state.get('job_analysis_complete', False):
        st.stop()
    
    # 如果有会话状态保存的结果，使用它们
    if st.session_state.get('job_analysis_complete', False) and not analyze_clicked:
        job_title = st.session_state.get('job_title')
        resilient_skills = st.session_state.get('resilient_skills')
        st.success(f"Using previously analyzed job: {job_title}")

# --- COMMON ANALYSIS FOR BOTH MODES ---
prompt = f"""
You are a career coach. Based on the resume below and the job title, create:
1. An AI automation risk score (1–10)
2. A 3-month personalized upskilling roadmap with weekly milestones.

Resume:
{resume_summary}

Matched Job Role:
{job_title}

Key Recommended Skills:
{resilient_skills}
"""

st.info("Generating career analysis with Gemini...")

model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
)

try:
    response = model.generate_content(prompt)
    roadmap_output = response.text
    st.success("Career analysis complete!")
except Exception as e:
    st.error(f"Error generating roadmap: {str(e)}")
    roadmap_output = "Error generating roadmap. Please try again with a different model or settings."

risk_match = re.search(r"risk score.*?(\d{1,2})", roadmap_output, re.IGNORECASE)
risk_score = risk_match.group(1) if risk_match else "N/A"

known_skills = [
    "Python", "Java", "C++", "JavaScript", "SQL", "Git", "Linux", "Networking", "Bash",
    "Tableau", "Excel", "Power BI", "Burp Suite", "Nmap", "Wireshark", "Prompt Engineering",
    "APIs", "Cloud", "Docker", "Kubernetes", "Ethical Hacking"
]
user_skills = [skill for skill in known_skills if skill.lower() in resume_text.lower()]
target_skills = [s.strip() for s in resilient_skills.split(",")]
try:
    plot_skill_gap_chart(user_skills, target_skills)
except Exception as e:
    st.error(f"Error creating skill gap chart: {str(e)}")

st.subheader("🌟 Job Analysis Results")
st.markdown(f"**Target Role:** {job_title}")
st.markdown(f"**Recommended Skills:** {resilient_skills}")
st.markdown(f"**⚠️ AI Automation Risk Score:** {risk_score}/10")

try:
    st.markdown("### 📈 Skill Gap Radar")
    st.image("skill_radar_chart.png")
except Exception as e:
    st.error(f"Error displaying skill chart: {str(e)}")

st.markdown("### 🧠 3-Month AI Roadmap")
st.text_area("Gemini Output:", value=roadmap_output, height=300)

if st.button("🗕️ Download Career Report"):
    try:
        file_path = generate_pdf_report(
            name="Candidate",
            role=job_title,
            skills=resilient_skills,
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
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")

if st.button("🔍 Suggest Resume Improvements"):
    try:
        improve_prompt = f"""
You are a professional resume advisor. Suggest improvements for the following resume content targeting the role of {job_title}:

{resume_summary}
"""
        improvements = model.generate_content(improve_prompt).text
        st.markdown("### ✨ Suggested Improvements")
        st.markdown(improvements)
    except Exception as e:
        st.error(f"Error generating resume improvements: {str(e)}")

st.markdown("### 🤖 Career Chatbot (Tutor Mode)")
example_qs = [
    "How can I learn Python for cybersecurity?",
    "What are good certifications for a data analyst?",
    "What should I master to become a cloud engineer?",
    "What's a good roadmap for AI in software engineering?"
]
selected_q = st.selectbox("Need inspiration?", ["-- Select --"] + example_qs)
user_query = st.text_input("Ask a career question:", value=selected_q if selected_q != "-- Select --" else "")

if user_query:
    try:
        tutor_prompt = f"You are a career tutor. Respond in {language}. Answer this question in under 150 words: '{user_query}'"
        tutor_response = model.generate_content(tutor_prompt).text
        st.markdown(f"<div style='background-color:#1e1e1e;padding:10px;border-radius:10px'><b>💡 Career Bot:</b> {tutor_response}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error getting career advice: {str(e)}")