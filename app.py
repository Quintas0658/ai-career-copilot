import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np
import pandas as pd
import re
from fpdf import FPDF
import matplotlib.pyplot as plt
import openai
import random
import requests  # 添加requests库导入

# --- CONFIG ---
st.set_page_config(page_title="AI Career Copilot", layout="wide")
st.title("🚀 AI-Ready: Your Career Copilot")
st.markdown("Welcome to your personalized AI career assistant. Upload your resume, get matched to future-proof careers, and build a custom roadmap.")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Config")

# 添加调试模式选项
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

# API选择
api_choice = st.sidebar.radio(
    "AI Provider:",
    ["Google Gemini", "OpenAI"]
)

# Gemini配置
if api_choice == "Google Gemini":
    model_name = st.sidebar.selectbox("Gemini Model", [
        "models/gemini-1.5-pro", "models/gemini-1.5-flash", "models/chat-bison-001"
    ])
    
# OpenAI配置
else:
    model_name = st.sidebar.selectbox("OpenAI Model", [
        "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"
    ])
    custom_base_url = st.sidebar.text_input(
        "OpenAI API Base URL (Optional)",
        value="https://api.openai-up.com",
        help="Override the default OpenAI API URL (https://api.openai.com)"
    )

temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Output Tokens", 100, 2048, 800)
language = st.sidebar.selectbox("🌐 Language", ["English", "Chinese", "Hindi", "Spanish", "French"])

uploaded_file = st.sidebar.file_uploader("📁 Upload Resume", type=["pdf"])

# 添加模拟模式
use_mock_data = st.sidebar.checkbox("Use Mock Data (No API calls)", value=False)

# 确保 force_direct_parsing 总是有定义
if debug_mode:
    st.sidebar.markdown("### Advanced Options")
    force_direct_parsing = st.sidebar.checkbox("Force direct text parsing (no API)", value=False)
    if force_direct_parsing:
        st.sidebar.info("Using direct text analysis instead of AI APIs")
else:
    force_direct_parsing = False # Default value when debug_mode is off

# --- API CONFIG ---
try:
    if api_choice == "Google Gemini" and not use_mock_data:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
        st.sidebar.success("✅ Gemini API key configured")
    elif api_choice == "OpenAI" and not use_mock_data:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        if custom_base_url:
            openai.base_url = custom_base_url
        openai.api_key = OPENAI_API_KEY
        st.sidebar.success("✅ OpenAI API key configured")
    elif use_mock_data:
        st.sidebar.success("✅ Using mock data mode (no API calls)")
except Exception as e:
    if not use_mock_data:
        st.sidebar.error(f"⚠️ API key error: {str(e)}")
        st.sidebar.warning("Enabling mock data mode")
        use_mock_data = True

# --- FUNCTIONS ---
def extract_text_from_resume(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])

def is_chinese_text(text):
    """检测文本是否主要为中文"""
    # 统计中文字符的数量
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    # 如果文本中有超过20%的中文字符，认为是中文文本
    return len(chinese_chars) > 0.2 * len(text.strip())

def get_mock_embedding():
    """Generate mock embedding for testing without API"""
    return [random.uniform(-1, 1) for _ in range(768)]

def get_gemini_embedding(text, model="models/embedding-001"):
    """Get embedding using Gemini API"""
    if use_mock_data:
        return get_mock_embedding()
    
    try:
        response = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        st.error(f"Gemini Embedding error: {str(e)}")
        return get_mock_embedding()

def get_openai_embedding(text, model="text-embedding-ada-002"):
    """Get embedding using OpenAI API"""
    if use_mock_data:
        return get_mock_embedding()
    
    try:
        # 直接返回mock数据，因为API解析存在问题
        if debug_mode:
            st.warning("Using mock embedding for OpenAI due to API compatibility issues")
        return get_mock_embedding()
        
        # 以下代码暂时不使用，因为解析问题
        """
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        # 处理可能的不同响应格式
        if hasattr(response, 'data'):
            # 标准OpenAI响应
            return response.data[0].embedding
        elif isinstance(response, dict) and 'data' in response:
            # 返回的是字典
            return response['data'][0]['embedding']
        elif isinstance(response, str):
            # 如果是字符串，可能是JSON字符串
            import json
            try:
                data = json.loads(response)
                return data['data'][0]['embedding']
            except:
                st.error("Unable to parse OpenAI response as JSON")
                return get_mock_embedding()
        else:
            st.error(f"Unexpected OpenAI response format: {type(response)}")
            return get_mock_embedding()
        """
    except Exception as e:
        st.error(f"OpenAI Embedding error: {str(e)}")
        return get_mock_embedding()

def get_text_embedding(text):
    """Get embedding based on selected API"""
    if api_choice == "Google Gemini":
        return get_gemini_embedding(text)
    else:
        return get_openai_embedding(text)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    # Prevent division by zero
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0
    return np.dot(a, b) / norm_product

def generate_pdf_report(name, role, skills, score, roadmap):
    # 使用FPDF2代替FPDF，更好地支持中文字符
    try:
        from fpdf import FPDF
        fpdf_version = 2  # 假设使用FPDF2
    except ImportError:
        st.warning("FPDF2 not available. Please run 'pip install fpdf2' for better CJK support.")
        from fpdf import FPDF
        fpdf_version = 1
        
    pdf = FPDF()
    
    # 添加中文支持
    if language == "Chinese":
        try:
            if fpdf_version >= 2:
                # FPDF2支持中文的方式
                pdf.add_font('fireflysung', '', '')  # 使用FPDF2内置的中文字体
                pdf.set_font('fireflysung', size=12)
            else:
                # 尝试添加中文支持 - 使用内置ArialUnicode如果可用
                pdf.add_font('ArialUnicode', '', '', uni=True)
                pdf.set_font('ArialUnicode', size=12)
        except Exception as e:
            # 如果找不到中文字体，使用标准字体并显示警告
            st.warning(f"中文字体不可用: {str(e)}。尝试使用标准字体。")
            pdf.set_font("Arial", size=12)
    else:
        # 非中文语言使用标准Arial字体
        pdf.set_font("Arial", size=12)
        
    pdf.add_page()
    
    # 设置标题
    if language == "Chinese":
        report_title = "AI职业发展报告"
        matched_role = "匹配职位"
        skills_label = "推荐技能"
        risk_score_label = "AI自动化风险评分"
        roadmap_label = "AI生成的3个月提升计划"
    else:
        report_title = "AI Career Report"
        matched_role = "Matched Role"
        skills_label = "Recommended Skills"
        risk_score_label = "AI Automation Risk Score"
        roadmap_label = "AI-Generated 3-Month Roadmap"
    
    pdf.set_title(report_title)

    # 移除可能导致编码问题的表情符号
    name = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', name)  # 保留ASCII和中文字符
    role = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', role)
    skills = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', skills)
    roadmap = re.sub(r'[^\x00-\x7F\u4e00-\u9fff]+', '', roadmap)

    pdf.multi_cell(0, 10, f"{report_title} - {name}")
    pdf.ln()

    pdf.cell(0, 10, f"{matched_role}: {role}", ln=True)
    pdf.cell(0, 10, f"{skills_label}: {skills}", ln=True)
    pdf.cell(0, 10, f"{risk_score_label}: {score}/10", ln=True)

    pdf.ln(10)
    pdf.multi_cell(0, 10, roadmap_label)
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

def get_mock_llm_response(prompt_text):
    """Generate mock LLM response for testing without API"""
    prompt_lower = prompt_text.lower()
    job_title_for_mock = "General Professional"
    skills_for_mock = "Communication, Organization, Problem Solving, MS Office Suite"

    title_match = re.search(r"Matched Job Role(?:\s*\(.*?\))?:\s*(.*?)(?:\n|$)", prompt_text, re.IGNORECASE)
    if not title_match: # Try another pattern if the main one fails (e.g. for resume improvements prompt)
        title_match = re.search(r"targeting the role of ([^\n]+)", prompt_text, re.IGNORECASE)
    if title_match:
        job_title_for_mock = title_match.group(1).strip()
    
    skills_match = re.search(r"Key Recommended Skills(?:\s*\(.*?\))?:\s*(.*?)(?:\n|$)", prompt_text, re.IGNORECASE)
    if skills_match:
        skills_for_mock = skills_match.group(1).strip()

    if "risk score" in prompt_lower and "roadmap" in prompt_lower: 
        risk_score = random.randint(2, 6) 
        month1_theme, week1_goal, week2_goal, week3_goal, week4_project = "", "", "", "", ""
        month2_theme, week1_m2, week2_m2, week3_m2, week4_project_m2 = "", "", "", "", ""
        month3_theme, week1_m3, week2_m3, week3_m3, capstone_project_m3 = "", "", "", "", ""
        job_title_lower = job_title_for_mock.lower()

        if "finance manager" in job_title_lower:
            month1_theme = "Foundational Financial Acumen for Finance Managers"
            week1_goal = "Understand core financial statements (Balance Sheet, Income Statement, Cash Flow Statements)"
            week2_goal = "Basics of financial modeling and forecasting in Excel (e.g., 3-statement model basics)"
            week3_goal = "Key financial ratios and performance metrics analysis"
            week4_project = "Project: Analyze a public company's annual report and present key financial insights"
            month2_theme = "Advanced Financial Analysis & Reporting Tools"
            week1_m2 = "Budgeting processes and variance analysis techniques"
            week2_m2 = "Introduction to financial planning & analysis (FP&A) software (e.g., Anaplan, Hyperion - conceptual)"
            week3_m2 = "Cost accounting fundamentals and profitability analysis"
            week4_project_m2 = "Project: Develop a departmental budget proposal with justifications"
            month3_theme = "Strategic Finance, Business Partnering, and Leadership"
            week1_m3 = "Capital budgeting and investment appraisal techniques (NPV, IRR)"
            week2_m3 = "Communicating financial data effectively to non-financial stakeholders"
            week3_m3 = "Introduction to risk management and internal controls in finance"
            capstone_project_m3 = "Capstone: Create a financial strategy presentation for a new business initiative"
        elif "data scientist" in job_title_lower or "data analyst" in job_title_lower:
            month1_theme = "Python, Statistics, and Data Fundamentals"
            week1_goal = "Python programming for data analysis (Pandas, NumPy, Matplotlib)"
            week2_goal = "Descriptive and Inferential Statistics core concepts"
            week3_goal = "SQL for data extraction and manipulation"
            week4_project = "Project: Exploratory Data Analysis (EDA) on a real-world dataset"
            month2_theme = "Machine Learning Foundations"
            week1_m2 = "Supervised Learning algorithms (Regression, Classification)"
            week2_m2 = "Unsupervised Learning algorithms (Clustering, Dimensionality Reduction)"
            week3_m2 = "Model evaluation techniques and feature engineering basics"
            week4_project_m2 = "Project: Build and evaluate a predictive model for a given problem"
            month3_theme = "Advanced Topics & Deployment"
            week1_m3 = "Time Series Analysis or Natural Language Processing (NLP) basics (choose one)"
            week2_m3 = "Introduction to Big Data technologies (e.g., Spark concept)"
            week3_m3 = "Communicating data insights and storytelling with data"
            capstone_project_m3 = "Capstone: End-to-end data science project with a presentation of findings"
        elif "recruiting coordinator" in job_title_lower or "hr coordinator" in job_title_lower or \
             ("coordinator" in job_title_lower and ("recruiting" in job_title_lower or "hr" in job_title_lower or "talent" in job_title_lower)) or \
             "talent acquisition coordinator" in job_title_lower or "human resources coordinator" in job_title_lower:
            month1_theme = "Recruiting Fundamentals & Tools Mastery"
            week1_goal = "Understanding the end-to-end recruitment lifecycle and RC role within it. Basics of Applicant Tracking Systems (ATS)."
            week2_goal = "Mastering scheduling tools (Outlook Calendar, Google Calendar) for complex interview arrangements."
            week3_goal = "Professional communication for candidate correspondence (email templates, phone etiquette). Basics of MS Excel/Google Sheets for tracking."
            week4_project = "Project: Create a mock interview schedule for 3 candidates, 5 interviewers with varying availability, and draft all candidate communication."
            month2_theme = "Candidate Experience & Process Efficiency"
            week1_m2 = "Techniques for improving candidate experience at each touchpoint. Handling scheduling conflicts gracefully."
            week2_m2 = "Introduction to data integrity in ATS and importance of accurate record keeping. Generating basic recruitment reports."
            week3_m2 = "Understanding job descriptions and basic screening criteria. Coordinating post-interview debrief logistics."
            week4_project_m2 = "Project: Design a checklist for ensuring a positive candidate experience for a virtual interview process."
            month3_theme = "Advanced Coordination & HR Acumen"
            week1_m3 = "Handling confidential information and understanding basic HR compliance relevant to recruiting."
            week2_m3 = "Time management and prioritization for handling multiple requisitions. Basics of SharePoint or similar for document management."
            week3_m3 = "Problem-solving common recruiting challenges (e.g., last-minute cancellations, unresponsive candidates/managers)."
            capstone_project_m3 = "Capstone: Develop a proposal to improve scheduling efficiency or candidate experience by 10% for the RC team, with actionable steps."
        else: 
            skills_list = [s.strip() for s in skills_for_mock.split(',') if s.strip()]
            month1_theme = "Foundations in Core Professional Skills"
            week1_goal = f"Effective Communication: Written and verbal. (Focus: {skills_list[0] if skills_list else 'Communication'})"
            week2_goal = f"Organizational Skills & Time Management. (Focus: {skills_list[1] if len(skills_list) > 1 else 'Organization'})"
            week3_goal = f"Proficiency in MS Office Suite (Word, Excel, Outlook/PowerPoint) or Google Workspace. (Focus: {skills_list[3] if len(skills_list) > 3 else 'Office Tools'})"
            week4_project = "Project: Organize a mock event/meeting including scheduling, communication, and document preparation."
            month2_theme = "Problem Solving & Process Improvement"
            week1_m2 = "Analytical thinking and problem-solving techniques. (Focus: Problem Solving)"
            week2_m2 = "Introduction to process mapping and identifying areas for efficiency."
            week3_m2 = f"Teamwork and collaboration skills. (Focus: {skills_list[2] if len(skills_list) > 2 else 'Teamwork'})"
            week4_project_m2 = "Project: Identify a common administrative bottleneck and propose a simple solution."
            month3_theme = "Advanced Professional Development"
            week1_m3 = "Developing industry-specific knowledge relevant to the company/role."
            week2_m3 = "Customer service orientation and stakeholder management."
            week3_m3 = "Presentation skills and professional networking basics."
            capstone_project_m3 = "Capstone: Create a professional development plan for yourself targeting a specific career goal within the organization."

        return f"""AI Automation Risk Score: {risk_score}/10

3-Month Personalized Upskilling Roadmap for {job_title_for_mock}:

Month 1: {month1_theme}
- Week 1: {week1_goal}
- Week 2: {week2_goal}
- Week 3: {week3_goal}
- Week 4: {week4_project}

Month 2: {month2_theme}
- Week 1: {week1_m2}
- Week 2: {week2_m2}
- Week 3: {week3_m2}
- Week 4: {week4_project_m2}

Month 3: {month3_theme}
- Week 1: {week1_m3}
- Week 2: {week2_m3}
- Week 3: {week3_m3}
- Week 4: {capstone_project_m3}
"""
    elif "job title" in prompt_lower and "skills" in prompt_lower: 
        job_title_lower = job_title_for_mock.lower()
        if "finance manager" in job_title_lower:
             return "Job Title: Finance Manager\nSkills: Financial Analysis, Forecasting, Budgeting, Excel, Financial Modeling, Reporting, Communication"
        elif "recruiting coordinator" in job_title_lower or ("coordinator" in job_title_lower and "recruiting" in job_title_lower) :
             return "Job Title: Recruiting Coordinator\nSkills: Scheduling, Communication, ATS, Organization, MS Outlook, MS Excel, Candidate Experience, Time Management"
        return f"""Job Title: {job_title_for_mock}
Skills: {skills_for_mock}
"""
    elif "improvements" in prompt_lower:
        job_title_lower = job_title_for_mock.lower()
        improvement_suggestions = [
            f"1. Quantify your achievements with specific numbers and metrics relevant to a {job_title_for_mock} role.",
            f"2. Tailor your skills section to precisely match the requirements typically found in {job_title_for_mock} job descriptions. Highlight transferable skills.",
            f"3. Write a compelling summary or objective statement at the top of your resume, clearly stating your career goal as a {job_title_for_mock} and your key qualifications.",
            "4. Use action verbs to start your bullet points (e.g., Managed, Coordinated, Developed, Implemented).",
            f"5. Ensure your resume is ATS-friendly by using standard fonts, clear section headings, and relevant keywords for a {job_title_for_mock}."
        ]
        if "finance manager" in job_title_lower:
            improvement_suggestions.append("6. Highlight experience with financial modeling, forecasting, budgeting software, and any specific financial regulations you are familiar with.")
        elif "recruiting coordinator" in job_title_lower:
            improvement_suggestions.append("6. Emphasize your organizational skills, experience with scheduling tools (Outlook, Google Calendar), and any familiarity with Applicant Tracking Systems (ATS). Mention specific examples of managing complex schedules or improving candidate experience.")
        elif "data scientist" in job_title_lower or "data analyst" in job_title_lower:
            improvement_suggestions.append("6. Showcase your portfolio of data projects (e.g., on GitHub), and list your proficiency in specific programming languages (Python, R, SQL) and data visualization tools (Tableau, Power BI).")
        
        return "Resume Improvement Suggestions (Mock):\n" + "\n".join(improvement_suggestions)
    else: 
        return f"""Mock Answer for {job_title_for_mock}:
        To build a career as a {job_title_for_mock}, focus on these key areas:
        1. Master the core relevant skills: {skills_for_mock}.
        2. Build practical projects or gain experience relevant to the role.
        3. Network with industry professionals in the {job_title_for_mock} field.
        """

def generate_gemini_content(prompt):
    """Generate content using Gemini API"""
    if use_mock_data:
        return get_mock_llm_response(prompt)
    
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini content generation error: {str(e)}")
        return get_mock_llm_response(prompt)

def generate_openai_content_direct(prompt):
    """使用requests直接调用OpenAI API，绕过SDK，模拟Postman的请求方式"""
    if use_mock_data:
        return get_mock_llm_response(prompt)
    
    try:
        # 构建与Postman完全相同的请求
        url = f"{custom_base_url}/v1/chat/completions"
        
        if debug_mode:
            st.write(f"Direct API call to URL: {url}")
            st.write(f"API Key (first 5 chars): {OPENAI_API_KEY[:5]}...")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful career coach assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # 发送HTTP请求，与Postman行为一致
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # 检查HTTP状态码
        if response.status_code == 200:
            # 成功获取响应
            if debug_mode:
                st.success("Direct API call successful!")
                
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            # API返回错误
            error_msg = f"API Error ({response.status_code}): {response.text[:100]}..."
            if debug_mode:
                st.error(error_msg)
            else:
                st.error("API Error. Enable debug mode for details.")
            
            return get_mock_llm_response(prompt)
    except Exception as e:
        st.error(f"Direct request error: {str(e)}")
        return get_mock_llm_response(prompt)

def generate_openai_content(prompt):
    """Generate content using OpenAI API"""
    # 直接调用新的直接请求函数，绕过SDK
    return generate_openai_content_direct(prompt)
    
    # 下面是原始的SDK实现，保留但注释掉，以防需要切换回来
    """
    if use_mock_data:
        return get_mock_llm_response(prompt)
    
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful career coach assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 处理可能的不同响应格式
        if hasattr(response, 'choices'):
            return response.choices[0].message.content
        elif isinstance(response, dict) and 'choices' in response:
            return response['choices'][0]['message']['content']
        elif isinstance(response, str):
            try:
                import json
                data = json.loads(response)
                return data['choices'][0]['message']['content']
            except Exception as json_error:
                st.error(f"OpenAI API Error: Unable to parse response as JSON. The response was: '{response[:100]}...'. Error: {str(json_error)}")
                return get_mock_llm_response(prompt)
        else:
            st.error(f"OpenAI API Error: Unexpected response format: {type(response)}. Response: '{str(response)[:100]}...'")
            return get_mock_llm_response(prompt)
    except Exception as e:
        st.error(f"OpenAI API Error: General content generation error: {str(e)}")
        return get_mock_llm_response(prompt)
    """

def generate_content(prompt):
    """Generate content based on selected API"""
    if api_choice == "Google Gemini":
        return generate_gemini_content(prompt)
    else:
        return generate_openai_content(prompt)

def extract_job_info_with_llm(job_description, force_direct_parsing_flag):
    try:
        if debug_mode:
            st.write("--- Debug: Entering extract_job_info_with_llm ---")
            st.write(f"force_direct_parsing_flag: {force_direct_parsing_flag}") # Log the passed flag
            st.write("Job Description (first 300 chars):", job_description[:300] + "...")

        description_lower = job_description.lower()
        extracted_title = None
        extracted_skills_list = []
        
        # 检测是否为中文职位描述
        is_chinese_jd = is_chinese_text(job_description)
        
        if is_chinese_jd:
            # 中文JD模式下的关键词对应表
            chinese_role_patterns = [
                (("招聘专员", "招聘人员", "招聘助理", "人才招聘专员"), "Recruiting Coordinator", ["排期", "沟通", "招聘系统", "组织能力", "办公软件", "候选人体验", "时间管理"]),
                (("人力资源专员", "人事专员", "HR专员"), "Human Resources Coordinator", ["HR管理", "入职流程", "员工档案", "沟通能力", "招聘系统", "办公软件", "日程安排"]),
                (("财务经理", "财务主管", "财会经理"), "Finance Manager", ["财务分析", "预算", "财务建模", "Excel", "报表", "沟通能力", "商业敏感度"]),
                (("数据科学家",), "Data Scientist", ["Python", "R", "机器学习", "SQL", "统计分析", "数据可视化", "大数据", "预测建模"]),
                (("数据分析师",), "Data Analyst", ["SQL", "Excel", "Tableau", "Power BI", "数据清洗", "数据可视化", "统计", "沟通能力"]),
                (("软件工程师", "软件开发", "程序员"), "Software Engineer", ["Python", "Java", "JavaScript", "Git", "SQL", "API", "问题解决", "数据结构"])
            ]
            
            # 使用中文匹配模式
            for keywords, title, skills in chinese_role_patterns:
                if isinstance(keywords, tuple):
                    if any(kw in description_lower for kw in keywords):
                        extracted_title = title
                        extracted_skills_list = skills
                        break
                elif keywords in description_lower:
                    extracted_title = title
                    extracted_skills_list = skills
                    break
                    
            # 中文职位名称的正则表达式
            if not extracted_title:
                chinese_title_patterns = [
                    r'职位[:：]\s*([\w\s]+)',
                    r'岗位[:：]\s*([\w\s]+)',
                    r'招聘[:：]\s*([\w\s]+)',
                    r'招聘([\w\s]{2,8})(?:岗位|人员|专员)',
                ]
                for pattern in chinese_title_patterns:
                    match = re.search(pattern, description_lower)
                    if match:
                        potential_title = match.group(1).strip()
                        if len(potential_title) >= 2 and len(potential_title) <= 10:
                            extracted_title = potential_title
                            # 将中文职位名转换为英文对应
                            if "招聘" in potential_title or "人才" in potential_title:
                                extracted_title = "Recruiting Coordinator"
                            elif "人力资源" in potential_title or "人事" in potential_title:
                                extracted_title = "HR Coordinator"
                            elif "财务" in potential_title or "会计" in potential_title:
                                extracted_title = "Finance Manager"
                            elif "数据" in potential_title and ("科学" in potential_title or "挖掘" in potential_title):
                                extracted_title = "Data Scientist"
                            elif "数据" in potential_title and "分析" in potential_title:
                                extracted_title = "Data Analyst"
                            elif "软件" in potential_title or "开发" in potential_title or "程序" in potential_title:
                                extracted_title = "Software Engineer"
                            break

        # 如果不是中文JD或中文JD未提取出信息，使用原有英文处理逻辑
        if not is_chinese_jd or not extracted_title:
            role_patterns = [
                ("recruiting coordinator", "Recruiting Coordinator", ["Scheduling", "Communication", "ATS", "Organization", "MS Outlook", "MS Excel", "Candidate Experience", "Time Management"]),
                (("hr coordinator", "human resources coordinator"), "Human Resources Coordinator", ["HR Administration", "Onboarding", "Employee Records", "Communication", "ATS", "MS Office", "Scheduling", "Problem Solving"]),
                ("talent acquisition coordinator", "Talent Acquisition Coordinator", ["Sourcing Support", "Candidate Engagement", "ATS Management", "Scheduling", "Reporting", "Communication", "Organization"]),
                (("finance manager", "financial manager"), "Finance Manager", ["Financial Analysis", "Forecasting", "Budgeting", "Excel", "Financial Modeling", "Reporting", "Communication", "Business Acumen"]),
                (("data scientist",), "Data Scientist", ["Python", "R", "Machine Learning", "SQL", "Statistics", "Data Visualization", "Big Data", "Predictive Modeling"]),
                (("data analyst",), "Data Analyst", ["SQL", "Excel", "Tableau", "Power BI", "Data Cleaning", "Data Visualization", "Statistics", "Communication"]),
                (("software engineer", "software developer"), "Software Engineer", ["Python", "Java", "JavaScript", "Git", "SQL", "APIs", "Problem Solving", "Data Structures"])
            ]

            for keywords, title, skills in role_patterns:
                if isinstance(keywords, tuple):
                    if any(kw in description_lower for kw in keywords):
                        extracted_title = title
                        extracted_skills_list = skills
                        break
                elif keywords in description_lower:
                    extracted_title = title
                    extracted_skills_list = skills
                    break
            
            if debug_mode and extracted_title:
                st.success(f"Direct keyword match successful. Title: {extracted_title}, Skills: {extracted_skills_list}")

            if not extracted_title:
                generic_title_patterns = [
                    r'seeking a(?:n)? ([\w\s]+?)(?:\sto|,|\swith|\[|,|\n)',
                    r'hiring a(?:n)? ([\w\s]+?)(?:\sto|,|\swith|\[|,|\n)',
                    r'is (?:looking|searching) for a(?:n)? ([\w\s]+?)(?:\sto|,|\swith|\[|,|\n)',
                    r'job title:?\s*([\w\s(]+?)(?:\n|,|\[)',
                    r'role:?\s*([\w\s(]+?)(?:\n|,|\[)', 
                    r'position:?\s*([\w\s(]+?)(?:\n|,|\[)'
                ]
                for pattern in generic_title_patterns:
                    match = re.search(pattern, description_lower, re.IGNORECASE)
                    if match:
                        potential_title = match.group(1).strip()
                        potential_title = re.sub(r'\s*\(.*?\)$' ,'', potential_title).strip()
                        potential_title = re.sub(r'\s+(to|with|for|as a|who|responsible for)$' ,'', potential_title, flags=re.IGNORECASE).strip()
                        if 2 < len(potential_title.split()) < 6 and len(potential_title) < 50:
                            extracted_title = potential_title.title()
                            if debug_mode:
                                st.info(f"Regex pattern matched. Potential Title: {extracted_title}")
                            if not extracted_skills_list:
                                extracted_skills_list = ["Communication", "Problem Solving", "Teamwork", "Organization", "Adaptability"]
                            break
        
        if debug_mode and force_direct_parsing_flag:
             st.warning("Force direct parsing is ON. Skipping LLM call for job info extraction.")
        
        if (force_direct_parsing_flag and extracted_title) or \
           (extracted_title and extracted_skills_list and not force_direct_parsing_flag and api_choice == "OpenAI"): 
            if debug_mode:
                st.success(f"Using directly parsed/keyword-matched job info. Title: {extracted_title}, Skills: {extracted_skills_list}")
            return {
                "job_title": extracted_title,
                "resilient_skills": ", ".join(extracted_skills_list)
            }

        if not extracted_title or not extracted_skills_list or (api_choice == "Google Gemini" and not force_direct_parsing_flag): 
            if debug_mode and (not extracted_title or not extracted_skills_list):
                st.info("Direct parsing failed or incomplete. Attempting LLM extraction...")
            elif debug_mode:
                 st.info("Proceeding with Gemini LLM extraction (or LLM if not forcing direct parse)...")
                 
            # 修改提示词以支持中英文职位描述
            if is_chinese_jd:
                prompt_llm = f"""
                从以下职位描述中提取准确的职位名称和关键技能。
                请特别注意描述中提及的实际职位名称，而不仅仅是关键词。
                对于财务、管理或非技术类角色，请确保正确识别。
                
                请严格按照以下格式输出您的回答:
                Job Title: [提取的职位名称，例如 Finance Manager, Recruiting Coordinator, Software Engineer 等]
                Skills: [5-8个该职位所需的关键技能，以逗号分隔。若为"招聘专员/Recruiting Coordinator"，技能应包括排期能力、沟通能力、招聘系统使用、组织能力、办公软件、候选人体验、时间管理等]

                职位描述:
                {job_description}
                """
            else:
                prompt_llm = f"""
                Extract the EXACT job title and key skills from the job description below.
                Pay close attention to the actual job title mentioned in the description, not just keywords.
                For financial, management, or non-technical roles, be sure to identify them correctly.
                
                Format your response EXACTLY as follows:
                Job Title: [extracted job title, e.g. Finance Manager, Recruiting Coordinator, Software Engineer, etc.]
                Skills: [comma-separated list of 5-8 key skills required for this role. For 'Recruiting Coordinator', skills should include Scheduling, Communication, ATS, Organization, MS Outlook, MS Excel, Candidate Experience, Time Management.]

                Job Description:
                {job_description}
                """
            
            if debug_mode:
                st.write("LLM Prompt for job info extraction:", prompt_llm)
            
            response_llm = generate_content(prompt_llm)
            
            if debug_mode:
                st.write("Raw LLM response for job info:", response_llm)
            
            job_title_match_llm = re.search(r"Job Title:\s*(.*?)(?:\n|$)", response_llm, re.IGNORECASE)
            skills_match_llm = re.search(r"Skills:\s*(.*?)(?:\n|$)", response_llm, re.IGNORECASE)
            
            llm_job_title = job_title_match_llm.group(1).strip() if job_title_match_llm else None
            llm_skills_text = skills_match_llm.group(1).strip() if skills_match_llm else None
            
            if llm_job_title and llm_skills_text:
                extracted_title = llm_job_title
                extracted_skills_list = [s.strip() for s in llm_skills_text.split(",") if s.strip()]
                if debug_mode:
                    st.success(f"LLM extraction successful. Title: {extracted_title}, Skills: {extracted_skills_list}")
            elif debug_mode:
                st.warning("LLM extraction failed to parse title/skills. Will use fallback.")

        if not extracted_title:
            extracted_title = "General Role"
            if debug_mode:
                st.warning("All extraction methods failed for title. Using 'General Role'.")
                
        if not extracted_skills_list:
            description_lower = job_description.lower()
            # 检测是否为中文职位描述
            if is_chinese_jd:
                if "财务" in description_lower or "会计" in description_lower:
                    extracted_skills_list = ["财务分析", "预测", "预算", "数据分析", "Excel"]
                elif "招聘" in description_lower or "人力资源" in description_lower or "人事" in description_lower:
                    extracted_skills_list = ["排期", "沟通", "招聘系统", "组织能力", "办公软件", "候选人体验"]
                elif "经理" in description_lower or "管理" in description_lower:
                    extracted_skills_list = ["领导力", "战略规划", "团队管理", "沟通能力", "项目管理"]
                else:
                    extracted_skills_list = ["沟通能力", "问题解决", "分析能力", "细节关注", "团队合作"]
            else:
                if "finance" in description_lower or "financial" in description_lower:
                    extracted_skills_list = ["Financial Analysis", "Forecasting", "Budgeting", "Data Analysis", "Excel"]
                elif "recruiting" in description_lower or "hr" in description_lower or "human resources" in description_lower:
                     extracted_skills_list = ["Scheduling", "Communication", "ATS", "Organization", "MS Office", "Candidate Support"]
                elif "manager" in description_lower or "management" in description_lower:
                    extracted_skills_list = ["Leadership", "Strategic Planning", "Team Management", "Communication", "Project Management"]
                else:
                    extracted_skills_list = ["Communication", "Problem Solving", "Analytical Skills", "Attention to Detail", "Teamwork"]
                    
            if debug_mode:
                st.warning(f"Using fallback skills for '{extracted_title}': {extracted_skills_list}")

        final_result = {
            "job_title": extracted_title,
            "resilient_skills": ", ".join(extracted_skills_list)
        }
        
        if debug_mode:
            st.write("--- Debug: Exiting extract_job_info_with_llm ---")
            st.json(final_result)
            
        return final_result

    except Exception as e:
        st.error(f"Critical error in extract_job_info_with_llm: {str(e)}")
        return {
            "job_title": "Error Processing Job",
            "resilient_skills": "Error, Issue, Problem"
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
resume_embedding = get_text_embedding(resume_summary)

# --- MODE 1: SYSTEM RECOMMENDATIONS ---
if mode == "Mode 1: System Recommendations":
    st.markdown("### 🔍 System Analyzing Your Resume Against Predefined Roles")
    
    if "job_embedding" not in df_jobs.columns:
        df_jobs["job_embedding"] = df_jobs["job_title"].apply(get_text_embedding)

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
                    # Ensure force_direct_parsing is correctly passed
                    job_info = extract_job_info_with_llm(job_desc, force_direct_parsing)
                    job_info["job_embedding"] = get_text_embedding(job_desc) 
                    job_info["similarity"] = cosine_similarity(resume_embedding, job_info["job_embedding"])
                    custom_jobs.append(job_info)
                    progress_bar.progress((i + 1) / len(job_list))
                except Exception as e:
                    # Display the actual error to help debug further if needed
                    st.error(f"Error processing job {i+1} ('{job_desc[:50]}...'): {str(e)}")
                    # Add a placeholder or skip this job to avoid crashing the whole loop
                    custom_jobs.append({
                        "job_title": f"Error in Job {i+1}", 
                        "resilient_skills": "Error", 
                        "job_embedding": get_mock_embedding(), # Use mock to prevent further errors
                        "similarity": 0
                    })
        
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
You are an expert career coach and curriculum designer specializing in creating highly tailored professional development plans.
Your task is to create a specific and actionable 3-month upskilling roadmap for a candidate aspiring to the "Matched Job Role" by intensely focusing on the "Key Recommended Skills" provided. Additionally, provide an AI automation risk score (1-10) for this role.

{'' if language != 'Chinese' else '请使用中文回答。' }
**Crucial Instructions for the Roadmap:**
1.  **Role Specificity:** The roadmap MUST be *directly and exclusively* relevant to the "{job_title}". Avoid generic advice.
2.  **Skill Integration:** Each weekly milestone must clearly contribute to learning or applying one or more of the "{resilient_skills}". Mention which skill(s) each week's activity targets.
3.  **Non-Technical Roles:** If "{job_title}" is a non-technical role (e.g., Finance Manager, Marketing Manager, HR Specialist), the roadmap must reflect non-technical skill development, acquisition of industry-specific knowledge, relevant software/tools (e.g., Excel for Finance, Salesforce for Sales), and soft skills pertinent to that role. *Absolutely do NOT provide a software development or generic IT roadmap for non-technical roles.*
4.  **Technical Roles:** If "{job_title}" is a technical role (e.g., Software Engineer, Data Scientist), focus on relevant programming languages, frameworks, tools, and project-based learning that are standard for that role.
5.  **Actionable Milestones:** Weekly goals should be specific, measurable, achievable, relevant, and time-bound (SMART) where possible. Instead of "Learn Python", suggest "Complete a Python basics course focusing on data structures and write 3 simple scripts".
6.  **Consider Resume (Implicitly):** While the resume summary is provided for context on the candidate's background, the roadmap's primary goal is to build proficiency for the "{job_title}" using the "{resilient_skills}".

**Output Format (Strict Adherence Required):**
{'' if language != 'Chinese' else '请用中文书写以下内容：' }

AI Automation Risk Score: [Score]/10

3-Month Personalized Upskilling Roadmap for {job_title}:

Month 1: [Theme for Month 1 - e.g., "Foundational Financial Acumen & Tools" for a Finance Manager or "Core Python & Data Manipulation" for Data Analyst]
- Week 1: [Specific, actionable goal related to skills for {job_title}]. (Targets: [Skill1, Skill2])
- Week 2: [Specific, actionable goal related to skills for {job_title}]. (Targets: [Skill2, Skill3])
- Week 3: [Specific, actionable goal related to skills for {job_title}]. (Targets: [Skill1, Skill4])
- Week 4: [Mini-project or practical application relevant to {job_title}]. (Integrates: [Skill1, Skill2, Skill3])

Month 2: [Theme for Month 2 - e.g., "Advanced Analysis & Reporting" for Finance or "Statistical Modeling & Machine Learning Basics" for Data Analyst]
- Week 1: [Goal]. (Targets: [SkillX])
- Week 2: [Goal]. (Targets: [SkillY])
- Week 3: [Goal]. (Targets: [SkillZ])
- Week 4: [Project]. (Integrates: [SkillX, SkillY, SkillZ])

Month 3: [Theme for Month 3 - e.g., "Strategic Application & Business Partnering" for Finance or "Specialized Techniques & Deployment" for Data Analyst]
- Week 1: [Goal]. (Targets: [SkillA])
- Week 2: [Goal]. (Targets: [SkillB])
- Week 3: [Goal]. (Targets: [SkillC])
- Week 4: [Capstone project or advanced application for {job_title}, demonstrating overall competency]. (Showcases: All key skills)

--- Context for AI --- 
Resume Summary (for background understanding of candidate's potential starting point):
{resume_summary}

Matched Job Role (The role the roadmap is for):
{job_title}

Key Recommended Skills (The skills the roadmap must focus on):
{resilient_skills}
"""

st.info(f"Generating career analysis with {api_choice}...")

try:
    roadmap_output = generate_content(prompt)
    st.success("Career analysis complete!")
except Exception as e:
    st.error(f"Error generating roadmap: {str(e)}")
    roadmap_output = "Error generating roadmap. Please try again with a different model or settings."

risk_match = re.search(r"risk score.*?(\d{1,2})", roadmap_output, re.IGNORECASE)
risk_score = risk_match.group(1) if risk_match else "N/A"

if debug_mode:
    st.write("--- Debug: Data for Radar Chart --- ")
    st.write(f"Job Title for Analysis: {job_title}")
    st.write(f"Resilient Skills String (to become target_skills): {resilient_skills}")

known_skills = [
    "Python", "Java", "C++", "JavaScript", "SQL", "Git", "Linux", "Networking", "Bash",
    "Tableau", "Excel", "Power BI", "Burp Suite", "Nmap", "Wireshark", "Prompt Engineering",
    "APIs", "Cloud", "Docker", "Kubernetes", "Ethical Hacking", 
    "Communication", "Management", "Financial Analysis", "Reporting", "Data Analysis", 
    "Project Management", "Problem Solving", "Organization", "Scheduling", "ATS", 
    "Candidate Experience", "Time Management", "HR Administration", "Onboarding", "Employee Records",
    "Sourcing Support", "Candidate Engagement", "ATS Management", 
    "Forecasting", "Budgeting", "Financial Modeling", "Business Acumen", 
    "R", "Machine Learning", "Statistics", "Data Visualization", "Big Data", "Predictive Modeling",
    "Data Cleaning", "Teamwork", "Adaptability"
]
user_skills = [skill for skill in known_skills if skill.lower() in resume_text.lower()]
target_skills = [s.strip() for s in resilient_skills.split(",") if s.strip()]

if debug_mode:
    st.write(f"User Skills (from resume & known_skills): {user_skills}")
    st.write(f"Target Skills (parsed from resilient_skills): {target_skills}")
    st.write("--- End Debug --- ")

try:
    plot_skill_gap_chart(user_skills, target_skills)
except Exception as e:
    st.error(f"Error creating skill gap chart: {str(e)}")

st.subheader("🌟 Job Analysis Results")
st.markdown(f"**Target Role:** {job_title}")
st.markdown(f"**Recommended Skills:** {resilient_skills}")
st.markdown(f"**AI Automation Risk Score:** {risk_score}/10")

try:
    st.markdown("### 📈 Skill Gap Radar")
    st.image("skill_radar_chart.png")
except Exception as e:
    st.error(f"Error displaying skill chart: {str(e)}")

st.markdown("### 🧠 3-Month AI Roadmap")
st.text_area("AI Output:", value=roadmap_output, height=300)

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
        # 根据语言选择确定提示词
        if language == "Chinese":
            improve_prompt = f"""
你是一位专业的简历顾问。为以下简历内容提供改进建议，目标职位是 {job_title}:

{resume_summary}

请提供5-6条具体的改进建议，包括如何更好地展示技能、成就量化、格式优化等方面。
"""
        else:
            improve_prompt = f"""
You are a professional resume advisor. Suggest improvements for the following resume content targeting the role of {job_title}:

{resume_summary}
"""
        improvements = generate_content(improve_prompt)
        st.markdown("### ✨ Suggested Improvements")
        st.markdown(improvements)
    except Exception as e:
        st.error(f"Error generating resume improvements: {str(e)}")

st.markdown("### 🤖 Career Chatbot (Tutor Mode)")

# 定义一个默认的问题列表，在所有情况下至少显示这些
if language == "Chinese":
    default_questions = [
        "如何准备职业转型？",
        "我应该优先发展哪些技能？",
        "如何更有效地展示我现有的技能？",
        "我应该了解哪些行业趋势？",
        "如何在职业发展中利用AI工具？"
    ]
else:
    default_questions = [
        "How can I prepare for a career transition?",
        "What skills should I prioritize developing next?",
        "How can I showcase my existing skills more effectively?",
        "What industry trends should I be aware of?",
        "How can I leverage AI tools in my career development?"
    ]

# 尝试生成针对性问题，但如果条件不满足则使用默认问题
try:
    # 检查是否有需要的上下文信息来生成个性化问题
    has_context = 'job_title' in locals() and 'resilient_skills' in locals() and 'resume_summary' in locals()
    
    if has_context and len(job_title) > 0:
        # 根据简历和JD生成预设问题
        if language == "Chinese":
            questions_prompt = f"""
            你是一位职业教练。根据这个人的简历和他们的目标职位，
            生成4个他们可能想问的关于职业转型的具体问题。
            问题应该针对他们需要为{job_title}发展的技能，以及他们
            当前技能与所需技能{resilient_skills}之间的差距。
            
            简历: {resume_summary}
            目标职位: {job_title}
            所需技能: {resilient_skills}
            
            格式: 仅返回一个包含4个字符串的Python列表，如下所示:
            ["问题1？", "问题2？", "问题3？", "问题4？"]
            """
        else:
            questions_prompt = f"""
            You are a career coach. Based on this person's resume and their target job role, 
            generate 4 specific questions they might want to ask about their career transition.
            Make questions specific to skills they need to develop for {job_title} and any gaps 
            between their current skills and {resilient_skills}.
            
            Resume: {resume_summary}
            Target Job: {job_title}
            Required Skills: {resilient_skills}
            
            Format: Return ONLY a Python list of 4 strings like this:
            ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]
            """
        questions_response = generate_content(questions_prompt)
        
        # 解析返回的问题列表字符串为实际列表
        import ast
        try:
            generated_qs = ast.literal_eval(questions_response)
            # 确保格式正确
            if isinstance(generated_qs, list) and len(generated_qs) > 0:
                example_qs = generated_qs
            else:
                example_qs = default_questions
        except:
            # 解析失败时使用默认问题
            example_qs = default_questions
    else:
        # 没有上下文时使用默认问题
        example_qs = default_questions
except Exception as e:
    # 出错时使用默认问题
    example_qs = default_questions

# 初始化session_state以存储所选问题
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""

def update_question():
    """当选择问题时更新session_state"""
    if st.session_state.inspiration_dropdown != "-- Select --":
        st.session_state.selected_question = st.session_state.inspiration_dropdown

# 使用callback来处理选择变化
selected_q = st.selectbox(
    "Need inspiration?", 
    ["-- Select --"] + example_qs,
    key="inspiration_dropdown",
    on_change=update_question
)
# 使用session_state中存储的问题作为输入框的默认值
user_query = st.text_input("Ask a career question:", value=st.session_state.selected_question)

if user_query:
    try:
        tutor_prompt = f"You are a career tutor. Respond in {language}. Answer this question in under 150 words: '{user_query}'"
        tutor_response = generate_content(tutor_prompt)
        st.markdown(f"<div style='background-color:#1e1e1e;padding:10px;border-radius:10px'><b>💡 Career Bot:</b> {tutor_response}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error getting career advice: {str(e)}")