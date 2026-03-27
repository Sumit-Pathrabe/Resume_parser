import streamlit as st
import spacy
import pdfplumber
import docx
import pandas as pd
import re
import json
from io import BytesIO

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Automated Resume Parser",
    page_icon="📄",
    layout="wide"
)

# ─────────────────────────────────────────────
# Load spaCy Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        st.stop()
    return nlp

nlp = load_nlp_model()

# ─────────────────────────────────────────────
# Predefined Skill Dictionary (Expanded)
# ─────────────────────────────────────────────
SKILLS_DB = [
    # Programming Languages
    "python", "java", "javascript", "c++", "c#", "c", "r", "swift", "kotlin",
    "typescript", "php", "ruby", "go", "rust", "scala", "perl", "matlab",
    # Web
    "html", "css", "react", "angular", "vue", "node.js", "django", "flask",
    "fastapi", "spring", "bootstrap", "jquery", "next.js", "express.js",
    # Data / ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "data visualization",
    "tensorflow", "pytorch", "keras", "scikit-learn", "opencv", "hugging face",
    "bert", "transformers", "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "xgboost", "genai", "llm", "ensemble learning",
    # Databases & Data Engineering
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis",
    "firebase", "cassandra", "elasticsearch", "apache kafka", "kafka", 
    "pyspark", "spark", "delta lake", "databricks", "etl", "hadoop",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "gitlab",
    "jenkins", "ci/cd", "terraform", "linux", "ubuntu", "bash",
    # Tools & Others
    "excel", "power bi", "tableau", "jira", "figma", "postman", "rest api",
    "graphql", "agile", "scrum", "airflow"
]

# ─────────────────────────────────────────────
# Text Extraction Functions
# ─────────────────────────────────────────────
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # 1. Grab the visible text
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            
            # 2. Grab the hidden clickable URLs (Fixes the missing hyphen issue)
            if page.hyperlinks:
                for link in page.hyperlinks:
                    if 'uri' in link and link['uri']:
                        text += link['uri'] + " \n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def extract_text(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        return extract_text_from_docx(uploaded_file)
    elif file_type == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload PDF, DOCX, or TXT.")
        return ""

# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

# ─────────────────────────────────────────────
# Information Extraction Functions
# ─────────────────────────────────────────────
def extract_name(text, doc):
    words = text.split()
    if len(words) >= 2:
        first_line_attempt = " ".join(words[:2])
        if re.match(r"^[A-Za-z\s\-\.]+$", first_line_attempt):
            return first_line_attempt.title()
            
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if len(name.split()) >= 2 and name.lower() not in ["apache kafka", "machine learning"]:
                return name.title()
    return "Not Found"

def extract_email(text):
    pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    matches = re.findall(pattern, text)
    return matches[0] if matches else "Not Found"

def extract_phone(text):
    pattern = r'\+?\d[\d\s\-()]{7,15}\d'
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].strip()
    return "Not Found"

def extract_skills(text):
    text_lower = text.lower()
    found_skills = []
    for skill in SKILLS_DB:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill.title())
    return list(set(found_skills))

def extract_education(raw_text):
    education_keywords = [
        "bachelor", "master", "phd", "doctorate", "b.sc", "m.sc", "b.tech",
        "m.tech", "mba", "b.e", "m.e", "b.com", "m.com", "diploma",
        "associate", "degree", "university", "college", "institute",
        "school of", "faculty of"
    ]
    lines = raw_text.split('\n')
    education_lines = []
    
    for line in lines:
        clean_line = line.strip()
        if any(kw in clean_line.lower() for kw in education_keywords):
            if len(clean_line) > 5:
                education_lines.append(clean_line)

    return education_lines[:10] if education_lines else ["Not Found"]

def extract_experience(raw_text):
    experience_keywords = [
        "experience", "work history", "employment", "internship",
        "worked at", "working at", "position", "role", "responsibilities",
        "projects", "technical projects"
    ]
    lines = raw_text.split('\n')
    experience_lines = []
    capture = False

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        if any(kw in clean_line.lower() for kw in experience_keywords):
            capture = True
        
        if capture:
            if len(clean_line) > 10: 
                experience_lines.append(clean_line)
            if len(experience_lines) > 25: 
                break

    return experience_lines if experience_lines else ["Not Found"]

def extract_linkedin(raw_text):
    pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[^\s\|]+'
    matches = re.findall(pattern, raw_text)
    if matches:
        # Prioritize the actual embedded URL (which usually starts with http)
        for m in matches:
            if m.startswith("http"):
                return m.rstrip(".,;")
        return matches[0].rstrip(".,;")
    return "Not Found"

def extract_github(raw_text):
    pattern = r'(?:https?://)?(?:www\.)?github\.com/[^\s\|]+'
    matches = re.findall(pattern, raw_text)
    if matches:
        # Prioritize the actual embedded URL
        for m in matches:
            if m.startswith("http"):
                return m.rstrip(".,;")
        return matches[0].rstrip(".,;")
    return "Not Found"

# ─────────────────────────────────────────────
# Master Parse Function
# ─────────────────────────────────────────────
def parse_resume(text):
    clean_text = preprocess_text(text)
    doc = nlp(clean_text)

    # Passing 'text' (RAW text) to contact and link extractors ensures
    # hyphens and special URL characters aren't stripped by the preprocessor!
    result = {
        "Name":       extract_name(clean_text, doc),
        "Email":      extract_email(text),
        "Phone":      extract_phone(text),
        "LinkedIn":   extract_linkedin(text),
        "GitHub":     extract_github(text),
        "Skills":     extract_skills(clean_text),
        "Education":  extract_education(text),  
        "Experience": extract_experience(text), 
    }
    return result

# ─────────────────────────────────────────────
# CSV Export Helper
# ─────────────────────────────────────────────
def convert_to_csv(data: dict) -> bytes:
    flat = {k: (", ".join(v) if isinstance(v, list) else v) for k, v in data.items()}
    df = pd.DataFrame([flat])
    return df.to_csv(index=False).encode("utf-8")

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
def main():
    st.title("📄 Automated Resume Parser")
    st.markdown("**NLP-powered resume information extractor** | Powered by spaCy & Python")
    st.markdown("---")

    with st.sidebar:
        st.header("ℹ️ About")
        st.info("This tool automatically extracts key information from resumes using NLP techniques.")
        st.header("📂 Supported Formats")
        st.success("✅ PDF\n\n✅ DOCX\n\n✅ TXT")

    uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")

        with st.spinner("🔍 Extracting and parsing resume..."):
            raw_text = extract_text(uploaded_file)

        if not raw_text.strip():
            st.error("Could not extract text from the file.")
            return

        parsed_data = parse_resume(raw_text)

        st.markdown("---")
        st.subheader("📊 Parsed Resume Information")

        metric_style = "<p style='font-size:14px; color:gray; margin-bottom:0px;'>{label}</p>"
        value_style = "<p style='font-size:1.8rem; font-weight:600;'>{value}</p>"
        link_style = "<a href='{href}' target='_blank' style='font-size:1.4rem; font-weight:600; text-decoration:none; color:#1f77b4;'>{value}</a>"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 Name", parsed_data["Name"])
            
        with col2:
            st.markdown(metric_style.format(label="📧 Email"), unsafe_allow_html=True)
            if parsed_data["Email"] != "Not Found":
                st.markdown(link_style.format(href=f"mailto:{parsed_data['Email']}", value=parsed_data["Email"]), unsafe_allow_html=True)
            else:
                st.markdown(value_style.format(value="Not Found"), unsafe_allow_html=True)
                
        with col3:
            st.metric("📞 Phone", parsed_data["Phone"])

        st.write("") 

        col4, col5 = st.columns(2)
        with col4:
            st.markdown(metric_style.format(label="🔗 LinkedIn"), unsafe_allow_html=True)
            if parsed_data["LinkedIn"] != "Not Found":
                url = parsed_data["LinkedIn"]
                href = url if url.startswith("http") else "https://" + url
                st.markdown(link_style.format(href=href, value=url), unsafe_allow_html=True)
            else:
                st.markdown(value_style.format(value="Not Found"), unsafe_allow_html=True)

        with col5:
            st.markdown(metric_style.format(label="🐙 GitHub"), unsafe_allow_html=True)
            if parsed_data["GitHub"] != "Not Found":
                url = parsed_data["GitHub"]
                href = url if url.startswith("http") else "https://" + url
                st.markdown(link_style.format(href=href, value=url), unsafe_allow_html=True)
            else:
                st.markdown(value_style.format(value="Not Found"), unsafe_allow_html=True)

        st.markdown("---")

        st.subheader("🛠️ Extracted Skills")
        if parsed_data["Skills"] and parsed_data["Skills"] != ["Not Found"]:
            skill_html = " ".join(
                [f'<span style="background-color:#1f77b4;color:white;padding:4px 10px; border-radius:12px;margin:4px;display:inline-block;font-size:14px;">{skill}</span>'
                 for skill in sorted(parsed_data["Skills"])]
            )
            st.markdown(skill_html, unsafe_allow_html=True)
            st.caption(f"Total skills found: **{len(parsed_data['Skills'])}**")
        else:
            st.warning("No skills detected.")

        st.markdown("---")

        col_edu, col_exp = st.columns(2)

        with col_edu:
            st.subheader("🎓 Education")
            for edu in parsed_data["Education"]:
                st.write(f"• {edu}")

        with col_exp:
            st.subheader("💼 Experience / Projects")
            for exp in parsed_data["Experience"]:
                st.write(f"• {exp}")

        st.markdown("---")

        with st.expander("📃 View Extracted Raw Text"):
            st.text_area("Raw Resume Text", raw_text, height=300)

        with st.expander("🧾 View Parsed Data as JSON"):
            st.json(parsed_data)

        st.markdown("---")
        st.subheader("⬇️ Download Parsed Data")

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(label="📥 Download as CSV", data=convert_to_csv(parsed_data), file_name="parsed_resume.csv", mime="text/csv")

        with dl_col2:
            st.download_button(label="📥 Download as JSON", data=json.dumps(parsed_data, indent=4), file_name="parsed_resume.json", mime="application/json")

    else:
        st.markdown("###")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image("https://img.icons8.com/clouds/200/resume.png", use_column_width=True)
            st.markdown("<h4 style='text-align:center;color:gray;'>Upload a resume above to get started!</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()