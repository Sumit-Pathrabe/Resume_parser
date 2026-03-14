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
# Predefined Skill Dictionary
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
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis",
    "firebase", "cassandra", "elasticsearch",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "github", "gitlab",
    "jenkins", "ci/cd", "terraform", "linux", "bash",
    # Tools & Others
    "excel", "power bi", "tableau", "jira", "figma", "postman", "rest api",
    "graphql", "agile", "scrum", "hadoop", "spark", "airflow"
]

# ─────────────────────────────────────────────
# Text Extraction Functions
# ─────────────────────────────────────────────
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
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
    # Remove excessive whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    return text.strip()

# ─────────────────────────────────────────────
# Information Extraction Functions
# ─────────────────────────────────────────────

def extract_name(text, doc):
    """Extract candidate name using spaCy NER."""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Usually the first PERSON entity is the candidate
            name = ent.text.strip()
            if len(name.split()) >= 2:  # full name check
                return name
    # Fallback: first line heuristic
    first_line = text.strip().split("\n")[0].strip()
    if 2 <= len(first_line.split()) <= 5 and first_line.replace(" ", "").isalpha():
        return first_line
    return "Not Found"

def extract_email(text):
    """Extract email using Regex."""
    pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    matches = re.findall(pattern, text)
    return matches[0] if matches else "Not Found"

def extract_phone(text):
    """Extract phone number using Regex."""
    pattern = r'(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)(\d{3}[\s\-]?\d{4})'
    matches = re.findall(pattern, text)
    if matches:
        phone = "".join(["".join(m) for m in matches[:1]])
        return phone.strip()
    return "Not Found"

def extract_skills(text):
    """Extract skills using predefined skill dictionary."""
    text_lower = text.lower()
    found_skills = []
    for skill in SKILLS_DB:
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill.title())
    return list(set(found_skills))

def extract_education(text, doc):
    """Extract education details using NER + keyword matching."""
    education_keywords = [
        "bachelor", "master", "phd", "doctorate", "b.sc", "m.sc", "b.tech",
        "m.tech", "mba", "b.e", "m.e", "b.com", "m.com", "diploma",
        "associate", "degree", "university", "college", "institute",
        "school of", "faculty of"
    ]
    education_lines = []
    lines = text.split("\n")
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in education_keywords):
            clean_line = line.strip()
            if clean_line and len(clean_line) > 5:
                education_lines.append(clean_line)

    # Also capture ORG entities near education context
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    return education_lines[:5] if education_lines else ["Not Found"]

def extract_experience(text):
    """Extract work experience sections using keyword matching."""
    experience_keywords = [
        "experience", "work history", "employment", "internship",
        "worked at", "working at", "position", "role", "responsibilities"
    ]
    experience_lines = []
    lines = text.split("\n")
    capture = False

    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in experience_keywords):
            capture = True
        if capture:
            clean = line.strip()
            if clean:
                experience_lines.append(clean)
            if len(experience_lines) > 10:
                break

    return experience_lines if experience_lines else ["Not Found"]

def extract_linkedin(text):
    pattern = r'(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+'
    match = re.search(pattern, text)
    return match.group(0) if match else "Not Found"

def extract_github(text):
    pattern = r'(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_%]+'
    match = re.search(pattern, text)
    return match.group(0) if match else "Not Found"

# ─────────────────────────────────────────────
# Master Parse Function
# ─────────────────────────────────────────────
def parse_resume(text):
    clean_text = preprocess_text(text)
    doc = nlp(clean_text)

    result = {
        "Name":       extract_name(clean_text, doc),
        "Email":      extract_email(clean_text),
        "Phone":      extract_phone(clean_text),
        "LinkedIn":   extract_linkedin(clean_text),
        "GitHub":     extract_github(clean_text),
        "Skills":     extract_skills(clean_text),
        "Education":  extract_education(clean_text, doc),
        "Experience": extract_experience(clean_text),
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
    # Header
    st.title("📄 Automated Resume Parser")
    st.markdown("**NLP-powered resume information extractor** | Powered by spaCy & Python")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This tool automatically extracts key information "
            "from resumes using NLP techniques including:\n\n"
            "- Named Entity Recognition (NER)\n"
            "- Regular Expressions\n"
            "- Keyword-based Skill Matching"
        )
        st.header("📂 Supported Formats")
        st.success("✅ PDF\n\n✅ DOCX\n\n✅ TXT")
        st.header("🔧 Tech Stack")
        st.code("Python | spaCy | NLTK\npdfplumber | python-docx\nStreamlit | Pandas")

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload your Resume",
        type=["pdf", "docx", "txt"],
        help="Drag and drop or click to upload a resume file."
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")

        with st.spinner("🔍 Extracting and parsing resume..."):
            raw_text = extract_text(uploaded_file)

        if not raw_text.strip():
            st.error("Could not extract text from the file. Please try a different file.")
            return

        parsed_data = parse_resume(raw_text)

        st.markdown("---")
        st.subheader("📊 Parsed Resume Information")

        # ── Row 1: Contact Info ──────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 Name", parsed_data["Name"])
        with col2:
            st.metric("📧 Email", parsed_data["Email"])
        with col3:
            st.metric("📞 Phone", parsed_data["Phone"])

        # ── Row 2: Social Profiles ───────────────────
        col4, col5 = st.columns(2)
        with col4:
            st.metric("🔗 LinkedIn", parsed_data["LinkedIn"])
        with col5:
            st.metric("🐙 GitHub", parsed_data["GitHub"])

        st.markdown("---")

        # ── Skills ────────────────────────────────────
        st.subheader("🛠️ Extracted Skills")
        if parsed_data["Skills"] and parsed_data["Skills"] != ["Not Found"]:
            skill_html = " ".join(
                [f'<span style="background-color:#1f77b4;color:white;padding:4px 10px;'
                 f'border-radius:12px;margin:4px;display:inline-block;font-size:14px;">'
                 f'{skill}</span>'
                 for skill in sorted(parsed_data["Skills"])]
            )
            st.markdown(skill_html, unsafe_allow_html=True)
            st.caption(f"Total skills found: **{len(parsed_data['Skills'])}**")
        else:
            st.warning("No skills detected from the predefined skill list.")

        st.markdown("---")

        # ── Education & Experience ────────────────────
        col_edu, col_exp = st.columns(2)

        with col_edu:
            st.subheader("🎓 Education")
            for edu in parsed_data["Education"]:
                st.write(f"• {edu}")

        with col_exp:
            st.subheader("💼 Experience")
            for exp in parsed_data["Experience"]:
                st.write(f"• {exp}")

        st.markdown("---")

        # ── Raw Text Expander ─────────────────────────
        with st.expander("📃 View Extracted Raw Text"):
            st.text_area("Raw Resume Text", raw_text, height=300)

        # ── JSON View ─────────────────────────────────
        with st.expander("🧾 View Parsed Data as JSON"):
            st.json(parsed_data)

        # ── Downloads ─────────────────────────────────
        st.markdown("---")
        st.subheader("⬇️ Download Parsed Data")

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            csv_data = convert_to_csv(parsed_data)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name="parsed_resume.csv",
                mime="text/csv"
            )

        with dl_col2:
            json_data = json.dumps(parsed_data, indent=4)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name="parsed_resume.json",
                mime="application/json"
            )

    else:
        # Landing placeholder
        st.markdown("###")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(
                "https://img.icons8.com/clouds/200/resume.png",
                use_column_width=True
            )
            st.markdown(
                "<h4 style='text-align:center;color:gray;'>"
                "Upload a resume above to get started!</h4>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()