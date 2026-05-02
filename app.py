import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            return text
    except Exception as e:
        return ""

st.set_page_config(page_title="AI Resume Scanner", layout="wide")
st.title("📄 AI Resume Screening System")
st.write("Upload resumes to find the best match for your job description.")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
job_desc = st.text_area("Enter Job Description", height=200)

if st.button("Analyze Resumes"):
    if uploaded_files and job_desc:
        with st.spinner('Processing Resumes...'):
            resumes = []
            file_names = []
            
            for file in uploaded_files:
                text = extract_text(file)
                if text.strip():
                    resumes.append(text)
                    file_names.append(file.name)
            
            if not resumes:
                st.error("Could not extract text from the uploaded PDFs.")
            else:
                # Vectorization
                all_text = resumes + [job_desc]
                vectorizer = TfidfVectorizer()
                matrix = vectorizer.fit_transform(all_text)
                
                # Similarity calculation
                scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
                
                st.subheader("📊 Analysis Results:")
                results = sorted(zip(file_names, scores), key=lambda x: x[1], reverse=True)
                
                for name, score in results:
                    match_percentage = round(score * 100, 2)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{name}**")
                        st.progress(score)
                    with col2:
                        st.write(f"**{match_percentage}% Match**")
    else:
        st.warning("Please upload resumes and enter a job description first.")
