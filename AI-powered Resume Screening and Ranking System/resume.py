import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    """Basic text preprocessing: lowercase, remove punctuation, etc."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_text_from_resume(resume_path):
    """Extracts text from a resume file (supports .txt, .pdf, .docx)."""
    try:
        if resume_path.endswith('.txt'):
            with open(resume_path, 'r', encoding='utf-8') as file:
                text = file.read()
        elif resume_path.endswith('.pdf'):
            import PyPDF2
            with open(resume_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() or "" for page in reader.pages)

        elif resume_path.endswith('.docx'):
            import docx
            doc = docx.Document(resume_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise ValueError("Unsupported file format.")
        return preprocess_text(text)

    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"Error processing file: {e}"

def create_job_description_vector(job_description):
    """Creates a TF-IDF vector for the job description."""
    vectorizer = TfidfVectorizer()
    job_vector = vectorizer.fit_transform([preprocess_text(job_description)])
    return job_vector, vectorizer

def calculate_resume_similarity(resume_text, job_vector, vectorizer):
    """Calculates the cosine similarity between a resume and the job description."""
    resume_vector = vectorizer.transform([resume_text])
    similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]
    return similarity_score

def rank_resumes(job_description, resume_paths):
    """Ranks resumes based on their similarity to the job description."""
    job_vector, vectorizer = create_job_description_vector(job_description)
    resume_scores = []

    for resume_path in resume_paths:
        resume_text = extract_text_from_resume(resume_path)
        if "Error processing file" in resume_text or "File not found" in resume_text:
            print(f"Skipping {resume_path}: {resume_text}")
            continue

        similarity_score = calculate_resume_similarity(resume_text, job_vector, vectorizer)
        resume_scores.append((resume_path, similarity_score))

    ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)
    return ranked_resumes

# Example Usage:

job_description = """
Software Engineer with experience in Python, machine learning, and cloud technologies.
Strong problem-solving skills and experience with data analysis.
Experience with REST APIs and database management.
"""

resume_paths = [
    "resume1.pdf", # Place your resume files here
    "resume2.pdf",
]

ranked_resumes = rank_resumes(job_description, resume_paths)

for resume_path, score in ranked_resumes:
    print(f"Resume: {resume_path}, Similarity Score: {score:.4f}")