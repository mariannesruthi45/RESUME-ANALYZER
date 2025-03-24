from transformers import pipeline
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import pickle
import pymupdf
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Initialize the Named Entity Recognition (NER) model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure uploads directory exists

# Predefined keywords for each job category
JOB_KEYWORDS = {
    "Advocate": ["law", "court", "legal", "litigation", "case", "judge", "attorney"],
    "Arts": ["painting", "sculpture", "drawing", "illustration", "graphic design", "visual arts"],
    "Automation Testing": ["selenium", "test automation", "pytest", "cypress", "Jenkins", "regression testing"],
    "Blockchain": ["blockchain", "cryptocurrency", "Ethereum", "smart contracts", "NFT", "decentralized"],
    "Business Analyst": ["business analysis", "requirements gathering", "data modeling", "stakeholder management"],
    "Civil Engineer": ["construction", "structural design", "AutoCAD", "building codes", "surveying", "reinforced concrete"],
    "Data Science": ["machine learning", "data analysis", "python", "statistics", "AI", "deep learning", "big data"],
    "Database": ["SQL", "NoSQL", "PostgreSQL", "MongoDB", "database management", "DBMS"],
    "DevOps Engineer": ["docker", "kubernetes", "CI/CD", "Jenkins", "AWS", "terraform", "ansible"],
    "DotNet Developer": [".NET", "C#", "ASP.NET", "MVC", "Entity Framework", "SQL Server"],
    "ETL Developer": ["ETL", "data pipeline", "Informatica", "Talend", "SSIS", "data warehousing"],
    "Electrical Engineering": ["circuit design", "power systems", "electrical wiring", "microcontrollers", "PLC"],
    "HR": ["recruitment", "payroll", "employee engagement", "performance management", "HR policies"],
    "Hadoop": ["big data", "Hadoop", "Spark", "HDFS", "MapReduce", "Hive", "Pig"],
    "Health and Fitness": ["nutrition", "personal training", "exercise science", "wellness", "dietitian"],
    "Java Developer": ["java", "spring", "hibernate", "microservices", "J2EE", "multithreading"],
    "Mechanical Engineer": ["CAD", "solidworks", "thermodynamics", "manufacturing", "design", "mechanical systems"],
    "Network Security Engineer": ["cybersecurity", "firewalls", "intrusion detection", "VPN", "penetration testing"],
    "Operations Manager": ["operations", "supply chain", "project management", "logistics", "process optimization"],
    "PMO": ["project management", "PMO", "stakeholder communication", "risk management", "resource planning"],
    "Python Developer": ["python", "django", "flask", "pandas", "numpy", "machine learning"],
    "SAP Developer": ["SAP", "ABAP", "SAP HANA", "SAP ERP", "Fiori", "SAP BW"],
    "Sales": ["sales", "CRM", "lead generation", "B2B", "negotiation", "customer acquisition"],
    "Testing": ["automation", "manual testing", "selenium", "test cases", "bug tracking", "QA"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "UI/UX", "responsive design", "frontend", "wireframing"]
}

def load_model_artifacts():
    """Loads the saved machine learning model artifacts."""
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("resume_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, label_encoder, model

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    doc = pymupdf.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.replace("\n", " ").strip()

def extract_name(resume_text):
    """Extracts a potential name from the resume."""
    name_pattern = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
    names = re.findall(name_pattern, resume_text.split("\n")[0])
    return names[0] if names else 'N/A'

def extract_linkedin_email(resume_text):
    """Extracts LinkedIn profile URL and email from a resume."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    linkedin_pattern = r"(https://www\.linkedin\.com/in/[A-Za-z0-9_-]+)"

    emails = re.findall(email_pattern, resume_text)
    linkedin_urls = re.findall(linkedin_pattern, resume_text)

    return linkedin_urls[0] if linkedin_urls else 'N/A', emails[0] if emails else 'N/A'

def assess_resume_quality(resume_text):
    """Assesses resume quality based on word count."""
    word_count = len(resume_text.split())
    return "Excellent" if word_count > 100 else "Good" if word_count > 50 else "Bad"

def match_keywords(resume_text, job_category):
    """Calculates keyword match percentage based on predefined keywords."""
    if not resume_text or not isinstance(resume_text, str):
        return 0  
    job_keywords = JOB_KEYWORDS.get(job_category, [])
    if not job_keywords:
        return 0  

    resume_words = set(resume_text.lower().split())
    job_words = set(word.lower() for word in job_keywords)

    common_words = resume_words.intersection(job_words)
    match_score = len(common_words) / len(job_words) * 100 if job_words else 0

    return round(match_score, 2)

def predict_resume_category(resume_text, vectorizer, model, label_encoder):
    """Predicts the job category of a resume using the trained model."""
    tfidf_vector = vectorizer.transform([resume_text])
    prediction = model.predict(tfidf_vector)
    return label_encoder.inverse_transform(prediction)[0]

@app.route('/')
def home():
    job_categories = list(JOB_KEYWORDS.keys())
    return render_template('index.html', job_categories=job_categories)

@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handles resume uploads, analysis, and displays results."""
    if 'resumes' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('resumes')
    job_category = request.form['job_category']
    resumes_data = []

    vectorizer, label_encoder, model = load_model_artifacts()

    for file in files:
        if file and file.filename.endswith(".pdf"):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            resume_text = extract_text_from_pdf(file_path)
            predicted_category = predict_resume_category(resume_text, vectorizer, model, label_encoder)
            
            if predicted_category.lower() == job_category.lower():
                name = extract_name(resume_text)
                linkedin, email = extract_linkedin_email(resume_text)
                resume_quality = assess_resume_quality(resume_text)
                
                # Calculate match percentage using predefined keywords
                match_percentage = match_keywords(resume_text, job_category)  
                
                # Append extracted data
                resumes_data.append([file.filename, predicted_category, resume_quality, name, linkedin, email, match_percentage])

    return render_template('results.html', resumes=resumes_data, job_category=job_category)

if __name__ == "__main__":
    app.run(debug=True)