# Resume Screening System

A web app that predicts the job category of a resume from pasted resume text, using TF-IDF features and an SVM classifier.

## Features
- Paste resume text and get the predicted job category
- 25 job categories (Data Science, Python Developer, Java Developer, HR, etc.)
- Web interface built with HTML/CSS/JavaScript, served by a FastAPI backend

## Technologies
Python, FastAPI, scikit-learn, NLTK, TF-IDF, Linear SVM

## Dataset
Kaggle UpdatedResumeDataSet (962 resumes, 25 categories)

## How to Run
1. pip install -r requirements.txt
2. uvicorn main:app --reload --port 8000
3. Open http://127.0.0.1:8000 in your browser

## Future Work
- PDF/DOCX resume upload


