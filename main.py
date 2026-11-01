# ============================================================
# main.py  —  FastAPI Backend
# Run: uvicorn main:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- App Setup ----
app = FastAPI(title="Resume Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, replace * with your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load Models ----
# All 3 pkl files must be inside a "models/" folder next to main.py

with open(os.path.join(BASE_DIR, "models/tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(BASE_DIR, "models/best_resume_model.pkl"), "rb") as f:
    clf = pickle.load(f)

with open(os.path.join(BASE_DIR, "models/label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

# ---- Serve Frontend ----
# index.html must be inside a "templates/" folder next to main.py
app.mount("/static", StaticFiles(directory="templates"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("templates/index.html")

# ---- Request Schema ----
class ResumeRequest(BaseModel):
    resume: str

# ---- Clean Text ----
def cleanResume(txt: str) -> str:
    # Must match the cleaning used while training the model
    txt = txt.lower()
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    words = [w for w in txt.split() if w not in stop_words]
    return " ".join(words)

# ---- Routes ----

@app.post("/predict")
def predict(req: ResumeRequest):
    if len(req.resume.strip()) < 20:
        raise HTTPException(status_code=400, detail="Resume text is too short (min 20 characters)")

    cleaned = cleanResume(req.resume)
    vectorized = tfidf.transform([cleaned]).toarray()
    prediction = clf.predict(vectorized)
    category = le.inverse_transform(prediction)[0]

    confidence = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vectorized)
        confidence = round(float(proba.max()) * 100, 2)

    return {
        "category": category,
        "confidence": confidence,
        "status": "success"
    }


@app.get("/categories")
def get_categories():
    return {"categories": le.classes_.tolist()}


@app.get("/health")
def health():
    return {"status": "API is running ✅"}