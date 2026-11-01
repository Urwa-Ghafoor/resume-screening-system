# ============================================================
# train_model.py  -  Retrain the resume classifier
# Run from the project folder:  python train_model.py
# Needs UpdatedResumeDataSet.csv in the same folder
# Creates: models/tfidf.pkl, models/best_resume_model.pkl, models/label_encoder.pkl
# ============================================================

import os
import re
import pickle
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else 'UpdatedResumeDataSet.csv'


# Same cleaning as main.py (must stay identical)
def cleanResume(txt):
    txt = txt.lower()
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    words = [w for w in txt.split() if w not in stop_words]
    return " ".join(words)


df = pd.read_csv(CSV_PATH)
df['clean'] = df['Resume'].apply(cleanResume)

# The Kaggle dataset has many duplicate resumes. Remove them, otherwise the same
# resume lands in both train and test and the accuracy looks higher than it is.
print("Rows:", len(df), "| Unique resumes:", df['clean'].nunique())
df = df.drop_duplicates(subset='clean')

le = LabelEncoder()
y = le.fit_transform(df['Category'])

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], y, test_size=0.2, stratify=y, random_state=42
)

# Honest evaluation on unseen resumes
tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=5000)
model = LinearSVC()
model.fit(tfidf.fit_transform(X_train), y_train)
pred = model.predict(tfidf.transform(X_test))
print("\nTest accuracy (unseen, de-duplicated resumes): %.1f%%" % (accuracy_score(y_test, pred) * 100))
print(classification_report(y_test, pred, labels=range(len(le.classes_)),
                            target_names=le.classes_, zero_division=0))

# Final model: train on all unique resumes
tfidf = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=5000)
model = LinearSVC()
model.fit(tfidf.fit_transform(df['clean']), y)

os.makedirs('models', exist_ok=True)
pickle.dump(tfidf, open('models/tfidf.pkl', 'wb'))
pickle.dump(model, open('models/best_resume_model.pkl', 'wb'))
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))
print("Saved models/tfidf.pkl, models/best_resume_model.pkl, models/label_encoder.pkl")
