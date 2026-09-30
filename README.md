# Complaint Router (PS-01: Smart Complaint Classification)

Complaint text ko category/department mein classify karne wala Streamlit app.

## Features
- TF-IDF + Logistic Regression vs Naive Bayes comparison (accuracy, F1, confusion matrix)
- Live prediction with confidence bars and "prediction ke peeche ke words"
- Har category ke top words
- Data explorer, aur apna CSV upload karne ka option

## Setup
    pip install -r requirements.txt
    streamlit run app.py

Bina upload kiye sample (synthetic) data use hota hai. Real dataset ke liye sidebar mein CSV upload karo.
