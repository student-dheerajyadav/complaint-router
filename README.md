# Complaint Router

Smart complaint classification for banking customer complaints (Hackathon PS-01).

## Problem
Support teams spend time manually sorting complaints into categories and departments. This app predicts the category of a complaint and the team it should be routed to.

## Solution
- Text is converted to features with TF-IDF (unigrams and bigrams).
- Two classifiers are trained and compared: Logistic Regression and Naive Bayes.
- The user enters a complaint and gets the predicted category, routing department, confidence, and the key terms behind the prediction.

## Components
- **Classify**: live prediction with department routing, confidence, class probabilities, key terms and an adjustable manual-review threshold.
- **Batch**: classify a whole CSV of complaints and download the results.
- **Model performance**: accuracy, macro F1, confusion matrix and per-category precision/recall on a held-out 20% test split.
- **Keywords**: most indicative terms for each category.
- **Dataset**: category distribution, filtering and search.

## Data
Banking consumer complaints (CFPB-derived, five product categories). Duplicate complaints and complaints carrying conflicting labels were removed, and classes were balanced to 1,462 complaints each (7,310 total) to avoid bias toward the majority class.

## Setup
    pip install -r requirements.txt
    streamlit run app.py

The app loads `clean_complaints.csv` automatically when present. Any other CSV can be uploaded from the sidebar by selecting its text and category columns.
