# COMPANY: CODTECH IT SOLUTIONS

# NAME: Addanki Raghavendra Chary

# INTERN ID: CT04DF169

# DOMAIN: Machine Learning

# DURATION: 4 WEEEKS

# MENTOR: NEELA SANTOSH

# SENTIMENT ANALYSIS WITH NLP

# PERFORM SENTIMENT ANALYSIS ON A DATASET OF CUSTOMER REVIEWS USING TF-IDF VECTORIZATION AND LOGISTIC REGRESSION

# Step 1: Import Libraries

import pandas as pd import numpy as np import re from sklearn.model_selection import train_test_split from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.linear_model import LogisticRegression from sklearn.metrics import accuracy_score, classification_report, confusion_matrix import matplotlib.pyplot as plt import seaborn as sns

# Step 2: Load Dataset

df = pd.read_csv('customer_reviews.csv') # Replace with your actual path df.head()

# Step 3: Data Cleaning

def clean_text(text): text = text.lower() text = re.sub(r'[^\w\s]', '', text) return text

df['cleaned_review'] = df['Review'].apply(clean_text)

# Step 4: TF-IDF Vectorization

tfidf = TfidfVectorizer(stop_words='english', max_features=5000) X = tfidf.fit_transform(df['cleaned_review']) y = df['Sentiment']

# Step 5: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Model

model = LogisticRegression() model.fit(X_train, y_train)

# Step 7: Evaluation

y_pred = model.predict(X_test) print("Accuracy:", accuracy_score(y_test, y_pred)) print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix

cm = confusion_matrix(y_test, y_pred) sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive']) plt.xlabel('Predicted') plt.ylabel('Actual') plt.title('Confusion Matrix') plt.show()

# Step 9: Predict Sentiment

def predict_sentiment(review): cleaned = clean_text(review) vector = tfidf.transform([cleaned]) prediction = model.predict(vector)[0] return "Positive" if prediction == 1 else "Negative"

# Example

predict_sentiment("The product was amazing and exceeded my expectations.")
