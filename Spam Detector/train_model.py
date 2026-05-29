import os
import pandas as pd
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

def train_and_save_model():
    print("Fetching SMS Spam Collection Dataset...")
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms.zip"
    
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("sms_data")
    
    df = pd.read_csv("sms_data/SMSSpamCollection", sep='\t', names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split to extract real holdout accuracy
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
    
    print("Training Machine Learning Pipeline...")
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
        ('nb', MultinomialNB())
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # Calculate operational accuracy score
    accuracy = model_pipeline.score(X_test, y_test) * 100
    print(f"Engine Accuracy Evaluated: {accuracy:.2f}%")
    
    # Package model metrics together
    payload = {
        'pipeline': model_pipeline,
        'accuracy': round(accuracy, 2)
    }
    
    joblib.dump(payload, 'spam_classifier_system.pkl')
    print("Model bundle successfully saved as 'spam_classifier_system.pkl'!")
    
    os.remove(zip_path)

if __name__ == "__main__":
    train_and_save_model()