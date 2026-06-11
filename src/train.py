
import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import the clean_text function we wrote in preprocess.py
from preprocess import clean_text

def train_model(data_path, models_dir):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['title', 'text', 'label'])
    
    print("Preprocessing text (this may take a minute due to dataset size)...")

    df['combined_text'] = df['title'] + " " + df['text']
    
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Split features (X) and target labels (y)
    X = df['cleaned_text']
    y = df['label']
    
    print("Splitting dataset into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    print("Evaluating model performance...")
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "dataset_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }
    
    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}\n")
    
    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the vectorizer, model, and metrics
    print("Saving model pkl file")
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    with open(os.path.join(models_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
        
    with open(os.path.join(models_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Model training pipeline completed successfully!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, "data", "merged.csv")
    models_folder = os.path.join(base_dir, "models")
    
    if os.path.exists(data_file):
        train_model(data_file, models_folder)
    else:
        print(f"Error: Dataset not found")
