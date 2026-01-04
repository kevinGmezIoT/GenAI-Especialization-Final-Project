import pandas as pd
import joblib
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from src.data.preprocess import create_preprocessor

def train_model(data_path, model_path):
    print(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return

    # Target mapping
    # Ensure target column exists
    if 'target' not in df.columns:
        print("Error: 'target' column not found in data.")
        return

    # Handle target mapping if it's string
    if df['target'].dtype == 'object':
        df['target'] = df['target'].map({'good risk': 0, 'bad risk': 1})
    
    # Drop rows with missing target if any
    df = df.dropna(subset=['target'])
    
    # Features and Target
    # We drop 'target' and 'description' (if exists)
    drop_cols = ['target']
    if 'description' in df.columns:
        drop_cols.append('description')
        
    X = df.drop(columns=drop_cols)
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create pipeline
    preprocessor = create_preprocessor()
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    
    # Train
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Save training features as baseline for monitoring (Drift detection)
    baseline_path = os.path.join(os.path.dirname(data_path), "train_features.csv")
    X_train.to_csv(baseline_path, index=False)
    print(f"Baseline features saved to {baseline_path}")

    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Define paths
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Default to enriched data, fallback to raw if enriched doesn't exist (but raw needs enrichment first usually)
    # For this script, we assume we want to train on the enriched data or the provided processed data.
    # The notebook saved to data/processed/output_target.csv. Let's check if we should use that or our new enriched one.
    # We'll prefer our new enriched one, but fallback to the one from the notebook if it exists.
    
    enriched_path = os.path.join(base_path, "data", "processed", "credit_risk_enriched.csv")
    notebook_processed_path = os.path.join(base_path, "data", "processed", "output_target.csv")
    
    if os.path.exists(enriched_path):
        data_path = enriched_path
    elif os.path.exists(notebook_processed_path):
        data_path = notebook_processed_path
    else:
        # Fallback to raw for demonstration, but it will fail without target
        data_path = os.path.join(base_path, "data", "raw", "credit_risk_reto.csv")
        print("Warning: Processed data not found. Training might fail if target is missing.")

    model_output_path = os.path.join(base_path, "models", "credit_risk_model.joblib")
    
    train_model(data_path, model_output_path)
