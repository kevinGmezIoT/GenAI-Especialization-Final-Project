
import argparse

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__== "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train-V-1.csv")
    parser.add_argument("--test-file", type=str, default="test-V-1.csv")
    
    args, _ = parser.parse_known_args()
    
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)
    
    print("[INFO] Reading data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_df.columns)
    label = features.pop(-1)
    
    print("Building training and testing datasets")
    print()
    
    X_train = train_df[features]
    X_test = test_df[features]
    
    y_train = train_df[label]
    y_test = test_df[label]
    
    print('Column order: ')
    print(features)
    print()
    
    print('Label column is:', label)
    print()
    
    print("Data Shape: ")
    print()
    print("----SHAPE OF TRAINING DATA (80%)")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("----SHAPE OF TESTING DATA (20%)")
    print(X_test.shape)
    print(y_test.shape)
    print()
    
    print("Training SVM Model....")
    print()
    model = SVC(kernel=args.kernel, C=args.C, gamma=args.gamma, verbose=True)
    model.fit(X_train, y_train)
    print()
    
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at "+ model_path)
    print()
    
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)
    
    print()
    print("--- METRICS RESULTS FOR TESTING DATA ---")
    print()
    print("Total Rows are:", X_test.shape[0])
    print("[TESTING] Model accuracy is: ", test_acc)
    print("[TESTING] Testing Report: ")
    print(test_rep)
    
