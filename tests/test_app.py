import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.preprocess import create_preprocessor
from models.predict import make_prediction

def test_create_preprocessor():
    preprocessor = create_preprocessor()
    assert preprocessor is not None
    # Check if it handles expected columns
    data = pd.DataFrame({
        'Age': [30],
        'Job': [2],
        'Credit amount': [1000],
        'Duration': [12],
        'Sex': ['male'],
        'Housing': ['own'],
        'Saving accounts': ['little'],
        'Checking account': ['moderate'],
        'Purpose': ['car']
    })
    processed = preprocessor.fit_transform(data)
    assert processed.shape[0] == 1
    assert processed.shape[1] > 0

def test_make_prediction_mock():
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.array([0]) # Good Risk
        def predict_proba(self, X):
            return np.array([[0.9, 0.1]]) # 90% Good, 10% Bad
            
    model = MockModel()
    input_data = {
        'Age': 30,
        'Job': 2,
        'Credit amount': 1000,
        'Duration': 12,
        'Sex': 'male',
        'Housing': 'own',
        'Saving accounts': 'little',
        'Checking account': 'moderate',
        'Purpose': 'car'
    }
    
    prediction, probability = make_prediction(model, input_data)
    assert prediction == "Good Risk"
    assert probability == 0.1

def test_make_prediction_bad_risk_mock():
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.array([1]) # Bad Risk
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]]) # 20% Good, 80% Bad
            
    model = MockModel()
    input_data = {} # Data doesn't matter for mock
    
    prediction, probability = make_prediction(model, input_data)
    assert prediction == "Bad Risk"
    assert probability == 0.8
