import joblib
import pandas as pd
import os

def load_model(model_path):
    """
    Loads the trained model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)

def make_prediction(model, input_data):
    """
    Makes a prediction for the given input data.
    
    Args:
        model: The trained Scikit-Learn pipeline.
        input_data: A dictionary or DataFrame containing the input features.
        
    Returns:
        prediction_label: 'Good Risk' or 'Bad Risk'
        probability: The probability of the predicted class (if available).
    """
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Get probability if available
    probability = None
    if hasattr(model, "predict_proba"):
        # Probability of class 1 (Bad Risk)
        prob_bad_risk = model.predict_proba(input_data)[0][1]
        probability = prob_bad_risk
    
    # Map prediction to label
    # 0: Good Risk, 1: Bad Risk (based on our training mapping)
    prediction_label = "Bad Risk" if prediction[0] == 1 else "Good Risk"
    
    return prediction_label, probability
