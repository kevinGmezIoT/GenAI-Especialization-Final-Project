def detect_prediction_drift(reference_preds, current_preds, threshold=0.1):
    """
    Detects drift in model outputs by comparing the rate of 'bad risk' predictions.
    """
    if len(reference_preds) == 0 or len(current_preds) == 0:
        return {"error": "Empty predictions"}

    # Calculate 'bad risk' rate (assuming 1 is bad risk, 0 is good risk)
    # If inputs are labels, map them
    def compute_rate(preds):
        if all(isinstance(x, str) for x in preds):
             # Map strings if needed
             preds = [1 if 'bad' in str(x).lower() else 0 for x in preds]
        return sum(preds) / len(preds)

    ref_rate = compute_rate(reference_preds)
    curr_rate = compute_rate(current_preds)

    drift = abs(curr_rate - ref_rate) > threshold

    return {
        "reference_rate": ref_rate,
        "current_rate": curr_rate,
        "drift": drift
    }
