import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.monitoring.data_drift import detect_data_drift
from src.monitoring.prediction_drift import detect_prediction_drift
from src.monitoring.alerts import send_alert

def run_monitoring_suite():
    print("--- Running Monitoring Suite ---")
    results = {"data_drift": {}, "prediction_drift": {}, "status": "success"}
    
    # Paths (Baseline vs Current)
    ref_data_path = BASE_DIR / "data/processed/train_features.csv"
    current_log_path = BASE_DIR / "logs/inference.log"
    
    if not ref_data_path.exists():
        return {"status": "error", "message": "Reference data not found. Please train the model."}

    if not current_log_path.exists():
        return {"status": "waiting", "message": "No inference logs found yet."}

    # Load Reference Data
    reference_df = pd.read_csv(ref_data_path)
    
    try:
        current_records = []
        with open(current_log_path, 'r') as f:
            for line in f:
                current_records.append(json.loads(line))
        
        if not current_records:
            return {"status": "waiting", "message": "Inference log is empty."}
            
        current_all = pd.DataFrame(current_records)
        current_data = pd.json_normalize(current_all['input_summary'])
        current_preds = current_all['prediction']
        
        # 1. Data Drift
        print("Checking for Data Drift...")
        data_drift = detect_data_drift(reference_df, current_data)
        results["data_drift"] = data_drift
        for feature, res in data_drift.items():
            if res["drift"]:
                send_alert(f"Data drift detected in {feature} (PSI={res['psi']:.2f})")

        # 2. Prediction Drift
        print("Checking for Prediction Drift...")
        # Baseline bad risk rate from training (we assume ~25% for this dataset)
        ref_preds = [1, 0, 0, 0] # Example baseline: 25% bad risk
        pred_drift = detect_prediction_drift(ref_preds, current_preds)
        results["prediction_drift"] = pred_drift
        
        if pred_drift.get("drift"):
            send_alert(f"Prediction drift detectado: {pred_drift['reference_rate']:.2f} → {pred_drift['current_rate']:.2f}")
            
        return results
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

import json
if __name__ == "__main__":
    import json
    report = run_monitoring_suite()
    print(json.dumps(report, indent=2))
