import numpy as np
import pandas as pd

def population_stability_index(expected, actual, bins=10):
    """
    Calculates the PSI for two distributions.
    """
    # Create histograms
    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    # Normalize to percentages
    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    # Calculate PSI
    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )
    return psi


def detect_data_drift(reference_df, current_df, threshold=0.2):
    """
    Detects drift in features comparing reference and current data.
    """
    drift_report = {}

    # Only check columns present in both
    cols_to_check = [c for c in reference_df.columns if c in current_df.columns]

    for col in cols_to_check:
        try:
            # Handle numeric and categorical data for PSI
            # For simplicity in this implementation, we assume numeric or pre-encoded data
            # or we cast to numeric if possible for the histogram
            ref_vals = pd.to_numeric(reference_df[col], errors='coerce').dropna().values
            curr_vals = pd.to_numeric(current_df[col], errors='coerce').dropna().values
            
            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            psi = population_stability_index(ref_vals, curr_vals)
            drift_report[col] = {
                "psi": psi,
                "drift": psi > threshold
            }
        except Exception as e:
            print(f"Error computing PSI for {col}: {e}")

    return drift_report
