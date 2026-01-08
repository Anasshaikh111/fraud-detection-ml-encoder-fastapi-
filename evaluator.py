import joblib
import numpy as np
from encoder import get_text_embedding

rf = joblib.load("supervised_rf.pkl")
iso = joblib.load("anomaly_if.pkl")
scaler = joblib.load("scaler.pkl")

def evaluate_transaction(features, description: str = ""):

    base_features = np.array(features)

    if description != "":
        text_vec = get_text_embedding(description)
        combined = np.concatenate([base_features, text_vec])
    else:
        combined = base_features

    combined = combined.reshape(1, -1)

    scaled = scaler.transform(combined)

    supervised_prob = rf.predict_proba(scaled)[0][1]
    anomaly_score = -iso.decision_function(scaled)[0]

    final_score = 0.7 * supervised_prob + 0.3 * anomaly_score

    result = "Fraudulent" if final_score > 0.5 else "Legitimate"

    return {
        "supervised_probability": float(supervised_prob),
        "anomaly_score": float(anomaly_score),
        "final_score": float(final_score),
        "prediction": result
    }
