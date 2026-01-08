import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib

from encoder import get_text_embedding   # NEW

# -------- load dataset ----------
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ---------- generate fake text (or add your real text column) ----------
fake_text = ["transaction" for _ in range(len(X))]

text_embeddings = np.array([get_text_embedding(t) for t in fake_text])

# ---- combine numeric + text embeddings ----
X = np.hstack((X.values, text_embeddings))

# ---------- scale ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- supervised ----------
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

# ---------- anomaly ----------
iso = IsolationForest(
    contamination=0.01,
    random_state=42
)
iso.fit(X_train)

# ---------- save ----------
joblib.dump(rf, "supervised_rf.pkl")
joblib.dump(iso, "anomaly_if.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Retrained models saved with encoder features ✔")