# --------------------------------------------------------
# Cognitive Load Index Estimation Model (Ashmit Prototype)
# --------------------------------------------------------
# Run this file in VSCode with Python 3.9+ and scikit-learn installed.
# pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Simulate Sensor Data
# ---------------------------
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    "EEG_theta_alpha": np.random.normal(1.5, 0.3, n_samples),     # EEG ratio (theta/alpha)
    "HRV_LF_HF": np.random.normal(2.0, 0.5, n_samples),           # HRV ratio (LF/HF)
    "pupil_change": np.random.normal(0.1, 0.05, n_samples),       # Δpupil (fractional change)
    "emotion_intensity": np.random.randint(0, 4, n_samples),      # 0–3 scale
})

# True (simulated) Cognitive Load based on weighted contribution
data["CLI_true"] = (
    0.4 * data["EEG_theta_alpha"] +
    0.3 * data["HRV_LF_HF"] +
    0.2 * data["pupil_change"] +
    0.1 * data["emotion_intensity"] +
    np.random.normal(0, 0.05, n_samples)   # noise term
)

# -----------------------------
# Step 2: Train the Regression Model
# -----------------------------
X = data[["EEG_theta_alpha", "HRV_LF_HF", "pupil_change", "emotion_intensity"]]
y = data["CLI_true"]

# Pipeline: Standardize + Regress
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# Step 3: Normalize CLI to 0–100
# -----------------------------
def normalize_cli(y_values):
    y_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    return y_norm * 100

CLI_normalized = normalize_cli(y_pred)

# -----------------------------
# Step 4: Evaluate Model
# -----------------------------
print("\nCognitive Load Index Model Results")
print("-----------------------------------")
print("R² Score: ", round(r2_score(y_test, y_pred), 3))
print("\nExample Predicted CLI (0–100):")
print(np.round(CLI_normalized[:10], 2))

# -----------------------------
# Step 5: Visualize
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, CLI_normalized, alpha=0.6, label="Predictions")
plt.xlabel("True Cognitive Load (Simulated)")
plt.ylabel("Predicted CLI (0–100)")
plt.title("Cognitive Load Index Prediction")
plt.legend()
plt.grid(True)
plt.show()
