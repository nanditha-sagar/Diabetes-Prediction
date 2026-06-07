from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import os

app = Flask(__name__)
CORS(app)

# ── Train the model on startup ───────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")
diabetes_dset = pd.read_csv(CSV_PATH)

X = diabetes_dset.drop(columns="Outcome", axis=1)
Y = diabetes_dset["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

FEATURE_NAMES = list(X.columns)
print("✅  Model trained and ready.")

# ── Symptom → Feature mapping reference values ───────────────────────────────
# These are medically-grounded estimates based on symptom severity (0–3 scale).
# Baseline = population average; each symptom nudges the relevant clinical features.

FEATURE_BASELINES = {
    "Glucose":                   107.0,
    "BloodPressure":              69.1,
    "SkinThickness":              20.5,
    "Insulin":                    79.8,
    "BMI":                        31.9,
    "DiabetesPedigreeFunction":    0.47,
}

# Each symptom contributes a delta to one or more features per severity point (0–3)
SYMPTOM_DELTAS = {
    # (feature, delta_per_severity_point)
    "polyuria":        [("Glucose", 12.0),  ("Insulin", 15.0)],
    "polydipsia":      [("Glucose", 10.0),  ("BloodPressure", 2.0)],
    "polyphagia":      [("Glucose",  8.0),  ("Insulin", -8.0)],
    "weight_loss":     [("BMI",     -1.8),  ("Insulin", -12.0)],
    "fatigue":         [("Glucose",  6.0),  ("Insulin",  -6.0)],
    "blurry_vision":   [("Glucose",  9.0),  ("BloodPressure", 3.0)],
    "slow_healing":    [("Glucose",  7.0),  ("DiabetesPedigreeFunction", 0.05)],
    "tingling":        [("Glucose",  8.0),  ("DiabetesPedigreeFunction", 0.08)],
}


def symptoms_to_features(payload: dict) -> dict:
    """
    Convert symptom severities + age + pregnancies into the 8 SVM features.

    payload keys:
      age (int), pregnancies (int)
      polyuria, polydipsia, polyphagia, weight_loss,
      fatigue, blurry_vision, slow_healing, tingling  ← each 0–3
    """
    features = dict(FEATURE_BASELINES)

    # Apply symptom deltas
    for symptom, deltas in SYMPTOM_DELTAS.items():
        severity = float(payload.get(symptom, 0))
        severity = max(0.0, min(3.0, severity))
        for feat, delta in deltas:
            features[feat] += delta * severity

    # Clamp to physiologically plausible ranges
    features["Glucose"]                   = max(50.0,  min(250.0, features["Glucose"]))
    features["BloodPressure"]             = max(40.0,  min(130.0, features["BloodPressure"]))
    features["SkinThickness"]             = max(0.0,   min(99.0,  features["SkinThickness"]))
    features["Insulin"]                   = max(0.0,   min(846.0, features["Insulin"]))
    features["BMI"]                       = max(15.0,  min(67.0,  features["BMI"]))
    features["DiabetesPedigreeFunction"]  = max(0.078, min(2.42,  features["DiabetesPedigreeFunction"]))

    features["Pregnancies"] = max(0, int(payload.get("pregnancies", 0)))
    features["Age"]         = max(1, int(payload.get("age", 30)))

    return features


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
      "age": 45,
      "pregnancies": 2,
      "polyuria": 2,      // 0=none, 1=mild, 2=moderate, 3=severe
      "polydipsia": 1,
      "polyphagia": 2,
      "weight_loss": 0,
      "fatigue": 3,
      "blurry_vision": 1,
      "slow_healing": 0,
      "tingling": 1
    }
    """
    try:
        payload = request.get_json(force=True)

        required = ["age", "pregnancies", "polyuria", "polydipsia", "polyphagia",
                    "weight_loss", "fatigue", "blurry_vision", "slow_healing", "tingling"]
        missing = [k for k in required if k not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = symptoms_to_features(payload)

        values = [features[f] for f in FEATURE_NAMES]
        input_array = np.asarray(values).reshape(1, -1)
        input_df = pd.DataFrame(input_array, columns=FEATURE_NAMES)
        std_data = scaler.transform(input_df)

        prediction = int(classifier.predict(std_data)[0])
        label = "Diabetic" if prediction == 1 else "Not Diabetic"

        # Count how many symptoms are elevated (severity >= 1)
        symptom_keys = ["polyuria", "polydipsia", "polyphagia", "weight_loss",
                        "fatigue", "blurry_vision", "slow_healing", "tingling"]
        symptom_count = sum(1 for k in symptom_keys if int(payload.get(k, 0)) >= 1)

        return jsonify({
            "prediction": prediction,
            "label": label,
            "diabetic": prediction == 1,
            "symptom_count": symptom_count,
            "estimated_glucose": round(features["Glucose"], 1),
            "estimated_bmi": round(features["BMI"], 1),
            "estimated_insulin": round(features["Insulin"], 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)