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

# ── Train model on startup ────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")
diabetes_dset = pd.read_csv(CSV_PATH)

X = diabetes_dset.drop(columns="Outcome")
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

# ── Symptom → clinical feature mapping ───────────────────────────────────────
BASELINES = {
    "Glucose": 107.0,
    "BloodPressure": 69.1,
    "SkinThickness": 20.5,
    "Insulin": 79.8,
    "BMI": 31.9,
    "DiabetesPedigreeFunction": 0.47,
}

SYMPTOM_DELTAS = {
    "polyuria":      [("Glucose", 12.0), ("Insulin", 15.0)],
    "polydipsia":    [("Glucose", 10.0), ("BloodPressure", 2.0)],
    "polyphagia":    [("Glucose",  8.0), ("Insulin", -8.0)],
    "weight_loss":   [("BMI",     -1.8), ("Insulin", -12.0)],
    "fatigue":       [("Glucose",  6.0), ("Insulin",  -6.0)],
    "blurry_vision": [("Glucose",  9.0), ("BloodPressure", 3.0)],
    "slow_healing":  [("Glucose",  7.0), ("DiabetesPedigreeFunction", 0.05)],
    "tingling":      [("Glucose",  8.0), ("DiabetesPedigreeFunction", 0.08)],
}

CLAMPS = {
    "Glucose":                  (50.0,  250.0),
    "BloodPressure":            (40.0,  130.0),
    "SkinThickness":            (0.0,    99.0),
    "Insulin":                  (0.0,   846.0),
    "BMI":                      (15.0,   67.0),
    "DiabetesPedigreeFunction": (0.078,   2.42),
}


def symptoms_to_features(payload: dict) -> dict:
    features = dict(BASELINES)

    # If user provided explicit insulin, use it; otherwise estimate
    explicit_insulin = payload.get("insulin_level")
    if explicit_insulin is not None:
        features["Insulin"] = float(explicit_insulin)

    for symptom, deltas in SYMPTOM_DELTAS.items():
        severity = max(0.0, min(3.0, float(payload.get(symptom, 0))))
        for feat, delta in deltas:
            if feat == "Insulin" and explicit_insulin is not None:
                continue  # don't override explicit value
            features[feat] += delta * severity

    for feat, (lo, hi) in CLAMPS.items():
        features[feat] = max(lo, min(hi, features[feat]))

    # Gender adjustment: males have 0 pregnancies for the model
    gender = payload.get("gender", "female").lower()
    preg = 0 if gender == "male" else max(0, int(payload.get("pregnancies", 0)))
    features["Pregnancies"] = preg
    features["Age"] = max(1, int(payload.get("age", 30)))

    return features


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body with:
      gender (male/female), age (int), pregnancies (int, females only),
      insulin_level (float, optional — μU/mL),
      polyuria, polydipsia, polyphagia, weight_loss,
      fatigue, blurry_vision, slow_healing, tingling  (each 0–3)
    """
    try:
        payload = request.get_json(force=True)
        required = ["gender", "age", "polyuria", "polydipsia", "polyphagia",
                    "weight_loss", "fatigue", "blurry_vision", "slow_healing", "tingling"]
        missing = [k for k in required if k not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        features = symptoms_to_features(payload)

        values = [features[f] for f in FEATURE_NAMES]
        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)
        std_data = scaler.transform(input_df)
        prediction = int(classifier.predict(std_data)[0])

        symptom_keys = ["polyuria", "polydipsia", "polyphagia", "weight_loss",
                        "fatigue", "blurry_vision", "slow_healing", "tingling"]
        symptom_count = sum(1 for k in symptom_keys if int(payload.get(k, 0)) >= 1)

        return jsonify({
            "prediction": prediction,
            "label": "Diabetic" if prediction == 1 else "Not Diabetic",
            "diabetic": prediction == 1,
            "symptom_count": symptom_count,
            "estimated_glucose": round(features["Glucose"], 1),
            "estimated_bmi": round(features["BMI"], 1),
            "estimated_insulin": round(features["Insulin"], 1),
            "gender": payload.get("gender", "female"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)