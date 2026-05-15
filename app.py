from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# ---------------- LOAD MODEL & SCALER ----------------
with open("Mrri.pkl", "rb") as f:
    model = pickle.load(f)

with open("scalar.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- LOAD COLUMN NAMES ----------------
df = pd.read_csv("Mri_dataset.csv")
feature_names = df.columns[:-1].tolist()

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        values = [float(request.form[f]) for f in feature_names]

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        prediction = "Disease Detected" if pred == 1 else "No Disease Detected"
        probability = round(prob * 100, 2)

    return render_template(
        "index.html",
        features=feature_names,
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)
