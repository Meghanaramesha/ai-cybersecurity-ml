from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/")
def home():
    return "AI Cybersecurity API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        features = np.array(data).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
