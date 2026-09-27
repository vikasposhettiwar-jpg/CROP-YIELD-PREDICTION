"""Flask backend for smart crop yield prediction and recommendation."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from model.predict import dashboard_data, load_model, predict_yield, recommend_crop

app = Flask(__name__)

model = None


def validate_payload(payload: dict) -> tuple[bool, str]:
    """Validate request payload values and return status with message."""
    required = ["temperature", "rainfall", "humidity", "soil_type"]
    for field in required:
        if field not in payload:
            return False, f"Missing required field: {field}"

    try:
        temperature = float(payload["temperature"])
        rainfall = float(payload["rainfall"])
        humidity = float(payload["humidity"])
        soil_type = str(payload["soil_type"]).strip()
    except (TypeError, ValueError):
        return False, "temperature, rainfall, and humidity must be numeric."

    if not (0 <= temperature <= 55):
        return False, "temperature must be between 0 and 55 C."
    if not (0 <= rainfall <= 600):
        return False, "rainfall must be between 0 and 600 mm."
    if not (0 <= humidity <= 100):
        return False, "humidity must be between 0 and 100."
    if not soil_type:
        return False, "soil_type cannot be empty."

    return True, "ok"


@app.route("/prediction")
def index():
    """Render web dashboard."""
    return render_template("index.html")


@app.route("/")
def home():
    """Render home page."""
    return render_template("home.html")


@app.route("/login")
def login():
    """Render login page."""
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """Render user dashboard."""
    return render_template("dashboard.html")


@app.get("/api/dashboard")
def get_dashboard():
    """Return model comparison and feature importance data."""
    return jsonify(dashboard_data())


@app.post("/predict")
def predict():
    """Predict yield for given climate and soil input."""
    payload = request.get_json(silent=True) or {}
    valid, message = validate_payload(payload)
    if not valid:
        return jsonify({"error": message}), 400

    try:
        result = predict_yield(model, payload)
        return jsonify({"predicted_yield_ton_per_hectare": result})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.post("/recommend")
def recommend():
    """Recommend best crop for given climate and soil input."""
    payload = request.get_json(silent=True) or {}
    valid, message = validate_payload(payload)
    if not valid:
        return jsonify({"error": message}), 400

    try:
        recommendation = recommend_crop(payload)
        return jsonify({"recommended_crop": recommendation["crop"], "confidence_score": recommendation["score"]})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Recommendation failed: {exc}"}), 500


def load_or_train_model():
    """Load model artifact and guide user if missing."""
    global model  # noqa: PLW0603
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Model artifact missing. Please run `python model/train_model.py` before starting Flask."
        ) from exc


if __name__ == "__main__":
    load_or_train_model()
    port = 5000
    print(f"Starting app on http://127.0.0.1:{port}")
    app.run(debug=True, port=port)

