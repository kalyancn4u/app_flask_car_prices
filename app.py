"""Flask REST API for the Car Price & Range Predictor.

Wraps the same RandomForest models the companion Streamlit project's apps use
behind a single POST /predict endpoint, so the app can be containerised and
deployed to AWS ECS (Fargate) for the class assignment. See README.md for the
full deployment walkthrough and example requests.

Run locally without Docker:
    pip install -r requirements.txt
    python app.py                     # Flask's own dev server, http://localhost:5000

Run through gunicorn instead (what the production Dockerfile uses):
    gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app
"""

from __future__ import annotations

from flask import Flask, jsonify, request

from car_model import CarPriceModel, ModelNotLoaded, OPTIONAL_FIELDS, REQUIRED_FIELDS

app = Flask(__name__)
app.json.ensure_ascii = False  # let ₹ render as a real character, not ₹

car_model = CarPriceModel()
try:
    car_model.load()
except ModelNotLoaded as exc:
    # Don't crash the process — /health will report the problem clearly
    # instead of the container looping through failed restarts.
    app.logger.error(str(exc))


@app.get("/")
def index():
    """GET / -- service info plus a usage example."""
    return jsonify({
        "service": "car-price-predictor",
        "status": "ok" if car_model.is_loaded else "models_not_loaded",
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "content_type": "application/json",
            "required_fields": list(REQUIRED_FIELDS),
            "optional_fields": list(OPTIONAL_FIELDS),
            "example_request": {"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000},
        },
    })


@app.get("/health")
def health():
    """GET /health -- 200 if the models are loaded, 503 otherwise (the ECS health check)."""
    if not car_model.is_loaded:
        return jsonify({"status": "unhealthy", "reason": "models not loaded"}), 503
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    """POST /predict -- return a price estimate and band for the posted car JSON."""
    if not car_model.is_loaded:
        return jsonify({"error": "Models are not loaded on the server. Check /health."}), 503

    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, dict):
        return jsonify({
            "error": "Request body must be a JSON object.",
            "hint": 'Send Content-Type: application/json and a body like '
                    '{"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000}',
        }), 400

    try:
        result = car_model.predict(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.errorhandler(405)
def method_not_allowed(_exc):
    """405 handler -- explain that /predict needs POST, not GET."""
    return jsonify({
        "error": "Method not allowed.",
        "hint": "/predict only accepts POST with a JSON body. A browser visiting "
                "this URL sends GET, which is why you see this error there.",
    }), 405


@app.errorhandler(404)
def not_found(_exc):
    """404 handler -- list the available endpoints."""
    return jsonify({
        "error": "Not found.",
        "available_endpoints": ["/", "/health", "/predict"],
    }), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
