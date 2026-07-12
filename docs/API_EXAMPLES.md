# 📗 API Examples — the Flask predictor by example

Two ways to use this project: the **Python class** (`car_model.CarPriceModel`) and the
**REST API** (`POST /predict`). Copy-paste recipes with expected output below; a runnable
version is in [`notebooks/api_examples.ipynb`](../notebooks/api_examples.ipynb).

Setup once:

```bash
pip install -r requirements.txt
```

---

## The Python API — `CarPriceModel`

```python
from car_model import CarPriceModel

m = CarPriceModel()
m.load()                     # reads the models/ artifacts once
m.is_loaded                  # True

m.predict({"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000})
# -> {'predicted_price_lakhs': 6.5,
#     'predicted_price_display': '₹6.50 Lakhs',
#     'price_range': {'label': 'Medium', 'low_lakhs': 3.99, 'high_lakhs': 6.75,
#                     'display': '₹3.99 Lakhs - ₹6.75 Lakhs'},
#     'input_used': {...auto-filled...}}
```

Only `make` + `model` are required — the rest is auto-filled:

```python
m.predict({"make": "HYUNDAI", "model": "I20 SPORTZ"})   # still returns a full result
```

Bad input raises a clear `ValueError`:

```python
m.predict({"make": "NOTABRAND", "model": "X"})     # ValueError: Unknown make ...
m.predict({"make": "MARUTI", "model": "SWIFT VDI", "fuel": "Hydrogen"})  # Invalid fuel ...
```

Helpers:

```python
import car_model
car_model.format_price(6.5)          # '₹6.50 Lakhs'
car_model.format_price(145)          # '₹1.45 Crore'
car_model.SELLER_FLAGS["Dealer"]     # []  (the dropped baseline sets no flag)
```

## The REST API — `POST /predict`

Start the server:

```bash
python app.py                        # dev server, http://localhost:5000
# or, production:  gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app
```

Call it with `curl`:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000}'
# {"predicted_price_lakhs": 6.5, ... "label": "Medium" ...}

curl http://localhost:5000/health      # {"status":"healthy"}
```

...or with Python `requests`:

```python
import requests
r = requests.post("http://localhost:5000/predict",
                  json={"make": "MARUTI", "model": "SWIFT VDI"})
r.status_code, r.json()["predicted_price_lakhs"]     # (200, 6.5)
```

## Testing the API without a server — the Flask test client

```python
from app import app
client = app.test_client()

client.get("/health").get_json()                    # {'status': 'healthy'}
client.post("/predict", json={"make": "MARUTI", "model": "SWIFT VDI"}).status_code   # 200
client.get("/predict").status_code                  # 405 (browsers send GET; /predict needs POST)
```

> The `405` on a browser visit is *expected* — see the README's "Testing the deployed app".
> Deploying this to AWS ECS/Fargate? Walkthrough:
> [`notebooks/deploy_docker_and_aws_ecs.ipynb`](../notebooks/deploy_docker_and_aws_ecs.ipynb).
