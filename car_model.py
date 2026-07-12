"""Shared model-loading and feature-engineering logic for the Flask API.

This reproduces, byte-for-byte, the one-hot encoding that
training/train_model.py used when it fit the two RandomForest models — the
same encoding the companion Streamlit project's apps (app.py / app_v1.py) use.
Keeping it in one place means app.py only has to worry about HTTP concerns
(routing, status codes, JSON parsing) and never touches feature engineering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent / "models"

# UI/API choice -> which pre-encoded flag columns to set to 1. An empty list
# means "this is the dropped baseline" (all flags stay 0) — exactly how the
# dataset was one-hot encoded during training. See training/train_model.py.
SELLER_FLAGS: Dict[str, List[str]] = {
    "Dealer": [],
    "Individual": ["Individual"],
    "Trustmark Dealer": ["Trustmark Dealer"],
}
FUEL_FLAGS: Dict[str, List[str]] = {
    "Petrol": ["Petrol"],
    "Diesel": ["Diesel"],
    "Electric": ["Electric"],
    "LPG": ["LPG"],
    "CNG": [],
}
TRANSMISSION_FLAGS: Dict[str, List[str]] = {
    "Manual": ["Manual"],
    "Automatic": [],
}
SEATS_FLAGS: Dict[str, List[str]] = {
    "5": ["Seats_5"],
    "More than 5": ["Seats_Above_5"],
    "Fewer than 5": [],
}

LAKH = 100_000        # 1 Lakh  = Rs 100,000
CRORE_IN_LAKHS = 100   # 1 Crore = 100 Lakhs

REQUIRED_FIELDS = ("make", "model")
OPTIONAL_FIELDS = (
    "age", "km_driven", "mileage", "engine", "max_power",
    "seller", "fuel", "transmission", "seats",
)


def format_price(value_lakhs: float) -> str:
    """Render a price (given in Lakhs) using Indian Crore/Lakh/Rupee notation."""
    if value_lakhs >= CRORE_IN_LAKHS:
        return f"₹{value_lakhs / CRORE_IN_LAKHS:.2f} Crore"
    if value_lakhs >= 1:
        return f"₹{value_lakhs:.2f} Lakhs"
    return f"₹{value_lakhs * LAKH:,.0f}"


class ModelNotLoaded(RuntimeError):
    """Raised when a prediction is requested before/without artifacts loaded."""


class CarPriceModel:
    """Loads the trained artifacts once and exposes a single `.predict(...)` call.

    Instantiate one of these at app start-up (see app.py) so the ~15 MB of
    pickled forests are read from disk exactly once per process, not once per
    request.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.price_model = None
        self.range_model = None
        self.feature_columns: List[str] | None = None
        self.range_config: Dict[str, Any] | None = None
        self.metadata: Dict[str, Any] | None = None

    def load(self) -> None:
        """Load the trained models and JSON config from `models_dir` (raises ModelNotLoaded if missing)."""
        try:
            self.price_model = joblib.load(self.models_dir / "price_model.pkl")
            self.range_model = joblib.load(self.models_dir / "range_model.pkl")
            self.feature_columns = json.loads(
                (self.models_dir / "feature_columns.json").read_text(encoding="utf-8")
            )
            self.range_config = json.loads(
                (self.models_dir / "range_config.json").read_text(encoding="utf-8")
            )
            self.metadata = json.loads(
                (self.models_dir / "metadata.json").read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise ModelNotLoaded(
                f"Model artifacts missing under {self.models_dir}. Either the "
                "models/ folder wasn't copied into the image, or you need to "
                "run training/train_model.py first to generate it."
            ) from exc

    @property
    def is_loaded(self) -> bool:
        """True once the models have been loaded."""
        return self.price_model is not None

    # -- metadata lookups --------------------------------------------------
    def _resolve_make_model(self, make: str, model: str) -> Tuple[str, str]:
        """Validate and uppercase the make/model, raising ValueError if unknown."""
        make = make.strip().upper()
        model = model.strip().upper()
        makes_models = self.metadata["makes_models"]
        if make not in makes_models:
            sample = list(makes_models)[:5]
            raise ValueError(f"Unknown make {make!r}. Example valid makes: {sample}")
        if model not in makes_models[make]:
            sample = makes_models[make][:5]
            raise ValueError(
                f"Unknown model {model!r} for make {make!r}. "
                f"Example valid models for {make}: {sample}"
            )
        return make, model

    def _auto_specs(self, make: str, model: str) -> Dict[str, float]:
        """Typical engine / power / mileage for this model (then make, then global)."""
        meta = self.metadata
        num = meta["numeric_features"]
        return (
            meta["model_specs"].get(make, {}).get(model)
            or meta["make_specs"].get(make)
            or {k: num[k]["default"] for k in ("mileage", "engine", "max_power")}
        )

    def _auto_options(self, make: str, model: str) -> Dict[str, List[str]]:
        """Fuel / transmission / seats this model actually came in (most-common first)."""
        meta = self.metadata
        cat = meta["categorical_features"]
        return (
            meta["model_options"].get(make, {}).get(model)
            or meta["make_options"].get(make)
            or {k: cat[k]["options"] for k in ("fuel", "transmission", "seats")}
        )

    def _numeric_default(self, name: str) -> float:
        """Return the median default for a numeric feature from the metadata."""
        return self.metadata["numeric_features"][name]["default"]

    # -- prediction ----------------------------------------------------------
    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Turn a car-description payload into a price estimate and a Low/Medium/High band."""
        if not self.is_loaded:
            raise ModelNotLoaded("Models are not loaded.")

        missing = [f for f in REQUIRED_FIELDS if f not in payload or payload[f] in (None, "")]
        if missing:
            raise ValueError(f"Missing required field(s): {missing}")

        make, model = self._resolve_make_model(str(payload["make"]), str(payload["model"]))

        # Anything not supplied is auto-filled from this model's real-world
        # typical values, exactly like the Streamlit "simple" app (app.py)
        # does — so a caller only has to know the car, not the raw ML schema.
        specs = self._auto_specs(make, model)
        opts = self._auto_options(make, model)

        try:
            age = float(payload.get("age", self._numeric_default("age")))
            km_driven = float(payload.get("km_driven", self._numeric_default("km_driven")))
            mileage = float(payload.get("mileage", specs["mileage"]))
            engine = float(payload.get("engine", specs["engine"]))
            max_power = float(payload.get("max_power", specs["max_power"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Numeric fields must be numbers: {exc}") from exc

        seller = str(payload.get("seller", "Dealer"))
        fuel = str(payload.get("fuel", opts["fuel"][0]))
        transmission = str(payload.get("transmission", opts["transmission"][0]))
        seats = str(payload.get("seats", opts["seats"][0]))

        for name, value, table in (
            ("seller", seller, SELLER_FLAGS),
            ("fuel", fuel, FUEL_FLAGS),
            ("transmission", transmission, TRANSMISSION_FLAGS),
            ("seats", seats, SEATS_FLAGS),
        ):
            if value not in table:
                raise ValueError(f"Invalid {name} {value!r}. Valid options: {list(table)}")

        row: Dict[str, Any] = {col: 0 for col in self.feature_columns}
        row.update(
            age=age, km_driven=km_driven, mileage=mileage,
            engine=engine, max_power=max_power, make=make, model=model,
        )
        for flag in (
            SELLER_FLAGS[seller] + FUEL_FLAGS[fuel]
            + TRANSMISSION_FLAGS[transmission] + SEATS_FLAGS[seats]
        ):
            if flag in row:
                row[flag] = 1

        X = pd.DataFrame([row])[self.feature_columns]
        price_lakhs = max(float(self.price_model.predict(X)[0]), 0.0)
        band = str(self.range_model.predict(X)[0])

        labels, edges = self.range_config["labels"], self.range_config["bin_edges"]
        idx = labels.index(band)
        low, high = edges[idx], edges[idx + 1]

        return {
            "input_used": {
                "make": make, "model": model, "age": age, "km_driven": km_driven,
                "mileage": mileage, "engine": engine, "max_power": max_power,
                "seller": seller, "fuel": fuel, "transmission": transmission,
                "seats": seats,
            },
            "predicted_price_lakhs": round(price_lakhs, 2),
            "predicted_price_display": format_price(price_lakhs),
            "price_range": {
                "label": band,
                "low_lakhs": low,
                "high_lakhs": high,
                "display": f"{format_price(low)} - {format_price(high)}",
            },
        }
