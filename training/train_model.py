"""
Car Price & Range Predictor - Model Training Pipeline
=====================================================

Trains two scikit-learn models on the Cars24 used-car dataset and writes every
artifact the Streamlit app needs to run *without* the raw CSV:

    models/price_model.pkl       RandomForestRegressor  -> exact price (in Lakhs)
    models/range_model.pkl       RandomForestClassifier -> Low / Medium / High
    models/feature_columns.json  Ordered feature names (the model's input schema)
    models/range_config.json     Price bin edges + labels + unit (for display)
    models/metadata.json         Everything the UI needs: make->model lists,
                                 numeric input ranges, categorical options and
                                 evaluation metrics.

Key facts about the dataset (see DESIGN_NOTES.md for the full rationale):

  * ``selling_price`` is expressed in **Lakhs of Rupees** (e.g. 4.75 == Rs 4.75 L),
    NOT raw rupees. Every downstream component treats price as Lakhs.
  * The categorical columns are already one-hot encoded *with a dropped baseline*:
        - Seller       : ``Individual`` / ``Trustmark Dealer`` (baseline = Dealer)
        - Fuel         : ``Petrol`` / ``Diesel`` / ``Electric`` / ``LPG`` (baseline = CNG/Other)
        - Transmission : ``Manual`` (baseline = Automatic)
        - Seats        : ``5`` / ``>5`` (baseline = fewer than 5)
    The app must be able to reproduce those baselines, so they are recorded in
    metadata.json rather than hard-coded.
  * Only ``make`` and ``model`` are free-text and get one-hot encoded inside the
    pipeline (``handle_unknown='ignore'`` so unseen values degrade gracefully).

This copy lives under app_car_prices_flask/training/ and is deliberately kept
separate from the Flask API in the parent folder: *training a model* and
*serving a model* are different workflows with different dependencies, run
cadence, and compute needs. The Docker image that serves predictions never
needs pandas' full training stack or the raw CSV baked in, only the small
.pkl/.json files this script produces, which is why they live in ../models
instead of here.

Run (from inside app_car_prices_flask/training/):
    python train_model.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # Paths (relative to this training/ folder; artifacts are written one
    # level up into app_car_prices_flask/models/ so the Flask app can load
    # them without any knowledge of how they were produced).
    # Stored gzip-compressed to keep the repo small (~81% smaller: 1.5 MB ->
    # ~290 KB — see README "Repository size"). pandas.read_csv() infers the
    # .gz extension and decompresses on the fly, so no separate unzip step
    # is needed anywhere in this script.
    "data_csv": "data/cars24-car-price-cleaned-new.csv.gz",
    "models_dir": "../models",
    "price_model_file": "../models/price_model.pkl",
    "range_model_file": "../models/range_model.pkl",
    "feature_columns_file": "../models/feature_columns.json",
    "range_config_file": "../models/range_config.json",
    "metadata_file": "../models/metadata.json",

    # Schema
    "target_price": "selling_price",
    "price_unit": "Lakhs",
    "text_features": ["make", "model"],
    "numeric_features": ["km_driven", "mileage", "engine", "max_power", "age"],
    "flag_features": [
        "Individual", "Trustmark Dealer",
        "Diesel", "Electric", "LPG", "Petrol",
        "Manual", "Seats_5", "Seats_Above_5",
    ],

    # Price-range binning (terciles -> Low / Medium / High)
    "range_quantiles": [0.0, 1 / 3, 2 / 3, 1.0],
    "range_labels": ["Low", "Medium", "High"],

    # Train / test split
    "test_size": 0.2,
    "random_state": 42,

    # Random-forest hyper-parameters. Depth and leaf size are *bounded* on
    # purpose: with ~3,200 one-hot model columns, unbounded trees overfit and
    # balloon the pickle to >100 MB. These caps keep accuracy high while
    # shrinking the artifacts ~10x and speeding up app start-up.
    "rf_common": {
        "n_estimators": 200,
        "max_depth": 18,
        "min_samples_leaf": 4,
        "n_jobs": -1,
        "random_state": 42,
    },
    "rf_regressor_extra": {"max_features": 0.5},
    "rf_classifier_extra": {"max_features": "sqrt", "class_weight": "balanced"},
}


# ---------------------------------------------------------------------------
# Small console helpers
# ---------------------------------------------------------------------------

def banner(text: str) -> None:
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def step(text: str) -> None:
    print(f"-> {text}")


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Read the raw CSV, exiting with a clear message if it is missing."""
    path = CONFIG["data_csv"]
    if not os.path.exists(path):
        print(f"ERROR: dataset not found at '{path}'.")
        print("Place the Cars24 CSV in the data/ folder and re-run.")
        sys.exit(1)

    df = pd.read_csv(path)
    step(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the awkward seat columns, drop empty/invalid rows."""
    df = df.rename(columns={"5": "Seats_5", ">5": "Seats_Above_5"})

    required = [CONFIG["target_price"], *CONFIG["text_features"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: dataset is missing required columns: {missing}")
        sys.exit(1)

    before = len(df)
    df = df.dropna(subset=[CONFIG["target_price"]])
    df = df[df[CONFIG["target_price"]] > 0]
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        step(f"Dropped {dropped:,} rows with missing/invalid values "
             f"({dropped / before * 100:.1f}%)")

    # Normalise the free-text columns so the dropdowns are tidy.
    for col in CONFIG["text_features"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def make_price_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Add a ``price_range`` column via tercile binning and return display config."""
    price = df[CONFIG["target_price"]]
    edges = [float(price.quantile(q)) for q in CONFIG["range_quantiles"]]
    # Guard against duplicate edges (would break pd.cut).
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 0.01

    df = df.copy()
    df["price_range"] = pd.cut(
        price, bins=edges, labels=CONFIG["range_labels"], include_lowest=True
    )

    range_config = {
        "labels": CONFIG["range_labels"],
        "bin_edges": [round(e, 2) for e in edges],
        "price_unit": CONFIG["price_unit"],
    }

    step("Price-range distribution:")
    for label in CONFIG["range_labels"]:
        n = int((df["price_range"] == label).sum())
        print(f"     {label:<7} {n:>6,} cars ({n / len(df) * 100:4.1f}%)")
    return df, range_config


def numeric_input_specs(df: pd.DataFrame) -> Dict[str, Dict]:
    """Compute UI-friendly slider bounds (robust to outliers) per numeric feature."""
    # Round-to-nice step for each feature so the sliders feel natural.
    steps = {"km_driven": 1000, "mileage": 0.5, "engine": 50,
             "max_power": 1.0, "age": 1}
    int_like = {"km_driven", "engine", "age"}

    specs: Dict[str, Dict] = {}
    for col in CONFIG["numeric_features"]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        lo = float(s.quantile(0.01))   # clip outliers off the slider ends
        hi = float(s.quantile(0.99))
        median = float(s.median())
        st = steps[col]

        # Snap bounds outward to a multiple of the step.
        ui_min = max(0.0, math.floor(lo / st) * st)
        ui_max = math.ceil(hi / st) * st
        default = round(median / st) * st
        default = min(max(default, ui_min), ui_max)

        if col in int_like:
            ui_min, ui_max, default, st = int(ui_min), int(ui_max), int(default), int(st)
            dtype = "int"
        else:
            ui_min, ui_max, default = round(ui_min, 1), round(ui_max, 1), round(default, 1)
            dtype = "float"

        specs[col] = {"min": ui_min, "max": ui_max, "default": default,
                      "step": st, "dtype": dtype}
    return specs


def build_auto_spec_tables(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Median engine/power/mileage per model (and per make, as a fallback).

    The Streamlit "simple" app (app.py) hides these technical inputs from novices and
    auto-fills them from the chosen car's typical values. They carry strong
    price signal — a luxury SUV must not be scored with a hatchback's engine —
    so we record the medians here rather than using one global default.
    """
    specs = ["mileage", "engine", "max_power"]

    model_table: Dict[str, Dict] = {}
    grouped = df.groupby(["make", "model"])[specs].median().round(1)
    for (make, model), row in grouped.iterrows():
        model_table.setdefault(make, {})[model] = {s: float(row[s]) for s in specs}

    make_table = {
        make: {s: float(v) for s, v in row.items()}
        for make, row in df.groupby("make")[specs].median().round(1).iterrows()
    }
    return model_table, make_table


def build_option_tables(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Valid fuel / transmission / seats actually observed for each model.

    These let the app constrain its dropdowns so a user can only choose a
    combination that exists in the data — a diesel-only variant never offers
    "Electric", etc. Almost every model has a single valid fuel/gearbox because
    the variant name already encodes it (e.g. "SWIFT VDI" -> Diesel/Manual).
    Options are ordered most-common-first, so option [0] is a safe default.
    """
    work = df.copy()

    fuel = pd.Series("CNG", index=work.index)        # CNG is the dropped baseline
    for f in ["Petrol", "Diesel", "Electric", "LPG"]:
        fuel[work[f] == 1] = f
    work["_fuel"] = fuel
    work["_transmission"] = work["Manual"].map({1: "Manual"}).fillna("Automatic")
    seats = pd.Series("Fewer than 5", index=work.index)
    seats[work["Seats_5"] == 1] = "5"
    seats[work["Seats_Above_5"] == 1] = "More than 5"
    work["_seats"] = seats

    def options(group) -> Dict:
        return {
            "fuel": group["_fuel"].value_counts().index.tolist(),
            "transmission": group["_transmission"].value_counts().index.tolist(),
            "seats": group["_seats"].value_counts().index.tolist(),
        }

    model_options: Dict[str, Dict] = {}
    for (make, model), group in work.groupby(["make", "model"]):
        model_options.setdefault(make, {})[model] = options(group)
    make_options = {make: options(group) for make, group in work.groupby("make")}
    return model_options, make_options


def build_metadata(df: pd.DataFrame, metrics: Dict) -> Dict:
    """Assemble everything the Streamlit UI needs to render itself."""
    makes_models = {
        make: sorted(group["model"].unique().tolist())
        for make, group in df.groupby("make")
    }
    model_specs, make_specs = build_auto_spec_tables(df)
    model_options, make_options = build_option_tables(df)
    return {
        "price_unit": CONFIG["price_unit"],
        "n_samples": int(len(df)),
        "makes_models": dict(sorted(makes_models.items())),
        # Per-model / per-make median specs for the simple app's auto-fill.
        "model_specs": model_specs,
        "make_specs": make_specs,
        # Per-model / per-make valid categorical options (prevents mismatches).
        "model_options": model_options,
        "make_options": make_options,
        "numeric_features": numeric_input_specs(df),
        # Categorical options + the dropped baselines, so the app can rebuild
        # the exact one-hot encoding the model was trained on.
        "categorical_features": {
            "seller": {
                "options": ["Dealer", "Individual", "Trustmark Dealer"],
                "default": "Dealer",
            },
            "fuel": {
                "options": ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
                "default": "Petrol",
            },
            "transmission": {
                "options": ["Manual", "Automatic"],
                "default": "Manual",
            },
            "seats": {
                "options": ["5", "More than 5", "Fewer than 5"],
                "default": "5",
            },
        },
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------

def build_pipelines() -> Tuple[Pipeline, Pipeline]:
    """Create the regression and classification pipelines (shared preprocessing)."""
    preprocessor = ColumnTransformer(
        transformers=[(
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CONFIG["text_features"],
        )],
        remainder="passthrough",
    )

    regressor = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            **CONFIG["rf_common"], **CONFIG["rf_regressor_extra"])),
    ])
    classifier = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            **CONFIG["rf_common"], **CONFIG["rf_classifier_extra"])),
    ])
    return regressor, classifier


def split_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return the feature matrix X (ordered) and the feature-name list."""
    feature_columns = (
        CONFIG["numeric_features"]
        + CONFIG["text_features"]
        + [c for c in CONFIG["flag_features"] if c in df.columns]
    )
    return df[feature_columns].copy(), feature_columns


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_artifacts(price_model, range_model, feature_columns, range_config, metadata):
    os.makedirs(CONFIG["models_dir"], exist_ok=True)
    # LZMA beats joblib's default zlib backend on both axes for these forests:
    # ~30% smaller AND faster to load (benchmarked on the real artifacts —
    # see README "Repository size"). The only cost is slower *saving*, which
    # only happens here, once, during training — never at API start-up.
    joblib.dump(price_model, CONFIG["price_model_file"], compress=("lzma", 9))
    joblib.dump(range_model, CONFIG["range_model_file"], compress=("lzma", 9))
    save_json(feature_columns, CONFIG["feature_columns_file"])
    save_json(range_config, CONFIG["range_config_file"])
    save_json(metadata, CONFIG["metadata_file"])

    for path in (CONFIG["price_model_file"], CONFIG["range_model_file"],
                 CONFIG["feature_columns_file"], CONFIG["range_config_file"],
                 CONFIG["metadata_file"]):
        size_mb = os.path.getsize(path) / 1024 / 1024
        step(f"saved {path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    banner("Car Price & Range Predictor - Training")

    df = load_dataset()
    df = clean_dataset(df)
    df, range_config = make_price_ranges(df)

    X, feature_columns = split_features(df)
    y_price = df[CONFIG["target_price"]]
    y_range = df["price_range"]

    step(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")
    step(f"Price unit: {CONFIG['price_unit']}  (e.g. 4.75 = Rs 4.75 Lakhs)")

    (X_tr, X_te,
     yp_tr, yp_te,
     yr_tr, yr_te) = train_test_split(
        X, y_price, y_range,
        test_size=CONFIG["test_size"], random_state=CONFIG["random_state"],
    )

    banner("Training models (this can take a minute)")
    price_model, range_model = build_pipelines()

    step("Fitting price regressor ...")
    price_model.fit(X_tr, yp_tr)

    step("Fitting range classifier ...")
    range_model.fit(X_tr, yr_tr)

    # Evaluation
    banner("Evaluation (held-out test set)")
    price_pred = price_model.predict(X_te)
    range_pred = range_model.predict(X_te)
    metrics = {
        "price_r2": round(float(r2_score(yp_te, price_pred)), 4),
        "price_mae_lakhs": round(float(mean_absolute_error(yp_te, price_pred)), 3),
        "range_accuracy": round(float(accuracy_score(yr_te, range_pred)), 4),
    }
    print(f"  Price  R^2            : {metrics['price_r2']:.4f}")
    print(f"  Price  MAE            : Rs {metrics['price_mae_lakhs']:.2f} Lakhs")
    print(f"  Range  accuracy       : {metrics['range_accuracy'] * 100:.2f}%")

    metadata = build_metadata(df, metrics)

    banner("Saving artifacts")
    save_artifacts(price_model, range_model, feature_columns, range_config, metadata)

    banner("Done - artifacts written to ../models/. Serve them with:  python app.py")


if __name__ == "__main__":
    main()
