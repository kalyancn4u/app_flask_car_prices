# 🚗 Car Price Predictor — Flask API

A minimal REST API that wraps the same two RandomForest models trained for a
companion Streamlit car-price predictor (`app.py` / `app_v1.py`, developed
alongside this API but maintained in its own separate repository) behind a single
`POST /predict` endpoint, packaged for containerised deployment to
**AWS ECS (Fargate)**.

This is a standalone project — it does not import or depend on anything
outside this repo. The `models/` folder below already contains everything
needed to run it; `training/` can reproduce those artifacts from scratch
without any of the Streamlit app's code.

> Given a car's make, model, age and kilometres driven (plus optional extra
> details), it returns an estimated selling price in ₹ Lakhs/Crores and a
> Low/Medium/High price band.

---

## Table of contents

1. [Project structure](#project-structure)
2. [Quick start (no Docker)](#quick-start-no-docker)
3. [The API](#the-api)
4. [Running with Docker — dev vs production, explained simply](#running-with-docker--dev-vs-production-explained-simply)
5. [Retraining the models](#retraining-the-models)
6. [Reproducible rebuilds & the Python-version policy](#reproducible-rebuilds--the-python-version-policy)
7. [Repository size & Git LFS — do we need it?](#repository-size--git-lfs--do-we-need-it)
8. [Deploying to AWS ECS (Fargate)](#deploying-to-aws-ecs-fargate)
9. [Testing the deployed app](#testing-the-deployed-app)
10. [Troubleshooting](#troubleshooting)
11. [Changelog & design decisions log](#changelog--design-decisions-log)

---

## Project structure

```
app_car_prices_flask/
├── app.py                  # Flask routes: /, /health, /predict
├── car_model.py             # Feature engineering + prediction (no Flask imports)
├── wsgi.py                  # Entry point gunicorn uses: `gunicorn wsgi:app`
├── requirements.txt          # Runtime deps for the SERVED app only
├── Dockerfile                # PRODUCTION image (gunicorn)   -> use this for ECS
├── Dockerfile.dev            # DEV image (Flask's own server) -> local demos only
├── .dockerignore
├── .gitignore
├── .gitattributes            # Git LFS rules (see § Repository size)
├── models/                   # Pre-trained artifacts the API loads at start-up
│   ├── price_model.pkl        # RandomForestRegressor -> exact price (Lakhs)
│   ├── range_model.pkl         # RandomForestClassifier -> Low / Medium / High
│   ├── feature_columns.json    # Ordered column list the models expect
│   ├── range_config.json       # Price-band edges + labels
│   └── metadata.json           # make->model list, medians, valid option tables
├── training/                 # A SEPARATE workflow — see § Retraining the models
│   ├── train_model.py
│   ├── requirements.txt
│   └── data/cars24-car-price-cleaned-new.csv.gz  # gzip-compressed, ~81% smaller (see § Repository size)
└── tests/
    └── test_app.py            # `pytest tests/` — no server or Docker needed
```

**Why `training/` is split off from the rest:** building a model and serving
one are different jobs with different dependencies, different run frequency,
and different compute needs. `training/` needs pandas' full data-wrangling
stack and the raw CSV; the API never touches either. Keeping them apart means
the Docker image you actually deploy only ever contains what a request
handler needs — smaller image, faster build, smaller attack surface, and
nothing in it can accidentally depend on training-only code.

---

## Quick start (no Docker)

```bash
cd app_car_prices_flask
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

# Sanity check the prediction logic against the real trained models:
pip install pytest
pytest tests/ -v
# -> 8 passed (tests/test_app.py) + skipped guided stubs (tests/test_stubs.py)

# Run the dev server:
python app.py
```

> 🧪 **Learning to test & debug?** [`docs/TESTING_GUIDE.md`](docs/TESTING_GUIDE.md) walks you
> from "what is a test?" to mastery, and [`tests/test_stubs.py`](tests/test_stubs.py) is a
> graded ladder of guided exercises (including regression tests for the 405 and auto-fill
> behaviours) — complete each stub to turn it green.

Then, in another terminal:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000}'
```

You should get back something like:

```json
{
  "input_used": {
    "make": "MARUTI", "model": "SWIFT VDI", "age": 3.0, "km_driven": 45000.0,
    "mileage": 22.9, "engine": 1248.0, "max_power": 74.0,
    "seller": "Dealer", "fuel": "Diesel", "transmission": "Manual", "seats": "5"
  },
  "predicted_price_lakhs": 6.5,
  "predicted_price_display": "₹6.50 Lakhs",
  "price_range": {
    "label": "Medium", "low_lakhs": 3.99, "high_lakhs": 6.75,
    "display": "₹3.99 Lakhs - ₹6.75 Lakhs"
  }
}
```

(This exact example was run against the real models while building this
README — see [Quick start] above; it isn't a hypothetical.)

---

## The API

> 📗 **Usage recipes:** [`docs/API_EXAMPLES.md`](docs/API_EXAMPLES.md) has copy-paste examples
> for the `CarPriceModel` class, the REST endpoints (curl / requests), and the Flask test
> client — runnable in [`notebooks/api_examples.ipynb`](notebooks/api_examples.ipynb).

| Method | Path       | Purpose                                             |
| :----- | :--------- | :--------------------------------------------------- |
| `GET`  | `/`        | Service info + a usage example (human-friendly)      |
| `GET`  | `/health`  | `200` if models are loaded, `503` otherwise — wire this up as the ECS/ALB health check |
| `POST` | `/predict` | The prediction endpoint                              |

### `POST /predict`

**Headers:** `Content-Type: application/json`

**Required fields**

| Field   | Type   | Notes                                                       |
| :------ | :----- | :------------------------------------------------------------ |
| `make`  | string | e.g. `"MARUTI"`. Case-insensitive — normalised to uppercase. |
| `model` | string | e.g. `"SWIFT VDI"`. Must be a model that exists for that make. |

**Optional fields** — omit any of these and the API fills it in for you from
that specific model's real-world typical values (median engine/power/mileage,
most-common fuel/transmission/seat-count actually observed for that model in
the training data). This is the same "auto-fill" trick `app.py` uses in the
Streamlit app, so you can test with a two-field request and still get a
realistic prediction:

| Field          | Type          | Default source if omitted                  |
| :------------- | :------------ | :------------------------------------------ |
| `age`          | number (years)| Global median age                            |
| `km_driven`    | number         | Global median km driven                       |
| `mileage`      | number (km/l)  | That model's median mileage                    |
| `engine`       | number (cc)    | That model's median engine size                |
| `max_power`    | number (bhp)   | That model's median power                      |
| `seller`       | `"Dealer"` \| `"Individual"` \| `"Trustmark Dealer"` | `"Dealer"` |
| `fuel`         | `"Petrol"` \| `"Diesel"` \| `"CNG"` \| `"LPG"` \| `"Electric"` | Most common fuel for that model |
| `transmission` | `"Manual"` \| `"Automatic"`  | Most common for that model |
| `seats`        | `"5"` \| `"More than 5"` \| `"Fewer than 5"` | Most common for that model |

**Example — minimal request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"make": "HYUNDAI", "model": "I20 SPORTZ"}'
```

**Example — full request:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "make": "MARUTI", "model": "SWIFT VDI",
        "age": 5, "km_driven": 60000,
        "mileage": 22.0, "engine": 1248, "max_power": 74,
        "seller": "Individual", "fuel": "Diesel",
        "transmission": "Manual", "seats": "5"
      }'
```

**Errors** are always JSON with an `"error"` key and, where useful, a
`"hint"`:

- `400` — missing `make`/`model`, an unknown make/model, or an invalid option
  (e.g. `"fuel": "Hydrogen"`).
- `405` — wrong HTTP method (e.g. opening `/predict` in a browser, which sends
  `GET`). The response explains why, since this is the exact error the
  assignment description mentions running into.
- `503` — models failed to load on the server (check `/health` for detail).

---

## Running with Docker — dev vs production, explained simply

There are **two Dockerfiles on purpose**, because "run a Flask app" and "run a
Flask app the way a real deployment should" are different jobs with different
answers. If you're new to this, read this section before picking one.

### The short version

- **Learning / a quick local demo → `Dockerfile.dev`**
- **The AWS ECS assignment / anything resembling production → `Dockerfile`**

### Why two files instead of one?

`app.py` ends with:

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

That `app.run(...)` line starts **Flask's own built-in web server**. It's
great for development because it needs zero setup — but Flask's own
documentation is explicit that it "is not designed to be particularly
efficient, stable, or secure" for real traffic, and by default it can only
handle **one request at a time**. If two people (or your test script and a
grader's script) hit the app at the same moment, the second one simply waits.

**`gunicorn`** is a separate, production-grade program whose only job is to
run a Python web app correctly under real traffic: it starts several worker
*processes* of your Flask app and spreads incoming requests across them, so
concurrent requests don't queue up behind each other, and a worker that
crashes gets replaced instead of taking the whole service down.

So:

| | `Dockerfile.dev` | `Dockerfile` |
| :--- | :--- | :--- |
| Server | Flask's built-in dev server | `gunicorn` (2 worker processes) |
| Start command | `python app.py` | `gunicorn -w 2 -b 0.0.0.0:5000 wsgi:app` |
| Handles concurrent requests? | No (effectively one at a time) | Yes |
| Meant for | Learning, local demos | ECS / any real deployment |
| Extra concepts to understand | None | `wsgi.py`, "workers" |

`wsgi.py` exists only because `gunicorn` needs to be told *which* `app`
object to serve; it doesn't know about `app.py`'s `if __name__ == "__main__"`
convenience block, so `wsgi.py` just re-imports the same Flask object under
the name gunicorn expects (`wsgi:app` = "the `app` variable inside `wsgi.py`").

### Building each one

Both listen on the **same port, 5000**, and expose the **same routes** — the
only difference is the server underneath, so nothing about how you *test* the
app changes based on which one you pick.

```bash
# Production image (recommended, and what the ECS instructions below use):
docker build -t car-price-api .
docker run -p 5000:5000 car-price-api

# Dev image (simpler, single-threaded):
docker build -f Dockerfile.dev -t car-price-api:dev .
docker run -p 5000:5000 car-price-api:dev
```

Either way, test it with:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000}'
```

---

## Retraining the models

The `models/` folder already contains trained artifacts copied from the
companion Streamlit project, so **you do not need to do this to run or deploy the API.**
Retrain only if you want to change the dataset or hyperparameters.

```bash
cd training
python -m venv venv && venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
```

This reads `training/data/cars24-car-price-cleaned-new.csv.gz`, trains both
models, and writes fresh artifacts straight into `../models/` — the exact
files `app.py` loads at start-up. Notice this is a **completely separate
command, environment, and folder** from running the API: that's the "training
and serving are different workflows" point from the project structure section
made concrete. Nothing under `training/` is copied into the Docker image
(see `.dockerignore`).

**Why `.gz` and not a plain `.csv`:** `pandas.read_csv()` inspects the file
extension and transparently decompresses `.gz` (also `.zip`, `.bz2`, `.xz`,
`.zst`) in memory — this has been built into pandas for years, so
`train_model.py` doesn't call `gzip` anywhere itself; the line
`pd.read_csv(CONFIG["data_csv"])` in `load_dataset()` is unchanged from when
the file was a raw `.csv`. Compressing the dataset was therefore a
**zero-code-change, one-time step**: `gzip -k -9 cars24-car-price-cleaned-new.csv`.
See the next section for the actual size saved and why `.gz` was chosen over
a `.zip` archive.

### Inspecting the dataset file (it won't open in Excel — that's expected)

The dataset here is `cars24-car-price-cleaned-new.csv.gz` — a normal CSV
(comma-separated plain text) that has simply been **gzip-compressed**. Two
things newcomers trip over:

- **Double-clicking it does *not* open Excel.** On Windows, a bare `.csv` is
  associated with Excel, so clicking one opens it there. But this file's
  extension is **`.gz`**, so Windows hands it to an **archive tool**
  (7-Zip / WinRAR / the built-in extractor) instead. That's correct — it's a
  compressed archive, not a spreadsheet. You don't need to extract it: the API's
  training script reads it directly.
- **Excel mangles CSVs anyway.** Even the uncompressed form is risky to open in
  Excel and re-save: it strips leading zeros (`007` → `7`), turns long numbers
  into scientific notation (`9.19E+09`), and reinterprets values like `3-4` as
  dates. Treat Excel as a last-resort *viewer*, never an editor, for raw data.

**How to look inside safely** (no full extraction needed):

| Goal | Command |
| :--- | :------ |
| Peek at the first rows | `zcat training/data/…csv.gz \| head` (Git Bash), or `gzip -dc file.csv.gz \| head` |
| Load a sample in Python | `pd.read_csv("training/data/…csv.gz", nrows=5)` — pandas decompresses on the fly |
| Confirm the true format | `file training/data/…csv.gz` (reads the file's magic bytes, ignoring its name) |
| See real extensions in Explorer | View → **File name extensions** (Windows hides them, so `data.csv` may really be `data.csv.gz`) |

### Storage-format alternatives — measured on *this* dataset

Compressing the CSV to `.gz` is the *cheapest* size win (§ next section covers
`.gz` vs `.zip`), but it isn't the only option. Here's the fuller landscape,
written from the same 19,820 × 17 table; read time is the fastest of 5 pandas
reads on one machine (absolute numbers vary — the **ranking** is the point):

| Format | File size | vs CSV | Read speed | Human-readable? | Excel double-click? | Keeps column types? |
| :----- | --------: | -----: | ---------: | :-------------- | :------------------ | :------------------ |
| **CSV (raw)** | 1,535 KB | 100 % | 25 ms | ✅ plain text | ✅ yes | ❌ everything is text |
| **CSV + gzip** (`.csv.gz`) — shipped here | 290 KB | 19 % | 28 ms | ⚠️ after unzip | ❌ archive tool | ❌ |
| **CSV + xz / LZMA** (`.csv.xz`) | 163 KB | 11 % | 34 ms | ⚠️ after unzip | ❌ | ❌ |
| **CSV + bzip2** (`.csv.bz2`) | 151 KB | 10 % | 74 ms | ⚠️ after unzip | ❌ | ❌ |
| **Parquet** (snappy) | 274 KB | 18 % | **7 ms** | ❌ binary | ❌ needs tools | ✅ yes |
| **Parquet** (zstd) | 227 KB | 14 % | **7 ms** | ❌ binary | ❌ | ✅ yes |
| **Feather / Arrow** | 867 KB | 57 % | **5 ms** | ❌ binary | ❌ | ✅ yes |

**How to read it:**

- **Smallest on disk:** `CSV + bzip2` (10 %), but the **slowest to read** (~3× a
  raw CSV) — good for cold archival, poor for a file you load often.
- **Best size/speed balance for a git repo:** **`CSV + gzip`** (what this project
  ships) — 5× smaller than raw CSV, reads just as fast, one command to produce,
  and pandas reads it with zero code change. That combination is exactly why it
  was chosen here over the alternatives.
- **Best for real data pipelines:** **Parquet** — nearly as small as gzip **and
  ~3–4× faster to read**, because it's *columnar* and preserves each column's
  type (no dtype guessing on load). Needs `pyarrow`; it's the analytics default.
  If this dataset grew to millions of rows, Parquet would be the switch to make.
- **Fastest read, size no object:** **Feather/Arrow** — near-instant but ~3×
  larger than Parquet; best for short-lived local hand-offs, not for shipping.
- **Most universal & readable:** **raw CSV** — opens anywhere with no library;
  the price is size and lost type information.

> 🧭 **Rules of thumb:** *human sharing / tiny files* → **CSV**; *shrinking a CSV
> in a repo with zero friction* → **CSV + gzip** (this project); *a real
> analytics workflow or a large dataset* → **Parquet**; *maximum read speed for
> temporary local files* → **Feather**. Avoid **pickle** for datasets — fast,
> but it runs arbitrary code on load (a security risk) and breaks across
> library versions.

---

## Reproducible rebuilds & the Python-version policy

The API serves **pickled** scikit-learn models, and a pickle only loads under
the *same* library versions it was saved with — a newer scikit-learn raises
`InconsistentVersionWarning` and then an `AttributeError` at load time. The
[`Makefile`](Makefile) removes that risk by making the environment that
**trains** the models the very same one that **pins** the dependencies.

```bash
make rebuild      # fresh conda env -> retrain -> verify the models load -> re-pin
make push         # commit the retrained models + refreshed pins, then push
make help         # list every target
```

`make rebuild` runs `env -> train -> verify -> freeze`:

- **env** creates a conda env on **Python 3.11** and installs both the serving
  (`requirements.txt`) and training (`training/requirements.txt`) dependencies.
- **train** runs `training/train_model.py`, writing fresh `models/*.pkl`.
- **verify** loads both pickles to prove they unpickle cleanly.
- **freeze** calls [`tools/pin_env.py`](tools/pin_env.py) to rewrite *both*
  requirement files (comments preserved) and `.python-version` to the exact
  versions that just produced the models — so the pins can never drift from the
  artifacts, and neither can the `python:3.11-slim` Docker image.

### Why Python 3.11?

`numpy==1.26.x` / `scikit-learn==1.6.x` publish wheels only for **Python
3.9–3.12**. 3.11 has wheels for every pin *and* matches the Dockerfile base
image, so local, Docker and ECS all agree on one runtime.

> ⚠️ **Do not** pair the current pins with Python 3.13/3.14 — numpy 1.26 has no
> wheel there and the install fails outright. It's a contradiction, not a tweak.

### If you ever need Python 3.13+

Upgrade the whole stack and **retrain** (newer numpy 2.x ⇒ newer scikit-learn ⇒
new `.pkl` files):

```bash
# Drop the ==... pins on numpy/pandas/scikit-learn/joblib in BOTH requirement
# files (requirements.txt and training/requirements.txt), then:
make rebuild PY=3.13    # a fresh 3.13 env installs the latest compatible set
make freeze             # recapture the resolved versions + .python-version
```

Also bump the Dockerfile base image to `python:3.13-slim` to keep it in step.

---

## Repository size & Git LFS — do we need it?

Short answer: **not right now, but the repo is wired to handle it
automatically if that ever changes.** Here's the reasoning.

**Current sizes** (the largest files in this folder):

| File | Size |
| :--- | ---: |
| `models/price_model.pkl` | ~8.5 MB |
| `models/metadata.json` | ~1.1 MB |
| `models/range_model.pkl` | ~0.7 MB |
| `training/data/cars24-car-price-cleaned-new.csv.gz` | ~0.28 MB (was ~1.48 MB raw) |
| **Whole repo** | **~11 MB** |

**The dataset is stored gzip-compressed** (`cars24-car-price-cleaned-new.csv.gz`
instead of the raw `.csv`) purely to shrink the repo, verified on this exact
file:

| Form | Size | Notes |
| :--- | ---: | :--- |
| Raw `.csv` | 1,552,203 bytes | what used to be committed |
| `.gz` (gzip, level 9) | 295,568 bytes | **81.0% smaller** — what's committed now |
| `.zip` (DEFLATE, level 9) | 296,097 bytes | measured for comparison — same algorithm as gzip, so essentially identical size |

`.gz` was picked over `.zip` because the two compress equally well here (both
use DEFLATE) but `.gz` is a single-file, single-command trick
(`gzip -k -9 file.csv`) with **native pandas support** — `.zip` would work too,
but only cleanly for an archive containing exactly one file, and gains nothing
in return. Either way, no unzip step needed anywhere in this project: pandas
does it in memory as it reads.

**The models are saved with LZMA compression** (`joblib.dump(model, path,
compress=("lzma", 9))` in `training/train_model.py`) instead of joblib's
zlib default. Benchmarked directly on these two artifacts before switching
(all numbers below are real, not estimated):

| Backend | `price_model.pkl` | `range_model.pkl` | Load time (price model) |
| :--- | ---: | ---: | ---: |
| Uncompressed | 45.73 MB | — | 0.09 s |
| zlib, level 3 (old default) | 14.16 MB | 1.02 MB | 0.35 s |
| zlib, level 9 | 13.95 MB | — | 0.35 s |
| bz2, level 9 | 11.27 MB | 0.85 MB | 1.59 s |
| **lzma, level 9 (current)** | **8.92 MB** | **0.69 MB** | **0.67 s** |

LZMA won on *every* axis that matters for a served model: **37% smaller than
the old default**, and still loads in well under a second — actually faster
than bz2, which compresses less and loads slower. The only downside is a
much slower **save**, but that only happens once, inside `train_model.py`,
never at API start-up or request time, so it's free in practice.

Two things this did *not* require: no code changes to `car_model.py` or
`app.py` — `joblib.load()` auto-detects the compression backend from the
file's header — and no accuracy trade-off, since compression here is
lossless (the exact same fitted trees come back out).

**GitHub's actual limits**, per GitHub's own documentation:

- Git itself will warn you when adding/updating a file **larger than 50 MiB**.
- GitHub **blocks** any push containing a file **larger than 100 MiB** outright.
- Repositories are recommended to stay **under 1 GB** ("strongly recommended"
  under 5 GB).
- Files that need to exceed the 100 MiB limit **must** use Git LFS instead.

  Source: [GitHub Docs — About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

At ~11 MB total, this repo isn't close to the 50 MiB warning threshold, let
alone the 100 MiB hard block — so plain `git add`/`git commit` is genuinely
fine here. **Don't reach for Git LFS reflexively; it adds a dependency
(everyone who clones needs `git lfs` installed) and, on GitHub's free tier,
its own quota (currently 10 GB storage + 10 GB bandwidth/month per account —
see [GitHub Docs — About Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)
and [GitHub billing docs](https://docs.github.com/billing/managing-billing-for-git-large-file-storage/about-billing-for-git-large-file-storage))
that an ~11 MB repo doesn't need.**

**Why `.gitattributes` is still included, then:** it ships the LFS rules for
`*.pkl`, `*.csv`, and `*.csv.gz` **ready-written but commented out**, so today
everything commits as a normal git blob (correct at ~11 MB), yet the moment an
artifact ever grows toward the 100 MiB limit you just uncomment one line instead
of researching LFS from scratch. The policy is pre-decided in version control,
so no one has to reinvent it under pressure later. (This isn't a hypothetical
risk: these same random forests, when first trained during development, were
once **187 MB combined** — `price_model.pkl` 122 MB + `range_model.pkl` 65 MB —
before depth-bounding was added, comfortably over GitHub's 100 MiB single-file
limit.)

To activate it (only needed once a tracked file is large enough to matter),
uncomment the relevant line(s) in `.gitattributes`, then:

```bash
# One-time per machine:
git lfs install

# From the repo root:
git add .gitattributes <the-large-file>
git commit -m "Track large files with Git LFS"
```

**Other size-reduction measures already applied here**, worth keeping if you
retrain or grow the dataset further:

- The dataset is committed as `.csv.gz`, not raw `.csv` — **81% smaller**
  (1.48 MB → 0.28 MB) for zero code changes, since pandas decompresses it
  automatically on read (see above).
- `training/train_model.py` caps `max_depth=18` and `min_samples_leaf=4` on
  the random forests specifically to bound pickle size — the comments in that
  file note unbounded trees can otherwise reach 100+ MB with this dataset's
  ~3,200 one-hot columns.
- Model artifacts are saved with `joblib.dump(..., compress=("lzma", 9))` —
  **37% smaller than joblib's zlib default**, benchmarked above.

**Further, more invasive options** if the models ever need to shrink a lot
more than this (none applied here — each trades something away, so treat
these as things to evaluate deliberately, not defaults):

- **Fewer/shallower trees** (lower `n_estimators` or `max_depth` further in
  `train_model.py`) shrinks the pickle roughly linearly with tree count/depth,
  but directly trades away prediction accuracy — re-check `price_r2` /
  `range_accuracy` in the training output before adopting.
- **A more compact model family**, e.g. swapping `RandomForestRegressor` for
  `HistGradientBoostingRegressor`: histogram-based boosting typically needs
  far fewer, shallower trees for comparable accuracy, so its pickles are
  usually a fraction of an equivalent random forest's size. This is a real
  retrain + re-evaluate, not a drop-in change.
- **ONNX export** (`skl2onnx`) converts a fitted sklearn model into a compact,
  language-agnostic binary format, often smaller than pickle and faster to
  load — but it adds a new serving dependency (`onnxruntime`) and changes how
  `car_model.py` calls `.predict()`, so it's a deliberate architecture change,
  not a compression tweak.

None of this was strictly necessary to stay under GitHub's limits at the
current dataset size — it's cheap insurance for when the dataset or models
grow, so a future `git push` doesn't hit the 50 MiB warning or 100 MiB block
by surprise.

---

## Deploying to AWS ECS (Fargate)

> 🔰 **New to Docker/AWS?** There's a **complete-beginner, click-by-click runbook** in
> [`notebooks/deploy_docker_and_aws_ecs.ipynb`](notebooks/deploy_docker_and_aws_ecs.ipynb)
> — it builds & tests the container locally (with real, captured output) and then walks the
> **AWS Console** (ECR → ECS Fargate) step by step, plus cleanup and troubleshooting. The
> CLI section below is the terminal-first equivalent of that same flow.

This mirrors the assignment's steps (build & push to ECR → create a Fargate
cluster → register a task definition → run the task → hit its public IP on
port 5000), spelled out with exact commands. Console-based steps work
identically; the AWS CLI is used here because it's copy-pasteable and
reviewable.

### Prerequisites

- AWS account + [AWS CLI installed and configured](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
  (`aws configure`, with a user/role that can use ECR and ECS).
- Docker Desktop (or equivalent) running locally.
- This repo checked out locally with `docker build` already working (see
  above).

### 1. Build the image for the right CPU architecture

Fargate's default runtime platform is **linux/x86_64 (amd64)**. If you're
building on an Apple Silicon Mac (arm64) or any other arm64 machine, you must
explicitly target amd64 or the task will fail to start on Fargate:

```bash
docker build --platform linux/amd64 -t car-price-api .
```

On a Windows or Intel/AMD machine this is usually the default already, but
passing `--platform linux/amd64` explicitly costs nothing and avoids a
confusing failure later.

### 2. Create an ECR repository and push the image

```bash
# Pick your region; used consistently below.
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

aws ecr create-repository --repository-name car-price-api --region $AWS_REGION

aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker tag car-price-api:latest \
  $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/car-price-api:latest

docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/car-price-api:latest
```

(PowerShell users: replace `export VAR=value` with `$env:VAR = "value"` and
`$(...)` with `$(aws ... )` assigned via `= (aws ...)`; the `aws`/`docker`
commands themselves are unchanged.)

### 3. Create an ECS cluster (Fargate)

```bash
aws ecs create-cluster --cluster-name car-price-cluster --region $AWS_REGION
```

### 4. Register a task definition

Save this as `task-definition.json` (fill in your account ID/region), then
register it:

```json
{
  "family": "car-price-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "car-price-api",
      "image": "<ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/car-price-api:latest",
      "portMappings": [{ "containerPort": 5000, "protocol": "tcp" }],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/car-price-api",
          "awslogs-region": "<AWS_REGION>",
          "awslogs-create-group": "true",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole"
}
```

> `ecsTaskExecutionRole` is a role AWS creates for you the first time you use
> ECS through the console; if it doesn't exist yet, create it once via
> IAM → Roles → attach the AWS-managed `AmazonECSTaskExecutionRolePolicy`.

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION
```

### 5. Open port 5000 in a security group

The task needs a security group that allows **inbound TCP 5000**. Using your
default VPC's security group as an example:

```bash
export SG_ID=$(aws ec2 describe-security-groups \
  --filters Name=group-name,Values=default \
  --query "SecurityGroups[0].GroupId" --output text --region $AWS_REGION)

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID --protocol tcp --port 5000 --cidr 0.0.0.0/0 \
  --region $AWS_REGION
```

`0.0.0.0/0` (anyone) is fine for a short-lived class assignment you'll tear
down afterwards; for anything longer-lived, restrict the CIDR to your own IP
(`curl ifconfig.me` to find it) or your grader's known IP range instead.

### 6. Run the task

```bash
export SUBNET_ID=$(aws ec2 describe-subnets \
  --filters Name=default-for-az,Values=true \
  --query "Subnets[0].SubnetId" --output text --region $AWS_REGION)

aws ecs run-task \
  --cluster car-price-cluster \
  --task-definition car-price-api \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_ID],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region $AWS_REGION
```

Note the `taskArn` in the output (or fetch it again with
`aws ecs list-tasks --cluster car-price-cluster`).

### 7. Get the task's public IP

```bash
export TASK_ARN=$(aws ecs list-tasks --cluster car-price-cluster --query "taskArns[0]" --output text --region $AWS_REGION)

export ENI_ID=$(aws ecs describe-tasks --cluster car-price-cluster --tasks $TASK_ARN \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text --region $AWS_REGION)

aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID \
  --query "NetworkInterfaces[0].Association.PublicIp" --output text --region $AWS_REGION
```

That last command prints the public IP you'll test against.

---

## Testing the deployed app

Same requests as local testing, just against the ECS task's public IP instead
of `localhost`:

```bash
curl -X POST http://<PUBLIC_IP>:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000}'
```

**Using Postman instead:**

1. Method: `POST`
2. URL: `http://<PUBLIC_IP>:5000/predict`
3. Headers: `Content-Type: application/json`
4. Body → raw → JSON:
   ```json
   {"make": "MARUTI", "model": "SWIFT VDI", "age": 3, "km_driven": 45000}
   ```
5. Send, and confirm you get a `200` with a `predicted_price_lakhs` field.

Also worth checking:

```bash
curl http://<PUBLIC_IP>:5000/health     # expect {"status":"healthy"}
curl http://<PUBLIC_IP>:5000/           # expect usage info, not an error
```

If you open `http://<PUBLIC_IP>:5000/predict` in a plain browser, you'll get
a `405 Method Not Allowed` with an explanatory hint — that's expected: browsers
send `GET`, and `/predict` only accepts `POST`. This is the exact behaviour
described in the assignment ("Method Not Allowed" from a browser visit); it
means the deployment is working correctly, not that something is broken.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| `curl: (7) Failed to connect` | Security group doesn't allow inbound 5000, or task has no public IP | Re-check step 5 (security group) and that `assignPublicIp=ENABLED` was set in step 6 |
| Task stops immediately after starting | Wrong CPU architecture image, or app crashed | Rebuild with `--platform linux/amd64`; check logs: `aws logs tail /ecs/car-price-api --follow` |
| `405 Method Not Allowed` when testing in a browser | Expected — browsers send `GET`, `/predict` needs `POST` | Use `curl -X POST` or Postman, as shown above |
| `{"error":"Models are not loaded..."}` (`503`) | `models/` wasn't copied into the image, or `.dockerignore` excluded it | Confirm `COPY models/ ./models/` is in the `Dockerfile` you built and that the folder exists locally before `docker build` |
| `400` "Unknown make" / "Unknown model" | Typo, or a make/model not present in the training data | Check spelling/casing against `models/metadata.json`'s `makes_models` map (input is auto-uppercased, so casing itself isn't the issue) |
| `git push` rejected for a large file | A retrained `.pkl` exceeds GitHub's 100 MiB hard limit | Run `git lfs install && git lfs track "*.pkl"` (already declared in `.gitattributes`) before committing — see [Repository size & Git LFS](#repository-size--git-lfs--do-we-need-it) |

---

## Changelog & design decisions log

*Last updated: 2026-07-09.* A chronological record of what was built, what
alternatives were weighed, and what was actually measured — kept in one place
so a decision doesn't have to be re-litigated (or re-benchmarked) later. Each
entry links to the section with the full detail; this is the index.

1. **Created this app as a separate, self-contained sibling to the parent
   Streamlit project**, in `app_car_prices_flask/`, so it can be split into
   its own git repository later without carrying anything it doesn't need.
   See [Project structure](#project-structure).

2. **`POST /predict` input format** — three options were considered: (a) a
   full raw feature vector matching the model's internal one-hot schema
   exactly, (b) a "friendly" shorthand (just `make`/`model`, everything else
   auto-filled from that model's real median specs), or (c) supporting both.
   **Chose (b)** — it mirrors `app.py`'s own auto-fill design in the companion
   Streamlit project, needs no knowledge of the internal encoding to test, and is what
   the assignment's own example curl command assumes. See
   [The API](#the-api).

3. **API surface** — considered a minimal price-only `/predict` with no other
   routes vs. a fuller response (exact price *and* price band) plus a
   `/health` endpoint. **Chose the fuller option**: `/health` doubles as the
   ECS task health check, and returning both the price and its Low/Medium/High
   band costs nothing extra since both models are already loaded. See
   [The API](#the-api).

4. **Two Dockerfiles, not one** — `Dockerfile` (gunicorn, 2 workers) for
   production/ECS and `Dockerfile.dev` (Flask's own dev server) for local
   learning/demos, rather than picking a single "correct" one. Reasoning and
   a side-by-side comparison in
   [Running with Docker](#running-with-docker--dev-vs-production-explained-simply).

5. **`training/` split from the served app** — training needs pandas' full
   stack and the raw dataset; the API needs neither. Verified the split
   works by actually retraining end-to-end from `training/` and confirming
   the regenerated artifacts still pass all 8 tests in `tests/test_app.py`
   and serve correct predictions from a rebuilt Docker image. See
   [Retraining the models](#retraining-the-models).

6. **AWS ECS (Fargate) deployment guide written** — exact `aws` CLI commands
   for ECR push, cluster/task-definition creation, security group, running
   the task, and retrieving its public IP, plus a troubleshooting table for
   the specific failure modes (wrong CPU architecture, closed security-group
   port, browser-GET-vs-POST confusion) that this class assignment tends to
   surface. See [Deploying to AWS ECS](#deploying-to-aws-ecs-fargate).

7. **Dataset compression** — the training CSV was raw (`1,552,203` bytes).
   Compared `.gz` (gzip -9) against `.zip` (DEFLATE -9): both landed within
   0.2% of each other (295,568 vs 296,097 bytes) since they use the same
   compression algorithm. **Chose `.gz`**: identical ratio, but a one-command
   change with zero code changes, since `pandas.read_csv()` already
   auto-decompresses `.gz` by extension. Result: **81% smaller**
   (1.48 MB → 0.28 MB), verified byte-for-byte identical after decompression
   (`df.equals()` → `True`) and by retraining fully from the compressed file.
   See [Repository size & Git LFS](#repository-size--git-lfs--do-we-need-it).

8. **Model artifact compression** — benchmarked joblib's compression backends
   directly on the real trained models (not estimated): uncompressed, zlib
   (the old default, levels 3 and 9), bz2, lzma, and lz4, at multiple levels,
   measuring size *and* load time for each. **Chose LZMA level 9** — it won
   on every axis that matters for a served model (smallest *and* fastest to
   load, beating bz2 on both counts), for a cost that only lands once, during
   training, never at request time:
   - `price_model.pkl`: 14.16 MB → **8.92 MB** (−37%)
   - `range_model.pkl`: 1.02 MB → **0.69 MB** (−32%)

   No code changed outside `training/train_model.py`'s two `joblib.dump(...)`
   calls — `joblib.load()` auto-detects the compression backend from the
   file header. Full benchmark table in
   [Repository size & Git LFS](#repository-size--git-lfs--do-we-need-it).

9. **Git LFS — evaluated, not adopted (yet)** — checked this repo's actual
   size (~11 MB) against GitHub's documented limits (50 MiB warning, 100 MiB
   hard block, Git LFS's 10 GB/month free quota) and found LFS unnecessary at
   the current size. **Compromise adopted:** `.gitattributes` ships the LFS
   rules for `*.pkl`, `*.csv`, and `*.csv.gz` **commented out** (so everything
   commits as a plain git blob now), ready to uncomment the moment a file grows
   large enough to need them — informed
   directly by these same forests' own history earlier in this project's
   development, when they once hit 187 MB combined before depth-bounding was
   introduced (see the benchmark table in
   [Repository size & Git LFS](#repository-size--git-lfs--do-we-need-it)).

10. **Further compression options considered and explicitly deferred** —
    each would save more space but trades away something else, so none were
    applied without a explicit go-ahead: fewer/shallower trees (trades
    prediction accuracy), swapping `RandomForestRegressor` for
    `HistGradientBoostingRegressor` (real retrain + re-evaluation, not a
    drop-in change), and ONNX export via `skl2onnx` (adds an `onnxruntime`
    dependency and changes how `car_model.py` calls `.predict()`). Listed
    with reasoning in
    [Repository size & Git LFS](#repository-size--git-lfs--do-we-need-it).

---

### 🔗 The Car Prices Quartet

Four sibling projects built on the same Cars24 dataset:

- 🎛️ **[Streamlit web app →](https://github.com/kalyancn4u/app_car_prices_streamlit)** — interactive price-predictor UI
- 🐳 **Flask REST API** — containerised API (Docker + AWS ECS/Fargate) · _you are here_
- 🔬 **[MLOps lifecycle →](https://github.com/kalyancn4u/app_car_prices_mlops)** — full SDLC: notebooks → production pipeline
- 🛠️ **[Pipeline starter →](https://github.com/kalyancn4u/app_car_prices_pipeline)** — beginner-friendly guide + test stubs to extend
