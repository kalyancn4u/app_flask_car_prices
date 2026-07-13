# =============================================================================
# Car Prices (Flask API) - clean rebuild & deploy pipeline
#
#   Fresh conda env -> retrain models -> verify they load -> pin versions -> ship
#
# Same discipline as the Streamlit app: the environment that TRAINS the models
# also writes requirements.txt + .python-version, so the Docker image (and any
# other target) installs the exact versions the .pkl files were saved with and
# can unpickle them without a scikit-learn version-mismatch crash.
#
# The serving image is python:3.11-slim (see Dockerfile), so PY defaults to 3.11.
# Run from a shell where `conda` works (Anaconda Prompt / Git Bash).
# Needs GNU make:  conda install -c conda-forge make
# =============================================================================

# --- Python version policy (why 3.11) ----------------------------------------
# The models are pinned to numpy 1.26.x / scikit-learn 1.6.x, and numpy 1.26
# only ships wheels for Python 3.9-3.12. So this project stays on 3.11, which
# also matches the Dockerfile base image (python:3.11-slim) -> every pin
# installs from a prebuilt wheel and the committed models load unchanged.
#
# Do NOT pair these pins with Python 3.13/3.14: no numpy 1.26 wheel exists there,
# so the install fails outright. To move to 3.13+ you must upgrade the stack AND
# retrain (unpin numpy/pandas/scikit-learn, `make rebuild PY=3.13`, then
# `make freeze`). See README -> "Reproducible rebuilds & the Python-version
# policy".
# -----------------------------------------------------------------------------

ENV   ?= car-flask       # conda env name
PY    ?= 3.11            # Python for the env; matches Dockerfile python:3.11-slim
CONDA ?= conda           # conda executable
IMAGE ?= car-price-api   # docker image tag

RUN := $(CONDA) run -n $(ENV) --no-capture-output

.DEFAULT_GOAL := help
.PHONY: help env train verify freeze rebuild test docker-build docker-run push clean

help:  ## Show this help
	@echo "Targets:"
	@echo "  make env          - create conda env '$(ENV)' (Python $(PY)) + serve & train deps"
	@echo "  make train        - retrain models (training/train_model.py) -> models/*.pkl"
	@echo "  make verify       - load the saved models to confirm they unpickle"
	@echo "  make freeze       - pin requirements.txt + training/requirements.txt + .python-version"
	@echo "  make rebuild      - env + train + verify + freeze (full clean run)"
	@echo "  make test         - run the test suite"
	@echo "  make docker-build - build the production image ($(IMAGE))"
	@echo "  make docker-run   - run the image locally on http://localhost:5000"
	@echo "  make push         - commit retrained models + pins and push"
	@echo "  make clean        - delete the conda env"
	@echo ""
	@echo "Override e.g.:  make rebuild PY=3.12 ENV=flask312"

env:  ## Create env and install serving + training dependencies
	$(CONDA) create -y -n $(ENV) python=$(PY)
	$(RUN) python -m pip install --upgrade pip
	$(RUN) python -m pip install -r requirements.txt -r training/requirements.txt

train:  ## Retrain models (runs in training/, writes ../models/*.pkl)
	cd training && $(RUN) python train_model.py

verify:  ## Confirm the freshly-saved models unpickle
	$(RUN) python -c "import joblib; joblib.load('models/price_model.pkl'); joblib.load('models/range_model.pkl'); print('OK: both models load cleanly')"

freeze:  ## Pin both requirement files + .python-version to this env
	$(RUN) python tools/pin_env.py requirements.txt training/requirements.txt

rebuild: env train verify freeze  ## Full clean run: env -> train -> verify -> pin
	@echo ""
	@echo "Rebuild complete. Review 'git diff', then 'make push' (or push manually)."
	@echo "Keep the Dockerfile base image (python:$(PY)-slim) == .python-version."

test:  ## Run the test suite
	$(RUN) python -m pytest -q

docker-build:  ## Build the production Docker image
	docker build -t $(IMAGE) .

docker-run:  ## Run the image locally (http://localhost:5000)
	docker run --rm -p 5000:5000 $(IMAGE)

push:  ## Commit retrained models + regenerated pins and push
	git add models/ requirements.txt training/requirements.txt .python-version
	git commit -m "Rebuild models and re-pin environment (Python $(PY))"
	git push origin HEAD

clean:  ## Remove the conda env
	-$(CONDA) env remove -y -n $(ENV)
