.PHONY: setup venv deps activate run lint test fmt clean

PYTHON_VERSION ?= 3.11.4                          # bump when you’re ready
VENV_NAME      ?= braze-extractor-env
REQUIREMENTS   ?= requirements.txt
MAIN           ?= src/main.py                     # entry-point script

setup: venv deps                                  ## Do everything once

venv:                                             ## Install Python & create venv once
	pyenv install --skip-existing $(PYTHON_VERSION)
	pyenv exec python -m venv $(VENV_NAME)
	@if [ -d "$(VENV_NAME)" ]; then \
		rm -rf $(VENV_NAME); \
	fi

install: venv
	pyenv exec python -m pip install --upgrade pip
	pyenv exec pip install -r $(REQUIREMENTS)

activate:                                         ## Print 'export PYENV_VERSION=…'
	@echo "Run this to activate the venv:" && \
	 echo "  export PYENV_VERSION=$(VENV_NAME)"

run:                                              ## python -m src.main
	pyenv exec python -m src.main

lint:                                             ## flake8 .
	pyenv exec flake8 src/

test:                                             ## Run unit tests
	pyenv exec python -m pytest tests/ -v

test-cov:                                         ## Run tests with coverage
	pyenv exec python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

fmt:                                              ## black .
	pyenv exec black src/

clean:                                            ## Delete venv & .python-version
	@pyenv virtualenv-delete -f $(VENV_NAME) || true

# Time-series forecasting targets
ingest:                                           ## Ingest daily Canvas data
	pyenv exec python src/ingest_daily.py

forecast:                                         ## Generate forecasts from existing data
	pyenv exec python src/forecast_quiet_dates.py --forecast-only

forecast-full:                                    ## Full pipeline: ingest data and forecast
	pyenv exec python src/forecast_quiet_dates.py

sample-data:                                      ## Create sample data for testing
	pyenv exec python src/forecast_quiet_dates.py --create-sample-data

ingest-historical:                                ## Ingest last 60 days of data
	pyenv exec python src/ingest_historical.py --days 60
