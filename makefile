.PHONY: setup venv install lint test fmt clean forecast ingest-historical

PYTHON_VERSION ?= 3.11.4
VENV_NAME      ?= braze-extractor-env
REQUIREMENTS   ?= requirements.txt

setup: venv install

venv:
	pyenv install --skip-existing $(PYTHON_VERSION)
	pyenv exec python -m venv $(VENV_NAME)
	@if [ -d "$(VENV_NAME)" ]; then \
		rm -rf $(VENV_NAME); \
	fi

install: venv
	pyenv exec python -m pip install --upgrade pip
	pyenv exec pip install -r $(REQUIREMENTS)

lint:
	pyenv exec flake8 src/

test:
	pyenv exec python -m pytest tests/ -v

test-cov:
	pyenv exec python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

fmt:
	pyenv exec black src/ tests/

clean:
	@pyenv virtualenv-delete -f $(VENV_NAME) || true

forecast:
	pyenv exec python -m src.forecast_quiet_dates --forecast-only --filter-prefix "transactional"

ingest-historical:
	pyenv exec python -m src.ingest_historical --days 90 --filter-prefix "transactional"
