.PHONY: setup venv install lint test fmt clean forecast ingest-historical typecheck check visualization-save fmt-fix clean-imports auto-fix check-fix

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

typecheck:
	pyenv exec mypy src/

check: lint typecheck test

test:
	pyenv exec python -m pytest tests/ -v

test-cov:
	pyenv exec python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

fmt:
	pyenv exec black src/ tests/

# Auto-fix formatting issues
fmt-fix:
	pyenv exec black src/ tests/
	pyenv exec isort src/ tests/

# Auto-fix import and unused code issues
clean-imports:
	pyenv exec autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src/ tests/

# Auto-fix all automatically fixable issues
auto-fix: fmt-fix clean-imports

# Enhanced check that includes auto-fixing
check-fix: auto-fix check

clean:
	@pyenv virtualenv-delete -f $(VENV_NAME) || true

forecast:
	pyenv exec python -m src.forecast_quiet_dates --filter-prefix "transactional"

ingest-historical:
	pyenv exec python -m src.ingest_historical --days 90 --filter-contains "1 day" "1 week" "3 hr"

visualization-save:
	pyenv exec python -m src.visualization.main --multi-model --output plots/ --no-display --filter-contains "1 day" "1 week" "3 hr"
