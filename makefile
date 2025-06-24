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

run:                                              ## python src/main.py
	pyenv exec python $(MAIN)

lint:                                             ## flake8 .
	pyenv exec flake8 src/

fmt:                                              ## black .
	pyenv exec black src/

clean:                                            ## Delete venv & .python-version
	@pyenv virtualenv-delete -f $(VENV_NAME) || true
