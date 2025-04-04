#!/usr/bin/env bash


# black for python
if ! command -v black; then
    echo 'black could not be found, install with "pip install black"'
else
  black \
    --target-version=py312 \
    --exclude venv \
    .
fi

# isort sorts python imports
if ! command -v isort; then
    echo 'isort could not be found, install with "pip install isort"'
else
    isort --profile black --skip venv .
fi

# pycln removes unused python imports
if ! command -v pycln; then
    echo 'pycln could not be found, install with "pip install pycln"'
else
    pycln -a --exclude venv .
fi
