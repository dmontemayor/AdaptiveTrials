#! /bin/bash
if [ -d ".pytest_cache" ]; then
  rm -rf .pytest_cache
fi
if [ -d "adaptivetrials/tests/__pycache__" ]; then
  rm -rf adaptivetrials/tests/__pycache__
fi
pylint adaptivetrials
pytest
