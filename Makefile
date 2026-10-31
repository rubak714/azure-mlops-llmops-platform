# Shortcuts for the commands I actually run.
#
# The real reason this exists is error 1 and error 10 in
# docs/errors_and_fixes.md. Both were invocation mistakes rather than code
# mistakes - "pip" instead of "python -m pip", and "python src/x.py"
# instead of "python -m src.x". Neither is guessable by reading the file.
# Putting the correct invocation in one place means I cannot get it wrong
# twice, and anyone cloning this does not have to find out the hard way.
#
# On Windows these need make installed - via Chocolatey, Git Bash, or WSL.
# Every recipe is a single command that can also be copied and pasted
# straight into a terminal, which is deliberate.

.PHONY: help install test lint format train evaluate register \
        prompts prompt-eval injection drift serve mlflow ci clean

help:
	@echo "Setup"
	@echo "  make install      install dependencies"
	@echo ""
	@echo "Checks"
	@echo "  make test         run the full test suite"
	@echo "  make lint         run flake8 with the same flags CI uses"
	@echo "  make ci           lint and test, what the PR pipeline runs"
	@echo ""
	@echo "Model pipeline"
	@echo "  make train        train and log runs to MLflow"
	@echo "  make evaluate     check the best run against the F1 threshold"
	@echo "  make register     promote the best run to the registry"
	@echo "  make serve        serve the promoted model on :8000"
	@echo ""
	@echo "Prompt pipeline"
	@echo "  make prompts      list the prompt versions and what changed"
	@echo "  make prompt-eval  score every prompt version, apply the gate"
	@echo "  make injection    check prompt boundary integrity"
	@echo ""
	@echo "Monitoring"
	@echo "  make drift        run the drift scenarios"
	@echo "  make mlflow       open the MLflow UI on :5000"

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=100 --extend-ignore=E501

format:
	autopep8 --in-place --aggressive --recursive src/ tests/

ci: lint test

train:
	python src/train.py

evaluate:
	python src/evaluate.py

register:
	python src/register_model.py

serve:
	uvicorn src.serve:app --reload

prompts:
	python -m src.prompt_registry

prompt-eval:
	python -m src.evaluate_prompts

injection:
	python -m src.check_injection

drift:
	python -m src.detect_drift

mlflow:
	mlflow ui

clean:
	python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"
