# Azure MLOps and LLMOps Platform

I have been working with Terraform since last year through assigned company tasks covering GCP, Kubernetes, and Helm. More recently I built a full Azure data platform from scratch using Databricks, Key Vault, and ADLS Gen2. I also have an ML background from my thesis on missing data imputation using GANs, VAEs, DAEs, and statistical methods on DHS health survey data, plus supervised learning from university research and coursework.

**This repo is the next step:** taking what I know about infrastructure and ML and learning how to operationalize models and LLMs properly on Azure. The focus is MLflow experiment tracking, a threshold-based model promotion pipeline, GitHub Actions CI, and my first hands-on work with LLMOps patterns.

I am documenting what I learn as I go, including a dedicated [LLM_LEARNING.md](./LLM_LEARNING.md) for the LLM concepts that are new to me.

## What this builds

- A scikit-learn training pipeline with MLflow experiment tracking across four parameter combinations including one deliberately weak run
- A threshold-based model promotion gate - `evaluate.py` checks the best run against F1 ≥ 0.85 before `register_model.py` promotes it to the MLflow registry
- GitHub Actions CI that runs flake8 and pytest on every PR and runs training and evaluation on every merge to main
- Basic prompt versioning for a text summarization task with notes on what changed between versions
- Unit tests covering threshold logic, boundary cases, data split sizes, and label alignment
- 7 real errors documented in [docs/errors_and_fixes.md](./docs/errors_and_fixes.md) as they happened

## Dataset

This project uses the **Iris dataset** from scikit-learn's built-in datasets (`sklearn.datasets.load_iris`).

**Why Iris:**
No external files, no downloads, no storage setup needed. It loads in one line and works offline. The dataset is a standard benchmark for classification pipelines, which makes it easy for me to focus on what this project is actually about > the MLflow tracking, evaluation logic and CI pipeline rather than data wrangling.

**What it contains:**
150 samples of iris flowers across 3 species (setosa, versicolor, virginica). Each sample has 4 numeric features: sepal length, sepal width, petal length, and petal width. The task is to predict the species from the measurements.

**What will be investigated:**

- Does changing the number of trees and tree depth improve the model's F1 score?
- Does the model reach the minimum F1 score (0.85) needed to be saved to the registry?
- What happens in the pipeline when a model is too weak to be promoted?

## How to run locally

```bash
git clone https://github.com/rubak714/azure-mlops-llmops-platform.git
cd azure-mlops-llmops-platform
python -m pip install -r requirements.txt

# run training -logs 4 runs to MLflow
python src/train.py

# check which run passes the threshold
python src/evaluate.py

# register the best run in the MLflow model registry
python src/register_model.py

# open the MLflow UI to see all runs
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser. Go to the `iris-random-forest` experiment in the left sidebar to see all 4 runs with their parameters and metrics.

## Related repo

[azure-data-platform-terraform](https://github.com/rubak714/azure-data-platform-terraform)
- the Terraform infrastructure layer this builds on top of.

## Status

Work in progress. The repo is being built incrementally.
