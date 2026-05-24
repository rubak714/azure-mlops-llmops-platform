# Azure MLOps and LLMOps Platform

I have been working with Terraform since last year through assigned company tasks covering GCP, Kubernetes, and Helm. More recently I built a full Azure data platform from scratch using Databricks, Key Vault, and ADLS Gen2. I also have an ML background from my thesis and university research projects.

**This repo is the next step:** I am implementing here what I know about infrastructure and ML and learning how to operationalize models and LLMs properly on Azure. The focus is MLflow experiment tracking, a basic model promotion pipeline and my first hands-on work with LLMOps patterns.

I am documenting what I learn as I go, including a dedicated LLM_LEARNING.md for the LLM concepts that are new to me.

## What this builds

- A scikit-learn training pipeline with MLflow experiment tracking
- GitHub Actions CI that runs training and evaluation on every push
- Basic prompt versioning for a text summarization task
- Unit tests for the evaluation logic

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

## Related repo

[azure-data-platform-terraform](https://github.com/rubak714/azure-data-platform-terraform)
- the Terraform infrastructure layer this builds on top of.

## Status

Work in progress. The repo is being built incrementally.
