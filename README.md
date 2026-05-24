# Azure MLOps and LLMOps Platform

I have been working with Terraform since last year through assigned company tasks covering GCP, Kubernetes, and Helm. More recently I built a full Azure data platform from scratch using Databricks, Key Vault, and ADLS Gen2. I also have an ML background from my thesis and university research projects.

**This repo is the next step:** I am implementing here what I know about infrastructure and ML and learning how to operationalize models and LLMs properly on Azure. The focus is MLflow experiment tracking, a basic model promotion pipeline and my first hands-on work with LLMOps patterns.

I am documenting what I learn as I go, including a dedicated LLM_LEARNING.md for the LLM concepts that are new to me.

## What this builds

- A scikit-learn training pipeline with MLflow experiment tracking
- GitHub Actions CI that runs training and evaluation on every push
- Basic prompt versioning for a text summarization task
- Unit tests for the evaluation logic

## Related repo

[azure-data-platform-terraform](https://github.com/rubak714/azure-data-platform-terraform)
- the Terraform infrastructure layer this builds on top of.

## Status

Work in progress. The repo is being built incrementally.
