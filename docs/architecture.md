# Architecture

A description of what this project builds, how the pieces fit together and why I made certain decisions along the way.

---

## Overview

This project adds an MLOps layer on top of the Azure data platform I built in my previous repo (azure-data-platform-terraform). That repo provisioned the infrastructure - Databricks workspace, ADLS Gen2 storage,Key Vault, and networking. This repo focuses on what runs on top of that infrastructure: a training pipeline, experiment tracking, and model promotion logic.

For now everything runs locally and on GitHub Actions. There is no live Azure deployment in this repo - the previous repo already demonstrates that I can provision and deploy Azure resources with Terraform. The focus here is the ML pipeline itself.

---

## How the pieces fit together

    src/train.py
        trains a RandomForestClassifier on Iris
        logs parameters and metrics to MLflow
        saves the model as an artifact in the run
            |
            v
    src/evaluate.py
        finds the best run by F1 score
        checks whether it meets the threshold (0.85)
        prints a clear pass or fail result
            |
            v
    src/register_model.py
        runs the same threshold check
        if the model passes, registers it in the MLflow model registry
        if it fails, exits without registering anything
            |
            v
    .github/workflows/train-evaluate.yml  
        runs train.py and evaluate.py automatically on every push
        so I can see immediately if a change breaks the pipeline

---

## Why RandomForest and not something else

I used RandomForest because I know it well from my thesis and research projects. The goal of this repo is to learn the MLflow and pipeline side of things, not to experiment with new model architectures. Using a familiar model meant I could focus on wiring up the tracking correctly without debugging model issues at the same time.

---

## Why a promotion threshold

In my thesis I tracked model performance in notebooks and compared runs manually. That works for one-off experiments but breaks down when you want to automate things. A threshold makes the promotion decision explicit and automatic - the pipeline either passes or fails, with no manual check needed.

0.85 is a reasonable minimum for a classifier on a clean dataset like Iris. On a real dataset with noise and class imbalance the threshold would need more thought.

---

## Why local MLflow and not Databricks MLflow

The Databricks workspace I provisioned in my previous repo costs money to keep running. I destroyed it after verifying the infrastructure worked. For this project I am running MLflow locally - the tracking code is identical either way, the only difference is the tracking URI. Pointing it at a Databricks workspace would be a one-line change.

---

## What is missing compared to a production setup

- No connection to a real data source - Iris is a stand-in
- No retraining trigger - training is kicked off manually or on push
- No serving layer - the registered model is not deployed to an endpoint
- No drift detection - that is planned for a later stage of this project
- MLflow runs locally - in production this would point to Databricks or Azure ML

These are honest gaps for a learning project. The structure of the pipeline - experiment tracking, threshold gates, registry promotion, CI integration - transfers directly to a production setup.

---

## Related repo

[azure-data-platform-terraform](https://github.com/rubak714/azure-data-platform-terraform) - the infrastructure layer this builds on. Covers Terraform modules for Databricks, ADLS Gen2, Key Vault, and VNet on Azure.