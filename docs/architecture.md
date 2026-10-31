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

There are now three pipelines in this repo, not one. They share the same
shape deliberately - produce candidates, score them, gate the best one on
a threshold, log everything to MLflow - because that shape is the part
that transfers.

---

## The prompt pipeline (LLMOps)

    llm_experiments/registry.json
        four prompt versions, each with its technique and what changed
            |
            v
    src/prompt_registry.py
        loads a version, substitutes the input text
        fails loudly if a prompt is missing its {text} placeholder
            |
            v
    src/llm_client.py
        sends the rendered prompt to the active provider
        offline stub by default, Azure OpenAI when AZURE_OPENAI_* is set
            |
            v
    src/prompt_metrics.py
        scores the output on four separate measures
        ROUGE-L, grounding, length compliance, format compliance
            |
            v
    src/evaluate_prompts.py
        runs every version over all 8 evaluation examples
        logs each version as its own MLflow run, prompt text as artifact
        best version by composite score, gated at 0.70
            |
            v
    the same decision evaluate.py makes, one level up

The thing that took me longest to accept about LLMOps is why this is
needed at all. A prompt is a text file - it felt like configuration, not
like part of the model. But editing a prompt changes the system's
behaviour exactly the way retraining does, and an untracked change that
alters behaviour is the problem that MLflow exists to solve. So the prompt
gets a version, a recorded score, and a gate, for the same reasons the
model does.

---

## The drift pipeline

    src/detect_drift.py
        reference window: the distribution the model was trained on
        current window:   what is arriving now
            |
            v
        per feature, two measures
            PSI - how much probability mass moved between bins
            KS  - largest gap between the two cumulative distributions
            |
            v
        gate at PSI >= 0.25, per feature
        logs every feature's PSI and KS to MLflow
            |
            v
    .github/workflows/drift-check.yml
        runs on push to main, on demand, and on a Monday cron

This one answers a different question from the other two. `evaluate.py`
asks whether the model is good on the data it was trained on. Drift
detection asks whether that data still resembles the data arriving now. A
model can hold a perfect F1 on its own test split for a year while the
live inputs move away underneath it, and until this existed nothing in the
project would have noticed. Model performance was monitored; model inputs
were not.

---

## Why RandomForest and not something else

I used RandomForest because I know it well from my thesis and research projects. The goal of this repo is to learn the MLflow and pipeline side of things, not to experiment with new model architectures. Using a familiar model meant I could focus on wiring up the tracking correctly without debugging model issues at the same time.

---

## Why a promotion threshold

In my thesis I tracked model performance in notebooks and compared runs manually. That works for one-off experiments but breaks down when you want to automate things. A threshold makes the promotion decision explicit and automatic - the pipeline either passes or fails, with no manual check needed.

0.85 is a reasonable minimum for a classifier on a clean dataset like Iris. On a real dataset with noise and class imbalance the threshold would need more thought.

---

## Why two different thresholds

F1 >= 0.85 gates the model. Composite >= 0.70 gates the prompt. The gap is
deliberate.

F1 on a clean classification task can genuinely reach 1.0, so demanding
0.85 is demanding most of the available range. Text overlap metrics
cannot. A summary scoring 1.0 on ROUGE-L would have reproduced my
reference wording exactly, which is not the goal and would not even be
desirable - there are many correct ways to word the same summary, and the
metric penalises all but one of them.

The derivation for 0.70: grounding, length and format carry 0.60 of the
composite between them, so a prompt that is merely obedient already
scores 0.60. The threshold has to sit above that, or it would promote a
prompt that follows every instruction and bears no relationship to the
reference summary.

Two thresholds that look inconsistent, measuring two things with
different achievable ranges. Writing down why is the only thing that
stops the lower one looking like a lowered bar.

---

## Why PSI and KS rather than one of them

They fail differently.

PSI bins the reference distribution, so a shift that happens to stay
inside the existing bin boundaries is invisible to it. The KS statistic
makes no assumption about shape and catches that case.

KS gives a yes/no answer against a critical value, which is statistically
honest but hard to act on - it has no notion of "a bit of drift". PSI has
conventional interpretation bands (0.10 and 0.25) that practitioners
already agree on, which is what makes it usable as an automated gate. I
did not have to invent a threshold and then defend it.

Both are computed in numpy rather than imported from scipy. Partly so the
maths is readable in a repo whose purpose is learning, partly so the
script adds no dependency that `train.py` did not already have. The same
reasoning sits behind writing out ROUGE-L and the LCS by hand in
`prompt_metrics.py`.

Bin edges come from the reference window's quantiles rather than the
combined data. This is the one decision in the drift code I would call
non-obvious: the reference is the baseline the model was trained on, so it
defines what normal means. Deriving bin edges from both windows together
would let drift partially hide by moving the goalposts along with it.

---

## Why an offline LLM stub, and what it costs

The default provider in `llm_client.py` is not a language model. It is a
deterministic extractive summarizer that reads the constraints out of a
prompt and obeys them, and deliberately misbehaves - inventing a closing
sentence, opening with a preamble - when the prompt does not forbid it.

The alternative designs were worse. Requiring a live Azure OpenAI
deployment would mean nobody can run this repo without an Azure
subscription and a bill, and CI could not run the evaluation at all
without secrets. Skipping prompt evaluation entirely would leave the
LLMOps half of this project as four text files and no measurement.

What it costs is real and I would rather state it than bury it: the
numbers measure how precisely each prompt states its requirements, not
output quality. The clearest evidence is that v3 and v4 score identically
even though v4 is the better prompt - once length, grounding and format
are pinned, the stub has nothing left to detect. That is written up as
error 12 in [errors_and_fixes.md](./errors_and_fixes.md).

Same trade as running MLflow locally instead of on Databricks, below. The
structure is correct and the backend is a stand-in.

---

## Why local MLflow and not Databricks MLflow

The Databricks workspace I provisioned in my previous repo costs money to keep running. I destroyed it after verifying the infrastructure worked. For this project I am running MLflow locally - the tracking code is identical either way, the only difference is the tracking URI. Pointing it at a Databricks workspace would be a one-line change.

---

## What is missing compared to a production setup

Closed since the first version of this document:

- ~~No drift detection~~ - `detect_drift.py` now checks PSI and KS per
  feature on a schedule
- ~~No LLM integration~~ - `llm_client.py` has a real Azure OpenAI
  adapter, though see the caveat below

Still open, honestly:

- **No connection to a real data source.** Iris is a stand-in on the ML
  side and eight texts I wrote myself are a stand-in on the LLM side.
- **No live LLM endpoint.** The Azure OpenAI adapter is written and its
  configuration and failure paths are tested, but it has never been run
  against a provisioned deployment - no Azure OpenAI resource, for the
  same cost reason the Databricks workspace was destroyed. Every number
  in the prompt docs comes from the offline stub.
- **No serving layer.** The registered model is not deployed behind an
  endpoint. This is now the largest remaining gap - the model gets
  promoted to the registry and then nothing consumes it.
- **No retraining trigger.** The drift check reports and alerts; it does
  not kick off `train.py`. Wiring drift to retraining is the obvious next
  step and is a bigger decision than it looks, because automatic
  retraining on drifted data can make a model worse rather than better.
- **No real drift data.** The four scenarios in `detect_drift.py` are
  constructed in a function, not observed. The detector is real; the
  drift is simulated.
- **MLflow runs locally.** In production this would point at Databricks
  or Azure ML. One line of configuration, but worth saying.
- **No LLM-as-judge.** Grounding by word overlap cannot catch a summary
  that reuses source vocabulary to say something the source never
  supported.

These are honest gaps for a learning project, and I would rather list them
than have someone find them. The structure of all three pipelines -
experiment tracking, threshold gates, registry promotion, scheduled
monitoring, CI integration - transfers directly to a production setup. The
backends behind that structure are stand-ins and every one of them is a
configuration change rather than a redesign.

---

## Related repo

[azure-data-platform-terraform](https://github.com/rubak714/azure-data-platform-terraform) - the infrastructure layer this builds on. Covers Terraform modules for Databricks, ADLS Gen2, Key Vault, and VNet on Azure.