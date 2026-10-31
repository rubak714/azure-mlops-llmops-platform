# LLM Learning Notes

My background is in machine learning - missing data imputation using deep generative models (GAN, VAE, DAE, AE) and statistical methods (MICE, KNN, mean imputation) on DHS health survey data for my thesis, plus supervised learning from university research and coursework. LLMs work differently in several important ways. I am writing down what I learn here as I go, so I can refer back to it and track how my understanding develops.

---

## 1. How LLMs are different from what I already know

In my thesis, the task was **missing data imputation**, which included estimating missing values in tabular health survey data using generative models and statistical baselines. The methods worked with structured numeric and categorical features. I trained them, evaluated on held-out missing values, and compared RMSE or MAE and other metrics across methods.

LLMs are different in three main ways.

**Input is natural language, not features.**
There are no features to engineer. Instead I write a prompt - a plain text instruction. The same model behaves very differently depending on how I phrase that instruction. Getting this right is called prompt engineering.

**I usually do not train from scratch.**
Models like GPT-4, Mistral, or Llama are already trained on billions of text samples. I can use them as they are, give them a few examples to guide their output, or fine-tune them on domain data. Fine-tuning is expensive - most real use cases use prompting instead.

**Evaluation is much harder.**
With imputation the evaluation was clear - how close are the imputed values to the real ones? With LLMs the output is text, and scoring text is still an open problem.

Common approaches are:

- ROUGE and BLEU: measure word overlap between generated and reference text. Simple but miss meaning.
- LLM-as-judge: ask a second LLM to score the output. Works surprisingly well but adds cost and another moving part.
- Human evaluation: still the most reliable, but slow and expensive.
- Task-specific metrics: for example faithfulness for summarization - did the model stick to the source text?

---

## 2. Words I kept seeing and had to look up

**Token.**
LLMs process text as tokens, not words. A token is roughly three quarters of a word on average. "Versicherung" might be two or three tokens. Models have a maximum number of tokens they can process in one call - this is the context window. Sending too much text causes an error or silent truncation.

**Temperature.**
Controls how random the output is. Temperature 0 gives the same output every time for the same input. Higher temperature gives more varied and creative output. For factual tasks like insurance claims I would want low temperature.

**Context window.**
Everything the model sees in one call - my instructions, any documents I pass in, the conversation history, and the question. The model has no memory between calls. If I want it to remember something from earlier I have to include it explicitly every time.

**RAG - Retrieval-Augmented Generation.**
Instead of fine-tuning a model on documents, I search for relevant documents at query time and include them in the prompt. Much cheaper than fine-tuning and easier to update when documents change. This is the standard approach for enterprise Q&A systems.

**Hallucination.**
LLMs sometimes produce confident, fluent, completely wrong answers. This is not a bug that gets fixed - it is a fundamental property of how they work. The main ways to reduce it are RAG (ground the answer in retrieved documents) and output validation.

---

## 3. What LLMOps adds on top of MLOps

MLOps - which I am also learning in this project - handles experiment tracking, model versioning, and automated pipelines.

LLMOps extends this for language models. The biggest difference is that the prompt is part of the system in the same way the model weights are. Changing a prompt changes behavior just like retraining would. So prompts need versioning and testing, the same way code does.

    MLOps concept          LLMOps equivalent
    ─────────────────────────────────────────────────────
    Model version          Prompt version + model version
    Accuracy and F1        ROUGE, faithfulness, toxicity
    Data drift             Topic drift, input distribution shift
    Retraining             Prompt iteration or fine-tuning
    Model registry         Prompt registry + model endpoint

---

## 4. Azure AI Foundry - what it is and why this JD mentions it

Azure AI Foundry is Microsoft's platform for deploying and managing LLM endpoints on Azure. It gives access to models like GPT-4, Mistral, Llama, and Phi through a single API, handles the infrastructure for serving them, and includes built-in tools for evaluation and content safety.

Why it matters for BarmeniaGothaer: they are building AI systems on Azure. AI Foundry is the natural place to deploy LLM endpoints in that environment. My previous repo (azure-data-platform-terraform) shows I can provision Azure resources with Terraform. Provisioning an AI Foundry project follows the same pattern.

I have not deployed to AI Foundry yet in this project. That is a next step once I am more confident with the LLM evaluation side.

---

## 5. What I am experimenting with in this repo

This section started as "two versions of a prompt in a folder". It is now a working prompt evaluation pipeline, and the gap between those two things is most of what I actually learned.

What is in the repo now:

- Four prompt versions in `llm_experiments/`, progressing from zero-shot through explicit constraints, few-shot with delimiters, to a full output contract
- `registry.json`, which records what changed in each version and why - the prompt equivalent of a model registry
- Eight evaluation texts with reference summaries I wrote by hand
- Four scoring metrics in `src/prompt_metrics.py`, combined into one composite
- A promotion gate at composite >= 0.70, mirroring the F1 >= 0.85 gate on the ML side
- Each prompt version logged as its own MLflow run, with the prompt text itself as the artifact

I still think the instinct in the original version of this section was right - understand the problem before reaching for tools that automate it. Writing ROUGE-L out by hand taught me more about what it does and does not measure than importing it would have, and the thing I got most out of this whole project came directly from that.

The write-up is in [docs/prompt_engineering.md](./docs/prompt_engineering.md).

### The three things that actually changed my mind

**A prompt is a specification, not an incantation.** I came in expecting prompt engineering to be about clever phrasing. It is about removing unstated assumptions. A model fills a gap in the instructions with something plausible rather than leaving it empty, so every gap I leave gets resolved for me, differently each time. One sentence - "do not add anything that is not in the original text" - moved the grounding score from 0.72 to 0.83. Nothing clever about it.

**A single metric will confidently rank a bad prompt first.** I had read that evaluating LLM output is hard and I thought I understood it. I did not understand it until ROUGE-L scored my worst prompt highest of four - 0.582, above every other version - because it wrote four sentences instead of two and more text means more overlap with the reference. That is in [errors_and_fixes.md](./docs/errors_and_fixes.md) as entry 11. It is the single most useful thing I got out of this project, and it changed how I read "we evaluate with ROUGE" in a paper.

**Knowing what your evaluation cannot see matters as much as the score.** My v3 and v4 prompts score identically even though v4 is clearly better. The offline simulator I built only detects constraints it can pattern match, so once a prompt pins length, grounding and format there is nothing left for it to measure. The temptation was to add a rule that would reward something only v4 has - which would be tuning the evaluation to produce a ranking I had already decided on. Writing the limitation down was the right answer and it was uncomfortable.

---

## 6. Things I do not fully understand yet

Updated from the original list, with what moved.

**Partly answered now:**

- *How to evaluate LLM output at scale without it becoming expensive.* I know more about the shape of this than I did. Cheap deterministic proxies - overlap, grounding by vocabulary, format and length compliance - catch a lot and cost nothing, and they are enough to gate the obvious regressions in CI. What they cannot catch is a summary that reuses source words to say something the source never supported, and catching that needs a second model reading the output. So the realistic answer looks like a layered one: cheap metrics on every run, expensive judging on a sample or on release candidates. I have not built the second layer.
- *How embedding drift detection works in practice.* I built distributional drift detection on tabular features in `src/detect_drift.py` - PSI and the two-sample KS test, per feature, with a threshold gate. That gave me the mechanism. Embedding drift is the same question asked of a 768-dimensional vector instead of four columns, and the part I still cannot picture is what "per feature" means when no individual dimension is interpretable. Per-dimension PSI over 768 dimensions would flag something constantly. I think the real approaches work on distances or cluster structure rather than per-dimension distributions, but I have not done it.

**Still open:**

- How to choose between RAG and fine-tuning for a specific use case. I understand the cost argument - retrieval is cheaper to run and far cheaper to update when documents change. What I cannot do yet is predict which one wins on quality for a given task.
- How AI Foundry and Databricks are meant to work together. Still have not used AI Foundry at all.
- How you decide a prompt is good enough to ship when the metric is a proxy. My 0.70 threshold has a derivation, and it is still a judgement call about a proxy measure. On the ML side F1 >= 0.85 means something concrete. I do not think there is a clean answer to this one, which is itself worth knowing.

**New questions this project created:**

- Prompt injection. Delimiters make the boundary between instruction and data explicit but do not enforce it. What does defence actually look like beyond "ask nicely in the prompt"?
- How prompt versions and model versions interact. A prompt tuned against one model is not guaranteed to be better on the next one, so an upgrade can silently regress every prompt at once. My registry versions the prompt and logs the provider, which records the pairing but does nothing about it.
- Whether drift should trigger retraining automatically. My drift check reports and alerts. Wiring it straight to `train.py` would be easy and I am not convinced it is right - retraining on drifted data can make a model worse, and the alert is asking a human to look at *why* the inputs moved.

---

*Started May 2026. Updated October 2026 after building the prompt evaluation and drift detection pipelines.*