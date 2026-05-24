# LLM Learning Notes

My background is in machine learning - missing data imputation using deep generative models (GAN, VAE, DAE, AE) and statistical methods (MICE, KNN, mean imputation) on DHS health survey data for my thesis, plus supervised learning from university research and coursework. LLMs work differently in several important ways. I am writing down what I learn here as I goo, so I can refer back to it and track how my understanding develops.

---

## 1. How LLMs are different from what I already know

In my thesis, the task was **missing data mputation** , which included estimating missing values in tabular health survey data using generative models and statistical baselines. The methods worked with structured numeric and categorical features. I trained them, evaluated on held-out missing values, and compared RMSE or MAE and other metrics across methods.

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

In the llm_experiments folder I am keeping two versions of a prompt for a simple text summarization task. The goal is to understand how much the output changes when I reword the prompt, and to start building the habit of treating prompts as versioned artifacts rather than throwaway text.

This is basic compared to a real LLMOps setup but it is a deliberate starting point - I want to understand the problem before reaching for tools that automate it.

---

## 6. Things I do not fully understand yet

- How to choose between RAG and fine-tuning for a specific use case
- How embedding drift detection works in practice
- How AI Foundry and Databricks are meant to work together 
- How to evaluate LLM output at scale without it becoming expensive

I will update this file as I work through these.

---

*Started May 2026. Updated as I learn.*