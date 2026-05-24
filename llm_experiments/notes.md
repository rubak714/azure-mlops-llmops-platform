# Prompt Versioning Notes

This folder contains two versions of a prompt for a text summarization task. The goal is to understand how much the output changes when the prompt is reworded, and to get into the habit of treating prompts as versioned artifacts rather than throwaway text.

## What changed between v1 and v2

v1 is very minimal - just the instruction, the input, and the output label. It gives the model a lot of freedom to decide what to include and how long to make the summary.

v2 adds three extra constraints:
- a sentence limit (under 3 sentences)
- an instruction to focus only on the most important points
- an instruction not to add anything not in the original text

The third constraint is specifically about hallucination - LLMs sometimes add details that were not in the source text. Making this explicit in the prompt is one way to reduce it.

## What I noticed

With a minimal prompt like v1, the model tends to produce longer summaries and sometimes rephrases things in ways that shift the meaning slightly. With v2 the output is shorter and stays closer to the source text.

This is a small manual observation, not a formal evaluation. A proper comparison would run both prompts on the same set of texts and score the outputs with a metric like ROUGE or faithfulness. That is something to do in a later iteration.

## Why I am keeping both versions

Deleting v1 once v2 was written would lose the record of what changed and why. Keeping both versions in the repo is the same idea as keeping old experiment runs in MLflow - the history is useful.