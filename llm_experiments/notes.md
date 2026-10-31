# Prompt Versioning Notes

This folder contains the versioned prompts for a text summarization task. The goal is to understand how much the output changes when the prompt is reworded, and to get into the habit of treating prompts as versioned artifacts rather than throwaway text.

It started as two prompts and a paragraph of impressions. It is now four prompts, a registry, a hand-written evaluation set, and actual numbers. The impressions turned out to be half right, which is the useful part.

## What is in here

| File | What it is |
|---|---|
| `registry.json` | The registry. Every version with its technique and what changed in it. |
| `prompt_v1.txt` | Zero-shot, minimal. |
| `prompt_v2.txt` | Zero-shot with explicit constraints. |
| `prompt_v3.txt` | Few-shot with `<text>` delimiters. |
| `prompt_v4.txt` | Role framing, delimiters, few-shot, explicit output contract. |
| `eval_dataset.json` | Eight short texts with reference summaries I wrote by hand. |

The techniques themselves, and why each one does what it does, are written up in [docs/prompt_engineering.md](../docs/prompt_engineering.md). This file is just the record of what changed between versions and what the scores said.

## What changed between versions

**v1 to v2.** Added a sentence limit, an instruction to focus on the most important points, and an instruction not to add anything that is not in the original text. The third one is specifically about hallucination - LLMs fill a gap in the instructions with something plausible rather than leaving it empty, so the gap has to be closed explicitly.

**v2 to v3.** Wrapped the input in `<text>` tags so the instructions cannot be confused with the data, and added one worked example. The example demonstrates the expected length and style instead of describing them.

**v3 to v4.** Added a role for the model, tightened the limit to 2 sentences, banned the "Here is a summary" preamble, forbade added background and opinions, and added a fallback rule for text too short to compress. This version specifies what the output must look like as well as what it must contain.

## What the numbers said

From `python -m src.evaluate_prompts`, averaged over all eight evaluation examples:

| Version | ROUGE-L | Grounding | Length | Format | Composite |
|---|---|---|---|---|---|
| v1 | 0.582 | 0.721 | 0.00 | 0.00 | 0.449 |
| v2 | 0.498 | 0.834 | 1.00 | 0.00 | 0.650 |
| v3 | 0.563 | 1.000 | 1.00 | 1.00 | 0.825 |
| v4 | 0.563 | 1.000 | 1.00 | 1.00 | 0.825 |

Threshold for promotion is 0.70 composite, so v1 and v2 both fail and v3 is selected as the best version.

## What I got wrong in my first pass

The original version of this file said that with a minimal prompt like v1 the model "tends to produce longer summaries and sometimes rephrases things in ways that shift the meaning slightly", and that v2 stays closer to the source.

That was right, and measuring it put numbers on it - v1's grounding is 0.721 against v2's 0.834, and v1 blows the length limit entirely.

What I did not see by eye was that **ROUGE-L alone ranks v1 first**, at 0.582, higher than any other version. v1 is the worst prompt of the four and a single-metric evaluation would have promoted it. The reason is mechanical: v1 writes four sentences instead of two, and more text means more overlap with the reference. ROUGE rewarded it for rambling.

That is the main thing I learned from scoring these properly rather than reading them, and it is why the composite is a weighted sum of four metrics instead of one number.

The other thing I missed: v2 scores 0.0 on format compliance across every example. I had read its outputs and not registered that every single one of them opened with "Here is a summary of the text:". v2 constrains the content of the output carefully and says nothing at all about its shape.

## What the scores cannot tell me

v3 and v4 tie at 0.825, to three decimal places.

v4 is the better prompt - it has a role, a tighter limit, an explicit preamble ban, a rule against added opinions, and a fallback for degenerate input. The evaluation cannot see any of that, because the default provider is an offline simulator that only reacts to constraints it can detect by pattern matching. Once length, grounding, and format are all pinned, it has nothing left to measure.

That limitation is written up properly in [docs/prompt_engineering.md](../docs/prompt_engineering.md). The short version: these scores measure how precisely a prompt states its requirements, which is a real property of a prompt and is not the same thing as output quality. Separating v3 from v4 needs a real model and harder inputs.

## Why I am keeping every version

Deleting v1 once v2 was written would lose the record of what changed and why. Keeping all four in the repo is the same idea as keeping old experiment runs in MLflow - the history is the useful part.

It turned out to be more than a tidiness argument. v1 being the ROUGE-L winner is only visible because v1 is still here to be scored. If I had deleted it as soon as v2 looked better, I would never have found the problem with single-metric evaluation.
