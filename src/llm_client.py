import hashlib
import os
import re

# This module is the single place where the project talks to a language model.
# Everything else - the prompt registry, the evaluation harness - calls
# complete() and does not care which provider is behind it.
#
# There are two providers:
#
#   "stub"  - an offline simulator that runs on my machine and in CI for free.
#             This is the default so that anyone cloning the repo can run the
#             whole prompt evaluation without an API key or an Azure subscription.
#
#   "azure" - a real Azure OpenAI deployment, used when the AZURE_OPENAI_*
#             environment variables are set.
#
# IMPORTANT, and I want this written down clearly rather than buried:
# the stub is NOT a language model. It is a deterministic extractive
# summarizer that reads the instructions out of the prompt and obeys them.
# It exists so I can build and test the evaluation pipeline offline. Scores
# produced with the stub say something about whether a prompt states its
# constraints clearly - they say nothing about how a real model would perform.
# See docs/prompt_engineering.md for why I built it this way anyway.

PROVIDER_STUB = "stub"
PROVIDER_AZURE = "azure"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256

# Azure OpenAI reads all of its configuration from the environment so that
# no key ever ends up in the repo. Worth noting that the deployment name is
# whatever the deployment was called in Azure, not the model name - "gpt-4o"
# is a model, "my-summarizer" might be the deployment serving it. Azure's
# API takes the deployment.
ENV_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_DEPLOYMENT = "AZURE_OPENAI_DEPLOYMENT"
ENV_API_VERSION = "AZURE_OPENAI_API_VERSION"

DEFAULT_API_VERSION = "2024-06-01"

# A generic sentence the stub appends when a prompt never tells it to stay
# inside the source text. This simulates the single most common LLM failure
# mode - adding a confident closing statement that was never in the input.
INVENTED_SENTENCE = "Overall this represents a significant and positive development."

PREAMBLE = "Here is a summary of the text:"


def get_provider():
    # LLM_PROVIDER lets me force a provider explicitly. If it is not set I
    # pick azure only when the credentials are actually present, otherwise
    # the stub. This means the scripts never fail with an auth error just
    # because someone has not configured Azure.
    requested = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if requested in (PROVIDER_STUB, PROVIDER_AZURE):
        return requested

    if os.environ.get(ENV_ENDPOINT) and os.environ.get(ENV_API_KEY):
        return PROVIDER_AZURE

    return PROVIDER_STUB


def describe_provider():
    # Used by the evaluation harness so the MLflow run records which
    # provider produced the scores. Without this a run is meaningless -
    # stub scores and real model scores are not comparable.
    provider = get_provider()

    if provider == PROVIDER_AZURE:
        return "azure:" + os.environ.get(ENV_DEPLOYMENT, "unknown-deployment")

    return "stub:extractive-simulator"


# ---------------------------------------------------------------------------
# Reading the input text back out of a rendered prompt
# ---------------------------------------------------------------------------

def extract_source_text(prompt):
    # The stub needs to know which part of the prompt is the text to
    # summarize, as opposed to the instructions around it.
    #
    # My prompt versions mark the input in one of two ways - either between
    # <text> tags or after a "Text:" label. I take the LAST match in both
    # cases, because a few-shot prompt contains an example that uses the
    # same markers and the real input is always the one at the end.
    tagged = re.findall(r"<text>(.*?)</text>", prompt, flags=re.DOTALL)
    if tagged:
        return tagged[-1].strip()

    labelled = re.findall(r"Text:(.*?)Summary:", prompt, flags=re.DOTALL)
    if labelled:
        return labelled[-1].strip()

    # Nothing recognisable - treat the whole prompt as the text. This only
    # happens if I write a prompt that does not mark its input, which the
    # registry tests are there to catch.
    return prompt.strip()


# ---------------------------------------------------------------------------
# Reading the instructions out of a prompt
# ---------------------------------------------------------------------------

def parse_example_summaries(prompt):
    # Find the worked examples in a few-shot prompt.
    #
    # Every one of my prompt versions ends with a bare "Summary:" that the
    # model is supposed to complete. In a few-shot prompt there are earlier
    # "Summary:" labels that already have text after them - those are the
    # demonstrations. So: any Summary: label with content after it is an
    # example, and the trailing empty one is the slot for the answer.
    matches = re.findall(r"Summary:[ \t]*(.*?)(?:\n\s*\n|\Z)", prompt, flags=re.DOTALL)
    return [match.strip() for match in matches if match.strip()]


def parse_directives(prompt):
    # This is the heart of the simulator. It looks for the specific
    # constraints a prompt states and returns them as a dict. A vague
    # prompt yields few directives and the stub then behaves badly - which
    # is exactly the behaviour I want to be able to measure.
    lowered = prompt.lower()

    examples = parse_example_summaries(prompt)
    stated_limit = _parse_sentence_limit(lowered)

    # A worked example demonstrates the expected length without stating it.
    # This is the actual mechanism behind few-shot prompting, so the stub
    # has to honour it or it cannot tell a few-shot prompt from a zero-shot
    # one. When a prompt both states a limit and shows an example, the
    # tighter of the two wins - that is what a model does in practice.
    example_limit = None
    if examples:
        example_limit = len(split_sentences(examples[-1]))

    if stated_limit and example_limit:
        max_sentences = min(stated_limit, example_limit)
    else:
        max_sentences = stated_limit or example_limit

    directives = {
        "max_sentences": max_sentences,
        "extractive_only": _states_extractive_constraint(lowered),
        # An example that starts straight in with the summary teaches the
        # format just as effectively as a rule forbidding a preamble.
        "no_preamble": (
            "preamble" in lowered or
            "start with the summary" in lowered or
            bool(examples)
        ),
        "focus_on_important": "most important" in lowered or "key point" in lowered,
        "has_example": bool(examples),
    }

    return directives


def _parse_sentence_limit(lowered):
    # Several phrasings mean the same thing and I want the stub to
    # understand all of them, because the whole point of v1 versus v2 is
    # that v2 states the limit and v1 only hints at it.
    patterns = [
        r"at most (\d+) sentence",
        r"under (\d+) sentence",
        r"no more than (\d+) sentence",
        r"maximum of (\d+) sentence",
        r"in (\d+) sentence",
        # "2-3 sentences" or "2 to 3 sentences" - take the upper bound
        r"\d+\s*(?:-|to)\s*(\d+) sentence",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            limit = int(match.group(1))
            # "under 3 sentences" means 2, "at most 3" means 3. I treat
            # "under" strictly because that is what the word means and a
            # real model that ignores it should be penalised.
            if pattern.startswith("under"):
                limit = max(1, limit - 1)
            return limit

    return None


def _states_extractive_constraint(lowered):
    # Any of these phrasings tells the model not to invent content.
    markers = [
        "do not add anything",
        "do not add",
        "only include information",
        "use only information",
        "use only facts",
        "only facts stated",
        "not in the original",
        "stick to the source",
        "do not add background",
        "without changing its meaning",
    ]

    return any(marker in lowered for marker in markers)


# ---------------------------------------------------------------------------
# The extractive summarizer behind the stub
# ---------------------------------------------------------------------------

def split_sentences(text):
    # Deliberately simple sentence splitting - split on . ! ? followed by
    # whitespace. Good enough for the evaluation texts I wrote, and I would
    # rather have 5 obvious lines here than pull in nltk for this.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _content_words(sentence):
    # Strip punctuation and drop very short words. Not a real stopword
    # list - just enough that scoring is driven by meaningful words.
    words = re.findall(r"[a-z0-9']+", sentence.lower())
    return [word for word in words if len(word) > 3]


def summarize_extractive(text, max_sentences):
    # Frequency-based extractive summarization. Score each sentence by how
    # many of the document's common words it contains, normalised by its
    # length so long sentences do not automatically win, then return the
    # top sentences in their original order so the summary still reads.
    sentences = split_sentences(text)

    if not sentences:
        return ""

    if max_sentences >= len(sentences):
        return " ".join(sentences)

    frequencies = {}
    for sentence in sentences:
        for word in _content_words(sentence):
            frequencies[word] = frequencies.get(word, 0) + 1

    scored = []
    for index, sentence in enumerate(sentences):
        words = _content_words(sentence)
        if words:
            score = sum(frequencies.get(word, 0) for word in words) / len(words)
        else:
            score = 0.0
        # The first sentence of a text usually carries the main point, so
        # it gets a small bonus. This is a standard trick in extractive
        # summarization and it stops the summary starting mid-story.
        if index == 0:
            score += 1.0
        scored.append((score, index, sentence))

    # Sort by score descending, then by position so ties are deterministic.
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(scored[:max_sentences], key=lambda item: item[1])

    return " ".join(item[2] for item in chosen)


def _stub_complete(prompt):
    # Put the pieces together: find the input, read the instructions,
    # summarize accordingly, then deliberately misbehave in the ways a
    # real model misbehaves when the prompt does not forbid it.
    source = extract_source_text(prompt)
    directives = parse_directives(prompt)

    # No stated limit means the model is free to ramble. Four sentences is
    # the stub's idea of rambling.
    limit = directives["max_sentences"] or 4

    summary = summarize_extractive(source, limit)

    # Simulated hallucination. If the prompt never says to stay inside the
    # source text, add a sentence that is not in it.
    if not directives["extractive_only"]:
        summary = summary + " " + INVENTED_SENTENCE

    # Simulated format drift. If the prompt does not forbid a preamble and
    # does not pin the output format, the model chats at me first.
    if not directives["no_preamble"]:
        summary = PREAMBLE + " " + summary

    return summary.strip()


def stub_fingerprint(prompt):
    # The stub is fully deterministic, so the same prompt always gives the
    # same output. I log this hash with each run as a cheap way to prove
    # that two runs really did use the same prompt text.
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# The real Azure OpenAI provider
# ---------------------------------------------------------------------------

def _azure_complete(prompt, temperature, max_tokens):
    # Imported here rather than at the top of the file on purpose. The
    # openai package is only needed by this one function, so the stub path
    # and the whole test suite run without it installed.
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise RuntimeError(
            "The azure provider needs the openai package. "
            "Install it with: python -m pip install openai"
        )

    endpoint = os.environ.get(ENV_ENDPOINT)
    api_key = os.environ.get(ENV_API_KEY)
    deployment = os.environ.get(ENV_DEPLOYMENT)

    missing = [
        name for name, value in [
            (ENV_ENDPOINT, endpoint),
            (ENV_API_KEY, api_key),
            (ENV_DEPLOYMENT, deployment),
        ] if not value
    ]

    if missing:
        raise RuntimeError(
            "The azure provider needs these environment variables: " +
            ", ".join(missing)
        )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.environ.get(ENV_API_VERSION, DEFAULT_API_VERSION),
    )

    # temperature=0 by default because this is an evaluation harness. If the
    # model gives a different answer every time I cannot tell whether a
    # score moved because I changed the prompt or because of sampling noise.
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# The one function the rest of the project calls
# ---------------------------------------------------------------------------

def complete(prompt, temperature=DEFAULT_TEMPERATURE,
             max_tokens=DEFAULT_MAX_TOKENS, provider=None):
    # Send a fully rendered prompt to whichever provider is active and
    # return the model's text. The provider argument is mainly for tests -
    # normally it comes from the environment.
    active = provider or get_provider()

    if active == PROVIDER_AZURE:
        return _azure_complete(prompt, temperature, max_tokens)

    if active == PROVIDER_STUB:
        return _stub_complete(prompt)

    raise ValueError("Unknown provider: " + str(active))


if __name__ == "__main__":
    # A quick manual check so I can see what the active provider does
    # without running the whole evaluation.
    EXAMPLE = (
        "The city council approved funding for three new bus routes in "
        "April. The routes connect the eastern suburbs to the main train "
        "station. Construction of the new bus stops begins in July. "
        "Officials expect the routes to carry 12,000 passengers a day."
    )

    DEMO_PROMPT = (
        "Summarize the text below in at most 2 sentences. "
        "Use only information that appears in the text. "
        "Do not begin with a preamble.\n\n"
        "<text>\n" + EXAMPLE + "\n</text>\n\nSummary:"
    )

    print("Active provider: " + describe_provider())
    print("Prompt fingerprint: " + stub_fingerprint(DEMO_PROMPT))
    print("\nOutput:")
    print(complete(DEMO_PROMPT))
