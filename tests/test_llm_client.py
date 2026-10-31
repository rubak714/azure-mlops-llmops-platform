import pytest

from src.llm_client import (
    INVENTED_SENTENCE,
    PREAMBLE,
    PROVIDER_AZURE,
    PROVIDER_STUB,
    complete,
    extract_source_text,
    get_provider,
    parse_directives,
    parse_example_summaries,
    split_sentences,
    stub_fingerprint,
    summarize_extractive,
)

# Tests for the offline stub and the provider selection logic.
#
# Nothing here makes a network call. The Azure path is only checked for
# how it fails when it is misconfigured, because testing the real thing
# would need a live deployment and would cost money on every CI run.

SOURCE = (
    "The city council approved funding for three new bus routes in April. "
    "The routes connect the eastern suburbs to the main train station. "
    "Construction of the new bus stops begins in July. "
    "Officials expect the routes to carry twelve thousand passengers a day."
)


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

def test_default_provider_is_stub(monkeypatch):
    # The most important test in this file. With no environment configured
    # the project must fall back to the offline stub, so that cloning the
    # repo and running the evaluation works with no API key and no bill.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    assert get_provider() == PROVIDER_STUB


def test_azure_selected_when_credentials_present(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "not-a-real-key")

    assert get_provider() == PROVIDER_AZURE


def test_endpoint_without_key_does_not_select_azure(monkeypatch):
    # Half-configured Azure must not be picked. Otherwise the scripts die
    # with an authentication error instead of quietly using the stub.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    assert get_provider() == PROVIDER_STUB


def test_llm_provider_env_var_overrides_credentials(monkeypatch):
    # Being able to force the stub while Azure credentials are set is how
    # I test changes to the harness without spending money.
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "not-a-real-key")

    assert get_provider() == PROVIDER_STUB


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        complete("anything", provider="some-provider-that-does-not-exist")


def test_azure_provider_without_config_raises_clear_error(monkeypatch):
    # No network call happens - the missing variables are caught first.
    # The point is that the error names what is missing instead of
    # surfacing an opaque SDK exception.
    for name in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                 "AZURE_OPENAI_DEPLOYMENT"]:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError) as error:
        complete("anything", provider=PROVIDER_AZURE)

    message = str(error.value)
    assert "AZURE_OPENAI_ENDPOINT" in message or "openai package" in message


# ---------------------------------------------------------------------------
# Finding the input text inside a rendered prompt
# ---------------------------------------------------------------------------

def test_extract_source_from_tags():
    prompt = "Summarize this.\n\n<text>\n" + SOURCE + "\n</text>\n\nSummary:"

    assert extract_source_text(prompt) == SOURCE


def test_extract_source_from_label():
    prompt = "Summarize the following text.\n\nText: " + SOURCE + "\n\nSummary:"

    assert extract_source_text(prompt) == SOURCE


def test_extract_source_takes_last_match_in_few_shot_prompt():
    # A few-shot prompt contains an example wrapped in the same tags as the
    # real input. The real input is always last, and getting this wrong
    # would mean every example in the evaluation set was scored against the
    # summary of the example in the prompt.
    prompt = (
        "<text>\nThis is the worked example text.\n</text>\n"
        "Summary: This is the example summary.\n\n"
        "<text>\n" + SOURCE + "\n</text>\n\nSummary:"
    )

    assert extract_source_text(prompt) == SOURCE


# ---------------------------------------------------------------------------
# Reading instructions out of a prompt
# ---------------------------------------------------------------------------

def test_detects_at_most_sentence_limit():
    assert parse_directives("Write at most 2 sentences.")["max_sentences"] == 2


def test_under_n_sentences_means_n_minus_one():
    # "under 3 sentences" means 2, taken literally. A model that writes 3
    # has not followed the instruction, so the stub should not either.
    assert parse_directives("Keep it under 3 sentences.")["max_sentences"] == 2


def test_detects_sentence_range():
    # Prompt v1 says "in 2-3 sentences" and the upper bound is the limit.
    assert parse_directives("Summarize in 2-3 sentences.")["max_sentences"] == 3


def test_no_stated_limit_is_none():
    assert parse_directives("Summarize this text.")["max_sentences"] is None


def test_detects_anti_hallucination_instruction():
    directive = parse_directives("Do not add anything that is not in the original text.")

    assert directive["extractive_only"] is True


def test_vague_prompt_has_no_extractive_constraint():
    assert parse_directives("Summarize this text.")["extractive_only"] is False


def test_example_summary_is_detected():
    # The trailing bare "Summary:" is the slot for the answer, not an
    # example, so only one example should be found here.
    prompt = (
        "<text>\nExample input.\n</text>\n"
        "Summary: The example summary.\n\n"
        "<text>\nReal input.\n</text>\n\nSummary:"
    )

    assert len(parse_example_summaries(prompt)) == 1


def test_worked_example_implies_no_preamble():
    # This is the mechanism behind few-shot prompting - a demonstration
    # pins the output format without any rule stating it. If the stub
    # ignored this it could not tell a few-shot prompt from a zero-shot one.
    prompt = (
        "Summarize the text.\n\n"
        "<text>\nExample input here.\n</text>\n"
        "Summary: The example summary.\n\n"
        "<text>\nReal input here.\n</text>\n\nSummary:"
    )

    directives = parse_directives(prompt)

    assert directives["has_example"] is True
    assert directives["no_preamble"] is True


def test_example_length_tightens_a_looser_stated_limit():
    # The prompt says up to 4 sentences but demonstrates 1. The tighter of
    # the two wins, which is what a real model does in practice.
    prompt = (
        "Summarize in at most 4 sentences.\n\n"
        "<text>\nExample input.\n</text>\n"
        "Summary: One short sentence.\n\n"
        "<text>\nReal input.\n</text>\n\nSummary:"
    )

    assert parse_directives(prompt)["max_sentences"] == 1


# ---------------------------------------------------------------------------
# The extractive summarizer
# ---------------------------------------------------------------------------

def test_split_sentences_counts_correctly():
    assert len(split_sentences(SOURCE)) == 4


def test_summarize_respects_sentence_limit():
    assert len(split_sentences(summarize_extractive(SOURCE, 2))) == 2


def test_summarize_keeps_original_order():
    # The selected sentences must come back in the order they appeared or
    # the summary reads like it was shuffled.
    summary = summarize_extractive(SOURCE, 3)
    positions = [SOURCE.index(sentence) for sentence in split_sentences(summary)]

    assert positions == sorted(positions)


def test_summarize_keeps_the_first_sentence():
    # The opening sentence of these texts carries the main point, so the
    # scorer gives it a bonus and it should survive any limit.
    summary = summarize_extractive(SOURCE, 2)

    assert summary.startswith("The city council approved funding")


def test_summarize_limit_larger_than_text_returns_everything():
    assert summarize_extractive(SOURCE, 99) == SOURCE


def test_summarize_empty_text_returns_empty():
    assert summarize_extractive("", 2) == ""


# ---------------------------------------------------------------------------
# The simulated failure modes
# ---------------------------------------------------------------------------

def test_vague_prompt_produces_preamble_and_invented_sentence():
    # A prompt with no format rule and no grounding rule triggers both
    # simulated failures. These are the behaviours the metrics are
    # designed to catch, so the stub has to actually produce them.
    prompt = "Summarize the following text.\n\nText: " + SOURCE + "\n\nSummary:"
    output = complete(prompt, provider=PROVIDER_STUB)

    assert output.startswith(PREAMBLE)
    assert INVENTED_SENTENCE in output


def test_constrained_prompt_produces_neither():
    prompt = (
        "Write at most 2 sentences. Use only facts stated in the text. "
        "Do not begin with a preamble.\n\n"
        "<text>\n" + SOURCE + "\n</text>\n\nSummary:"
    )
    output = complete(prompt, provider=PROVIDER_STUB)

    assert not output.startswith(PREAMBLE)
    assert INVENTED_SENTENCE not in output


def test_stub_is_deterministic():
    # The evaluation harness is only meaningful if the same prompt gives
    # the same output every time. Otherwise a score change could be
    # sampling noise rather than a real effect of my edit.
    prompt = "Write at most 2 sentences.\n\n<text>\n" + SOURCE + "\n</text>\n\nSummary:"

    assert complete(prompt, provider=PROVIDER_STUB) == complete(prompt, provider=PROVIDER_STUB)


def test_fingerprint_differs_for_different_prompts():
    assert stub_fingerprint("prompt one") != stub_fingerprint("prompt two")


def test_fingerprint_is_stable_for_same_prompt():
    assert stub_fingerprint("same prompt") == stub_fingerprint("same prompt")
