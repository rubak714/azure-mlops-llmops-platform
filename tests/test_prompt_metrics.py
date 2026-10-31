from src.prompt_metrics import (
    METRIC_WEIGHTS,
    PROMPT_PROMOTION_THRESHOLD,
    average_scores,
    composite_score,
    count_sentences,
    format_compliance,
    grounding_score,
    is_above_threshold,
    length_compliance,
    longest_common_subsequence_length,
    rouge_l,
)

# These tests check the scoring logic in isolation, the same way
# test_evaluate.py checks the F1 threshold logic without touching MLflow.
# No language model is involved anywhere in this file - every input is a
# fixed string I wrote, so every expected score is something I can work
# out by hand and argue about.

SOURCE = (
    "The council approved the new cycle lane in March. "
    "Construction is due to start in September and will take four months."
)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def test_lcs_of_identical_sequences_is_full_length():
    assert longest_common_subsequence_length([1, 2, 3], [1, 2, 3]) == 3


def test_lcs_ignores_gaps():
    # The subsequence does not have to be contiguous - that is the whole
    # reason ROUGE-L uses LCS instead of exact phrase matching.
    assert longest_common_subsequence_length([1, 2, 3, 4], [1, 3]) == 2


def test_lcs_of_nothing_in_common_is_zero():
    assert longest_common_subsequence_length([1, 2], [3, 4]) == 0


def test_rouge_l_identical_text_scores_one():
    # A summary identical to the reference is the best possible case.
    assert rouge_l(SOURCE, SOURCE) == 1.0


def test_rouge_l_unrelated_text_scores_zero():
    assert rouge_l("completely different words here", "nothing shared") == 0.0


def test_rouge_l_is_penalised_for_padding():
    # Same reference, two candidates. The one that says it in fewer words
    # must score higher, because precision drops as padding is added.
    # Without this property a model could score well by dumping the whole
    # source text as its summary.
    reference = "The council approved the cycle lane in March."
    concise = "The council approved the cycle lane in March."
    padded = concise + " " + SOURCE + " This is additional unrelated filler text."

    assert rouge_l(concise, reference) > rouge_l(padded, reference)


def test_rouge_l_empty_candidate_scores_zero():
    # An empty answer should score zero, not crash.
    assert rouge_l("", SOURCE) == 0.0


# ---------------------------------------------------------------------------
# Grounding - the hallucination check
# ---------------------------------------------------------------------------

def test_grounding_of_copied_text_is_one():
    # Every content word came from the source, so nothing was invented.
    assert grounding_score(SOURCE, SOURCE) == 1.0


def test_grounding_detects_invented_content():
    # This is the metric's entire job. The summary below is fluent and
    # reads well but "unanimously" and "residents" are not in the source.
    summary = "The council unanimously approved the cycle lane after residents complained."

    assert grounding_score(summary, SOURCE) < 1.0


def test_grounding_of_fully_invented_text_is_low():
    summary = "Scientists discovered an entirely unrelated chemical compound yesterday."

    assert grounding_score(summary, SOURCE) < 0.3


def test_grounding_ignores_short_words():
    # "the" and "in" appear everywhere and matching them proves nothing,
    # so only words of four letters or more are counted.
    summary = "The council approved the cycle lane in March."

    assert grounding_score(summary, SOURCE) == 1.0


# ---------------------------------------------------------------------------
# Length compliance
# ---------------------------------------------------------------------------

def test_count_sentences_handles_all_terminators():
    assert count_sentences("One. Two! Three?") == 3


def test_length_compliance_within_limit_is_one():
    assert length_compliance("One sentence only.", max_sentences=2) == 1.0


def test_length_compliance_exactly_at_limit_is_one():
    # Boundary case. Exactly at the limit complies - "at most 2" includes 2.
    # The F1 gate had this exact bug once (> instead of >=) and the
    # boundary test is what caught it, so it is tested here from the start.
    assert length_compliance("One. Two.", max_sentences=2) == 1.0


def test_length_compliance_one_over_loses_half():
    assert length_compliance("One. Two. Three.", max_sentences=2) == 0.5


def test_length_compliance_far_over_is_zero():
    # Three sentences over the limit is a zero, not a negative number.
    assert length_compliance("One. Two. Three. Four. Five.", max_sentences=2) == 0.0


def test_length_compliance_with_no_limit_is_one():
    # If the prompt set no limit there is nothing to violate.
    assert length_compliance("One. Two. Three. Four.", max_sentences=None) == 1.0


# ---------------------------------------------------------------------------
# Format compliance
# ---------------------------------------------------------------------------

def test_format_compliance_clean_answer_passes():
    assert format_compliance("The council approved the cycle lane in March.") == 1.0


def test_format_compliance_catches_preamble():
    # This is the exact string the stub produces when a prompt does not
    # forbid a preamble, and it is what prompt v4 was written to stop.
    assert format_compliance("Here is a summary of the text: The council approved it.") == 0.0


def test_format_compliance_catches_chatty_opening():
    assert format_compliance("Sure, the council approved the cycle lane.") == 0.0


def test_format_compliance_is_case_insensitive():
    assert format_compliance("HERE IS A SUMMARY: the council approved it.") == 0.0


# ---------------------------------------------------------------------------
# Composite score and the promotion gate
# ---------------------------------------------------------------------------

def test_metric_weights_sum_to_one():
    # If the weights drift away from 1.0 the composite stops being
    # comparable to the threshold and every past score becomes meaningless.
    assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9


def test_composite_of_perfect_scores_is_one():
    perfect = {name: 1.0 for name in METRIC_WEIGHTS}

    assert abs(composite_score(perfect) - 1.0) < 1e-9


def test_composite_of_zero_scores_is_zero():
    zeros = {name: 0.0 for name in METRIC_WEIGHTS}

    assert composite_score(zeros) == 0.0


def test_grounding_outweighs_format():
    # A deliberate check on the weighting decision, not just the maths.
    # Losing grounding must cost more than losing format compliance,
    # because an invented fact matters more than a stray preamble.
    assert METRIC_WEIGHTS["grounding"] > METRIC_WEIGHTS["format_compliance"]


def test_strong_prompt_passes_threshold():
    assert is_above_threshold(0.90) is True


def test_prompt_exactly_at_threshold_passes():
    # The same boundary case as the F1 gate. Exactly on the threshold
    # passes, because the gate is >= and not >.
    assert is_above_threshold(PROMPT_PROMOTION_THRESHOLD) is True


def test_prompt_just_below_threshold_fails():
    assert is_above_threshold(PROMPT_PROMOTION_THRESHOLD - 0.01) is False


def test_weak_prompt_fails_threshold():
    assert is_above_threshold(0.45) is False


# ---------------------------------------------------------------------------
# Averaging across the evaluation set
# ---------------------------------------------------------------------------

def test_average_scores_averages_each_metric():
    scores = [
        {"rouge_l": 0.4, "grounding": 1.0},
        {"rouge_l": 0.6, "grounding": 0.0},
    ]

    averaged = average_scores(scores)

    assert averaged["rouge_l"] == 0.5
    assert averaged["grounding"] == 0.5


def test_average_scores_of_empty_list_is_empty():
    # Guard against dividing by zero if the evaluation set is ever empty.
    assert average_scores([]) == {}
