import re

# Scoring LLM output is the part I found hardest to get my head around.
# On the ML side of this project a prediction is either right or wrong and
# F1 falls out of that. Here the output is a paragraph of text and there is
# no single correct answer, so "how good is it" has to be broken into
# several narrower questions that each have a defensible answer:
#
#   rouge_l         - does it overlap with a reference summary I wrote?
#   grounding       - is every content word in it actually in the source?
#   length_compliance - did it respect the sentence limit it was given?
#   format_compliance - did it obey the output format it was given?
#
# None of these alone says "good summary". Together they catch the four
# things that actually went wrong in my runs: missing the point, inventing
# facts, ignoring the length limit, and chatting at me before answering.
#
# Everything here is plain Python on purpose. I wanted to understand what
# ROUGE actually computes before importing something that computes it for
# me - same reason I wrote the PSI calculation by hand in detect_drift.py.

# A prompt version has to clear this to be promoted. It is lower than the
# 0.85 F1 threshold on the ML side and that is not a mistake - text overlap
# metrics are bounded well below 1.0 even for genuinely good summaries,
# because there are many correct ways to word the same summary. A prompt
# scoring 1.0 here would mean it reproduced my reference wording exactly,
# which is not the goal.
#
# 0.70 is a judgement call. The reasoning: a version that gets grounding,
# length, and format all right scores 0.60 from those three alone, so the
# threshold has to sit above that to mean anything - it should require
# some genuine overlap with the reference as well, not just obedience.
PROMPT_PROMOTION_THRESHOLD = 0.70

# The weights are a judgement call, so I am writing down the reasoning.
# Grounding is weighted heavily because a fluent summary containing an
# invented fact is worse than a clumsy summary containing none - that is
# the failure mode that would actually matter in an insurance or health
# context. Format is weighted lightest because a stray preamble is
# annoying but harmless and is trivially fixable in post-processing.
METRIC_WEIGHTS = {
    "rouge_l": 0.40,
    "grounding": 0.30,
    "length_compliance": 0.20,
    "format_compliance": 0.10,
}

# Openings that mean the model started talking instead of answering.
# The first four are the ones that actually show up in my runs - the stub
# produces "Here is a summary of the text:" whenever a prompt does not pin
# the format. The rest are the common chatty openings, included so the
# metric does not have to be revisited the first time a real model says
# "Certainly!" instead.
PREAMBLE_PATTERNS = [
    r"^here is a summary",
    r"^here's a summary",
    r"^here is the summary",
    r"^summary\s*:",
    r"^sure[,!]",
    r"^certainly[,!]",
    r"^of course[,!]",
    r"^the text describes",
    r"^this text is about",
    r"^in summary[,:]",
]

# Words short enough that matching them says nothing about meaning.
# A crude stopword filter - anything of 4 letters or more is kept.
MIN_CONTENT_WORD_LENGTH = 4


def tokenize(text):
    # Lowercase word tokens with punctuation stripped. Used by ROUGE, where
    # every word counts including short ones, because word order matters.
    return re.findall(r"[a-z0-9']+", text.lower())


def content_words(text):
    # Only the words long enough to carry meaning. Used by grounding, where
    # matching "the" between summary and source would prove nothing.
    return [word for word in tokenize(text) if len(word) >= MIN_CONTENT_WORD_LENGTH]


def split_sentences(text):
    # Same simple splitter as the stub uses. Split on . ! ? plus
    # whitespace. It miscounts abbreviations like "e.g." but none of my
    # evaluation texts contain any, and I would rather the counting rule be
    # visible here than hidden in a dependency.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def count_sentences(text):
    return len(split_sentences(text))


def longest_common_subsequence_length(first, second):
    # Standard dynamic programming LCS. The subsequence does not have to be
    # contiguous, which is the point - it rewards a summary for keeping
    # words in the same order as the reference without demanding the exact
    # same phrasing.
    #
    # Only two rows of the table are kept at a time. The full table would
    # be fine at these text lengths, but two rows is the same amount of
    # code and does not grow with the input.
    if not first or not second:
        return 0

    previous = [0] * (len(second) + 1)

    for item in first:
        current = [0] * (len(second) + 1)
        for index, other in enumerate(second, start=1):
            if item == other:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current

    return previous[-1]


def rouge_l(candidate, reference):
    # ROUGE-L is the F1 of the longest common subsequence between the
    # generated summary and the reference summary.
    #
    #   precision = LCS / length of candidate   (did it pad with extra words?)
    #   recall    = LCS / length of reference   (did it leave things out?)
    #
    # F1 of the two, so a summary cannot score well by being very short or
    # by copying the whole source text.
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)

    if not candidate_tokens or not reference_tokens:
        return 0.0

    lcs = longest_common_subsequence_length(candidate_tokens, reference_tokens)

    if lcs == 0:
        return 0.0

    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def grounding_score(summary, source):
    # The hallucination check, and the metric I care most about.
    #
    # Every content word in the summary should appear somewhere in the
    # source text. A score of 1.0 means the summary introduced no new
    # vocabulary at all. Anything below means words appeared that were not
    # in the input - which for a summarization task is usually invention.
    #
    # This is a proxy, not a proof. A model could reorder source words into
    # a sentence that means something the source never said and still score
    # 1.0 here. Catching that needs an LLM-as-judge or a human, which is
    # noted as a limitation in docs/prompt_engineering.md.
    summary_words = content_words(summary)

    if not summary_words:
        return 0.0

    source_words = set(content_words(source))
    grounded = sum(1 for word in summary_words if word in source_words)

    return grounded / len(summary_words)


def length_compliance(summary, max_sentences):
    # Did the output respect the sentence limit the prompt asked for?
    #
    # Within the limit scores 1.0. Each sentence over costs half the score,
    # so one extra sentence is a real penalty and three extra is a zero.
    # A hard pass/fail would hide the difference between slightly too long
    # and four times too long.
    if max_sentences is None or max_sentences <= 0:
        return 1.0

    sentences = count_sentences(summary)

    if sentences == 0:
        return 0.0

    if sentences <= max_sentences:
        return 1.0

    over = sentences - max_sentences
    return max(0.0, 1.0 - 0.5 * over)


def format_compliance(summary):
    # Did the output start with the summary, or with the model clearing its
    # throat first? Binary, because either there is a preamble or there is
    # not. This is the metric that improved the most when I pinned the
    # output format in prompt v4.
    stripped = summary.strip().lower()

    for pattern in PREAMBLE_PATTERNS:
        if re.match(pattern, stripped):
            return 0.0

    return 1.0


def score_summary(summary, reference, source, max_sentences):
    # Run every metric on one summary and return them plus the weighted
    # composite. Returning the parts as well as the total matters - a
    # composite of 0.65 tells me a prompt failed but not why, and the
    # whole point of the per-metric breakdown is to tell me what to change.
    scores = {
        "rouge_l": rouge_l(summary, reference),
        "grounding": grounding_score(summary, source),
        "length_compliance": length_compliance(summary, max_sentences),
        "format_compliance": format_compliance(summary),
    }

    scores["composite"] = composite_score(scores)

    return scores


def composite_score(scores):
    # Weighted sum of the individual metrics using METRIC_WEIGHTS.
    return sum(scores[name] * weight for name, weight in METRIC_WEIGHTS.items())


def is_above_threshold(composite):
    # The promotion gate for prompts.
    #
    # >= not >, for the same reason as the F1 gate in evaluate.py - a score
    # landing exactly on the threshold passes. I got this wrong with the
    # F1 gate once and the boundary test caught it, so it is written the
    # right way round here from the start and tested the same way.
    return composite >= PROMPT_PROMOTION_THRESHOLD


def average_scores(per_example_scores):
    # Mean of each metric across all evaluation examples. A prompt version
    # is judged on its average over the whole set, not on its best example.
    if not per_example_scores:
        return {}

    names = list(per_example_scores[0].keys())

    return {
        name: sum(scores[name] for scores in per_example_scores) / len(per_example_scores)
        for name in names
    }
