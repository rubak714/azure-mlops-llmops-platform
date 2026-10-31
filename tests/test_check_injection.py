from src.check_injection import (
    REQUIRED_INTEGRITY,
    boundary_held,
    check_version,
    integrity_rate,
    load_injection_tests,
)
from src.prompt_registry import list_versions

# Tests for the prompt boundary checks.
#
# These pin down findings as much as they check code. The two cases that
# matter most are summary-label and closing-tag, because between them they
# show that adding delimiters to a prompt traded one vulnerability for
# another rather than removing one. If someone later changes the prompt
# format or the extractor, these tests say what that change costs.

UNDELIMITED = ["v1", "v2"]
DELIMITED = ["v3", "v4"]


def _case(case_id):
    for case in load_injection_tests()["cases"]:
        if case["id"] == case_id:
            return case

    raise AssertionError("No injection case with id " + case_id)


# ---------------------------------------------------------------------------
# The control case
# ---------------------------------------------------------------------------

def test_benign_text_survives_every_prompt_version():
    # The control, and the most important test here for the same reason
    # the no-drift scenario is the most important drift test. A check that
    # cannot pass on harmless input is measuring something other than what
    # it claims to.
    benign = _case("benign-control")["text"]

    for version in list_versions():
        assert boundary_held(version, benign) is True


# ---------------------------------------------------------------------------
# The two findings
# ---------------------------------------------------------------------------

def test_undelimited_prompts_are_cut_short_by_their_own_output_label():
    # v1 and v2 use "Text: ... Summary:" with nothing marking where the
    # input ends. Any document containing "Summary:" - minutes, a report
    # with headings, a forwarded email - truncates the input silently.
    # No attacker required.
    hostile = _case("summary-label")["text"]

    for version in UNDELIMITED:
        assert boundary_held(version, hostile) is False


def test_delimited_prompts_survive_the_output_label():
    # The payoff for adding <text> tags in v3.
    hostile = _case("summary-label")["text"]

    for version in DELIMITED:
        assert boundary_held(version, hostile) is True


def test_delimited_prompts_are_cut_short_by_their_own_closing_tag():
    # The cost of adding <text> tags, and the finding I did not expect.
    # A delimiter separates instructions from data only while the data
    # cannot contain the delimiter. An attacker who knows the format knows
    # exactly what to close.
    hostile = _case("closing-tag")["text"]

    for version in DELIMITED:
        assert boundary_held(version, hostile) is False


def test_undelimited_prompts_survive_the_closing_tag():
    # The mirror image. v1 and v2 do not use tags, so a closing tag is
    # just text to them. The two formats fail on opposite attacks, which
    # is the whole point.
    hostile = _case("closing-tag")["text"]

    for version in UNDELIMITED:
        assert boundary_held(version, hostile) is True


def test_an_injected_opening_tag_breaks_every_version():
    # Text containing a complete <text>...</text> region hijacks the
    # extractor regardless of whether the prompt itself uses tags,
    # because the extractor looks for tags first.
    hostile = _case("nested-opening-tag")["text"]

    for version in list_versions():
        assert boundary_held(version, hostile) is False


def test_no_version_resists_everything():
    # The headline. Stated as a test so that if a future prompt version
    # does resist everything, this fails loudly and the claim in the docs
    # gets revisited rather than quietly going stale.
    tests = load_injection_tests()

    rates = [
        integrity_rate(check_version(version, tests))
        for version in list_versions()
    ]

    assert all(rate < REQUIRED_INTEGRITY for rate in rates)


def test_a_plain_instruction_cannot_break_the_parsing():
    # "Ignore all previous instructions" imitates no delimiter, so it
    # cannot truncate anything. Whether it works at all depends on the
    # model, which is exactly why boundary integrity is not the whole
    # story and the docs say so.
    hostile = _case("plain-override")["text"]

    for version in list_versions():
        assert boundary_held(version, hostile) is True


# ---------------------------------------------------------------------------
# The measure itself
# ---------------------------------------------------------------------------

def test_integrity_rate_counts_held_cases():
    results = [
        {"boundary_held": True},
        {"boundary_held": True},
        {"boundary_held": False},
        {"boundary_held": False},
    ]

    assert integrity_rate(results) == 0.5


def test_integrity_rate_of_all_held_is_one():
    assert integrity_rate([{"boundary_held": True}]) == 1.0


def test_integrity_rate_of_empty_is_zero():
    assert integrity_rate([]) == 0.0


def test_required_integrity_allows_no_partial_credit():
    # Deliberate. "The input survived 83% of the time" is not a property
    # anyone would accept in a system handling real documents, so the bar
    # is all or nothing.
    assert REQUIRED_INTEGRITY == 1.0


# ---------------------------------------------------------------------------
# The test data
# ---------------------------------------------------------------------------

def test_every_case_has_the_required_fields():
    for case in load_injection_tests()["cases"]:
        assert case["id"]
        assert case["text"]
        assert case["attack"]
        assert case["note"]


def test_case_ids_are_unique():
    ids = [case["id"] for case in load_injection_tests()["cases"]]

    assert len(ids) == len(set(ids))


def test_a_canary_is_defined():
    assert load_injection_tests()["canary"]


def test_every_hostile_case_contains_the_canary():
    # Except the control, which is supposed to be harmless. If an attack
    # case did not contain the canary the leak check would pass for a
    # reason unrelated to the prompt working.
    tests = load_injection_tests()
    canary = tests["canary"]

    for case in tests["cases"]:
        if case["id"] == "benign-control":
            assert canary not in case["text"]
        else:
            assert canary in case["text"]


def test_check_version_returns_one_result_per_case():
    tests = load_injection_tests()

    results = check_version("v4", tests)

    assert len(results) == len(tests["cases"])

    for result in results:
        assert "boundary_held" in result
        assert "canary_leaked" in result
