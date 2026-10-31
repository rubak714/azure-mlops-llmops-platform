import pytest

from src.prompt_registry import (
    PROMPT_PLACEHOLDER,
    get_version_metadata,
    list_versions,
    load_eval_dataset,
    load_prompt_template,
    load_registry,
    render_prompt,
)

# These tests guard the registry itself rather than any scoring logic.
#
# The failure they are really there to prevent is a silent one: a prompt
# file that loads fine but is missing its {text} placeholder, or a registry
# entry pointing at a file that no longer exists. Either one produces an
# evaluation run full of numbers that mean nothing, which is worse than a
# crash because nothing looks wrong.


def test_registry_has_versions():
    assert len(list_versions()) > 0


def test_registry_declares_a_task():
    assert load_registry()["task"] == "text-summarization"


def test_version_ids_are_unique():
    versions = list_versions()

    assert len(versions) == len(set(versions))


def test_every_version_has_required_metadata():
    # Each entry has to say which file it is and what changed in it.
    # A version with no explanation of what changed is the thing this
    # whole registry exists to prevent.
    for entry in load_registry()["versions"]:
        assert entry["file"]
        assert entry["technique"]
        assert entry["what_changed"]
        assert entry["created"]


def test_every_registered_prompt_file_loads():
    # Catches a registry entry pointing at a renamed or deleted file.
    for version in list_versions():
        assert load_prompt_template(version).strip()


def test_every_prompt_has_the_text_placeholder():
    # Without the placeholder every example would be sent the identical
    # prompt and the resulting scores would be meaningless.
    for version in list_versions():
        assert PROMPT_PLACEHOLDER in load_prompt_template(version)


def test_unknown_version_raises_with_a_helpful_message():
    with pytest.raises(KeyError) as error:
        get_version_metadata("v999")

    # The error should list what does exist, not just say no.
    assert "v1" in str(error.value)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_inserts_the_text():
    rendered = render_prompt("v1", "A short input text.")

    assert "A short input text." in rendered
    assert PROMPT_PLACEHOLDER not in rendered


def test_render_handles_text_containing_braces():
    # str.format() would raise on this input, which is exactly why
    # render_prompt uses str.replace instead. This test is the record of
    # that decision.
    rendered = render_prompt("v1", "A text with {braces} in it.")

    assert "{braces}" in rendered


def test_rendering_is_repeatable():
    assert render_prompt("v2", "Same text.") == render_prompt("v2", "Same text.")


# ---------------------------------------------------------------------------
# The evaluation dataset
# ---------------------------------------------------------------------------

def test_eval_dataset_has_examples():
    assert len(load_eval_dataset()["examples"]) > 0


def test_eval_dataset_declares_a_sentence_target():
    assert load_eval_dataset()["target_max_sentences"] > 0


def test_every_example_has_id_text_and_reference():
    for example in load_eval_dataset()["examples"]:
        assert example["id"]
        assert example["text"]
        assert example["reference"]


def test_example_ids_are_unique():
    # Duplicated ids would make the per-example outputs logged to MLflow
    # impossible to match back to their input.
    ids = [example["id"] for example in load_eval_dataset()["examples"]]

    assert len(ids) == len(set(ids))


def test_references_are_shorter_than_their_source():
    # A reference "summary" longer than the text it summarizes would make
    # the ROUGE scores nonsense. Cheap sanity check on data I wrote by hand.
    for example in load_eval_dataset()["examples"]:
        assert len(example["reference"]) < len(example["text"]) * 1.5
