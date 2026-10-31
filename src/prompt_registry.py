import json
import os

# A prompt registry, in the same spirit as the MLflow model registry in
# register_model.py. The idea I took from the ML side of this project is
# that an artifact you cannot trace back to its exact version is not worth
# much - so every prompt version lives in a file, and registry.json records
# what changed between them.
#
# Keeping the metadata in JSON rather than hardcoding it in Python means
# the registry is readable without running anything, and adding a v5 is a
# file plus one entry rather than a code change.

PROMPT_PLACEHOLDER = "{text}"

# Paths are resolved relative to this file, not to the working directory.
# Doing it the other way round meant the scripts only worked when run from
# the repo root, which broke as soon as pytest ran them from elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(_HERE, "..", "llm_experiments")

REGISTRY_FILE = os.path.join(EXPERIMENTS_DIR, "registry.json")
EVAL_DATASET_FILE = os.path.join(EXPERIMENTS_DIR, "eval_dataset.json")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registry():
    # Return the whole registry file as a dict.
    return _read_json(REGISTRY_FILE)


def load_eval_dataset():
    # Return the evaluation set as a dict with "examples" and
    # "target_max_sentences".
    return _read_json(EVAL_DATASET_FILE)


def list_versions():
    # Version ids in registry order, for example ["v1", "v2", "v3", "v4"].
    return [entry["version"] for entry in load_registry()["versions"]]


def get_version_metadata(version):
    # Look up one registry entry by its version id.
    for entry in load_registry()["versions"]:
        if entry["version"] == version:
            return entry

    raise KeyError(
        "Version " + str(version) + " is not in registry.json. "
        "Known versions: " + ", ".join(list_versions())
    )


def load_prompt_template(version):
    # Read the raw prompt text for a version, placeholder still in place.
    metadata = get_version_metadata(version)
    path = os.path.join(EXPERIMENTS_DIR, metadata["file"])

    with open(path, "r", encoding="utf-8") as handle:
        template = handle.read()

    # A prompt template with no {text} placeholder would silently send the
    # same prompt for every example and produce meaningless scores. Better
    # to fail loudly here than to debug flat metrics later.
    if PROMPT_PLACEHOLDER not in template:
        raise ValueError(
            metadata["file"] + " does not contain the " +
            PROMPT_PLACEHOLDER + " placeholder, so the input text would "
            "never be inserted."
        )

    return template


def render_prompt(version, text):
    # Substitute one input text into a prompt template.
    #
    # Plain str.replace, not str.format - the prompts contain literal
    # braces in places and .format() would try to interpret them. This was
    # the first thing that broke when I wrote prompt v4.
    template = load_prompt_template(version)
    return template.replace(PROMPT_PLACEHOLDER, text)


if __name__ == "__main__":
    # Print the registry so I can see all versions and what changed
    # without opening the JSON file.
    registry = load_registry()
    dataset = load_eval_dataset()

    print("Task: " + registry["task"])
    print("Prompt versions: " + str(len(registry["versions"])))
    print("Evaluation examples: " + str(len(dataset["examples"])))

    for entry in registry["versions"]:
        print("\n" + entry["version"] + " (" + entry["file"] + ")")
        print("  technique   : " + entry["technique"])
        print("  what changed: " + entry["what_changed"])
