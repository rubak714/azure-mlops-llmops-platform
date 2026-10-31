import sys

import mlflow

from src.llm_client import (
    DEFAULT_TEMPERATURE,
    complete,
    describe_provider,
    stub_fingerprint,
)
from src.prompt_metrics import (
    METRIC_WEIGHTS,
    PROMPT_PROMOTION_THRESHOLD,
    average_scores,
    is_above_threshold,
    score_summary,
)
from src.prompt_registry import (
    get_version_metadata,
    list_versions,
    load_eval_dataset,
    load_prompt_template,
    render_prompt,
)

# This is the LLMOps mirror of evaluate.py.
#
# evaluate.py trains nothing - it reads the runs train.py logged, finds the
# best one by F1, and checks it against a threshold. This script does the
# same thing one level up: it runs every prompt version over the whole
# evaluation set, scores the outputs, logs each version as its own MLflow
# run, finds the best version by composite score, and checks it against a
# promotion threshold.
#
# The reason this is worth building at all is the thing that took me
# longest to accept about LLMOps: the prompt is part of the system in the
# same way model weights are. Editing a prompt changes behaviour just like
# retraining does, so it needs the same treatment - a version, a recorded
# score, and a gate standing between it and production.

EXPERIMENT_NAME = "prompt-summarization"


def evaluate_version(version, dataset):
    # Run one prompt version over every example in the evaluation set and
    # return the averaged scores plus the raw outputs.
    target_max_sentences = dataset["target_max_sentences"]

    per_example_scores = []
    outputs = []

    for example in dataset["examples"]:
        prompt = render_prompt(version, example["text"])
        summary = complete(prompt, temperature=DEFAULT_TEMPERATURE)

        scores = score_summary(
            summary=summary,
            reference=example["reference"],
            source=example["text"],
            max_sentences=target_max_sentences,
        )

        per_example_scores.append(scores)

        # Keeping the generated text, not just the score, is the whole
        # point. A composite of 0.62 means nothing on its own - I need to
        # read what the model actually wrote to know what to fix. These get
        # logged to MLflow as an artifact.
        outputs.append({
            "id": example["id"],
            "summary": summary,
            "scores": scores,
        })

    return average_scores(per_example_scores), outputs


def log_version_run(version, dataset, averages, outputs):
    # One MLflow run per prompt version, structured the same way train.py
    # structures its runs - parameters describe the configuration, metrics
    # describe the result, artifacts hold the evidence.
    metadata = get_version_metadata(version)
    template = load_prompt_template(version)

    with mlflow.start_run(run_name=version):
        # The prompt version and technique are parameters in exactly the
        # same sense that n_estimators is a parameter on the ML side.
        mlflow.log_param("prompt_version", version)
        mlflow.log_param("technique", metadata["technique"])
        mlflow.log_param("provider", describe_provider())
        mlflow.log_param("temperature", DEFAULT_TEMPERATURE)
        mlflow.log_param("eval_examples", len(dataset["examples"]))
        mlflow.log_param("target_max_sentences", dataset["target_max_sentences"])

        # A hash of the prompt text. If two runs disagree I want to be able
        # to prove whether the prompt actually differed, and comparing
        # twelve hex characters is easier than diffing two paragraphs.
        mlflow.log_param("prompt_sha", stub_fingerprint(template))

        for name, value in averages.items():
            mlflow.log_metric(name, value)

        # The prompt itself is the artifact here, the way the pickled model
        # is the artifact in train.py. Without it the run is not
        # reproducible.
        mlflow.log_text(template, "prompt.txt")
        mlflow.log_dict({"outputs": outputs}, "generated_summaries.json")


def print_version_result(version, averages):
    metadata = get_version_metadata(version)

    print("\n" + version + " - " + metadata["technique"])

    for name in METRIC_WEIGHTS:
        # Pad the metric names so the numbers line up in the terminal.
        print("  " + name.ljust(18) + ": " + str(round(averages[name], 3)))

    print("  " + "composite".ljust(18) + ": " + str(round(averages["composite"], 3)))


def evaluate_all_versions():
    # Evaluate every registered prompt version and return
    # {version: averaged scores}.
    dataset = load_eval_dataset()
    versions = list_versions()

    print("Provider : " + describe_provider())
    print("Examples : " + str(len(dataset["examples"])))
    print("Versions : " + ", ".join(versions))
    print("Threshold: " + str(PROMPT_PROMOTION_THRESHOLD) + " composite")

    results = {}

    for version in versions:
        averages, outputs = evaluate_version(version, dataset)
        log_version_run(version, dataset, averages, outputs)
        print_version_result(version, averages)
        results[version] = averages

    return results


def get_best_version(results):
    # Highest composite score wins. Same selection rule as evaluate.py
    # uses for F1, just over prompt versions instead of training runs.
    if not results:
        return None

    return max(results.items(), key=lambda item: item[1]["composite"])[0]


def report_promotion_decision(results):
    best_version = get_best_version(results)

    if best_version is None:
        print("\nNo prompt versions were evaluated.")
        return False

    composite = results[best_version]["composite"]

    print("\n" + "-" * 58)
    print("Best version : " + best_version)
    print("Composite    : " + str(round(composite, 3)))
    print("Threshold    : " + str(PROMPT_PROMOTION_THRESHOLD))

    if is_above_threshold(composite):
        print("\n  Result: PASSED - prompt meets the threshold.")
        print("  This version would be promoted as the active prompt.")
        return True

    print("\n  Result: FAILED - prompt is below the threshold.")
    print("  No version is good enough to promote.")
    print("  Next step: add a stricter version to the registry.")
    return False


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = evaluate_all_versions()
    passed = report_promotion_decision(all_results)

    print("\nOpen the MLflow UI and look at the " +
          EXPERIMENT_NAME + " experiment to compare versions.")

    # --strict turns the gate into a build failure, the way a failing test
    # does. Without it the script only reports, which is what I want when
    # I am iterating on a prompt by hand.
    if "--strict" in sys.argv and not passed:
        sys.exit(1)
