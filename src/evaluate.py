import mlflow

# This script checks whether the best run from the experiment
# meets the minimum F1 score needed to be promoted to the registry.
# I set the threshold at 0.85 - anything below that is not good
# enough to be registered as a usable model version.

PROMOTION_THRESHOLD = 0.85
EXPERIMENT_NAME = "iris-random-forest"


def get_best_run(experiment_name):
    # search all runs in the experiment and return the one
    # with the highest F1 score
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print("Experiment not found: " + experiment_name)
        print("Run train.py first to create the experiment.")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"]
    )

    if runs.empty:
        print("No runs found in this experiment.")
        return None

    best_run = runs.iloc[0]
    return best_run


def evaluate_best_run():
    best_run = get_best_run(EXPERIMENT_NAME)

    if best_run is None:
        return

    f1 = best_run["metrics.f1_score"]
    run_id = best_run["run_id"]
    n_estimators = best_run["params.n_estimators"]
    max_depth = best_run["params.max_depth"]

    print("\nBest run found:")
    print("  run_id      : " + str(run_id))
    print("  n_estimators: " + str(n_estimators))
    print("  max_depth   : " + str(max_depth))
    print("  f1_score    : " + str(round(f1, 3)))
    print("  threshold   : " + str(PROMOTION_THRESHOLD))

    if f1 >= PROMOTION_THRESHOLD:
        print("\n  Result: PASSED — model meets the threshold.")
        print("  This run would be promoted to the model registry.")
    else:
        print("\n  Result: FAILED — model is below the threshold.")
        print("  This run would not be promoted.")
        print("  Next step: try different parameters in train.py.")


if __name__ == "__main__":
    evaluate_best_run()
