import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

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
        print(f"Experiment '{experiment_name}' not found.")
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

    print(f"\nBest run found:")
    print(f"  run_id      : {run_id}")
    print(f"  n_estimators: {n_estimators}")
    print(f"  max_depth   : {max_depth}")
    print(f"  f1_score    : {f1:.3f}")
    print(f"  threshold   : {PROMOTION_THRESHOLD}")

    if f1 >= PROMOTION_THRESHOLD:
        print(f"\n  Result: PASSED - model meets the threshold.")
        print(f"  This run would be promoted to the model registry.")
    else:
        print(f"\n  Result: FAILED - model is below the threshold.")
        print(f"  This run would not be promoted.")
        print(f"  Next step: try different parameters in train.py.")


if __name__ == "__main__":
    evaluate_best_run()