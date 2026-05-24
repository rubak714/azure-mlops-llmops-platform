import mlflow
from mlflow import MlflowClient

# This script takes the best run from the experiment and registers
# it in the MLflow model registry - but only if it passed the promotion threshold in evaluate.py.
#
# In a real pipeline this would run automatically after evaluate.py
# confirms the model is good enough. Here I am running it manually to understand how the registry works.

PROMOTION_THRESHOLD = 0.85
EXPERIMENT_NAME = "iris-random-forest"
MODEL_NAME = "iris-classifier"


def get_best_run(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        print("Run train.py first.")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"]
    )

    if runs.empty:
        print("No runs found in this experiment.")
        return None

    return runs.iloc[0]


def register_if_passing():
    best_run = get_best_run(EXPERIMENT_NAME)

    if best_run is None:
        return

    f1 = best_run["metrics.f1_score"]
    run_id = best_run["run_id"]

    print(f"\nChecking best run: {run_id}")
    print(f"F1 score : {f1:.3f}")
    print(f"Threshold: {PROMOTION_THRESHOLD}")

    if f1 < PROMOTION_THRESHOLD:
        print("\nModel did not pass the threshold. Skipping registration.")
        print("Go back to train.py and try different parameters.")
        return

    # the model artifact path inside the run is called "model"
    # because that is what we named it in mlflow.sklearn.log_model
    model_uri = f"runs:/{run_id}/model"

    print(f"\nModel passed. Registering as '{MODEL_NAME}'...")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"\nRegistered successfully.")
    print(f"  Name   : {result.name}")
    print(f"  Version: {result.version}")
    print(f"\nCheck the Models tab in the MLflow UI to see it.")


if __name__ == "__main__":
    register_if_passing()