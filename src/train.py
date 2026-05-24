import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# I am trying three different combinations of n_estimators and max_depth
# to see how much the F1 score changes between runs.
# These are logged to MLflow so I can compare them side by side.

EXPERIMENTS = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    # deliberately weak settings to test the threshold gate
    # n_estimators=2 and max_depth=1 is too weak for even a clean
    # dataset — added to show what a failed promotion looks like
    {"n_estimators": 2, "max_depth": 1},
]


def load_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_log(n_estimators, max_depth, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        # log the parameters so I can see what was used in each run
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # log the results so I can compare runs in the MLflow UI
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # save the model as an artifact in this run
        mlflow.sklearn.log_model(model, "model")

        print(f"n_estimators={n_estimators} | max_depth={max_depth} "
              f"| accuracy={accuracy:.3f} | f1={f1:.3f}")

        return f1


if __name__ == "__main__":
    mlflow.set_experiment("iris-random-forest")

    X_train, X_test, y_train, y_test = load_data()

    for params in EXPERIMENTS:
        train_and_log(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
