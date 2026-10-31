import os

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# A serving layer for the model that register_model.py promotes.
#
# This was the last gap in architecture.md and the one that bothered me
# most. The pipeline trained a model, scored it, gated it on a threshold,
# and registered it - and then nothing consumed it. A model sitting in a
# registry is not in production, it is in storage.
#
# What this adds is the step that makes the registry mean something: an
# HTTP endpoint that loads the promoted version and answers prediction
# requests against it.
#
# Run it with:
#
#     uvicorn src.serve:app --reload
#
# then open http://127.0.0.1:8000/docs for the generated API docs.

MODEL_NAME = os.environ.get("MODEL_NAME", "iris-classifier")

# "latest" rather than a pinned version number, because the whole point of
# the registry is that register_model.py decides what the current best
# model is. Pinning a version here would move that decision into the
# serving code, where it does not belong.
MODEL_STAGE = os.environ.get("MODEL_STAGE", "latest")

# Iris targets are 0, 1, 2 and the model returns those integers. Returning
# a bare 2 to a caller is not useful, so the names are mapped back here.
# The order is fixed by sklearn's load_iris and must not be reordered.
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

N_FEATURES = len(FEATURE_NAMES)

# Iris measurements are in centimetres and are always positive. A negative
# petal length is not a model problem, it is a bad request, and it should
# be rejected with a 422 rather than quietly scored. The upper bound is
# deliberately loose - it is there to catch a unit mix-up or a stray
# sensor value, not to second-guess the caller.
MIN_FEATURE_VALUE = 0.0
MAX_FEATURE_VALUE = 100.0

app = FastAPI(
    title="Iris Classifier",
    description=(
        "Serves the model that evaluate.py gated and register_model.py "
        "promoted to the MLflow registry."
    ),
    version="1.0.0",
)

# Module-level cache for the loaded model. Loading from the registry is
# slow and there is no reason to do it per request.
#
# It is deliberately lazy rather than loaded at import time. Importing
# this module must not require a populated MLflow registry, otherwise the
# tests cannot import the app at all without first training and
# registering a model, and /health could never report that the model is
# missing - the process would have already failed to start.
_model = None


class PredictionRequest(BaseModel):
    # One or more flower measurements to classify.
    instances: list[list[float]] = Field(
        ...,
        description=(
            "List of measurements. Each is 4 floats in the order: "
            + ", ".join(FEATURE_NAMES)
        ),
        examples=[[[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]],
    )

    @field_validator("instances")
    @classmethod
    def check_instances(cls, instances):
        # Validating the shape here rather than letting it reach the model
        # means a caller gets "expected 4 features, got 3" instead of
        # whatever sklearn raises three layers down.
        if not instances:
            raise ValueError("instances must not be empty")

        for index, instance in enumerate(instances):
            if len(instance) != N_FEATURES:
                raise ValueError(
                    f"instance {index} has {len(instance)} features, "
                    f"expected {N_FEATURES}: {FEATURE_NAMES}"
                )

            for position, value in enumerate(instance):
                if not MIN_FEATURE_VALUE <= value <= MAX_FEATURE_VALUE:
                    raise ValueError(
                        f"instance {index} value {value} for "
                        f"'{FEATURE_NAMES[position]}' is outside the "
                        f"plausible range "
                        f"{MIN_FEATURE_VALUE} to {MAX_FEATURE_VALUE}"
                    )

        return instances


class Prediction(BaseModel):
    class_id: int
    class_name: str


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    model_name: str
    model_stage: str


def model_uri():
    # The reference string, used for display and in error messages. It is
    # what was asked for, which is not necessarily what gets loaded - see
    # resolve_model_uri below.
    return f"models:/{MODEL_NAME}/{MODEL_STAGE}"


def latest_version_number():
    # Find the highest registered version number for the model.
    #
    # Doing this by hand rather than letting MLflow resolve "latest",
    # because "models:/name/latest" goes through the stage-based lookup
    # that MLflow deprecated in 2.9 and plans to remove. Asking for the
    # versions and taking the highest number uses the numeric URI form,
    # which is not deprecated.
    #
    # The deprecation warning was the useful part here: it pointed at the
    # fact that "latest" was never really a version, it was a stage alias.
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not versions:
        raise RuntimeError(f"No versions registered for model '{MODEL_NAME}'")

    return max(int(version.version) for version in versions)


def resolve_model_uri():
    # Turn the reference string into a concrete versioned URI.
    #
    # Anything other than "latest" is taken literally, so setting
    # MODEL_STAGE to "3" pins version 3 and bypasses all of this.
    if MODEL_STAGE == "latest":
        return f"models:/{MODEL_NAME}/{latest_version_number()}"

    return model_uri()


def load_model():
    # Pull the promoted model out of the MLflow registry.
    #
    # Separated from get_model so the tests can replace the loading step
    # without needing a real registry, and so the caching logic is not
    # tangled up with the MLflow call.
    return mlflow.pyfunc.load_model(resolve_model_uri())


def get_model():
    # Return the cached model, loading it on first use.
    global _model

    if _model is None:
        _model = load_model()

    return _model


def reset_model_cache():
    # Used by the tests, and useful in a notebook after re-registering a
    # model. Without this the process would keep serving the version it
    # happened to load first.
    global _model
    _model = None


def model_is_available():
    # Whether a model can actually be served right now.
    #
    # This is what /health reports on. An endpoint that returns 200 while
    # being unable to answer a single prediction is worse than useless in
    # a deployment - it tells a load balancer to send it traffic.
    try:
        get_model()
        return True
    except Exception:
        # Deliberately broad. MLflow raises several different exception
        # types for "no registry", "no such model", and "no versions yet",
        # and from the caller's point of view they are the same situation.
        return False


@app.get("/health")
def health():
    # Liveness and readiness in one response. Kept cheap enough to be
    # polled, because after the first call the model is cached.
    available = model_is_available()

    return {
        "status": "ok" if available else "degraded",
        "model_loaded": available,
        "model_uri": model_uri(),
    }


@app.get("/model")
def model_info():
    # What is actually being served. The first thing I want to know when a
    # prediction looks wrong is which model version produced it.
    if not model_is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No model available at {model_uri()}. "
                "Run train.py, evaluate.py, then register_model.py first."
            ),
        )

    # resolved_uri carries the concrete version number that actually
    # answered, which is the thing worth knowing when a prediction looks
    # wrong. "latest" tells you nothing a week later.
    try:
        resolved = resolve_model_uri()
    except Exception:
        resolved = None

    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_uri": model_uri(),
        "resolved_uri": resolved,
        "features": FEATURE_NAMES,
        "classes": CLASS_NAMES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # 503 and not 500. The service is fine - there is simply no promoted
    # model to serve yet, which is a deployment state rather than a bug,
    # and the message says exactly which command fixes it.
    if not model_is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                f"No model available at {model_uri()}. "
                "Run train.py, evaluate.py, then register_model.py first."
            ),
        )

    model = get_model()

    try:
        raw_predictions = model.predict(request.instances)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to produce a prediction: {error}",
        )

    predictions = []
    for value in raw_predictions:
        class_id = int(value)

        # Guard against a model that returns a label outside the range the
        # class list covers. Returning an index error to the caller would
        # be a worse failure than saying the label is unknown.
        if 0 <= class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = "unknown"

        predictions.append(Prediction(class_id=class_id, class_name=class_name))

    return PredictionResponse(
        predictions=predictions,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )
