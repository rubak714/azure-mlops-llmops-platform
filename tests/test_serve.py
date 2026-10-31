import pytest
from fastapi.testclient import TestClient

from src import serve
from src.serve import CLASS_NAMES, FEATURE_NAMES, N_FEATURES, app

# Tests for the serving API.
#
# None of these need a trained model or a populated MLflow registry. That
# is the reason load_model and get_model are separate functions - the
# tests replace the loading step with a stub, so the endpoints can be
# tested on a clean checkout and in CI, where no model has ever been
# registered.
#
# What is being tested is the API contract: does it validate input, does
# it say something useful when there is no model, does it map class ids
# back to names. The model's accuracy is train.py and evaluate.py's
# problem and is covered there.

client = TestClient(app)

VALID_SETOSA = [5.1, 3.5, 1.4, 0.2]
VALID_VIRGINICA = [6.7, 3.0, 5.2, 2.3]


class StubModel:
    # Stands in for the pyfunc model MLflow would return. It only has to
    # do what the endpoint actually uses - take a list of instances and
    # return one label per instance.
    def __init__(self, labels):
        self.labels = labels
        self.received = None

    def predict(self, instances):
        self.received = instances
        return self.labels[:len(instances)]


@pytest.fixture(autouse=True)
def clear_cache():
    # The model cache is module level, so without this a stub installed by
    # one test would leak into the next one.
    serve.reset_model_cache()
    yield
    serve.reset_model_cache()


@pytest.fixture
def served_model(monkeypatch):
    # Install a stub model as though it had been loaded from the registry.
    def _install(labels):
        stub = StubModel(labels)
        monkeypatch.setattr(serve, "load_model", lambda: stub)
        return stub

    return _install


@pytest.fixture
def no_model(monkeypatch):
    # Simulate an empty registry. MLflow raises several different
    # exception types for this, so the code catches broadly and this
    # fixture only has to raise something.
    def _raise():
        raise RuntimeError("Registered Model with name=iris-classifier not found")

    monkeypatch.setattr(serve, "load_model", _raise)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_reports_ok_when_a_model_is_loaded(served_model):
    served_model([0])

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_health_reports_degraded_when_no_model(no_model):
    # Still 200 - the service itself is alive and answering. The body is
    # what says it cannot serve predictions yet.
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_health_does_not_raise_when_the_registry_is_empty(no_model):
    # The failure this guards against: an unhandled exception in the
    # health check means the container reports unhealthy for a reason
    # nobody can see without reading the logs.
    assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Serving with no model registered
# ---------------------------------------------------------------------------

def test_predict_returns_503_when_no_model(no_model):
    # 503 and not 500. There is no bug - there is simply no promoted model
    # yet, which is a deployment state.
    response = client.post("/predict", json={"instances": [VALID_SETOSA]})

    assert response.status_code == 503


def test_503_message_says_how_to_fix_it(no_model):
    # The error should name the commands that populate the registry,
    # because "no model available" on its own leaves the reader guessing.
    detail = client.post("/predict", json={"instances": [VALID_SETOSA]}).json()["detail"]

    assert "register_model.py" in detail


def test_model_info_returns_503_when_no_model(no_model):
    assert client.get("/model").status_code == 503


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_returns_a_class_name_not_just_a_number(served_model):
    # Returning a bare 0 to a caller is not useful.
    served_model([0])

    body = client.post("/predict", json={"instances": [VALID_SETOSA]}).json()

    assert body["predictions"][0]["class_id"] == 0
    assert body["predictions"][0]["class_name"] == "setosa"


def test_predict_handles_a_batch(served_model):
    served_model([0, 2])

    body = client.post(
        "/predict",
        json={"instances": [VALID_SETOSA, VALID_VIRGINICA]},
    ).json()

    assert len(body["predictions"]) == 2
    assert body["predictions"][1]["class_name"] == "virginica"


def test_predict_passes_the_instances_through_unchanged(served_model):
    # Catches a reshaping or reordering bug between the request and the
    # model, which would produce confident and completely wrong answers.
    stub = served_model([0])

    client.post("/predict", json={"instances": [VALID_SETOSA]})

    assert stub.received == [VALID_SETOSA]


def test_response_reports_which_model_answered(served_model):
    # The first thing I want to know when a prediction looks wrong.
    served_model([0])

    body = client.post("/predict", json={"instances": [VALID_SETOSA]}).json()

    assert body["model_name"] == serve.MODEL_NAME
    assert body["model_stage"] == serve.MODEL_STAGE


def test_unknown_class_id_does_not_crash(served_model):
    # A model returning a label outside the known range should produce
    # "unknown", not an IndexError and a 500.
    served_model([99])

    body = client.post("/predict", json={"instances": [VALID_SETOSA]}).json()

    assert body["predictions"][0]["class_name"] == "unknown"


def test_model_failure_becomes_a_500_with_a_message(served_model, monkeypatch):
    stub = served_model([0])

    def _explode(instances):
        raise ValueError("feature mismatch")

    monkeypatch.setattr(stub, "predict", _explode)

    response = client.post("/predict", json={"instances": [VALID_SETOSA]})

    assert response.status_code == 500
    assert "feature mismatch" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def test_too_few_features_is_rejected(served_model):
    served_model([0])

    response = client.post("/predict", json={"instances": [[5.1, 3.5, 1.4]]})

    assert response.status_code == 422


def test_too_many_features_is_rejected(served_model):
    served_model([0])

    response = client.post("/predict", json={"instances": [[5.1, 3.5, 1.4, 0.2, 9.9]]})

    assert response.status_code == 422


def test_validation_error_names_the_expected_feature_count(served_model):
    # The caller should learn what was wrong without reading the source.
    served_model([0])

    body = client.post("/predict", json={"instances": [[5.1, 3.5, 1.4]]}).json()

    assert str(N_FEATURES) in str(body)


def test_empty_instances_is_rejected(served_model):
    served_model([0])

    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_negative_measurement_is_rejected(served_model):
    # A negative petal length is a bad request, not a model problem. It
    # should never reach the model.
    served_model([0])

    response = client.post("/predict", json={"instances": [[5.1, 3.5, -1.4, 0.2]]})

    assert response.status_code == 422


def test_implausibly_large_measurement_is_rejected(served_model):
    # Catches a unit mix-up - millimetres sent where centimetres were
    # expected - which would otherwise be scored silently and wrongly.
    served_model([0])

    response = client.post("/predict", json={"instances": [[51.0, 35.0, 914.0, 2.0]]})

    assert response.status_code == 422


def test_validation_rejects_before_the_model_is_called(served_model):
    # The model must not see an invalid instance at all.
    stub = served_model([0])

    client.post("/predict", json={"instances": [[5.1, 3.5, 1.4]]})

    assert stub.received is None


def test_missing_instances_key_is_rejected(served_model):
    served_model([0])

    assert client.post("/predict", json={}).status_code == 422


def test_non_numeric_value_is_rejected(served_model):
    served_model([0])

    response = client.post(
        "/predict",
        json={"instances": [["not", "a", "number", "at all"]]},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Model info and caching
# ---------------------------------------------------------------------------

def test_model_info_lists_features_and_classes(served_model):
    served_model([0])

    body = client.get("/model").json()

    assert body["features"] == FEATURE_NAMES
    assert body["classes"] == CLASS_NAMES


def test_model_is_loaded_once_and_cached(monkeypatch):
    # Loading from the registry is slow, so it must not happen per
    # request. Counting calls is the only way to catch a regression here.
    calls = {"count": 0}

    def _counting_load():
        calls["count"] += 1
        return StubModel([0])

    monkeypatch.setattr(serve, "load_model", _counting_load)

    client.post("/predict", json={"instances": [VALID_SETOSA]})
    client.post("/predict", json={"instances": [VALID_SETOSA]})
    client.post("/predict", json={"instances": [VALID_SETOSA]})

    assert calls["count"] == 1


def test_reset_clears_the_cache(served_model):
    served_model([0])
    client.post("/predict", json={"instances": [VALID_SETOSA]})

    serve.reset_model_cache()

    assert serve._model is None


def test_model_uri_uses_the_registry_scheme():
    # "models:/" and not "runs:/". The serving layer must read whatever
    # register_model.py promoted, not a hardcoded training run.
    assert serve.model_uri().startswith("models:/")


# ---------------------------------------------------------------------------
# Version resolution
# ---------------------------------------------------------------------------

class FakeVersion:
    def __init__(self, version):
        self.version = str(version)


def _fake_registry(monkeypatch, versions):
    class FakeClient:
        def search_model_versions(self, query):
            return [FakeVersion(v) for v in versions]

    monkeypatch.setattr(serve.mlflow, "MlflowClient", FakeClient)


def test_latest_version_picks_the_highest_number(monkeypatch):
    # Deliberately out of order, and crossing the 9-to-10 boundary where
    # string sorting and numeric sorting disagree. "10" sorts before "9"
    # as text, so comparing version strings would serve a stale model.
    _fake_registry(monkeypatch, [3, 10, 9, 1])

    assert serve.latest_version_number() == 10


def test_latest_resolves_to_a_numeric_uri(monkeypatch):
    # The point of the whole thing: avoid "models:/name/latest", which
    # goes through the stage-based lookup MLflow deprecated in 2.9.
    _fake_registry(monkeypatch, [1, 2])
    monkeypatch.setattr(serve, "MODEL_STAGE", "latest")

    assert serve.resolve_model_uri() == f"models:/{serve.MODEL_NAME}/2"


def test_a_pinned_stage_is_taken_literally(monkeypatch):
    # Setting MODEL_STAGE to a number pins that version and skips the
    # lookup entirely.
    monkeypatch.setattr(serve, "MODEL_STAGE", "3")

    assert serve.resolve_model_uri() == f"models:/{serve.MODEL_NAME}/3"


def test_empty_registry_raises_when_resolving(monkeypatch):
    _fake_registry(monkeypatch, [])
    monkeypatch.setattr(serve, "MODEL_STAGE", "latest")

    with pytest.raises(RuntimeError):
        serve.latest_version_number()


def test_model_info_reports_the_resolved_version(served_model, monkeypatch):
    # "latest" tells you nothing a week later. The concrete version
    # number is what matters when a prediction looks wrong.
    served_model([0])
    _fake_registry(monkeypatch, [1, 2, 7])
    monkeypatch.setattr(serve, "MODEL_STAGE", "latest")

    body = client.get("/model").json()

    assert body["resolved_uri"].endswith("/7")
