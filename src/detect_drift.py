import sys

import mlflow
import numpy as np
from sklearn.datasets import load_iris

# Drift detection - the last gap I listed in architecture.md.
#
# The problem it solves: train.py and evaluate.py tell me whether a model
# is good on the data I had when I trained it. They say nothing about
# whether that data still looks like the data arriving now. A model can
# hold a perfect F1 on its own test split for a year while the live inputs
# drift away underneath it, and nothing in the pipeline so far would
# notice. Model performance is monitored; model *inputs* are not.
#
# So this script compares two windows of data - a reference window, which
# is what the model was trained on, and a current window, which stands in
# for what is arriving now - and asks whether their distributions still
# match, feature by feature.
#
# Two measures, because they catch different things:
#
#   PSI (Population Stability Index) bins the reference distribution and
#   asks how much probability mass moved between bins. It is the standard
#   measure in credit risk and it has conventional interpretation bands,
#   which is exactly what a threshold gate needs.
#
#   The Kolmogorov-Smirnov statistic is the largest gap between the two
#   cumulative distributions. It makes no assumption about shape and it
#   catches a shift that PSI's bin edges happen to straddle.
#
# Both are written out in numpy rather than imported from scipy. Partly so
# the maths is visible in a repo whose point is learning, partly so this
# script has no dependency that train.py does not already have. Same
# reasoning as writing out ROUGE-L by hand in prompt_metrics.py.

EXPERIMENT_NAME = "iris-drift-detection"

# The conventional PSI interpretation bands. These are not something I
# tuned - they are the standard thresholds used in industry, which is a
# large part of why PSI is a good choice for an automated gate. Having a
# number everyone already agrees on beats inventing my own.
#
#   PSI < 0.10         distributions are effectively the same
#   0.10 <= PSI < 0.25 moderate shift, worth looking at
#   PSI >= 0.25        significant shift, investigate and probably retrain
PSI_NO_DRIFT = 0.10
DRIFT_PSI_THRESHOLD = 0.25

# Number of bins for PSI. Ten is the usual choice. Too few and a real
# shift hides inside a bin; too many and small samples give empty bins
# that make the logarithm blow up.
PSI_BINS = 10

# PSI divides by the reference proportion and takes a log, so a bin that
# is empty in either window would give a division by zero or a log of
# zero. Substituting a small floor is the standard fix.
EPSILON = 1e-6

# Significance level for the KS test. 1.36 is the standard critical
# coefficient for alpha = 0.05 - it comes from the Kolmogorov
# distribution, and hardcoding the one value I need avoids pulling in
# scipy for a single constant.
KS_CRITICAL_COEFFICIENT = 1.36

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Population Stability Index
# ---------------------------------------------------------------------------

def psi_bin_edges(reference, bins=PSI_BINS):
    # Bin edges come from the quantiles of the REFERENCE window, not from
    # the combined data. That matters: the reference is the baseline the
    # model was trained on, so it defines what "normal" means. Letting the
    # current window influence the bin edges would let drift hide by
    # moving the goalposts with it.
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(reference, quantiles)

    # Quantile binning on a feature with repeated values can produce
    # duplicate edges, which would give zero-width bins. np.unique both
    # sorts and de-duplicates.
    edges = np.unique(edges)

    # Open out the outer edges so values beyond the reference range still
    # land in the first or last bin instead of falling outside every bin.
    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


def _proportions(values, edges):
    # Fraction of the sample falling in each bin, with the zero floor
    # applied so PSI stays finite.
    counts, _ = np.histogram(values, bins=edges)
    proportions = counts / max(len(values), 1)

    return np.where(proportions == 0, EPSILON, proportions)


def population_stability_index(reference, current, bins=PSI_BINS):
    # PSI = sum over bins of (actual - expected) * ln(actual / expected)
    #
    # Each bin contributes the amount of probability mass that moved,
    # weighted by the log ratio of how much moved. Identical
    # distributions give 0. The measure is not symmetric - swapping the
    # arguments gives a slightly different number - because "expected"
    # and "actual" are not interchangeable roles.
    edges = psi_bin_edges(reference, bins=bins)

    expected = _proportions(reference, edges)
    actual = _proportions(current, edges)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


# ---------------------------------------------------------------------------
# Two-sample Kolmogorov-Smirnov
# ---------------------------------------------------------------------------

def ks_statistic(reference, current):
    # The KS statistic is the largest vertical distance between the two
    # empirical cumulative distribution functions.
    #
    # Evaluating both ECDFs at every observed value is enough, because the
    # largest gap between two step functions always occurs at a step.
    # searchsorted with side="right" gives the count of values <= x, which
    # divided by the sample size is the ECDF at x.
    reference_sorted = np.sort(reference)
    current_sorted = np.sort(current)

    all_values = np.concatenate([reference_sorted, current_sorted])

    reference_cdf = np.searchsorted(reference_sorted, all_values, side="right") / len(reference_sorted)
    current_cdf = np.searchsorted(current_sorted, all_values, side="right") / len(current_sorted)

    return float(np.max(np.abs(reference_cdf - current_cdf)))


def ks_critical_value(n_reference, n_current):
    # Above this value the two samples are unlikely to have come from the
    # same distribution at alpha = 0.05. The critical value shrinks as the
    # samples grow, which is the property that matters - the same gap is
    # more convincing from 1000 points than from 30.
    #
    # float() on the way out, not decoration. np.sqrt returns a np.float64,
    # which then makes the comparison below return a np.bool_ rather than a
    # bool, and np.bool_(False) is not False - so "assert ... is False"
    # fails on a correct result. Every numpy value that leaves this module
    # gets converted back to a Python type at the boundary.
    return float(KS_CRITICAL_COEFFICIENT * np.sqrt(
        (n_reference + n_current) / (n_reference * n_current)
    ))


def ks_indicates_drift(reference, current):
    return bool(ks_statistic(reference, current) > ks_critical_value(
        len(reference), len(current)
    ))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def is_drifted(psi):
    # The drift gate. >= and not >, so a feature landing exactly on 0.25
    # counts as drifted.
    #
    # Note the direction is the opposite of the F1 gate in evaluate.py -
    # there a high number is good and the gate is a floor, here a high
    # number is bad and the gate is a ceiling. The boundary convention is
    # the same either way: the threshold value itself triggers the gate.
    return psi >= DRIFT_PSI_THRESHOLD


def classify_psi(psi):
    # Turn a PSI into the conventional label, for printing.
    if psi < PSI_NO_DRIFT:
        return "no drift"

    if psi < DRIFT_PSI_THRESHOLD:
        return "moderate"

    return "SIGNIFICANT"


def analyse_features(reference, current, feature_names):
    # Run both measures on every feature and return one result dict per
    # feature. Per-feature results matter more than a single overall
    # number - "the data drifted" is not actionable, "petal length drifted
    # and nothing else did" points straight at a recalibrated instrument.
    results = []

    for index, name in enumerate(feature_names):
        reference_column = reference[:, index]
        current_column = current[:, index]

        psi = population_stability_index(reference_column, current_column)

        results.append({
            "feature": name,
            "psi": psi,
            "classification": classify_psi(psi),
            "drifted": is_drifted(psi),
            "ks": ks_statistic(reference_column, current_column),
            "ks_critical": ks_critical_value(len(reference_column), len(current_column)),
            "ks_drifted": ks_indicates_drift(reference_column, current_column),
        })

    return results


# ---------------------------------------------------------------------------
# Building the windows to compare
# ---------------------------------------------------------------------------

def load_reference_window():
    # The reference window is the data the model was trained on. Iris
    # again, for the same reasons given in the README - it loads in one
    # line and works offline, which keeps the focus on the drift logic.
    iris = load_iris()
    return iris.data, list(iris.feature_names), iris.target


def make_current_window(reference, target, scenario, seed=RANDOM_SEED):
    # Iris is a fixed dataset, so there is no "data arriving now" to
    # compare against. I have to construct the current window, and I would
    # rather do that explicitly in a named function than hide it.
    #
    # Each scenario is a different kind of drift, because they do not all
    # look the same and a detector that only catches one is not much use.
    rng = np.random.default_rng(seed)

    if scenario == "no-drift":
        # A bootstrap resample of the reference itself. Same underlying
        # distribution, different sample. This is the control case and the
        # most important of the three - a detector that fires here is
        # producing false alarms, and a monitoring system people learn to
        # ignore is worse than none.
        indices = rng.choice(len(reference), size=len(reference), replace=True)
        return reference[indices]

    if scenario == "measurement-shift":
        # One feature shifts and the rest do not. This is what a
        # recalibrated or miscalibrated sensor looks like, and it is the
        # case where per-feature reporting earns its keep.
        current = reference.copy()
        current[:, 2] = current[:, 2] + 0.8
        return current

    if scenario == "population-shift":
        # The input population changes - one class stops appearing. No
        # individual measurement is wrong, but the mix of what arrives has
        # changed. A seasonal product, a new market, a referral source
        # that dried up.
        keep = target != 0
        return reference[keep]

    if scenario == "noise-increase":
        # Same centre, wider spread. Measurements got noisier without the
        # mean moving, so a monitor watching only the mean sees nothing.
        noise = rng.normal(loc=0.0, scale=0.6, size=reference.shape)
        return reference + noise

    raise ValueError("Unknown scenario: " + str(scenario))


SCENARIOS = [
    "no-drift",
    "measurement-shift",
    "population-shift",
    "noise-increase",
]


# ---------------------------------------------------------------------------
# Reporting and MLflow logging
# ---------------------------------------------------------------------------

def print_scenario_report(scenario, results):
    print("\n" + "=" * 72)
    print("Scenario: " + scenario)
    print("=" * 72)
    # f-strings with the columns padded into local variables first, rather
    # than one long concatenation split over several lines. Same reason
    # train.py formats its run lines this way - a line break before a +
    # trips flake8's W503, and a line break after it trips W504, so an
    # expression split across lines cannot satisfy both rules at once.
    print(f"  {'feature'.ljust(26)}{'PSI'.rjust(8)}   {'KS'.rjust(6)}   verdict")
    print("  " + "-" * 62)

    for result in results:
        flag = "DRIFT" if result["drifted"] else "ok"
        psi = str(round(result["psi"], 4)).rjust(8)
        ks = str(round(result["ks"], 3)).rjust(6)
        verdict = f"{flag} ({result['classification']})"

        print(f"  {result['feature'].ljust(26)}{psi}   {ks}   {verdict}")

    drifted = [result for result in results if result["drifted"]]

    if drifted:
        names = ", ".join(result["feature"] for result in drifted)

        print(f"\n  Result: DRIFT DETECTED in {len(drifted)} "
              f"of {len(results)} features.")
        print(f"  Features: {names}")
        print("  In a real pipeline this would open an alert and trigger retraining.")
    else:
        print("\n  Result: NO DRIFT - all features within threshold.")
        print("  The model's input distribution is still the one it was trained on.")


def log_scenario_run(scenario, reference, current, results):
    # One MLflow run per scenario, so drift checks sit in the same tracking
    # store as the training runs and the prompt runs. Running this on a
    # schedule would build up a history of PSI over time, and the trend
    # matters more than any single value - PSI creeping from 0.04 to 0.09
    # over six months is a warning even though it never crosses 0.25.
    with mlflow.start_run(run_name=scenario):
        mlflow.log_param("scenario", scenario)
        mlflow.log_param("reference_rows", len(reference))
        mlflow.log_param("current_rows", len(current))
        mlflow.log_param("psi_bins", PSI_BINS)
        mlflow.log_param("psi_threshold", DRIFT_PSI_THRESHOLD)

        for result in results:
            # MLflow metric names cannot contain some characters that the
            # Iris feature names do, for example the brackets in
            # "sepal length (cm)". Slugify before logging.
            key = _metric_key(result["feature"])
            mlflow.log_metric("psi_" + key, result["psi"])
            mlflow.log_metric("ks_" + key, result["ks"])

        drifted = [result for result in results if result["drifted"]]

        mlflow.log_metric("max_psi", max(result["psi"] for result in results))
        mlflow.log_metric("features_drifted", len(drifted))
        mlflow.log_metric("drift_detected", 1 if drifted else 0)

        mlflow.log_dict({"scenario": scenario, "features": results},
                        "drift_report.json")


def _metric_key(feature_name):
    # "sepal length (cm)" -> "sepal_length"
    cleaned = feature_name.replace("(cm)", "").strip()
    return cleaned.replace(" ", "_").replace("(", "").replace(")", "")


def run_all_scenarios():
    reference, feature_names, target = load_reference_window()

    print(f"Reference window: {len(reference)} rows, "
          f"{len(feature_names)} features")
    print(f"PSI threshold   : {DRIFT_PSI_THRESHOLD} (>= this counts as drift)")

    any_drift = False

    for scenario in SCENARIOS:
        current = make_current_window(reference, target, scenario)
        results = analyse_features(reference, current, feature_names)

        log_scenario_run(scenario, reference, current, results)
        print_scenario_report(scenario, results)

        if any(result["drifted"] for result in results):
            any_drift = True

    return any_drift


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    drift_found = run_all_scenarios()

    print("\n" + "=" * 72)
    print(f"Open the MLflow UI and look at the {EXPERIMENT_NAME} "
          "experiment to compare scenarios.")

    # --strict makes drift a build failure. It is off by default because
    # this script runs four scenarios and three of them are SUPPOSED to
    # drift - that is the demonstration. Pointed at one real window of
    # production data, --strict is how it would gate a pipeline.
    if "--strict" in sys.argv and drift_found:
        sys.exit(1)
