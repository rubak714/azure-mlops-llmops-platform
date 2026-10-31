import numpy as np
import pytest

from src.detect_drift import (
    DRIFT_PSI_THRESHOLD,
    PSI_NO_DRIFT,
    SCENARIOS,
    analyse_features,
    classify_psi,
    is_drifted,
    ks_critical_value,
    ks_indicates_drift,
    ks_statistic,
    load_reference_window,
    make_current_window,
    population_stability_index,
    psi_bin_edges,
)

# Tests for the drift detection maths and the gate around it.
#
# I wrote PSI and the KS statistic by hand rather than importing them, so
# these tests are doing real work - they are the only thing standing
# between a subtle indexing mistake and a monitoring system that quietly
# never fires. Every expected value here is either 0, 1, or a comparison
# between two cases, so none of it depends on my arithmetic being right.

RANDOM_SEED = 7


def _normal_sample(mean, scale, size=500, seed=RANDOM_SEED):
    # Fixed seed everywhere. A drift test that passes intermittently is
    # worse than no test, because the failure looks like real drift.
    return np.random.default_rng(seed).normal(loc=mean, scale=scale, size=size)


# ---------------------------------------------------------------------------
# PSI binning
# ---------------------------------------------------------------------------

def test_bin_edges_are_open_at_both_ends():
    # Values outside the reference range must still land in a bin.
    # Without the infinite outer edges they fall outside every bin,
    # np.histogram drops them, and an extreme shift would read as no
    # change at all - the exact case drift detection exists to catch.
    edges = psi_bin_edges(_normal_sample(0, 1))

    assert edges[0] == -np.inf
    assert edges[-1] == np.inf


def test_bin_edges_are_sorted_and_unique():
    edges = psi_bin_edges(_normal_sample(0, 1))

    assert list(edges) == sorted(edges)
    assert len(edges) == len(set(edges))


def test_bin_edges_survive_a_feature_with_repeated_values():
    # Quantile binning on a mostly-constant feature produces duplicate
    # edges, which would give zero-width bins. This is the case np.unique
    # in psi_bin_edges is there to handle.
    repetitive = np.array([1.0] * 95 + [2.0, 3.0, 4.0, 5.0, 6.0])

    edges = psi_bin_edges(repetitive)

    assert len(edges) == len(set(edges))


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def test_psi_of_identical_distributions_is_zero():
    # The most important PSI test. If a distribution compared against
    # itself does not score zero, every number this script produces is
    # offset by an unknown amount.
    sample = _normal_sample(0, 1)

    assert population_stability_index(sample, sample) == 0.0


def test_psi_is_never_negative():
    # PSI is a sum of (actual - expected) * log(actual / expected). Both
    # factors carry the same sign in every bin, so each term and the total
    # are non-negative. A negative result would mean an indexing bug.
    reference = _normal_sample(0, 1)
    current = _normal_sample(1.5, 2.0, seed=RANDOM_SEED + 1)

    assert population_stability_index(reference, current) >= 0.0


def test_psi_detects_a_mean_shift():
    reference = _normal_sample(0, 1)
    shifted = reference + 3.0

    assert population_stability_index(reference, shifted) >= DRIFT_PSI_THRESHOLD


def test_psi_detects_a_variance_change_with_the_same_mean():
    # Drift is not only about the mean. A monitor that watched the average
    # would see nothing here, which is the reason to use a distributional
    # measure in the first place.
    reference = _normal_sample(0, 1)
    wider = _normal_sample(0, 4, seed=RANDOM_SEED + 2)

    assert population_stability_index(reference, wider) >= DRIFT_PSI_THRESHOLD


def test_psi_grows_with_the_size_of_the_shift():
    # Monotonicity. A bigger shift must score higher, otherwise the number
    # cannot be compared against a threshold in any meaningful way.
    reference = _normal_sample(0, 1)

    small = population_stability_index(reference, reference + 0.3)
    large = population_stability_index(reference, reference + 3.0)

    assert large > small


def test_psi_of_a_resample_stays_below_the_no_drift_band():
    # The false alarm check. A fresh sample from the same distribution
    # must come in under 0.10, or the detector cries wolf on normal
    # sampling variation and people stop listening to it.
    rng = np.random.default_rng(RANDOM_SEED)
    reference = rng.normal(0, 1, 2000)
    resample = rng.normal(0, 1, 2000)

    assert population_stability_index(reference, resample) < PSI_NO_DRIFT


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov
# ---------------------------------------------------------------------------

def test_ks_of_identical_samples_is_zero():
    sample = _normal_sample(0, 1)

    assert ks_statistic(sample, sample) == 0.0


def test_ks_of_completely_separated_samples_is_one():
    # No overlap at all means the two cumulative distributions are as far
    # apart as they can be, so the statistic is exactly 1.0.
    low = np.array([1.0, 2.0, 3.0, 4.0])
    high = np.array([100.0, 200.0, 300.0, 400.0])

    assert ks_statistic(low, high) == 1.0


def test_ks_is_bounded_between_zero_and_one():
    reference = _normal_sample(0, 1)
    current = _normal_sample(2, 3, seed=RANDOM_SEED + 3)

    statistic = ks_statistic(reference, current)

    assert 0.0 <= statistic <= 1.0


def test_ks_is_symmetric():
    # Unlike PSI, the KS statistic does not care which sample is the
    # baseline - it is just the largest gap between two curves.
    reference = _normal_sample(0, 1)
    current = _normal_sample(1, 1, seed=RANDOM_SEED + 4)

    assert ks_statistic(reference, current) == ks_statistic(current, reference)


def test_ks_critical_value_shrinks_as_samples_grow():
    # The same gap is more convincing from more data. If this were the
    # wrong way round, large samples would be harder to flag than small
    # ones, which is backwards.
    assert ks_critical_value(1000, 1000) < ks_critical_value(30, 30)


def test_ks_flags_a_clear_shift():
    reference = _normal_sample(0, 1)

    assert ks_indicates_drift(reference, reference + 3.0) is True


def test_ks_does_not_flag_an_identical_sample():
    sample = _normal_sample(0, 1)

    assert ks_indicates_drift(sample, sample) is False


# ---------------------------------------------------------------------------
# The gate and the interpretation bands
# ---------------------------------------------------------------------------

def test_high_psi_is_drift():
    assert is_drifted(0.80) is True


def test_psi_exactly_at_threshold_is_drift():
    # The boundary case, written the same way as the F1 gate in
    # evaluate.py even though the comparison runs the other direction.
    # There a high score is good and the threshold is a floor; here a high
    # score is bad and the threshold is a ceiling. Either way the
    # threshold value itself triggers the gate.
    assert is_drifted(DRIFT_PSI_THRESHOLD) is True


def test_psi_just_below_threshold_is_not_drift():
    assert is_drifted(DRIFT_PSI_THRESHOLD - 0.001) is False


def test_low_psi_is_not_drift():
    assert is_drifted(0.01) is False


def test_classification_bands():
    assert classify_psi(0.05) == "no drift"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.90) == "SIGNIFICANT"


def test_moderate_band_does_not_trigger_the_gate():
    # A deliberate check on a real limitation. PSI between 0.10 and 0.25
    # is labelled moderate and does NOT fire the gate, so a genuine but
    # small shift passes silently. The noise-increase scenario hits this
    # with sepal length. Watching the trend over time is the answer, not
    # a lower threshold.
    assert classify_psi(0.15) == "moderate"
    assert is_drifted(0.15) is False


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------

def test_reference_window_is_iris():
    reference, feature_names, target = load_reference_window()

    assert reference.shape == (150, 4)
    assert len(feature_names) == 4
    assert len(target) == 150


def test_unknown_scenario_raises():
    reference, _, target = load_reference_window()

    with pytest.raises(ValueError):
        make_current_window(reference, target, "not-a-real-scenario")


def test_no_drift_scenario_raises_no_alarm():
    # The control case, and the single most important test in this file.
    # A monitoring system that fires on data which has not changed is
    # worse than no monitoring, because people learn to ignore it.
    reference, feature_names, target = load_reference_window()
    current = make_current_window(reference, target, "no-drift")

    results = analyse_features(reference, current, feature_names)

    assert not any(result["drifted"] for result in results)


def test_measurement_shift_isolates_the_affected_feature():
    # Only petal length is shifted, so only petal length should be
    # flagged. Per-feature reporting is the whole point - "the data
    # drifted" is not actionable, "petal length drifted and nothing else
    # did" points straight at one instrument.
    reference, feature_names, target = load_reference_window()
    current = make_current_window(reference, target, "measurement-shift")

    results = analyse_features(reference, current, feature_names)
    drifted = [result["feature"] for result in results if result["drifted"]]

    assert drifted == ["petal length (cm)"]


def test_population_shift_is_detected():
    reference, feature_names, target = load_reference_window()
    current = make_current_window(reference, target, "population-shift")

    results = analyse_features(reference, current, feature_names)

    assert any(result["drifted"] for result in results)


def test_population_shift_window_is_smaller():
    # One of the three classes stops appearing, so a third of the rows go.
    reference, _, target = load_reference_window()
    current = make_current_window(reference, target, "population-shift")

    assert len(current) == 100


def test_noise_increase_is_detected():
    reference, feature_names, target = load_reference_window()
    current = make_current_window(reference, target, "noise-increase")

    results = analyse_features(reference, current, feature_names)

    assert any(result["drifted"] for result in results)


def test_scenarios_are_reproducible():
    # Same seed, same window. Otherwise the numbers in the docs would not
    # match what someone running the script sees.
    reference, _, target = load_reference_window()

    first = make_current_window(reference, target, "noise-increase")
    second = make_current_window(reference, target, "noise-increase")

    assert np.array_equal(first, second)


def test_analyse_returns_one_result_per_feature_with_both_measures():
    reference, feature_names, target = load_reference_window()
    current = make_current_window(reference, target, "measurement-shift")

    results = analyse_features(reference, current, feature_names)

    assert len(results) == len(feature_names)

    for result in results:
        assert "psi" in result
        assert "ks" in result
        assert "drifted" in result
        assert "classification" in result


def test_every_declared_scenario_actually_runs():
    # Catches a scenario added to the SCENARIOS list but never implemented
    # in make_current_window, which would crash the script halfway
    # through its run rather than at the start.
    reference, _, target = load_reference_window()

    for scenario in SCENARIOS:
        current = make_current_window(reference, target, scenario)
        assert len(current) > 0
