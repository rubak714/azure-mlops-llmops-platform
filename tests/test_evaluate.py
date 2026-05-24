from src.evaluate import PROMOTION_THRESHOLD

# These tests check the promotion threshold logic in isolation.
# I am not testing MLflow itself - just the decision logic:
# does a given F1 score pass or fail the threshold?
# This is the kind of thing that can break silently without tests
# for example if someone changes the threshold constant by accident.


def is_above_threshold(f1_score):
    # accidentally using > instead of >= means a model scoring
    # exactly 0.85 will fail when it should pass
    return f1_score > PROMOTION_THRESHOLD


def test_strong_model_passes():
    # a well-performing model should always pass
    assert is_above_threshold(0.95) is True


def test_model_exactly_at_threshold_passes():
    # edge case - exactly at the threshold should pass, not fail
    assert is_above_threshold(0.85) is True


def test_weak_model_fails():
    # a model just below the threshold should not be promoted
    assert is_above_threshold(0.84) is False


def test_very_weak_model_fails():
    # a model that clearly did not train well should fail
    assert is_above_threshold(0.50) is False


def test_perfect_model_passes():
    # Iris often gives perfect scores - this should always pass
    assert is_above_threshold(1.00) is True
