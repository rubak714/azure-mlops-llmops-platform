from src.train import load_data

# These tests check that the data loading and splitting in train.py
# works as expected. Simple checks but useful - if someone changes
# the test_size or random_state in load_data() by accident, these
# will catch it immediately.


def test_train_test_split_sizes():
    # 150 samples total, 80/20 split means 120 train, 30 test
    X_train, X_test, y_train, y_test = load_data()
    assert len(X_train) == 120
    assert len(X_test) == 30


def test_feature_count():
    # Iris has exactly 4 features - if this changes something is wrong
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[1] == 4


def test_labels_match_samples():
    # number of labels must match number of samples in both sets
    X_train, X_test, y_train, y_test = load_data()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_only_three_classes():
    # Iris has exactly 3 species - setosa, versicolor, virginica
    _, _, y_train, y_test = load_data()
    all_labels = set(y_train) | set(y_test)
    assert len(all_labels) == 3
