import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from name_matching.name_matching_optimiser import NameMatchingOptimiser


def test_perform_name_matching(optimiser):
    metrics = ["weighted_jaccard", "discounted_levenshtein", "jaro_winkler"]
    result = optimiser._perform_name_matching(metrics)
    assert isinstance(result, pd.DataFrame)


def test_annotate_matches(optimiser):
    metrics = ["weighted_jaccard", "discounted_levenshtein", "jaro_winkler"]
    matches = optimiser._perform_name_matching(metrics)
    optimiser._annotated_data = {}
    optimiser._annotate_matches(matches)
    assert isinstance(optimiser._annotated_data, dict)


def test_split_list_random_uneven():
    lst = list(range(10))
    splits = NameMatchingOptimiser.split_list_random_uneven(lst, 3)
    assert len(splits) == 3
    assert sum(len(split) for split in splits) == len(lst)


def test_preselect_matches(optimiser):
    metrics = ["weighted_jaccard", "discounted_levenshtein", "jaro_winkler"]
    matches = optimiser._perform_name_matching(metrics)
    filtered = optimiser._preselect_matches(matches, lower_bound=70.0)
    assert isinstance(filtered, pd.DataFrame)
    assert all(filtered["max_scr"] > 70.0)


def test_export_import_annotation(optimiser, tmp_path):
    optimiser._annotated_data = {"TestName": "MatchedName"}
    file_path = tmp_path / "annotations.csv"
    optimiser.export_annotation(str(file_path))
    assert file_path.exists()

    optimiser._annotated_data = None
    optimiser.import_annotation(str(file_path))
    assert "TestName" in optimiser._annotated_data


def test_cross_validate_model(optimiser, annotated_data):
    optimiser._annotated_data = annotated_data
    optimiser.fit(preprocess=True)
    result = optimiser.cross_validate_model()
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_cross_validate_trains_on_complementary_folds(optimiser, monkeypatch):
    optimiser._true_index = {i: np.array([float(i), 1.0]) for i in range(10)}
    optimiser._false_index = {i: np.array([float(i + 10), 0.0]) for i in range(10)}
    optimiser._all_false = {i: np.array([[float(i + 20), 0.0]]) for i in range(10)}
    optimiser.scaler = StandardScaler()
    optimiser.model = DummyClassifier(strategy="uniform", random_state=0)
    fitted_scaler = optimiser.scaler

    original_split = optimiser._generate_balanced_split
    train_sizes = []

    def record_split(train_split, train_idx=None, scaler_instance=None):
        train_sizes.append(len(train_idx))
        return original_split(train_split, train_idx, scaler_instance)

    monkeypatch.setattr(optimiser, "_generate_balanced_split", record_split)

    optimiser.cross_validate_model(n_folds=5)

    assert train_sizes == [8] * 5
    assert optimiser.scaler is fitted_scaler


def test_cross_validate_rejects_invalid_fold_count(optimiser):
    optimiser._true_index = {i: np.array([float(i)]) for i in range(5)}

    with pytest.raises(ValueError, match="n_folds must be between 2"):
        optimiser.cross_validate_model(n_folds=1)

    with pytest.raises(ValueError, match="n_folds must be between 2"):
        optimiser.cross_validate_model(n_folds=6)


def test_balanced_split_fits_scaler_on_training_data(optimiser):
    optimiser._true_index = {0: np.array([10.0]), 1: np.array([1000.0])}
    optimiser._false_index = {0: np.array([20.0]), 1: np.array([2000.0])}
    optimiser._all_false = {
        0: np.array([[30.0]]),
        1: np.array([[3000.0]]),
    }
    scaler = StandardScaler()

    optimiser._generate_balanced_split(
        train_split=0.5,
        train_idx=[0],
        scaler_instance=scaler,
    )

    assert scaler.mean_[0] == 15.0


def test_predict(optimiser, annotated_data):
    optimiser._annotated_data = annotated_data
    optimiser.fit(preprocess=True)
    predictions = optimiser.predict()
    assert isinstance(predictions, dict)
    assert all(isinstance(k, str) for k in predictions.keys())


def test_plot_curves(optimiser, annotated_data):
    optimiser._annotated_data = annotated_data
    optimiser.fit(preprocess=True)
    data = optimiser.plot_curves(return_data=True)
    assert isinstance(data, dict)
    assert "Precision" in data and "Recall" in data


def test_integration_workflow(optimiser, annotated_data):
    optimiser._annotated_data = annotated_data
    optimiser.fit(preprocess=True)
    predictions = optimiser.predict()
    assert isinstance(predictions, dict)
    assert all(isinstance(k, str) for k in predictions.keys())
    result_df = optimiser.export_annotation()
    assert isinstance(result_df, pd.DataFrame)
