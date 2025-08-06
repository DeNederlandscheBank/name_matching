import pytest
import pandas as pd
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
