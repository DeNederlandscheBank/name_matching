import numpy as np
import pytest
import pandas as pd

from name_matching.match_annotator import MatchAnnotator


def test_init_with_defaults(sample_results):
    annotator = MatchAnnotator(results=sample_results)
    assert isinstance(annotator._results_data, pd.DataFrame)
    assert annotator._name_col == "original_name"
    assert isinstance(annotator._annotated_results, dict)


def test_init_with_annotated_results(sample_results):
    annotated = {"Alpha Inc": "Alpha Incorporated"}
    annotator = MatchAnnotator(results=sample_results, annotated_results=annotated)
    assert annotator._annotated_results == annotated


def test_annotated_results_property(annotator):
    assert isinstance(annotator.annotated_results, dict)


def test_define_possible_nodes_random(annotator):
    nodes = annotator._define_possible_nodes("random")
    assert isinstance(nodes, list)
    assert len(nodes) == len(annotator._results_data)


def test_possible_names(annotator):
    annotator._possible_nodes = [0]
    annotator._index = 0
    names = annotator._possible_names()
    assert isinstance(names, np.ndarray)
    assert set(names).issubset(set(["Alpha Incorporated", "Alpha Inc."]))


def test_save_result(annotator):
    class DummyButton:
        description = "Alpha Incorporated"

    annotator._possible_nodes = [0]
    annotator._index = 0
    annotator._save_result(DummyButton())
    assert annotator._annotated_results["Alpha Inc"] == "Alpha Incorporated"


def test_no_match(annotator):
    class DummyButton:
        pass

    annotator._possible_nodes = [0]
    annotator._index = 0
    annotator._skip = lambda x: setattr(annotator, "_index", annotator._index + 1)
    annotator._no_match(DummyButton())
    assert annotator._annotated_results["Alpha Inc"] == -1
    assert annotator._index == 1


def test_add_button(annotator):
    buttons = []
    annotator._save_result = lambda x: None
    annotator._add_button(buttons, "Test Name")
    assert any(b.description == "Test Name" for b in buttons)


def test_start_fresh(annotator):
    annotator.start(fresh_start=True)
    assert annotator._index == 0
    assert isinstance(annotator._possible_nodes, list)


def test_skip_and_stop(annotator):
    annotator._possible_nodes = [0, 1]
    annotator._index = 0
    annotator._skip(None)
    assert annotator._index == 1


# Integration test
def test_full_annotation_session(sample_results):
    annotator = MatchAnnotator(results=sample_results)
    annotator._possible_nodes = [0, 1, 2]
    annotator._index = 0

    class DummyButton:
        def __init__(self, description):
            self.description = description

    for i in range(3):
        name = sample_results.loc[i, "original_name"]
        match = sample_results.loc[i, "match_name_1"]
        annotator._save_result(DummyButton(match))

    assert len(annotator.annotated_results) == 3
    for i in range(3):
        name = sample_results.loc[i, "original_name"]
        match = sample_results.loc[i, "match_name_1"]
        assert annotator.annotated_results[name] == match
