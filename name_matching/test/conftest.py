import pytest
import matplotlib
import os.path as path
import pandas as pd
import numpy as np

from scipy.sparse import csc_matrix

from name_matching.name_matching_optimiser import NameMatchingOptimiser
from name_matching.match_annotator import MatchAnnotator
import name_matching.name_matcher as nm


@pytest.fixture(autouse=True)
def set_matplotlib_backend():
    matplotlib.use("Agg")


@pytest.fixture
def name_match():
    package_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    data = pd.read_csv(path.join(package_dir, "test", "test_names.csv"))
    name_matcher = nm.NameMatcher()
    name_matcher.load_and_process_master_data(
        "company_name", data, start_processing=False, transform=False
    )
    return name_matcher


@pytest.fixture
def original_name():
    package_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    return pd.read_csv(path.join(package_dir, "test", "test_names.csv"), index_col=0)


@pytest.fixture
def adjusted_name():
    package_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    return pd.read_csv(
        path.join(package_dir, "test", "adjusted_test_names.csv"), index_col=0
    )


@pytest.fixture
def sample_results():
    data = {
        "original_name": ["Alpha Inc", "Beta LLC", "Gamma Corp"],
        "match_name_1": ["Alpha Incorporated", "Beta Limited", "Gamma Corporation"],
        "match_name_2": ["Alpha Inc.", "Beta LLC", "Gamma Co"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def annotator(sample_results):
    return MatchAnnotator(results=sample_results)


@pytest.fixture
def optimiser(name_match, original_name):
    return NameMatchingOptimiser(
        df_matching_data=name_match._df_matching_data,
        matching_col="company_name",
        df_to_be_matched=original_name,
        to_be_matched_col="company_name",
        name_matcher=name_match,
    )


@pytest.fixture
def annotated_data():
    return {
        "torphy-corkery": "Torphy-Corkery",
        "casper-schaden": "Casper-Schaden",
        "cormier-stanton": -1,
        "terry and sons": "Terry and Sons",
        "swaniawski ltd": "Swaniawski Ltd",
        "bauch anderson and schiller": "Bauch, Anderson and Schiller",
        "jenkins group": -1,
        "hilpert-gibson": "Hilpert-Gibson",
        "veum-carroll": "Veum-Carroll",
        "emard inc": "Emard Inc",
        "zboncak heathcote and": " sanfordzboncak, heathcote and sa",
        "yost group": "yost .group",
        "skiles-huels": "Skiles-Huels",
        "ankunding-harber": "Ankunding-Harber",
        "hettinger llc": "Hettinger LLC",
    }


@pytest.fixture
def words():
    return [
        "fun",
        "small",
        "pool",
        "fun",
        "small",
        "pool",
        "sign",
        "small",
        "pool",
        "sign",
        "sign",
        "small",
        "pool",
        "sign",
        "paper",
        "oppose",
        "paper",
        "oppose",
        "brown",
        "pig",
        "fat",
        "oppose",
        "paper",
        "oppose",
        "brown",
        "pig",
        "fat",
        "snail",
    ]


@pytest.fixture
def mat_a():
    return csc_matrix(
        np.array(
            [
                [0.0, 0.0, 0.26, 0.0, 0.0, 0.35, 0.26, 0.17, 0.38, 0.49],
                [0.0, 0.0, 0.0, 0.0, 0.46, 0.54, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.61, 0.13, 0.0, 0.93, 0.0, 0.0, 0.52, 0.0],
                [0.0, 0.13, 0.0, 0.24, 0.0, 0.68, 0.0, 0.19, 0.0, 0.11],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.37, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.25, 0.0],
                [0.75, 0.0, 0.0, 0.62, 0.32, 0.92, 0.0, 0.33, 0.0, 0.54],
                [0.94, 0.9, 0.0, 0.37, 0.93, 0.91, 0.0, 0.0, 0.0, 0.0],
                [0.93, 0.5, 0.0, 0.0, 0.0, 0.54, 0.49, 0.0, 0.0, 0.78],
                [0.12, 0.0, 0.0, 0.28, 0.0, 0.45, 0.0, 0.96, 0.0, 0.77],
            ]
        )
    )


@pytest.fixture
def mat_b():
    return csc_matrix(
        np.array(
            [
                [0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.9, 0.9, 0.0, 0.1, 0.2, 0.6, 0.0, 0.0],
                [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.4, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.6, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.3, 0.6, 0.0, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.8, 0.3, 0.0, 0.0],
            ]
        )
    )


@pytest.fixture
def mat_c():
    return csc_matrix(
        np.array(
            [
                [0.2, 0.5, 0.2, 0.1, 0.5, 0.0],
                [0.2, 0.9, 0.3, 0.4, 0.4, 0.7],
                [0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.3, 0.8, 0.0],
                [0.7, 0.9, 0.0, 0.7, 0.9, 0.2],
                [0.2, 0.1, 0.8, 0.0, 0.0, 0.1],
            ]
        )
    )


@pytest.fixture
def mat_d():
    return csc_matrix(
        np.array(
            [
                [0.8, 0.0, 0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
                [0.3, 0.4, 0.0, 0.0, 0.0, 0.7],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.4, 0.4, 0.0, 0.0],
                [0.8, 0.0, 0.5, 0.8, 0.2, 0.0],
            ]
        )
    )
