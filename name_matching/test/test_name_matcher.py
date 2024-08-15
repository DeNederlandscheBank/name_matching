import numpy as np
import pandas as pd
import os.path as path
import pytest
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from cleanco.termdata import terms_by_country, terms_by_type
import functools
import operator
import re
import name_matching.name_matcher as nm
from distances import (
    Indel,
    DiscountedLevenshtein,
    CormodeLZ,
    Tichy,
    IterativeSubString,
    BaulieuXIII,
    Clement,
    DiceAsymmetricI,
    KuhnsIII,
    Overlap,
    PearsonII,
    WeightedJaccard,
    WarrensIV,
    Bag,
    RougeL,
    RatcliffObershelp,
    NCDbz2,
    FuzzyWuzzyPartialString,
    FuzzyWuzzyTokenSort,
    FuzzyWuzzyTokenSet,
    Editex,
    Typo,
    LIG3,
    SSK,
    Levenshtein,
    DoubleMetaphone,
    RefinedSoundex,
    PhoneticDistance,
)


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
    return pd.read_csv(path.join(package_dir, "test", "test_names.csv"))


@pytest.fixture
def adjusted_name():
    package_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    return pd.read_csv(path.join(package_dir, "test", "adjusted_test_names.csv"))


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


def number_of_words_in_legal_list(preprocess: bool) -> int:
    """
    Get the number of words in the legal word list. To allow for updates of the cleanco
    package.

    Parameters
    -------
    preprocess: bool
        a bool determining if the punctuations should be removed

    Returns
    -------
    int
        The number of legal words in the list
    """

    if preprocess:
        set_of_words = set(
            [
                re.sub(r"[^\w\s]", "", s).strip()
                for s in functools.reduce(
                    operator.iconcat, terms_by_country.values(), []
                )
            ]
        )
        set_of_words.update(
            [
                re.sub(r"[^\w\s]", "", s).strip()
                for s in functools.reduce(operator.iconcat, terms_by_type.values(), [])
            ]
        )
    else:
        set_of_words = set(
            [
                s.strip()
                for s in functools.reduce(
                    operator.iconcat, terms_by_country.values(), []
                )
            ]
        )
        set_of_words.update(
            [
                s.strip()
                for s in functools.reduce(operator.iconcat, terms_by_type.values(), [])
            ]
        )

    return len(set_of_words)


@pytest.mark.parametrize("method", ["", None, "no_method"])
def test_make_distance_metrics_error(name_match, method):
    with pytest.raises(TypeError):
        name_match.set_distance_metrics([method])


@pytest.mark.parametrize(
    "method, result",
    [
        ["indel", Indel()],
        ["discounted_levenshtein", DiscountedLevenshtein()],
        ["tichy", Tichy()],
        ["cormodeL_z", CormodeLZ()],
        ["iterative_sub_string", IterativeSubString()],
        ["baulieu_xiii", BaulieuXIII()],
        ["clement", Clement()],
        ["dice_asymmetricI", DiceAsymmetricI()],
        ["kuhns_iii", KuhnsIII()],
        ["overlap", Overlap()],
        ["pearson_ii", PearsonII()],
        ["weighted_jaccard", WeightedJaccard()],
        ["warrens_iv", WarrensIV()],
        ["bag", Bag()],
        ["rouge_l", RougeL()],
        ["ratcliff_obershelp", RatcliffObershelp()],
        ["ncd_bz2", NCDbz2()],
        ["fuzzy_wuzzy_partial_string", FuzzyWuzzyPartialString()],
        ["fuzzy_wuzzy_token_sort", FuzzyWuzzyTokenSort()],
        ["fuzzy_wuzzy_token_set", FuzzyWuzzyTokenSet()],
        ["editex", Editex()],
        ["typo", Typo()],
        ["lig_3", LIG3()],
        ["ssk", SSK()],
        [
            "refined_soundex",
            PhoneticDistance(
                transforms=RefinedSoundex(max_length=30),
                metric=Levenshtein(),
                encode_alpha=True,
            ),
        ],
        [
            "double_metaphone",
            PhoneticDistance(
                transforms=DoubleMetaphone(max_length=30),
                metric=Levenshtein(),
                encode_alpha=True,
            ),
        ],
    ],
)
def test_make_distance_metrics(name_match, method, result):
    name_match.set_distance_metrics([method])
    assert type(name_match._distance_metrics.popitem()[1][0]) == type(result)


@pytest.mark.parametrize(
    "kwargs_str, result_1, result_2, result_3, result_4",
    [
        [{"ngrams": (4, 5)}, -1, False, (4, 5), 5000],
        [{"low_memory": True}, -1, True, (2, 3), 5000],
        [{"legal_suffixes": True}, 0, False, (2, 3), 5000],
        [
            {"legal_suffixes": True, "number_of_rows": 8, "ngrams": (1, 2, 3)},
            0,
            False,
            (1, 2, 3),
            8,
        ],
    ],
)
def test_initialisation(kwargs_str, result_1, result_2, result_3, result_4):
    name_match = nm.NameMatcher(**kwargs_str)
    number_of_words = 1
    if result_1 > -1:
        number_of_words = number_of_words_in_legal_list(
            name_match._preprocess_punctuations
        )
    assert len(name_match._word_set) == number_of_words + result_1
    assert name_match._low_memory == result_2
    assert name_match._vec.ngram_range == result_3
    assert name_match._number_of_rows == result_4


@pytest.mark.parametrize(
    "occ, result_1, result_2, result_3, result_4, result_5",
    [
        [1, "", "", "", "", ""],
        [
            2,
            "Schiller",
            "Sch-ster, an[",
            "Runolfsson, Tashirian Will",
            "Hyats, S|nger",
            "Ankunding-Harb-er",
        ],
        [
            3,
            "Schiller",
            "Sch-ster, Raynor an[ Hermann",
            "Runolfsson, Tashirian Will",
            "Hyats, Durgan S|nger",
            "Ankunding-Harb-er",
        ],
    ],
)
def test_preprocess_reduce(
    name_match, adjusted_name, occ, result_1, result_2, result_3, result_4, result_5
):

    name_match._column_matching = "company_name"
    new_names = name_match._preprocess_reduce(adjusted_name, occurrence_count=occ)
    assert new_names.loc[166, "company_name"] == result_1
    assert new_names.loc[423, "company_name"] == result_2
    assert new_names.loc[268, "company_name"] == result_3
    assert new_names.loc[59, "company_name"] == result_4
    assert new_names.loc[18, "company_name"] == result_5


@pytest.mark.parametrize(
    "col, start_pro, transform",
    [
        ["company_name", False, False],
        ["no_name", False, False],
        ["company_name", True, False],
        ["company_name", True, True],
        ["company_name", True, True],
    ],
)
def test_load_and_process_master_data(adjusted_name, col, start_pro, transform):
    name_matcher = nm.NameMatcher()
    name_matcher.load_and_process_master_data(
        column=col,
        df_matching_data=adjusted_name,
        start_processing=start_pro,
        transform=transform,
    )

    assert name_matcher._column == col
    pd.testing.assert_frame_equal(name_matcher._df_matching_data, adjusted_name)
    assert name_matcher._preprocessed == start_pro
    if transform & start_pro:
        assert type(name_matcher._n_grams_matching) == csc_matrix


@pytest.mark.parametrize(
    "trans, common",
    [
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_process_matching_data(name_match, trans, common):
    name_match._postprocess_common_words = common
    name_match._process_matching_data(transform=trans)

    assert name_match._preprocessed
    if trans:
        assert type(name_match._n_grams_matching) == csc_matrix
    else:
        assert name_match._n_grams_matching is None
    if common:
        assert len(name_match._word_set) > 0
    else:
        assert len(name_match._word_set) == 0


@pytest.mark.parametrize(
    "lower_case, punctuations, ascii, result_1, result_2, result_3",
    [
        [
            False,
            False,
            False,
            "Schumm PLC",
            "Towne, Johnston and Murray",
            "Ösinski-Schinner",
        ],
        [
            True,
            False,
            False,
            "schumm plc",
            "towne, johnston and murray",
            "ösinski-schinner",
        ],
        [
            False,
            True,
            False,
            "Schumm PLC",
            "Towne Johnston and Murray",
            "ÖsinskiSchinner",
        ],
        [
            False,
            False,
            True,
            "Schumm PLC",
            "Towne, Johnston and Murray",
            "Osinski-Schinner",
        ],
        [
            False,
            True,
            True,
            "Schumm PLC",
            "Towne Johnston and Murray",
            "OsinskiSchinner",
        ],
        [
            True,
            False,
            True,
            "schumm plc",
            "towne, johnston and murray",
            "osinski-schinner",
        ],
        [
            True,
            True,
            False,
            "schumm plc",
            "towne johnston and murray",
            "ösinskischinner",
        ],
        [
            True,
            True,
            True,
            "schumm plc",
            "towne johnston and murray",
            "osinskischinner",
        ],
    ],
)
def test_preprocess(
    name_match, lower_case, punctuations, ascii, result_1, result_2, result_3
):
    name_match._preprocess_lowercase = lower_case
    name_match._preprocess_punctuations = punctuations
    name_match._preprocess_ascii = ascii
    new_df = name_match.preprocess(name_match._df_matching_data, "company_name")
    assert new_df.loc[0, "company_name"] == result_1
    assert new_df.loc[2, "company_name"] == result_2
    assert new_df.loc[432, "company_name"] == result_3


@pytest.mark.parametrize(
    "low_memory, ngrams, result_1, result_2, result_3",
    [
        [1, (5, 6), 0.00689, 0.00892, 0.0293],
        [6, (2, 3), 0.01044, 0.01092, 0.035],
        [8, (1, 2), 0.02729, 0.02783, 0.0360],
        [0, (5, 6), 0.00689, 0.00892, 0.0293],
        [0, (2, 3), 0.01044, 0.01092, 0.035],
        [0, (1, 2), 0.02729, 0.02783, 0.036],
    ],
)
def test_transform_data(name_match, low_memory, ngrams, result_1, result_2, result_3):
    name_match._low_memory = low_memory
    name_match._vec = TfidfVectorizer(
        lowercase=False, analyzer="char", ngram_range=ngrams
    )
    name_match._process_matching_data(transform=False)
    name_match.transform_data()

    assert name_match._n_grams_matching.data[10] == pytest.approx(result_1, 0.001)
    assert name_match._n_grams_matching.data[181] == pytest.approx(result_2, 0.001)
    assert name_match._n_grams_matching.data[1000] == pytest.approx(result_3, 0.001)


@pytest.mark.parametrize(
    "to_be_matched, possible_matches, metrics, result",
    [
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandse Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard"],
            2,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandse Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "discounted_levenshtein"],
            5,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandse Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "discounted_levenshtein", "iterative_sub_string"],
            7,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandse Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "overlap", "iterative_sub_string"],
            6,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandse Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "overlap", "bag"],
            11,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandsche Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard"],
            2,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandsche Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "discounted_levenshtein"],
            4,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandsche Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "discounted_levenshtein", "iterative_sub_string"],
            6,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandsche Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "overlap", "iterative_sub_string"],
            6,
        ),
        (
            "De Nederlandsche Bank",
            [
                "Nederlandsche Bank",
                "De Nederlancsh Bank",
                "De Nederlandsche Bank",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "overlap", "bag"],
            6,
        ),
        (
            "Schumm PLC",
            [
                "Torphy-Corkery",
                "Hansen, Hoppe and Tillman",
                "Gerlach and Sons",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard"],
            2,
        ),
        (
            "Schumm PLC",
            [
                "Torphy-Corkery",
                "Hansen, Hoppe and Tillman",
                "Gerlach and Sons",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "discounted_levenshtein"],
            4,
        ),
        (
            "Schumm PLC",
            [
                "Torphy-Corkery",
                "Hansen, Hoppe and Tillman",
                "Gerlach and Sons",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "discounted_levenshtein", "iterative_sub_string"],
            6,
        ),
        (
            "Schumm PLC",
            [
                "Torphy-Corkery",
                "Hansen, Hoppe and Tillman",
                "Gerlach and Sons",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "overlap", "iterative_sub_string"],
            8,
        ),
        (
            "Schumm PLC",
            [
                "Torphy-Corkery",
                "Hansen, Hoppe and Tillman",
                "Gerlach and Sons",
                "Bank de Nederlandsche",
            ],
            ["weighted_jaccard", "overlap", "bag"],
            8,
        ),
    ],
)
def test_score_matches(to_be_matched, possible_matches, metrics, result):
    name_match = nm.NameMatcher()
    name_match.set_distance_metrics(metrics)
    assert (
        np.argmax(name_match._score_matches(to_be_matched, possible_matches)) == result
    )


@pytest.mark.parametrize(
    "number_of_matches, match_score, metrics, result",
    [
        (1, np.array([[0.9, 0.3, 0.5, 0.2, 0.1]]), ["weighted_jaccard"], [0]),
        (
            2,
            np.array([[0.9, 0.3, 0.5, 0.2, 0.1], [0.6, 0.7, 0.8, 0.4, 0.5]]),
            ["weighted_jaccard", "discounted_levenshtein"],
            [0, 2],
        ),
        (
            3,
            np.array(
                [
                    [0.9, 0.3, 0.5, 0.2, 0.1],
                    [0.6, 0.7, 0.8, 0.4, 0.5],
                    [1, 0.2, 0.3, 0.2, 0.1],
                ]
            ),
            ["weighted_jaccard", "discounted_levenshtein", "iterative_sub_string"],
            [0, 2, 0],
        ),
        (
            2,
            np.array(
                [
                    [0.9, 0.3, 0.5, 0.2, 0.1],
                    [0.6, 0.7, 0.8, 0.4, 0.5],
                    [1, 0.2, 0.3, 0.2, 0.1],
                ]
            ),
            ["tichy", "overlap", "bag"],
            [0, 0],
        ),
        (
            2,
            np.array([[0.9, 0.3, 0.5, 0.2, 0.1], [0.6, 0.7, 0.8, 0.4, 0.5]]),
            ["overlap", "bag"],
            [0, 2],
        ),
        (
            1,
            np.array(
                [
                    [0.9, 0.3, 0.5, 0.2, 0.1],
                    [0.6, 0.7, 0.8, 0.4, 0.5],
                    [1, 0.2, 0.3, 0.2, 0.1],
                ]
            ),
            ["weighted_jaccard", "overlap", "iterative_sub_string"],
            [0],
        ),
        (
            2,
            np.array(
                [
                    [0.9, 0.3, 0.5, 0.2, 0.1],
                    [0.6, 0.7, 0.8, 0.4, 0.5],
                    [1, 0.2, 0.3, 0.2, 0.1],
                ]
            ),
            ["weighted_jaccard", "overlap", "bag"],
            [0, 2],
        ),
        (1, np.array([[0.3, 0.3, 0.8, 0.2, 0.2]]), ["weighted_jaccard"], [2]),
        (
            3,
            np.array([[0.3, 0.4, 0.8, 0.2, 0.2], [0.3, 0.3, 0.8, 0.1, 0.1]]),
            ["weighted_jaccard", "discounted_levenshtein"],
            [2, 1, 0],
        ),
        (
            2,
            np.array([[0.3, 0.3, 0.2, 0.1, 0.02], [0.1, 0.1, 0.2, 0.3, 0.02]]),
            ["weighted_jaccard", "iterative_sub_string"],
            [0, 3],
        ),
        (
            1,
            np.array([[0.3, 0.3, 0.2, 0.1, 0.02], [0.3, 0.3, 0.2, 0.3, 0.02]]),
            ["overlap", "iterative_sub_string"],
            [0],
        ),
        (1, np.array([[-0.5, -0.8, -0.3, -0.7, 0, 2]]), ["bag"], [5]),
        (1, np.array([[-0.5, -0.8, -0.3, -0.7, 0, 2]]), ["BAG"], [5]),
        (3, np.array([[10, 8, 7, 6, 12, 15, 14, 88]]), ["weighted_jaccard"], [7,5,6]),
        (
            2,
            np.array([[1, 0.3], [0.1, 0.4]]),
            ["weighted_jaccard", "discounted_levenshtein"],
            [0, 1],
        ),
    ],
)
def test_rate_matches(number_of_matches, match_score, metrics, result):
    name_match = nm.NameMatcher()
    name_match._number_of_matches = number_of_matches
    name_match.set_distance_metrics(metrics)
    ind = name_match._rate_matches(match_score.T)
    assert len(ind) == np.min([number_of_matches, match_score.shape[1]])
    assert list(ind) == result


def test_vectorise_data(name_match):
    name_match._vectorise_data(transform=False)
    assert len(name_match._vec.vocabulary_) > 0


@pytest.mark.parametrize(
    "match, number_of_matches, word_set, score, result",
    [
        (
            pd.Series(
                ["Nederandsche", 0, 2, "De Nederlandsche Bank"],
                index=["match_name_0", "score_0", "match_index_0", "original_name"],
            ),
            1,
            set(["De", "Bank", "nl"]),
            0,
            94.553,
        ),
        (
            pd.Series(
                ["Nederandsche", 0, 2, "De Nederlandsche Bank"],
                index=["match_name_0", "score_0", "match_index_0", "original_name"],
            ),
            1,
            set(["komt", "niet", "voor"]),
            0,
            69.713,
        ),
        (
            pd.Series(
                [
                    "nederandsche",
                    0,
                    2,
                    "de nederand bank",
                    0.4,
                    3,
                    "De Nederlandsche Bank",
                ],
                index=[
                    "match_name_0",
                    "score_0",
                    "match_index_0",
                    "match_name_1",
                    "score_1",
                    "match_index_1",
                    "original_name",
                ],
            ),
            1,
            set(["De", "Bank", "nl"]),
            1,
            0.4,
        ),
        (
            pd.Series(
                [
                    "nederandsche",
                    0,
                    2,
                    "de nederand bank",
                    0.4,
                    3,
                    "De Nederlandsche Bank",
                ],
                index=[
                    "match_name_0",
                    "score_0",
                    "match_index_0",
                    "match_name_1",
                    "score_1",
                    "match_index_1",
                    "original_name",
                ],
            ),
            1,
            set(["De", "Bank", "nl"]),
            0,
            86.031,
        ),
    ],
)
def test_postprocess(name_match, match, number_of_matches, word_set, score, result):
    name_match._number_of_matches = number_of_matches
    name_match._word_set = word_set
    new_match = name_match.postprocess(match)
    assert new_match.loc[f"score_{score}"] == pytest.approx(result, 0.0001)


@pytest.mark.parametrize(
    "indicator, punctuations, word_set, cut_off, result_1, result_2",
    [
        ("legal", False, set(), 0.01, "plc.", "bedrijf"),
        ("legal", True, set(), 0.01, "plc", "bedrijf"),
        ("legal", True, set(["bedrijf"]), 0.01, "bedrijf", "Group"),
        ("common", True, set(), 0.01, "Group", "bedrijf"),
        ("common", True, set(), 0.3, "and", "Group"),
        ("common", True, set(["West"]), 0.3, "West", "bedrijf"),
        ("someting", True, set(["key"]), 0.01, "key", "val"),
    ],
)
def test_make_no_scoring_words(
    name_match, indicator, punctuations, word_set, cut_off, result_1, result_2
):
    name_match._preprocess_punctuations = punctuations
    new_word_set = name_match._make_no_scoring_words(indicator, word_set, cut_off)
    print(new_word_set)
    assert new_word_set.issuperset(set([result_1]))
    assert not new_word_set.issuperset(set([result_2]))


def test_search_for_possible_matches_error(adjusted_name):
    name_matcher = nm.NameMatcher()
    with pytest.raises(RuntimeError):
        name_matcher._search_for_possible_matches(adjusted_name)


@pytest.mark.parametrize(
    "top_n, low_memory, number_of_rows, result_1, result_2",
    [
        (10, True, 55, 469, 144),
        (50, True, 112, 499, 6),
        (100, True, 112, 499, 1),
        (1, True, 112, 44, 144),
        (10, False, 500, 469, 144),
        (50, False, 1500, 499, 6),
        (100, False, 500, 499, 1),
        (1, False, 500, 44, 144),
    ],
)
def test_search_for_possible_matches(
    name_match, adjusted_name, top_n, low_memory, number_of_rows, result_1, result_2
):
    name_match._column_matching = "company_name"
    name_match._low_memory = low_memory
    name_match._number_of_rows = number_of_rows
    name_match._top_n = top_n
    name_match._process_matching_data(True)
    possible_match = name_match._search_for_possible_matches(adjusted_name)
    assert possible_match.shape[1] == top_n
    assert np.max(possible_match) < len(adjusted_name)
    assert np.all(possible_match.astype(int) == possible_match)
    assert np.max(possible_match[44, :]) == result_1
    assert np.min(possible_match[144, :]) == result_2


@pytest.mark.parametrize(
    "common_words, num_matches, possible_matches, matching_series, result_0, result_1",
    [
        (
            True,
            3,
            np.array([29, 343, 126, 238, 445]),
            pd.Series(["Company and Sons"], index=["company_name"]),
            31.77,
            31.33,
        ),
        (
            False,
            2,
            np.array(
                [
                    29,
                    343,
                    126,
                ]
            ),
            pd.Series(["Company and Sons"], index=["company_name"]),
            71.28,
            68.6,
        ),
        (
            False,
            2,
            np.array([29, 343]),
            pd.Series(["Company and Sons"], index=["company_name"]),
            71.28,
            68.6,
        ),
        (
            ["Sons", "and"],
            3,
            np.array([29, 343, 126, 238, 445]),
            pd.Series(["Company and Sons"], index=["company_name"]),
            31.77,
            31.33,
        ),
        (
            False,
            2,
            np.array([[29, 343], [0, 0]]),
            pd.Series(["Company and Sons"], index=["company_name"]),
            71.28,
            68.6,
        ),
        (
            False,
            2,
            np.array([29, 343, 126, 238, 445]),
            pd.Series(["Company and Sons"], index=["company_name"]),
            71.28,
            68.6,
        ),
    ],
)
def test_fuzzy_matches(
    name_match,
    common_words,
    num_matches,
    possible_matches,
    matching_series,
    result_0,
    result_1,
):
    name_match._column_matching = "company_name"
    name_match._number_of_matches = num_matches
    name_match._postprocess_common_words = common_words
    if isinstance(common_words, list):
        name_match._word_set = set(common_words)
    elif common_words:
        name_match._word_set = set(["Sons", "and"])
    else:
        name_match._word_set = set()
    match = name_match.fuzzy_matches(possible_matches, matching_series)
    assert match["score_0"] == pytest.approx(result_0, 0.0001)
    assert match["score_1"] == pytest.approx(result_1, 0.0001)
    assert match["match_index_0"] in possible_matches
    assert match["match_index_1"] in possible_matches


def test_do_name_matching_split(name_match, adjusted_name):
    name_match._preprocess_split = True
    result = name_match.match_names(adjusted_name.iloc[44, :], "company_name")
    assert np.any(result["match_index"] == 44)


def test_do_name_matching_series(name_match, adjusted_name):
    result = name_match.match_names(adjusted_name.iloc[44, :], "company_name")
    assert np.any(result["match_index"] == 44)


def test_do_name_matching_full(name_match, adjusted_name):
    result = name_match.match_names(adjusted_name, "company_name")
    assert np.sum(result["match_index"] == result.index) == 494


@pytest.mark.parametrize(
    "old_index, new_index, adjust, size_a, size_b, match_result",
    [
        [10, "new", False, 20, 20, "new"],
        [10, "new", True, 20, 20, 10],
        [10, 526, False, 20, 20, 526],
        [10, 526, True, 20, 20, 10],
        [4, 201, True, 20, 50, 4],
        [8, 201, False, 20, 50, 201],
        [8, 44, True, 50, 20, 8],
        [4, 44, False, 50, 20, 44],
    ],
)
def test_do_name_matching_switch_index(
    original_name, old_index, new_index, adjust, size_a, size_b, match_result
):
    name_match = nm.NameMatcher(row_numbers=adjust, verbose=False)
    adjusted_name = original_name.copy()
    original_name = original_name.rename(index={old_index: new_index})
    name_match.load_and_process_master_data(
        "company_name",
        original_name.iloc[:size_a, :],
        start_processing=False,
        transform=False,
    )
    result = name_match.match_names(adjusted_name.iloc[:size_b, :], "company_name")
    assert result.loc[old_index, "match_index"] == match_result


def test_do_name_matching_error(adjusted_name):
    name_match = nm.NameMatcher()
    with pytest.raises(ValueError):
        name_match.match_names(adjusted_name, "company_name")


@pytest.mark.parametrize("verbose", [True, False])
def test_do_name_matching_print(capfd, name_match, adjusted_name, verbose):
    name_match._verbose = verbose
    name_match.match_names(adjusted_name.iloc[:5].copy(), "company_name")
    out, err = capfd.readouterr()
    if verbose:
        assert out.find("preprocessing") > -1
        assert out.find("searching") > -1
        assert out.find("possible") > -1
        assert out.find("fuzzy") > -1
        assert out.find("done") > -1
    else:
        assert out == ""


@pytest.mark.parametrize(
    "word, occurrence_count, result",
    [
        ["fun snail pool", 2, "snail"],
        ["fun snail pool", 3, "fun snail"],
        ["fun snail pool", 1, ""],
        ["fun small pool", 3, "fun small pool"],
        ["fun snail", 3, "fun snail"],
        ["fun small pool", 5, "fun small pool"],
    ],
)
def test_select_top_words(word, words, occurrence_count, result):
    word_counts = pd.Series(words).value_counts()
    name_match = nm.NameMatcher()
    new_word = name_match._select_top_words(word.split(), word_counts, occurrence_count)
    assert new_word == result


@pytest.mark.parametrize(
    "match, num_of_matches, result",
    [
        [
            {"match_name_1": "fun", "match_name_2": "dog", "match_name_0": "cat"},
            3,
            ["cat", "fun", "dog"],
        ],
        [
            {"match_name_1": "fun", "match_name_2": "dog", "match_name_0": "cat"},
            2,
            ["cat", "fun"],
        ],
        [{"match_name_1": "fun", "match_name_0": "cat"}, 2, ["cat", "fun"]],
        [{"match_name_1": "fun", "match_name_2": "dog", "match_name_0": "cat"}, 0, []],
    ],
)
def test_get_alternative_names(match, num_of_matches, result):
    name_match = nm.NameMatcher(number_of_matches=num_of_matches)
    res = name_match._get_alternative_names(pd.Series(match))
    assert res == result


@pytest.mark.parametrize(
    "preprocess_punctuations, output, input, x",
    [
        [True, "_blame_", {"test": ["fun...", "done"], "num": ["_.blame._"]}, 2],
        [True, "done", {"test": ["fun. . . ", "done"], "num": ["_.blame._"]}, 1],
        [True, "fun", {"test": ["fun. . . ", "done"], "num": ["_.blame._"]}, 0],
        [False, "fun. . .", {"test": ["fun. . . ", "done"], "num": ["_.blame._"]}, 0],
        [False, "fun. . .", {"num": ["_.blame._"], "test": ["fun. . . ", "done"]}, 1],
    ],
)
def test_preprocess_word_list(preprocess_punctuations, output, input, x):
    name_match = nm.NameMatcher(punctuations=preprocess_punctuations)
    res = name_match._preprocess_word_list(input)
    print(res)
    assert res[x] == output


@pytest.mark.parametrize(
    "num_matches, match_score, match, result, y",
    [
        [
            3,
            np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]]),
            pd.Series(dtype=float),
            100,
            0,
        ],
        [2, np.array([[1, 1], [0.4, 0.4], [0, 0]]), pd.Series(dtype=float), 40, 1],
        [1, np.array([[1, 1], [1, 1], [0, 0]]), pd.Series(dtype=float), 100, 0],
    ],
)
def test_adjust_scores(num_matches, match_score, match, result, y):
    name_match = nm.NameMatcher(number_of_matches=num_matches)
    match = name_match._adjust_scores(match_score, match)
    assert match.iloc[y] == result


@pytest.mark.parametrize(
    "string, stringlist, result_1, result_2, y",
    [
        [
            "know sign first",
            ["know", "know sign", "know sign first"],
            "know first",
            "know first",
            2,
        ],
        [
            "know sign first",
            ["know", "know sign", "know sign first"],
            "know first",
            "know",
            1,
        ],
        [
            "know sign first",
            ["know", "know sign", "know sign first"],
            "know first",
            "know",
            0,
        ],
        ["know first", ["know", "know", "know"], "know first", "know", 1],
        ["pool sign small", ["sign small", "small pool sign", "small"], "", "", 0],
        [
            "pool sign small know",
            ["sign small", "small pool sign", "small"],
            "know",
            "",
            0,
        ],
        [
            "know pool sign small",
            ["sign small", "small pool sign", "small"],
            "know",
            "",
            0,
        ],
        [
            "pool sign small",
            ["sign small", "small pool know sign", "small"],
            "",
            "know",
            1,
        ],
    ],
)
def test_process_words(words, string, stringlist, result_1, result_2, y):
    name_match = nm.NameMatcher()
    name_match._word_set = set(words)
    string, stringlist = name_match._process_words(string, stringlist)
    assert string == result_1
    assert stringlist[y] == result_2


@pytest.mark.parametrize(
    "word_set, cut_off, result_1, result_2",
    [
        [set(), 0, 639, "Group"],
        [set(), 0, 639, "and"],
        [set(), 0.1, 7, "Group"],
        [set(), 0.1, 7, "LLC"],
        [set(), 0.12, 7, "LLC"],
        [set(), 0.2, 1, "and"],
        [set(["apple"]), 1, 1, "apple"],
        [set(["apple"]), 0, 640, "apple"],
        [set(["apple"]), 0, 640, "Group"],
    ],
)
def test_process_common_words(name_match, word_set, cut_off, result_1, result_2):
    words = name_match._process_common_words(word_set, cut_off)
    assert result_2 in words
    assert len(words) == result_1


@pytest.mark.parametrize(
    "common_words, error",
    [[True, False], [[], False], [set(), False], [dict(), True], ["", True]],
)
def test_common_words_type_error(common_words, error):
    if error:
        with pytest.raises(TypeError):
            nm.NameMatcher(common_words=common_words)
    else:
        name_matcher = nm.NameMatcher(common_words=common_words)
        if isinstance(common_words, bool):
            assert name_matcher._postprocess_common_words == True
        else:
            assert name_matcher._word_set == set(common_words)


@pytest.mark.parametrize(
    "common_words, legal_suffixes",
    [
        [["Cherry", "Stream", "Puzzle", "Balloon", "Candle", "Mirror"], False],
        [["Cherry", "Stream", "Puzzle", "Balloon", "Candle", "Mirror"], True],
        [["Cherry"], False],
        [["Cherry"], True],
        [["limited", "gmbh"], False],
        [["limited", "gmbh"], True],
    ],
)
def test_common_words_addition(original_name, common_words, legal_suffixes):
    name_matcher = nm.NameMatcher(
        common_words=common_words, legal_suffixes=legal_suffixes
    )
    name_matcher.load_and_process_master_data(
        "company_name", original_name, start_processing=False, transform=False
    )
    name_matcher._process_matching_data(transform=False)
    for word in common_words:
        assert word in name_matcher._word_set


@pytest.mark.parametrize(
    "word_set, preprocess, result_1, result_2, result_3",
    [
        [set(), True, 0, "company", True],
        [set(), True, 0, "3ao", True],
        [set(), True, 0, "g.m.b.h.", False],
        [set(), False, 0, "& company", True],
        [set(), False, 0, "3ao", True],
        [set(), False, 0, "g.m.b.h.", True],
        [set(["apple"]), True, 1, "apple", True],
        [set(["apple"]), False, 1, "apple", True],
        [set(["apple.."]), True, 1, "apple..", True],
        [set(["apple.."]), False, 1, "apple..", True],
    ],
)
def test_process_legal_words(word_set, preprocess, result_1, result_2, result_3):
    name_match = nm.NameMatcher()
    name_match._preprocess_punctuations = preprocess
    words = name_match._process_legal_words(word_set)

    number_of_words = number_of_words_in_legal_list(preprocess)
    assert (result_2 in words) == result_3
    assert len(words) == number_of_words + result_1
