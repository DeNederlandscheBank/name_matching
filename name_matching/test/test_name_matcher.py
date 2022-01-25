from _pytest.python_api import approx
import numpy as np
import pandas as pd
import abydos.distance as abd
import abydos.phonetic as abp
import pytest
from scipy.sparse.csc import csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import name_matching.name_matcher as nm

@pytest.fixture
def name_match():
    data = pd.read_csv('test\\test_names.csv')
    name_matcher = nm.NameMatcher()
    name_matcher.load_and_process_master_data('company_name', data, start_processing=False, transform=False)
    return name_matcher

@pytest.fixture
def adjusted_name():
    return pd.read_csv('test\\adjusted_test_names.csv')
    
@pytest.mark.parametrize("method", 
                          ["",
                          None,
                          'no_method']
                        )
def test_make_distance_metrics_error(name_match, method):
    with pytest.raises(TypeError):
        name_match.set_distance_metrics([method])
        
@pytest.mark.parametrize("method, result", 
                          [['indel',abd.Indel()],
                            ['discounted_levenshtein',abd.DiscountedLevenshtein()],
                            ['tichy',abd.Tichy()],
                            ['cormodeL_z',abd.CormodeLZ()],
                            ['iterative_sub_string',abd.IterativeSubString()],
                            ['baulieu_xiii',abd.BaulieuXIII()],
                            ['clement',abd.Clement()],
                            ['dice_asymmetricI',abd.DiceAsymmetricI()],
                            ['kuhns_iii',abd.KuhnsIII()],
                            ['overlap',abd.Overlap()],
                            ['pearson_ii',abd.PearsonII()],
                            ['weighted_jaccard',abd.WeightedJaccard()],
                            ['warrens_iv',abd.WarrensIV()],
                            ['bag',abd.Bag()],
                            ['rouge_l',abd.RougeL()],
                            ['ratcliff_obershelp',abd.RatcliffObershelp()],
                            ['ncd_bz2',abd.NCDbz2()],
                            ['fuzzy_wuzzy_partial_string',abd.FuzzyWuzzyPartialString()],
                            ['fuzzy_wuzzy_token_sort',abd.FuzzyWuzzyTokenSort()],
                            ['fuzzy_wuzzy_token_set',abd.FuzzyWuzzyTokenSet()],
                            ['editex',abd.Editex()],
                            ['typo',abd.Typo()],
                            ['lig_3',abd.LIG3()],
                            ['ssk',abd.SSK()],
                            ['refined_soundex',abd.PhoneticDistance(transforms=abp.RefinedSoundex(max_length=30), metric=abd.Levenshtein(), encode_alpha=True)],
                            ['double_metaphone',abd.PhoneticDistance(transforms=abp.DoubleMetaphone(max_length=30), metric=abd.Levenshtein(), encode_alpha=True)]]
                        )
def test_make_distance_metrics(name_match, method, result):
    name_match.set_distance_metrics([method])
    assert type(name_match._distance_metrics.popitem()[1][0]) == type(result)
        
@pytest.mark.parametrize("kwargs_str, result_1, result_2, result_3",
                        [[{"ngrams":(4, 5)}, 0, 200, (4, 5)],
                        [{"memory_usage":0}, 0, 0, (2, 3)],
                        [{"legal_suffixes":True}, 247, 200, (2, 3)],
                        [{"legal_suffixes":True, "memory_usage": 8, "ngrams":(1,2,3)}, 247, 500, (1,2,3)],
                        ]) 
def test_initialisation(kwargs_str, result_1, result_2, result_3):
    name_match = nm.NameMatcher(**kwargs_str)
    assert len(name_match._word_set )==result_1
    assert name_match._low_memory ==result_2
    assert name_match._vec.ngram_range ==result_3

@pytest.mark.parametrize("lm",[-1, 11, 0, 10])
def test_initialisation_error(lm):
    if 0 <= lm <= 10:
        name_match = nm.NameMatcher(memory_usage=lm)
        assert type(name_match) == nm.NameMatcher
    else:
        with pytest.raises(ValueError):
            nm.NameMatcher(memory_usage=lm)

@pytest.mark.parametrize("occ, result_1, result_2, result_3, result_4, result_5",
                        [[1, '', '', '', '', ''],
                        [2, 'a-nd', 'Hndkiewicz,2Nicolas', 'Tashirian', 'Hpdson Sbns', 'Marquardt,'],
                        [3, 'Dickens a-nd', 'Hndkiewicz,2Nicolas', 'Runolfsson, Tashirian Will', 'Hpdson Sbns', 'Hermiston Marquardt,'],
                        ]) 
def test_preprocess_reduce(name_match, adjusted_name, occ, result_1, result_2, result_3, result_4, result_5):

    name_match._column_matching = 'company_name'
    new_names = name_match._preprocess_reduce(adjusted_name, occurence_count=occ)
    assert new_names.loc[1866, 'company_name'] == result_1
    assert new_names.loc[1423, 'company_name'] == result_2
    assert new_names.loc[268, 'company_name'] == result_3
    assert new_names.loc[859, 'company_name'] == result_4
    assert new_names.loc[1918, 'company_name'] == result_5

@pytest.mark.parametrize("col, start_pro, transform",
                        [['company_name', False, False],
                        ['no_name', False, False],
                        ['company_name', True, False],
                        ['company_name', True, True],
                        ['company_name', True, True],
                        ]) 
def test_load_and_process_master_data(adjusted_name, col, start_pro, transform):
    name_matcher = nm.NameMatcher()
    name_matcher.load_and_process_master_data(
                                     column=col,
                                     df_matching_data=adjusted_name,
                                     start_processing=start_pro,
                                     transform=transform)
                            
    assert name_matcher._column == col
    pd.testing.assert_frame_equal(name_matcher._df_matching_data, adjusted_name)
    assert name_matcher._preprocessed == start_pro
    if transform & start_pro:
        assert type(name_matcher._n_grams_matching) == csc_matrix


@pytest.mark.parametrize("trans, common",
                        [[False, False],
                        [True, False],
                        [False, True],
                        [True, True],
                        ]) 
def test_process_matching_data(name_match, trans, common):
    name_match._postprocess_common_words = common
    name_match._process_matching_data(transform = trans)

    assert name_match._preprocessed
    if trans:
        assert type(name_match._n_grams_matching) == csc_matrix
    else:
        assert name_match._n_grams_matching is None
    if common:
        assert len(name_match._word_set) > 0
    else:
        assert len(name_match._word_set) == 0


@pytest.mark.parametrize("lower_case, punctuations, ascii, result_1, result_2, result_3",
                        [[False, False, False, 'Schumm PLC', 'Towne, Johnston and Murray', 'Ösinski-Schinner'],
                        [True, False, False, 'schumm plc', 'towne, johnston and murray', 'ösinski-schinner'],
                        [False, True, False, 'Schumm PLC', 'Towne Johnston and Murray', 'ÖsinskiSchinner'],
                        [False, False, True, 'Schumm PLC', 'Towne, Johnston and Murray', 'Osinski-Schinner'],
                        [False, True, True, 'Schumm PLC', 'Towne Johnston and Murray', 'OsinskiSchinner'],
                        [True, False, True, 'schumm plc', 'towne, johnston and murray', 'osinski-schinner'],
                        [True, True, False, 'schumm plc', 'towne johnston and murray', 'ösinskischinner'],
                        [True, True, True, 'schumm plc', 'towne johnston and murray', 'osinskischinner'],
                        ]) 
def test_preprocess(name_match, lower_case, punctuations, ascii, result_1, result_2, result_3):
    name_match._preprocess_lowercase = lower_case
    name_match._preprocess_punctuations = punctuations
    name_match._preprocess_ascii = ascii
    new_df = name_match.preprocess(name_match._df_matching_data, 'company_name')
    print(new_df['company_name'])
    assert new_df.loc[0,'company_name'] == result_1
    assert new_df.loc[2,'company_name'] == result_2
    assert new_df.loc[784,'company_name'] == result_3


@pytest.mark.parametrize("low_memory, ngrams, result_1, result_2, result_3",
                        [[1, (5,6), 0.02579, 0.00781, 0.01738],
                        [6, (2,3), 0.009695, 0.01022, 0.01120],
                        [8, (1,2), 0.027087, 0.02765, 0.02910],
                        [0, (5,6), 0.02579, 0.00781, 0.01738],
                        [0, (2,3), 0.009695, 0.01022, 0.01120],
                        [0, (1,2), 0.027087, 0.02765, 0.02910],
                        ]) 
def test_transform_data(name_match, low_memory, ngrams, result_1, result_2, result_3):
    name_match._low_memory = low_memory
    name_match._vec = TfidfVectorizer(lowercase=False, analyzer="char", ngram_range=ngrams)
    name_match._process_matching_data(transform=False)
    name_match.transform_data()

    assert name_match._n_grams_matching.data[10] == pytest.approx(result_1, 0.001)
    assert name_match._n_grams_matching.data[181] == pytest.approx(result_2, 0.001)
    assert name_match._n_grams_matching.data[1000] == pytest.approx(result_3, 0.001)
    

@pytest.mark.parametrize("to_be_matched, possible_matches, metrics, result",
                        [('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandse Bank', 'Bank de Nederlandsche'], ['weighted_jaccard'], 2), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandse Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'discounted_levenshtein'], 5), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandse Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'discounted_levenshtein', 'iterative_sub_string'], 7), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandse Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'overlap', 'iterative_sub_string'], 6), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandse Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'overlap', 'bag'], 11), 
						 ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandsche Bank', 'Bank de Nederlandsche'], ['weighted_jaccard'], 2), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandsche Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'discounted_levenshtein'], 4), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandsche Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'discounted_levenshtein', 'iterative_sub_string'], 6), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandsche Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'overlap', 'iterative_sub_string'], 6), 
                         ('De Nederlandsche Bank', ['Nederlandsche Bank', 'De Nederlancsh Bank', 'De Nederlandsche Bank', 'Bank de Nederlandsche'], ['weighted_jaccard', 'overlap', 'bag'], 6), 
                         ('Schumm PLC', ['Torphy-Corkery', 'Hansen, Hoppe and Tillman', 'Gerlach and Sons', 'Bank de Nederlandsche'], ['weighted_jaccard'], 2), 
                         ('Schumm PLC', ['Torphy-Corkery', 'Hansen, Hoppe and Tillman', 'Gerlach and Sons', 'Bank de Nederlandsche'], ['weighted_jaccard', 'discounted_levenshtein'], 4), 
                         ('Schumm PLC', ['Torphy-Corkery', 'Hansen, Hoppe and Tillman', 'Gerlach and Sons', 'Bank de Nederlandsche'], ['weighted_jaccard', 'discounted_levenshtein', 'iterative_sub_string'], 6), 
                         ('Schumm PLC', ['Torphy-Corkery', 'Hansen, Hoppe and Tillman', 'Gerlach and Sons', 'Bank de Nederlandsche'], ['weighted_jaccard', 'overlap', 'iterative_sub_string'], 8), 
                         ('Schumm PLC', ['Torphy-Corkery', 'Hansen, Hoppe and Tillman', 'Gerlach and Sons', 'Bank de Nederlandsche'], ['weighted_jaccard', 'overlap', 'bag'], 8)
                         ]) 
def test_score_matches(to_be_matched, possible_matches, metrics, result):
    name_match = nm.NameMatcher()
    name_match.set_distance_metrics(metrics)
    assert np.argmax(name_match._score_matches(to_be_matched, possible_matches))==result
    
@pytest.mark.parametrize("number_of_matches, match_score, metrics, result",
                        [(1, np.array([[0.9, 0.3, 0.5, 0.2, 0.1]]), ['weighted_jaccard'], [0]), 
                         (2, np.array([[0.9, 0.3, 0.5, 0.2, 0.1],[0.6, 0.7, 0.8, 0.4, 0.5]]), ['weighted_jaccard', 'discounted_levenshtein'], [0, 1]), 
                         (3, np.array([[0.9, 0.3, 0.5, 0.2, 0.1],[0.6, 0.7, 0.8, 0.4, 0.5],[1, 0.2, 0.3, 0.2, 0.1]]), ['weighted_jaccard', 'discounted_levenshtein', 'iterative_sub_string'], [2, 1, 1]), 
                         (2, np.array([[0.9, 0.3, 0.5, 0.2, 0.1],[0.6, 0.7, 0.8, 0.4, 0.5],[1, 0.2, 0.3, 0.2, 0.1]]), ['tichy', 'overlap', 'bag'], [2, 1]), 
                         (2, np.array([[0.9, 0.3, 0.5, 0.2, 0.1],[0.6, 0.7, 0.8, 0.4, 0.5]]), ['overlap', 'bag'], [0, 2]), 
                         (1, np.array([[0.9, 0.3, 0.5, 0.2, 0.1],[0.6, 0.7, 0.8, 0.4, 0.5],[1, 0.2, 0.3, 0.2, 0.1]]), ['weighted_jaccard', 'overlap', 'iterative_sub_string'], [1]), 
                         (2, np.array([[0.9, 0.3, 0.5, 0.2, 0.1],[0.6, 0.7, 0.8, 0.4, 0.5],[1, 0.2, 0.3, 0.2, 0.1]]), ['weighted_jaccard', 'overlap', 'bag'], [1, 0]), 
						 (1, np.array([[0.3, 0.3, 0.8, 0.2, 0.2]]), ['weighted_jaccard'], [0]), 
                         (3, np.array([[0.3, 0.3, 0.8, 0.2, 0.2],[0.3, 0.3, 0.8, 0.1, 0.1]]), ['weighted_jaccard', 'discounted_levenshtein'], [0, 1]), 
                         (2, np.array([[0.3, 0.3, 0.2, 0.1, 0.02],[0.1, 0.1, 0.2, 0.3, 0.02]]), ['weighted_jaccard', 'iterative_sub_string'], [0, 0]), 
                         (1, np.array([[0.3, 0.3, 0.2, 0.1, 0.02],[0.3, 0.3, 0.2, 0.3, 0.02]]), ['overlap', 'iterative_sub_string'], [1]), 
                         (1, np.array([[-0.5, -0.8, -0.3, -0.7, 0,2]]), ['bag'], [0]), 
                         (3, np.array([[10, 8, 7 ,6, 12, 15, 14, 88]]), ['weighted_jaccard'], [0]), 
                         (2, np.array([[1, 0.3],[0.1, 0.4]]), ['weighted_jaccard', 'discounted_levenshtein'], [0, 1])
						 ]) 
def test_rate_matches(number_of_matches, match_score, metrics, result):
    name_match = nm.NameMatcher()
    name_match._number_of_matches = number_of_matches
    name_match.set_distance_metrics(metrics)
    ind = name_match._rate_matches(match_score)
    print(ind)
    assert len(ind)==np.min([number_of_matches, match_score.shape[0]])
    assert list(ind)==result


def test_vectorise_data(name_match):
    name_match._vectorise_data(transform=False)
    assert len(name_match._vec.vocabulary_)>0


@pytest.mark.parametrize("match, number_of_matches, word_set, score, result",
                        [(pd.Series(['Nederandsche', 0, 2, 'De Nederlandsche Bank'],index=['match_name_0', 'score_0', 'match_index_0', 'original_name']), 1, set(['De','Bank', 'nl']), 0, 94.553), 
                         (pd.Series(['Nederandsche', 0, 2, 'De Nederlandsche Bank'],index=['match_name_0', 'score_0', 'match_index_0', 'original_name']), 1, set(['komt','niet', 'voor']), 0, 69.713), 
                         (pd.Series(['nederandsche', 0, 2, 'de nederand bank', 0.4, 3, 'De Nederlandsche Bank'],index=['match_name_0', 'score_0', 'match_index_0', 'match_name_1', 'score_1', 'match_index_1', 'original_name']), 1, set(['De','Bank', 'nl']), 1, 0.4), 
                         (pd.Series(['nederandsche', 0, 2, 'de nederand bank', 0.4, 3, 'De Nederlandsche Bank'],index=['match_name_0', 'score_0', 'match_index_0', 'match_name_1', 'score_1', 'match_index_1', 'original_name']), 1, set(['De','Bank', 'nl']), 0, 86.031), 
						 ]) 
def test_postprocess(name_match, match, number_of_matches, word_set, score, result):
    name_match._number_of_matches = number_of_matches
    name_match._word_set = word_set
    new_match = name_match.postprocess(match)
    assert new_match.loc[f'score_{score}'] == pytest.approx(result, 0.0001)

@pytest.mark.parametrize("indicator, punctuations, word_set, cut_off, result_1, result_2",
                        [('legal', False, set(), 0.01, 'plc.', 'bedrijf'), 
                         ('legal', True, set(), 0.01, 'plc', 'bedrijf'), 
                         ('legal', True, set(['bedrijf']), 0.01, 'bedrijf', 'Group'), 
                         ('common', True, set(), 0.01, 'Group', 'West'), 
                         ('common', True, set(), 0.3, 'and', 'Group'), 
                         ('common', True, set(['West']), 0.3, 'West', 'bedrijf'), 
                         ('someting', True, set(['key']), 0.01, 'key', 'val') 
						 ]) 
def test_make_no_scoring_words(name_match, indicator, punctuations, word_set, cut_off, result_1, result_2):
    name_match._preprocess_punctuations = punctuations
    new_word_set = name_match._make_no_scoring_words(indicator, word_set, cut_off)
    assert new_word_set.issuperset(set([result_1]))
    assert not new_word_set.issuperset(set([result_2]))


def test_search_for_possible_matches_error(adjusted_name):
    name_matcher = nm.NameMatcher()
    with pytest.raises(RuntimeError):
        name_matcher._search_for_possible_matches(adjusted_name)


@pytest.mark.parametrize("top_n, low_memory, result_1, result_2",
                        [(10, 0, 1518, 144), 
                         (50, 0, 1992, 9), 
                         (100, 0, 1999, 6), 
                         (1, 0, 44, 144),
                         (10, 8, 1518, 144), 
                         (50, 8, 1992, 9), 
                         (100, 8, 1999, 6), 
                         (1, 8, 44, 144)
						 ]) 
def test_search_for_possible_matches(name_match, adjusted_name, top_n, low_memory, result_1, result_2):
    name_match._column_matching = 'company_name'
    name_match._low_memory = low_memory
    name_match._top_n = top_n
    name_match._process_matching_data(True)
    possible_match = name_match._search_for_possible_matches(adjusted_name)
    assert possible_match.shape[1] == top_n
    assert np.max(possible_match) < len(adjusted_name)
    assert np.all(possible_match.astype(int) == possible_match)
    assert np.max(possible_match[44,:]) == result_1
    assert np.min(possible_match[144,:]) == result_2


@pytest.mark.parametrize("common_words, num_matches, possible_matches, matching_series, result_0, result_1",
                        [(True, 3, np.array([29, 343, 727, 855, 1702]), pd.Series(['Company and Sons'],index=['company_name']), 36.03, 31.33), 
                         (False, 2, np.array([29, 343, 727, ]), pd.Series(['Company and Sons'],index=['company_name']), 71.28, 68.6), 
                         (False, 2, np.array([29, 343]), pd.Series(['Company and Sons'],index=['company_name']), 71.28, 68.6), 
                         (False, 2, np.array([[29, 343],[0, 0]]), pd.Series(['Company and Sons'],index=['company_name']), 71.28, 68.6), 
                         (False, 2, np.array([29, 343, 727, 855, 1702]), pd.Series(['Company and Sons'],index=['company_name']), 72.28, 71.28)
						 ]) 
def test_fuzzy_matches(name_match, common_words, num_matches, possible_matches, matching_series, result_0, result_1):
    name_match._column_matching = 'company_name'
    name_match._number_of_matches = num_matches
    name_match._postprocess_common_words = common_words
    name_match._word_set = set(['Sons', 'and'])
    match = name_match.fuzzy_matches(possible_matches, matching_series)
    assert match['score_0'] == pytest.approx(result_0, 0.0001)
    assert match['score_1'] == pytest.approx(result_1, 0.0001)
    assert match['match_index_0'] in possible_matches
    assert match['match_index_1'] in possible_matches


def test_do_name_matching_full(name_match, adjusted_name):
    result = name_match.do_name_matching(adjusted_name, 'company_name')
    assert np.sum(result['match_index'] == result.index) == 1922

def test_do_name_matching_split(name_match, adjusted_name):
    name_match._preprocess_split = True
    result = name_match.do_name_matching(adjusted_name.iloc[44,:], 'company_name')
    assert np.any(result['match_index'] == 44)
    
def test_do_name_matching_series(name_match, adjusted_name):
    result = name_match.do_name_matching(adjusted_name.iloc[44,:], 'company_name')
    assert np.any(result['match_index'] == 44)

def test_do_name_matching_error(adjusted_name):
    name_match = nm.NameMatcher()
    with pytest.raises(ValueError):
        name_match.do_name_matching(adjusted_name, 'company_name')

@pytest.mark.parametrize("verbose",[True, False])
def test_do_name_matching_print(capfd, name_match, adjusted_name, verbose):
    name_match._verbose = verbose
    name_match.do_name_matching(adjusted_name.iloc[:5], 'company_name')
    out, err = capfd.readouterr()
    if verbose:
        assert out.find('preprocessing') > -1
        assert out.find('searching') > -1
        assert out.find('possible') > -1
        assert out.find('fuzzy') > -1
        assert out.find('done') > -1
    else:
        assert out == ''
