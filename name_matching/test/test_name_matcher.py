import numpy as np
import pandas as pd
import abydos.distance as abd
import abydos.phonetic as abp
import pytest

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