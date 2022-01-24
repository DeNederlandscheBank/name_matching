import numpy as np
import pandas as pd
import pytest

from name_matching.distance_metrics import make_distance_metrics

@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 36/39),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 55/77),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.95), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.9), 
                            ('De Nederlandsche Bank', 'De Bank', 0.5), 
                            ('De Nederlandsche Bank', 'Bank', 0.32), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 4/7),
                            ('De Nederlandsche Bank', 'Federal Reserve', 7/18)]
                        )
def test_indel(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(indel=True).popitem()[1][0].sim(string_a, string_b)) == expected

@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 10360/13259),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 14171/31229),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 12668/13693), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 14285/16126), 
                            ('De Nederlandsche Bank', 'De Bank', 73895/197251), 
                            ('De Nederlandsche Bank', 'Bank', 0.18443356121294618), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.4850080059940999),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.3134817407970336)]
                        )
def test_discounted_levenshtein(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(discounted_levenshtein=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 17/18),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 17/21),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 17/19), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 14/19), 
                            ('De Nederlandsche Bank', 'De Bank', 5/7), 
                            ('De Nederlandsche Bank', 'Bank', 0.75), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.50),
                            ('De Nederlandsche Bank', 'Federal Reserve', 4/15)]
                        )
def test_tichy(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(tichy=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 19/21),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 6/7),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 6/7),  
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 5/7), 
                            ('De Nederlandsche Bank', 'De Bank', 3/7), 
                            ('De Nederlandsche Bank', 'Bank', 2/7), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 3/7),
                            ('De Nederlandsche Bank', 'Federal Reserve', 5/21)]
                        )
def test_cormodeL_z(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(cormodeL_z=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 25/26),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.9456378640464952),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 197/200), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.9147134187457855), 
                            ('De Nederlandsche Bank', 'De Bank', 1643/2210), 
                            ('De Nederlandsche Bank', 'Bank', 0.66), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.7153211009174312),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.3081299056671707)]
                        )
def test_iterative_sub_string(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(iterative_sub_string=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 3546/3551),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 289/290),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 2147/2149), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 232/233), 
                            ('De Nederlandsche Bank', 'De Bank', 68/75), 
                            ('De Nederlandsche Bank', 'Bank', 4/23), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 234/253),
                            ('De Nederlandsche Bank', 'Federal Reserve', 3/19)]
                        )
def test_baulieu_xiii(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(baulieu_xiii=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.8232342408134744),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.7788978053198099),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.8674165216369765), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.734759771488919), 
                            ('De Nederlandsche Bank', 'De Bank', 0.38144806847096924), 
                            ('De Nederlandsche Bank', 'Bank', 0.20468230928444348), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.42541053393936973),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.16003090928720642)]
                        )
def test_clement(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(clement=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 9/11),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 17/22),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 19/22), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 8/11), 
                            ('De Nederlandsche Bank', 'De Bank', 4/11), 
                            ('De Nederlandsche Bank', 'Bank', 2/11), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 9/22),
                            ('De Nederlandsche Bank', 'Federal Reserve', 3/22)]
                        )
def test_dice_asymmetricI(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(dice_asymmetricI=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 3067/3100), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.8271005106727322),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.7115907789232533),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.8594338161878166), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.7014070603349739), 
                            ('De Nederlandsche Bank', 'De Bank', 0.5170872111993288), 
                            ('De Nederlandsche Bank', 'Bank', 0.376527052407862), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.48246333174338174),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.30535291331122694)]
                        )
def test_kuhns_iii(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(kuhns_iii=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 18/19),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 17/22),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.95), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.8), 
                            ('De Nederlandsche Bank', 'De Bank', 1.0), 
                            ('De Nederlandsche Bank', 'Bank', 0.8), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.6),
                            ('De Nederlandsche Bank', 'Federal Reserve', 3/16)]
                        )
def test_overlap(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(overlap=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.9326379507404536),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.860116027428689),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.9479333464498336), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.8530617633487101), 
                            ('De Nederlandsche Bank', 'De Bank', 0.7254387697419673), 
                            ('De Nederlandsche Bank', 'Bank', 0.495978140334987), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.6158120209632525),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.19529216149425904)]
                        )
def test_pearson_ii(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(pearson_ii=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 54/59),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 51/61),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 57/61), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 24/29), 
                            ('De Nederlandsche Bank', 'De Bank', 12/19), 
                            ('De Nederlandsche Bank', 'Bank', 12/31), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 27/46),
                            ('De Nederlandsche Bank', 'Federal Reserve', 9/41)]
                        )
def test_weighted_jaccard(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(weighted_jaccard=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.9336347104909842),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.8693019343986543),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.9488186399633484), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.8624113475177305), 
                            ('De Nederlandsche Bank', 'De Bank', 0.6934422509643748), 
                            ('De Nederlandsche Bank', 'Bank', 0.4558455621522721), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.6518716705286544),
                            ('De Nederlandsche Bank', 'Federal Reserve', 282/1037)]
                        )
def test_warrens_iv(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(warrens_iv=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 6/7),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 20/21),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 19/21), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 19/21), 
                            ('De Nederlandsche Bank', 'De Bank', 1/3), 
                            ('De Nederlandsche Bank', 'Bank', 4/21), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 10/21),
                            ('De Nederlandsche Bank', 'Federal Reserve', 10/21)]
                        )
def test_bag(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(bag=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.8590308370044052),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.7142857142857144),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.9060895084372709), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.8584005869405722), 
                            ('De Nederlandsche Bank', 'De Bank', 65/193), 
                            ('De Nederlandsche Bank', 'Bank', 65/337), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 325/679),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.33480500367917587)]
                        )
def test_rouge_l(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(rouge_l=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 12/13),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 5/7),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.95), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.90), 
                            ('De Nederlandsche Bank', 'De Bank', 0.5), 
                            ('De Nederlandsche Bank', 'Bank', 0.32), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 4/7),
                            ('De Nederlandsche Bank', 'Federal Reserve', 7/18)]
                        )
def test_ratcliff_obershelp(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(ratcliff_obershelp=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 5/6),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 41/49),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 7/8), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 41/48), 
                            ('De Nederlandsche Bank', 'De Bank', 17/24), 
                            ('De Nederlandsche Bank', 'Bank', 5/8), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 35/48),
                            ('De Nederlandsche Bank', 'Federal Reserve', 5/8)]
                        )
def test_ncd_bz2n(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(ncd_bz2=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 1),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 5/7),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 17/19), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 16/19), 
                            ('De Nederlandsche Bank', 'De Bank', 6/7), 
                            ('De Nederlandsche Bank', 'Bank', 1), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.5),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.4)]
                        )
def test_fuzzy_wuzzy_partial_string(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(fuzzy_wuzzy_partial_string=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 12/13),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 6/7),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.95), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.90), 
                            ('De Nederlandsche Bank', 'De Bank', 0.5), 
                            ('De Nederlandsche Bank', 'Bank', 0.32), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 18/35),
                            ('De Nederlandsche Bank', 'Federal Reserve', 7/18)]
                        )
def test_fuzzy_wuzzy_token_sort(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(fuzzy_wuzzy_token_sort=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 1),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 20/21),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.95), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.90), 
                            ('De Nederlandsche Bank', 'De Bank', 1), 
                            ('De Nederlandsche Bank', 'Bank', 1), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 8/11),
                            ('De Nederlandsche Bank', 'Federal Reserve', 8/19)]
                        )
def test_fuzzy_wuzzy_token_set(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(fuzzy_wuzzy_token_set=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 6/7),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 11/21),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 19/21), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 37/42), 
                            ('De Nederlandsche Bank', 'De Bank', 8/21), 
                            ('De Nederlandsche Bank', 'Bank', 5/21), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 13/21),
                            ('De Nederlandsche Bank', 'Federal Reserve', 1/3)]
                        )
def test_editexn(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(editex=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 6/7),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 43/84),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 19/21), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 37/42), 
                            ('De Nederlandsche Bank', 'De Bank', 1/3), 
                            ('De Nederlandsche Bank', 'Bank', 4/21), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.49642190479096915),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.2664967491513207)]
                        )
def test_typo(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(typo=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.4),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 4/15),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 13/14), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 8/9), 
                            ('De Nederlandsche Bank', 'De Bank', 0.3), 
                            ('De Nederlandsche Bank', 'Bank', 0.0), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 6/17),
                            ('De Nederlandsche Bank', 'Federal Reserve', 2/17)]
                        )
def test_lig_3(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(lig_3=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.9228829701817861),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.7711353570271147),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.9377381283200467), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.9411991279605375), 
                            ('De Nederlandsche Bank', 'De Bank', 0.5230048581284561), 
                            ('De Nederlandsche Bank', 'Bank', 0.24487267643945035), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.45427562753677897),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.47482190774345556)]
                        )
def test_ssk(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(ssk=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 10/11),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 4/11),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 1.0), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 10/11), 
                            ('De Nederlandsche Bank', 'De Bank', 4/11), 
                            ('De Nederlandsche Bank', 'De Nederlandsche Benk', 1.0), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 7/11),
                            ('De Nederlandsche Bank', 'Federal Reserve', 4/11)]
                        )
def test_refined_soundex(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(refined_soundex=True).popitem()[1][0].sim(string_a, string_b)) == expected
    
@pytest.mark.parametrize("string_a, string_b, expected", 
                            [('De Nederlandsche Bank', 'De Nederlandsche Bank', 1), 
                            ('De Nederlandsche Bank', 'Nederlandsche Bank', 0.5),  
                            ('De Nederlandsche Bank', 'Bank de Nederlandsche', 0.5),  
                            ('De Nederlandsche Bank', 'De Nederlandse Bank', 0.5), 
                            ('De Nederlandsche Bank', 'De Nederlancsh Bank', 0.5), 
                            ('De Nederlandsche Bank', 'De Bank', 0.5), 
                            ('De Nederlandsche Bank', 'De Nederlandsche Benk', 1.0), 
                            ('De Nederlandsche Bank', 'De Duitse Bank', 0.5),
                            ('De Nederlandsche Bank', 'Federal Reserve', 0.5)]
                        )
def test_double_metaphone(string_a, string_b, expected):
    assert pytest.approx(make_distance_metrics(double_metaphone=True).popitem()[1][0].sim(string_a, string_b)) == expected


