import pandas as pd
import os.path as path
import pytest

import name_matching.run_nm as run_nm

@pytest.fixture
def original_name():
    package_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    return pd.read_csv(path.join(package_dir, 'test','test_names.csv'))

@pytest.fixture
def adjusted_name():
    package_dir = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    return pd.read_csv(path.join(package_dir, 'test','adjusted_test_names.csv'))

@pytest.mark.parametrize("series, column, group_column",
                        [[False, '', ''],
                        [False, 'column', ''],
                        [False, 'company_name', 'column'],
                        [True, 'company_name', 'column'],
                        [True, '', 'company_name']
                        ]) 
def test_match_names_check_data_errors(adjusted_name, series, column, group_column):
    if series:
        adjusted_name = adjusted_name['company_name']
    with pytest.raises(ValueError):
        run_nm._match_names_check_data(adjusted_name, column, group_column)

        
@pytest.mark.parametrize("series, column, group_column",
                        [[False, 'company_name', ''],
                        [True, 'company_name', '']
                        ]) 
def test_match_names_check_data(adjusted_name, series, column, group_column):
    if series:
        adjusted_name = adjusted_name['company_name']
    data = run_nm._match_names_check_data(adjusted_name, column, group_column)
    assert 'name_matching_data' in data
    assert type(data) == pd.DataFrame


@pytest.mark.parametrize("case_sensitive, punctuation_sensitive, special_character_sensitive, result_1, result_2",
                        [[True, True, True, 'Ösinski-Schinner', 'Osinski-Schinneg'],
                        [False, True, True, 'ösinski-schinner', 'osinski-schinneg'],
                        [True, False, True, 'ÖsinskiSchinner', 'OsinskiSchinneg'],
                        [True, True, False, 'Osinski-Schinner', 'Osinski-Schinneg'],
                        [False, False, True, 'ösinskischinner', 'osinskischinneg'],
                        [False, True, False, 'osinski-schinner', 'osinski-schinneg'],
                        [True, False, False, 'OsinskiSchinner', 'OsinskiSchinneg'],
                        [False, False, False, 'osinskischinner', 'osinskischinneg']
                        ]) 
def test_match_names_preprocess_data(original_name, adjusted_name, case_sensitive, punctuation_sensitive, special_character_sensitive, result_1, result_2):
    data_a, data_b = run_nm._match_names_preprocess_data('company_name', original_name, adjusted_name, case_sensitive, punctuation_sensitive, special_character_sensitive)
    assert data_a['company_name'][784] == result_1
    assert data_b['company_name'][784] == result_2

@pytest.mark.parametrize("n_equal",
                        [341, 342])
def test_match_names_combine_data(original_name, adjusted_name, n_equal):
    if n_equal ==342:
        adjusted_name.loc[89,'company_name'] = original_name.loc[89,'company_name']
    data_a = run_nm._match_names_combine_data(original_name, adjusted_name, 'company_name', 'company_name')
    assert data_a['score'][784] == 100