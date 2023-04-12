[![name_matching](https://github.com/DeNederlandscheBank/name_matching/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/DeNederlandscheBank/name_matching/actions/workflows/python-app.yml)

# Name matching

Name matching is a Python package for the matching of company names. This package has been developed to match the names of companies from different databases together to allow them to be merged. The package has a number of options to determine how exact the matches should be and also for the selection of different name matching algorithms.

For a more in-depth discussion of the name matching package please see the [company name matching](https://medium.com/dnb-data-science-hub/company-name-matching-6a6330710334) medium post


## Installation

The package can be installed via PiPy:

```bash
pip install name_matching
```

Alternatively you could install the package by downloading the repo, navigating to the folder and run the setup in pip locally

```bash
pip install .
```

## Usage

To see example usage of the package you can use the notebook folder. An example of the usage is also given below
```python
import pandas as pd
from name_matching.name_matcher import NameMatcher

# define a dataset with bank names
df_companies_a = pd.DataFrame({'Company name': [
        'Industrial and Commercial Bank of China Limited',
        'China Construction Bank',
        'Agricultural Bank of China',
        'Bank of China',
        'JPMorgan Chase',
        'Mitsubishi UFJ Financial Group',
        'Bank of America',
        'HSBC',
        'BNP Paribas',
        'Crédit Agricole']})

# alter each of the bank names a bit to test the matching
df_companies_b = pd.DataFrame({'name': [
        'Bank of China Limited',
        'Mitsubishi Financial Group',
        'Construction Bank China',
        'Agricultural Bank',
        'Bank of Amerika',
        'BNP Parisbas',
        'JP Morgan Chase',
        'HSCB',
        'Industrial and Commercial Bank of China',
        'Credite Agricole']})

# initialise the name matcher
matcher = NameMatcher(number_of_matches=1, 
                      legal_suffixes=True, 
                      common_words=False, 
                      top_n=50, 
                      verbose=True)

# adjust the distance metrics to use
matcher.set_distance_metrics(['bag', 'typo', 'refined_soundex'])

# load the data to which the names should be matched
matcher.load_and_process_master_data(column='Company name',
                                     df_matching_data=df_companies_a, 
                                     transform=True)

# perform the name matching on the data you want matched
matches = matcher.match_names(to_be_matched=df_companies_b, 
                              column_matching='name')

# combine the datasets based on the matches
combined = pd.merge(df_companies_a, matches, how='left', left_index=True, right_on='match_index')
combined = pd.merge(combined, df_companies_b, how='left', left_index=True, right_index=True)

```

## Contributing
All contributions are welcome. For more substantial changes, please open an issue first to discuss what you would like to change.

## License
The code is licensed under the MIT/X license an extended version of the licence: [MIT](https://choosealicense.com/licenses/mit/)

## Thanks
Thanks to the work of implementing name matching algorithms done in the [Abydos package](https://github.com/chrislit/abydos). These form the basis of the name matching algorithms used in this package.
