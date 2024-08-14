# Please Note: This package is in progress!

# Name matching for company
This is a forked repository from:
[![name_matching](https://github.com/DeNederlandscheBank/name_matching/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/DeNederlandscheBank/name_matching/actions/workflows/python-app.yml)

This is a package that use to match company names in your database. You should be providing:
        - a database that has all company names, and
        - It's aliases and search string

This package only supports any alphabetical language and Chinese, please make sure to not contain any other language in the database.




## Requirements

The following package needs to install before running this package.

fuzzychinese[([[https://github.com/DeNederlandscheBank/name_matching/actions/workflows/python-app.yml/badge.svg?branch=main]]([https://github.com/znwang25/fuzzychinese/tree/master]))]
```bash
pip install fuzzychinese
```

2.jieba
```bash
pip install jieba
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
