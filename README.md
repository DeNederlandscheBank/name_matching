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

[fuzzychinese](https://github.com/znwang25/fuzzychinese)
```bash
pip install fuzzychinese
```

## Usage
The user should upload it's database(recommend in jsonl form) with those required information. Import them into the code. When run the code, the user will be asked to enter a company name, when finished enter the name press<kbd>return</kbd>, the name matching package will run and give you the best match.
input: Alibaba
output:alibabacom


## Contributing
All contributions are welcome. For more substantial changes, please open an issue first to discuss what you would like to change.

## License
The code is licensed under the MIT/X license an extended version of the licence: [MIT](https://choosealicense.com/licenses/mit/)

## Thanks
Thanks to the work of implementing name matching algorithms done in the [Abydos package](https://github.com/chrislit/abydos),the base of this program [name-matching](https://github.com/DeNederlandscheBank/name_matching), and contributors of fuzzychinese[fuzzychinese](https://github.com/znwang25/fuzzychinese). These form the basis algorithms used in this package.
