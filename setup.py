from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
   name='name_matching',
   version='0.8.2',
   description='A package for the matching of company names',
   author='Michiel Nijhuis',
   author_email='m.nijhuis@ndb.nl',
   packages=['name_matching'],
   python_requires='<3.10',
   install_requires = ['abydos==0.5.0',
			'cleanco',
			'numba',
			'scikit-learn', 
         'pandas',
         'numpy',
			'tqdm'],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
