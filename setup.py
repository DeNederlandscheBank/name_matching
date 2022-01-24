from setuptools import setup

setup(
   name='name_matching',
   version='0.8',
   description='A package for the matching of company names',
   author='Michiel Nijhuis',
   author_email='m.nijhuis@ndb.nl',
   packages=['name_matching'],
   install_requires = ['abydos',
			'cleanco',
			'numba',
			'scikit-learn', 
			'tqdm'],
)