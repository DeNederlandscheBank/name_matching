from setuptools import setup

setup(
   name='name_matching',
   version='0.8',
   description='A package for the matching of company names',
   author='Michiel Nijhuis',
   author_email='m.nijhuis@ndb.nl',
   packages=['name_matching'],
   python_requires='<3.9',
   install_requires = ['abydos==0.5.0',
			'cleanco',
			'numba',
			'scikit-learn', 
         'pandas',
         'numpy',
			'tqdm'],
)
