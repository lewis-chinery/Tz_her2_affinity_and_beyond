from setuptools import setup, find_packages

setup(
    name='Tz_her2_affinity_and_beyond',
    version='1.0.0',
    description='Design and optimisation of high affinity antibody libraries',
    license='BSD 3-clause license',
    maintainer='Lewis Chinery',
    long_description_content_type='text/markdown',
    maintainer_email='lewis.chinery@dtc.ox.ac.uk',
    packages=find_packages(include=('src', 'src.*')),
    install_requires=[
        'blosum',
        'numpy',
        'pandas',
        'logomaker',
        'ablang',
        'fair-esm',
        'ipykernel',
        'tensorflow',
        'scikit-learn',
        'seaborn',
        'flaml==2.1.1',
        'xgboost==1.7.6',
        'lightgbm==4.2.0',
    ],
)