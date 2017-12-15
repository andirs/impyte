# impyte

[![Documentation Status](https://readthedocs.org/projects/impyte/badge/?version=latest)](http://impyte.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/andirs/impyte.svg?branch=master)](https://travis-ci.org/andirs/impyte)

Python module to impute missing values by prediction using machine learning algorithms.

## Documentation
A full documentation can be found on [ReadTheDocs](https://impyte.readthedocs.org) or in `docs/_build/html/index.html`. The symlink `documentation.html` in the root directory leads to this file. 

For additional tutorials and usage scenarios please head over to `tutorials` where you'll find a static tutorial version as well as an interactive jupyter notebook.

## Value Imputation
One essential problem for any person dealing with data is missing values. There are several possibilities to deal with missing information, ranging from dropping data points to estimating the value based on other values in that column (i.e. average or median values). A more recent method involves machine-learning algorithms. This module offers a lightweight Python solution to calculate missing information based on the underlying relationship between data points.

## Requirements
- [Python 3.6](https://www.python.org/)
- [sklearn 0.19](https://scikit-learn.org/)
- [pandas 0.21](http://pandas.pydata.org/)
- [scipy 0.19](https://www.scipy.org/)
- [pathlib 1.0.1](https://pypi.python.org/pypi/pathlib/)

## Files
Below are the most important files and a quick one line summary:

- `docs/`
    - `_build/html/index.html` - static documentation
- `impyte/`
    - `impyte.py` - contains main classes
- `requirements.txt` - requirements file, install dependencies with `pip install -r requirements.txt` 
- `tests/`
    - `testing.ipynb` - interactive testing notebook
    - `testing.html` - html version of jupyter notebook
    - `test_impyte.py` - automated pytest script
- `tools/` - contains scripts for development (i.e. fake data generation)
- `tutorials/`
    - `tutorials.ipynb` - notebook with common tutorial tasks
    - `tutorials.html` - static html version of notebook

## Functions
impyte focuses on two main goals: 

1) Easy to interpret visualization of missing patterns
2) Easy imputation of missing values

## Usage

    df = pd.read_csv("missing_values.csv")
    imp = impyte.Impyter(df)

    # show nan-patterns of data in one data frame
    imp.pattern() # shows nan-patterns

    # imputation of all single-nans using random forest
    imp.impute(estimator='rf')

    # imputation of all nan-patterns
    imp.impute(estimator='rf', multi_nans=True)
    
    # use f1 and r2 thresholds
    imp.impute(estimator='rf', threshold={"r2": .7, "f1_macro": .7})

## Limits and Notes
The current version is a work in progress. If you discover any errors or bugs don't hesitate to reach out!
