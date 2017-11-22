# impyte
Python module to impute missing values by prediction using machine learning algorithms.

## Value Imputation
One essential problem for any person dealing with data is missing values. There are several possibilities to deal with missing information, ranging from dropping data points to estimating the value based on other values in that column (i.e. average or median values). Another more recent method involves machine-learning algorithms. This module offers a lightweight Python module, to calculate missing information based on the underlying relationship between data points.

## Requirements
- [Python 2.7](https://www.python.org/)
- [sklearn 0.19](https://scikit-learn.org/)
- [pandas 0.21](http://pandas.pydata.org/)

## Files
- `impyte.py` - contains main classes
- `sandbox.ipynb` - documentation and examples
- `testing.ipynb` - additional functionality tests
- `tools/` - secondary tools for module
- `requirements.txt` - requirements file, install dependencies with `pip install -r requirements.txt` 

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
    imp.impyte(estimator='rf')

    # imputation of all nan-patterns
    imp.impyte(estimator='rf', multi_nans=True)
    
    # use f1 and r2 thresholds
    imp.impyte(estimator='rf', threshold=[.7, .7])  # f1, r2

## Limits and Notes
The current version is a work in progress. First imputation methods are implemented but pre-processing steps need additional documentation. 

## Python 3.6
There is a Python 3.6 version under development. To use impyte with python3.6 use impyte3.py. All operations work the same as with impyte. Future development might concentrate only on Python3.6>=.
