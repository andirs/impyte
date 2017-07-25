"""
Module to impute missing values using machine learning algorithms.
author: Andreas Rubin-Schwarz
"""

import pandas as pd


class Imputer:
    """
    Value imputation class.
    
    Parameters
    ----------
    data = pandas DataFrame
    
    Examples
    ----------
    Importing DataFrame from numpy ndarray:
    
    >>> import numpy as np
    >>> imputer = Imputer(np.random.randint(low=0, high=10, size=(4,4)))
    >>> imputer
       0  1  2  3
    0  1  5  1  1
    1  1  9  9  4
    2  5  7  2  1
    3  9  7  5  3
    """

    def __init__(self, data=None):
        if data is None:
            self.data = {}

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            try:
                self.data = pd.DataFrame(data)
            except ValueError as e:
                print "Value Error: {}".format(e)
                pass

    def __str__(self):
        if self.data is not None:
            return str(self.data)
