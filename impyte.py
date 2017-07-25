"""
Module to impute missing values using machine learning algorithms.
author: Andreas Rubin-Schwarz
"""

import pandas as pd
from sklearn.externals import joblib
from datetime import date


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
        self.clf = {}

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

    def load_model(self, model):
        """
        Load a stored machine learning model to perform value imputation.
        :param model: pickle object or filename of model. 
        """
        try:
            self.clf = joblib.load(model)
        except IOError as e:
            print "File not found: {}".format(e)

    def save_model(self, name=None):
        """
        Save a learned machine learning model to disk.
        :param name: Name of file.  
        """
        if name is None:
            name = str(date.today()) + "-impyte-mdl.pkl"
            print name
        joblib.dump(self.clf, name)
