"""
Module to impute missing values using machine learning algorithms.
author: Andreas Rubin-Schwarz
"""

import math
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from datetime import date


class Imputer:
    """
    Value imputation class.
    
    Parameters
    ----------
    data = pandas DataFrame
    clf = machine learning estimator 
    
    Examples
    ----------
    Importing DataFrame from numpy ndarray:
    
    >>> imputer = Imputer(np.random.randint(low=0, high=10, size=(4,4)))
    >>> imputer
       0  1  2  3
    0  1  5  1  1
    1  1  9  9  4
    2  5  7  2  1
    3  9  7  5  3
    
    Testing list for NaN values
    
    >>> nan_array = ["Test", None, '', 23, [None, "42"]]
    >>> imputer = Imputer()
    >>> print imputer.is_nan(nan_array)
    
    [False, True, True, False, [True, False]]
    
    """

    def __init__(self, data=None):
        # check if data is set in constructor otherwise load empty set.
        if data is None:
            self.data = pd.DataFrame()
        else:
            self.data = self._data_check(data)
        # initialize machine learning estimator
        self.clf = {}


    @staticmethod
    def _data_check(data):
        # perform instance check on data if available in constructor
        if isinstance(data, pd.DataFrame):
            return data
        # if data is not a DataFrame, try turning it into one
        else:
            try:
                return_data = pd.DataFrame(data)
                return return_data
            except ValueError as e:
                print "Value Error: {}".format(e)
                return pd.DataFrame()


    def __str__(self):
        """
        String representation of Imputer class.
        :return: stored DataFrame
        """
        if self.data is not None:
            return str(self.data)

    def load_data(self, data):
        """
        Function to load data into Imputer class.
        
        :param data: preferably pandas DataFrame 
        :return: pandas DataFrame
        """
        self.data = self._data_check(data)

    def is_nan(self,
               data,
               nan_vals=None,
               recursive=True):
        """
        Detect missing values (NaN in numeric arrays, empty strings in string arrays).
        
        Parameters
        ----------
        data : ndarray
        nan_vals : array of values that count as NaN values - if empty, "" and None are being used
        recursive : handle lists in recursive manner

        Returns
        -------
        result : array-like of bool or bool
            Array or bool indicating whether an object is null or if an array is
            given which of the element is null.
        """

        # Use immutable object as standard object
        if nan_vals is None:
            nan_vals = ["", None]

        result = []
        # if data is a list, evaluate all objects
        if isinstance(data, list) or hasattr(data, '__array__') and not np.isscalar(data):
            for item in data:
                # if item is a list, call function recursively
                if isinstance(item, list) or hasattr(item, '__array__'):
                    if recursive:
                        result.append(self.is_nan(item))
                    else:
                        raise ValueError("List in a list detected. Set recursive to True.")
                # If item is string, evaluate if empty
                elif isinstance(item, basestring):
                    if item in nan_vals:
                        result.append(True)
                    else:
                        result.append(False)
                # Check for None value explicitly
                elif item is None:
                    result.append(True)
                # Check if NaN when float
                elif isinstance(item, float) and math.isnan(item):
                    result.append(True)
                else:
                    result.append(False)
            # if result is not a list, turn into a list
            if isinstance(result, list):
                return result
            else:
                return np.array(result)
        # if data is not a list, evaluate single value
        elif isinstance(data, basestring):
            return data in nan_vals
        elif data is None:
            return True
        else:
            return math.isnan(data)


    @staticmethod
    def nan_check(row):
        """
        Function to evaluate row on its NaN value patterns.
        Works with is_nan function to determine whether a value is empty or not.

        Parameters
        ----------
        row: any row of a data set

        Returns
        -------
        tuple with pattern indicator
        """
        tmp_label = []
        for idx, value in enumerate(row):
            # For each value, check if NaN
            if self.is_nan(value):
                # Add NaN-indicator to label
                tmp_label.append('NaN')
            else:
                # Add complete-indicator to label
                tmp_label.append(1)

        return tuple(tmp_label)

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
